"""The "obvious rules" mechanism layer.

Every mechanism is `Callable[[list[Intent], World], list[Intent]]`. The
canonical pipeline order is below; `DEFAULT_MECHANISMS` matches it.

Pipeline order (rationale comments next to `DEFAULT_MECHANISMS`):

    validate   — drop unsafe intents up front so nothing downstream
                 computes against bad data.
    arrival_size  (3.5.B) — recompute `ships` for enemy targets accounting
                 for production growth during fleet flight. MUST run before
                 `lead_aim` because lead time depends on fleet size, which
                 depends on ship count.
    lead_aim   — populate `aim_angle` with an orbit-aware lead for orbiting
                 non-comet targets; comets and statics fall through to
                 current-position atan2.
    comet_aim  (3.5.C) — populate `aim_angle` for comet targets via
                 path-indexed prediction. Runs AFTER `lead_aim` so lead_aim
                 can no-op on comet targets and let comet_aim own them.
    sun_avoid  (3.5.D) — if direct path crosses the sun, route via waypoint
                 or drop the intent. Runs LAST because it needs the angle
                 set by lead_aim/comet_aim.

Step 3.5.A only includes `validate` and `lead_aim` — the parity-preserving
subset. `arrival_size`, `comet_aim`, `sun_avoid` are added by 3.5.B/C/D.
"""

from __future__ import annotations

import math

from lib.aim import aim_orbiting, swept_pair_hit
from lib.fleet import speed as fleet_speed
from lib.geometry import BOARD_SIZE, path_clears_sun
from lib.intent import Intent, World
from lib.orbit import is_orbiting, predict_relative
from lib.trajectory import predict_fleet_fate
from lib.world_model import WorldModel


# ---------------------------------------------------------------------------
# gang_up_size constants — v3.6 multi-source coordination (Plan: 7-step
# problem-solving iteration). Off by default; opt-in for A/B.
# ---------------------------------------------------------------------------
# Phase-0 idle-source decomposition (audit/2026-05-11-idle-breakdown-v3-snipe-
# phase0.md) showed ~96% of all idle classifications come from `intent.ships
# > src.ships` in validate + arrival_size — a single source can't fund the
# capture alone. The combat rule (lib/combat.py::resolve_arrivals) confirms
# same-owner same-step arrivals sum ships before combat resolution, so two
# small sources arriving simultaneously CAN cover a target neither alone
# could. `gang_up_size` is a new mechanism that runs BEFORE `validate` so
# unaffordable single-source intents survive long enough to be paired.
GANG_UP_ENABLED = 0              # default OFF; opt-in for A/B
GANG_UP_ETA_TOLERANCE = 0        # ±turns allowed in shared-eta match
GANG_UP_MIN_SHARE_THRESHOLD = 2  # min sources to form a gang
GANG_UP_RESERVE = 0              # garrison kept home per source (defense)
GANG_UP_MAX_PASSES = 3           # convergence safety belt


# ---------------------------------------------------------------------------
# Fleet-size over-commit (H19 / TID 697397). Per Gemini's Day-2 writeup,
# sending 10 % more ships than the minimum capture amount gives a
# log-curve speed boost (fleet_speed = 1.0 + (max_speed-1.0) ·
# (log(ships)/log(1000))^1.5) — earlier arrival, more reliable captures,
# small extra defender on arrival. Off by default (= 1.0 identity).
# 2026-05-13 falsified on v7 (3-variant Rule-21 sweep) — the K=10 rollout
# already sizes optimally; pre-rollout inflation drains source garrisons
# without producing the lift Gemini's heuristic-only setup observed.
# Kept as a flag for future use on simpler agents.
FLEET_OVERCOMMIT = 1.0


# ---------------------------------------------------------------------------
# Pre-reinforce window (H21 / TID 698478). The discussion thread describes
# the "you take it, they take it back at no cost" pattern: we capture a
# planet at t=eta with the minimum garrison, then enemy fleet arrives at
# t=eta+1 and recaptures because our post-capture garrison is ~0. The fix
# is to query the WorldModel arrival ledger for enemy fleets landing in
# the window (eta, eta+window] and add buffer to our intent so the
# post-capture garrison survives them. Off by default (= 0). A/B
# candidate values: 2 (one production tick + one arrival), 3, 5.
PRE_REINFORCE_WINDOW = 0


# Extra production-tick buffer applied in `arrival_size` for dynamic targets
# (comets or orbiting planets with non-zero world.omega). Was hard-coded to 1
# at line 194 ("prod_ticks = eta + 1 if is_dynamic"). PI observed under-commit
# by 1 ship on moving planets — `fleet_target_planet` is a non-orbiting
# ray-cast that can be off by "a step or two" (lib/world_model.py:51-54). +2
# covers the worst case; +1 left us bouncing when actual arrival was 2 ticks
# past the ray-cast eta. Sweep candidates: 1 / 2 / 3.
DYNAMIC_PROD_BUFFER = 2


# ---------------------------------------------------------------------------
# validate — drop intents that violate ownership / garrison constraints
# ---------------------------------------------------------------------------


def validate(intents: list[Intent], world: World) -> list[Intent]:
    """Pass through intents whose src is owned and garrison covers ships.

    Drops intents where:
    - src planet is unknown (env hasn't surfaced it),
    - src is not owned by us (strategy bug, defensive),
    - target is the source itself (self-target),
    - ships <= 0 or ships > current src.ships.

    Note: this enforces the *current* garrison sufficiency. If `arrival_size`
    later bumps ships above the garrison, that intent gets dropped at the
    final emission step in `realize()` (ships <= 0 check is the safety net).
    To reject early-bumped-too-large intents BEFORE lead_aim wastes work,
    rerun validate as the final stage (added when arrival_size lands).
    """
    out: list[Intent] = []
    for intent in intents:
        src = world.planets_by_id.get(intent.src_id)
        if src is None:
            continue
        if src.owner != world.my_id:
            continue
        if intent.target_id == intent.src_id:
            continue
        if intent.ships <= 0 or intent.ships > src.ships:
            continue
        out.append(intent)
    return out


# ---------------------------------------------------------------------------
# arrival_size — production-aware fleet sizing for enemy targets
# ---------------------------------------------------------------------------


def arrival_size(intents: list[Intent], world: World, model=None) -> list[Intent]:
    """Bump `ships` so an enemy-owned target's expected garrison at arrival
    is covered, accounting for in-flight adversary fleet stacking.

    Two sources for the "expected garrison at arrival":
    1. **Static estimate** (always available): `target.ships +
       target.production * eta + 1`. Assumes no enemy fleets reach the
       target before we do.
    2. **WorldModel estimate** (when `model` is provided): the simulator
       in `lib/world_model.py` already integrates in-flight adversary
       fleets and same-step combat into `ships_at(target_id, eta)`. This
       is the fix for the v3_snipe bounce-rate doubling
       (audit/2026-05-11-v3-snipe-critical-review.md §4.1): without the
       model, a two-attacker stack walking into our target leaves our
       arriving fleet under-sized by exactly the second attacker's count.

    We take `max(static, model)` so we never go below the static estimate
    (defensive against WorldModel mis-predictions for orbiting planets,
    `lib/world_model.py:46-51`). If `owner_at(target, eta) == world.my_id`
    the planet flips to us en route — drop the intent.

    Neutral targets and our own planets are pass-through.

    The bump is monotonic. If even our full garrison can't cover the
    needed size, drop — sending an under-sized fleet is pure waste.
    """
    out: list[Intent] = []
    for intent in intents:
        src = world.planets_by_id.get(intent.src_id)
        target = world.planets_by_id.get(intent.target_id)
        if src is None or target is None:
            out.append(intent)
            continue
        if target.owner == world.my_id:
            # Reinforce — pass through unchanged (boost would just crowd
            # the home combat resolution).
            out.append(intent)
            continue
        if target.owner == -1:
            # Neutral target — no garrison growth to integrate, but still
            # apply the over-commit boost so the fleet-speed bonus kicks
            # in and we deposit a non-trivial defender on arrival.
            if FLEET_OVERCOMMIT > 1.0:
                boosted = math.ceil(FLEET_OVERCOMMIT * intent.ships)
                intent.ships = min(boosted, int(src.ships))
            out.append(intent)
            continue
        d = math.hypot(target.x - src.x, target.y - src.y)
        v = fleet_speed(intent.ships)
        eta = math.ceil(d / v) if v > 0 else 0
        # Targeted off-by-one for dynamic targets (orbiting + comets):
        # the swept-pair collision check resolves combat at the entry-turn
        # position, which is one production tick AFTER the production tick
        # computed at eta. For static planets, eta is over-estimated by
        # (r_src + r_target)/v (fleet captures on radius-entry), so adding
        # the extra tick over-sizes. Audit:
        # `audit/2026-05-11-v3-snipe-games-analysis.md` items A + games §5.
        target_tuple = [
            target.id, target.owner, target.x, target.y,
            target.radius, target.ships, target.production,
        ]
        # is_orbiting() is geometric (target sits inside the rotation
        # radius); a target actually MOVES only when world.omega != 0.
        # Match lead_aim's gate (mechanism.py:254) so static-from-zero-omega
        # tests stay unchanged.
        is_dynamic = (
            target.id in world.comet_ids
            or (is_orbiting(target_tuple) and world.omega != 0.0)
        )
        prod_ticks = eta + (DYNAMIC_PROD_BUFFER if is_dynamic else 0)
        static_needed = target.ships + target.production * prod_ticks + 1
        needed = static_needed
        if model is not None:
            pred_owner = model.owner_at(target.id, eta)
            if pred_owner == world.my_id:
                # Already ours by then — let the planner skip.
                continue
            pred_ships = model.ships_at(target.id, eta)
            if pred_ships is not None:
                needed = max(static_needed, int(math.ceil(pred_ships)) + 1)
        intent.ships = max(intent.ships, needed)
        # H21 / [F] pre-reinforce: scan the WorldModel arrival ledger
        # for ENEMY fleets landing in (eta, eta + PRE_REINFORCE_WINDOW]
        # and add buffer to our intent so the post-capture garrison
        # survives the strongest such follow-up (TID 698478, "they
        # take it back at no cost" pattern). Per-target production
        # during the window offsets some of the threat.
        if PRE_REINFORCE_WINDOW > 0 and model is not None:
            strongest = 0
            strongest_eta = eta + 1
            for f_eta, f_owner, f_ships in model.ledger.get(target.id, ()):
                if f_owner == world.my_id:
                    continue
                if eta < f_eta <= eta + PRE_REINFORCE_WINDOW and f_ships > strongest:
                    strongest = int(f_ships)
                    strongest_eta = int(f_eta)
            if strongest > 0:
                # During (eta, strongest_eta] we own + grow the planet.
                prod_during = int(target.production) * (strongest_eta - eta)
                deficit = strongest - prod_during
                if deficit > 0:
                    intent.ships = max(intent.ships, intent.ships + deficit + 1)
        # 1.1× over-commit AFTER the production-aware sizing. Falsified
        # at v7; preserved as a dormant flag.
        if FLEET_OVERCOMMIT > 1.0:
            boosted = math.ceil(FLEET_OVERCOMMIT * intent.ships)
            intent.ships = min(boosted, int(src.ships))
        if intent.ships > src.ships:
            continue
        out.append(intent)
    return out


# ---------------------------------------------------------------------------
# gang_up_size — multi-source coordination (v3.6)
# ---------------------------------------------------------------------------


def _max_ships_for_eta(distance: float, target_eta: int) -> int:
    """Return the largest ship count whose fleet_speed yields eta == target_eta.

    `fleet_speed(s)` is monotone non-decreasing in s and bounded above
    (max_speed). To get a specific eta we want the LARGEST s such that
    `ceil(distance / fleet_speed(s)) <= target_eta`. Larger ships → faster →
    smaller eta, so we binary-search the upper bound.

    Returns 1 if even 1 ship would still beat target_eta (target_eta is too
    generous; caller should accept the lone ship which arrives earlier).
    Returns 1000 (the saturation point of fleet_speed) if even max_speed
    can't reach target_eta — caller will need to widen tolerance.
    """
    if target_eta <= 0:
        return 1
    lo, hi = 1, 1000
    while lo < hi:
        mid = (lo + hi + 1) // 2
        v = fleet_speed(mid)
        eta = math.ceil(distance / v) if v > 0 else target_eta + 1
        if eta <= target_eta:
            lo = mid
        else:
            hi = mid - 1
    return lo


def gang_up_size(
    intents: list[Intent], world: World, model: WorldModel | None = None,
) -> list[Intent]:
    """Coordinate multiple this-turn intents at the same target so their
    combined ships cover the predicted garrison.

    Default-off (`GANG_UP_ENABLED = 0`): pure pass-through. When enabled,
    runs BEFORE `validate` in `DEFAULT_MECHANISMS` so unaffordable single-
    source intents (which `validate` would otherwise drop on `intent.ships
    > src.ships`) survive long enough to be paired with siblings.

    Algorithm per target group of size >= GANG_UP_MIN_SHARE_THRESHOLD:
    1. Anchor eta = max(eta_solo) across the group. Slower sources can't
       speed up by sending more ships (fleet_speed is bounded), but
       faster sources CAN slow down by sending fewer ships.
    2. needed_total = max(static_at_anchor + 1, model.ships_at(target,
       anchor) + 1). Static = target.ships + production*anchor + 1.
    3. share_i proportional to src_i.ships, capped at src_i.ships -
       GANG_UP_RESERVE. Throttled DOWN so the source arrives at anchor
       (via _max_ships_for_eta).
    4. Up to GANG_UP_MAX_PASSES iterations: any source whose share_i
       implies a higher eta than the anchor → re-anchor up. Cap at 3;
       on failure to converge, drop the gang group (per-intent
       arrival_size handles them individually as today).
    5. Sources with share_i < 1 after capping are dropped from the gang
       and redistributed. If survivors == 1, the lone intent exits
       gang-up unmodified (preserves sole-source bit-identity).

    Single-intent targets pass through unmodified — sole-source path is
    a no-op (assertion enforced by test_gangup_sole_source_noop).
    """
    if not GANG_UP_ENABLED or not intents:
        return intents

    # Bucket intents by target_id.
    by_target: dict[int, list[Intent]] = {}
    for intent in intents:
        by_target.setdefault(intent.target_id, []).append(intent)

    # Process each multi-source group; collect modified intents.
    modified_intent_ids: set[int] = set()
    for target_id, group in by_target.items():
        if len(group) < GANG_UP_MIN_SHARE_THRESHOLD:
            continue
        target = world.planets_by_id.get(target_id)
        if target is None:
            continue
        # Neutrals don't grow during flight; gang-up is mostly relevant
        # for enemy targets, but we still allow it for neutrals when a
        # single source can't afford the static cost (very rare but
        # possible for early-game large neutrals).
        # Compute per-source eta_solo using their CURRENT intent.ships.
        infos = []
        for intent in group:
            src = world.planets_by_id.get(intent.src_id)
            if src is None:
                continue
            d = math.hypot(target.x - src.x, target.y - src.y)
            v = fleet_speed(intent.ships)
            eta_solo = math.ceil(d / v) if v > 0 else 0
            infos.append({
                "intent": intent, "src": src, "distance": d, "eta": eta_solo,
            })
        if len(infos) < GANG_UP_MIN_SHARE_THRESHOLD:
            continue

        # Convergence loop: anchor on slowest source; shrink faster
        # siblings to match; re-check until stable (or give up).
        converged = False
        for _ in range(GANG_UP_MAX_PASSES):
            anchor_eta = max(info["eta"] for info in infos)
            # needed_total at anchor_eta.
            if target.owner == -1 or target.owner == world.my_id:
                # Neutrals & own planets: no production growth during
                # flight; needed is just target.ships + 1.
                needed_total = max(1, int(target.ships) + 1)
            else:
                static_needed = (
                    int(target.ships)
                    + int(target.production) * anchor_eta
                    + 1
                )
                needed_total = static_needed
                if model is not None:
                    pred_ships = model.ships_at(target.id, anchor_eta)
                    if pred_ships is not None:
                        needed_total = max(
                            static_needed, int(math.ceil(pred_ships)) + 1,
                        )

            # Allocate shares proportional to src.ships, capped by
            # src.ships - reserve.
            total_src_ships = sum(info["src"].ships for info in infos)
            if total_src_ships <= 0:
                converged = True
                break

            new_etas = []
            for info in infos:
                raw_share = math.ceil(
                    needed_total * info["src"].ships / total_src_ships
                )
                cap = max(0, info["src"].ships - GANG_UP_RESERVE)
                # Also cap by _max_ships_for_eta so this source actually
                # arrives at anchor_eta (slower fleets need fewer ships).
                throttle = _max_ships_for_eta(info["distance"], anchor_eta)
                share = min(raw_share, cap, throttle)
                info["share"] = max(0, share)
                # Recompute eta with new share.
                v = fleet_speed(max(1, info["share"]))
                new_eta = math.ceil(info["distance"] / v) if v > 0 else 0
                new_etas.append(new_eta)

            new_anchor = max(new_etas)
            if new_anchor <= anchor_eta + GANG_UP_ETA_TOLERANCE:
                # Update anchor and apply throttles definitively.
                converged = True
                # Apply shares to intents; drop any source with share==0
                # from the gang (re-routed to per-intent arrival_size).
                gang_share_total = 0
                survivors = []
                for i, info in enumerate(infos):
                    info["eta"] = new_etas[i]
                    if info["share"] <= 0:
                        continue
                    survivors.append(info)
                    gang_share_total += info["share"]
                # If combined survivors still cover needed AND we have
                # ≥ min sources, write the throttled ships back to each
                # intent. Otherwise drop the gang (fall back to per-
                # intent arrival_size handling).
                if (
                    len(survivors) >= GANG_UP_MIN_SHARE_THRESHOLD
                    and gang_share_total >= needed_total
                ):
                    for info in survivors:
                        info["intent"].ships = int(info["share"])
                        modified_intent_ids.add(id(info["intent"]))
                break
            # Re-anchor on the new max eta and retry.
            for i, info in enumerate(infos):
                info["eta"] = new_etas[i]

        # If we ran out of passes without converging, leave intents
        # unmodified — per-intent arrival_size will drop them as today.
        _ = converged

    return intents


def _comet_path_lookup(world: World) -> dict[int, tuple[list, int]]:
    """Build {planet_id: (path, path_index)} for every comet in the obs.

    `obs["comets"]` is a list of groups, each `{planet_ids, paths, path_index}`.
    `paths[i]` is the trajectory of `planet_ids[i]` — a list of `[x, y]`
    pairs. `path_index` is shared across the group; advances 1 per turn.
    """
    raw = world.obs_raw
    comets = (
        raw.get("comets", []) if isinstance(raw, dict) else getattr(raw, "comets", [])
    )
    out: dict[int, tuple[list, int]] = {}
    for group in comets:
        if hasattr(group, "keys"):
            planet_ids = list(group["planet_ids"])
            paths = list(group["paths"])
            path_index = int(group["path_index"])
        else:
            planet_ids = list(group.planet_ids)
            paths = list(group.paths)
            path_index = int(group.path_index)
        for idx, pid in enumerate(planet_ids):
            out[int(pid)] = (paths[idx], path_index)
    return out


def comet_aim(intents: list[Intent], world: World) -> list[Intent]:
    """Populate `aim_angle` for comet targets via the path-indexed lead.

    Comets follow pre-computed elliptical paths, NOT the rotation formula —
    so `lead_aim`'s orbit prediction would mis-aim them. This mechanism
    fires on targets in `world.comet_ids`, looks up the comet's path,
    projects to `path_index + eta_turns`, and aims at the projected point.

    If `path_index + eta_turns` exceeds the path length the comet exits
    the board before our fleet arrives — drop the intent (sending an
    on-the-way fleet at an exit-bound comet would be wasted).

    **Status: experimental, NOT in DEFAULT_MECHANISMS.** The 3.5.C ablation
    tournament showed this single-pass version loses 9/40 = 22.5% vs the
    parity baseline. See the rationale comment near `DEFAULT_MECHANISMS`
    for the diagnosis. Kept as a registered mechanism so tournament panels
    can opt it in for future experiments (e.g. paired with a
    `search_safe_intercept` fallback at v3).
    """
    if not world.comet_ids:
        return intents
    paths_by_id = _comet_path_lookup(world)

    out: list[Intent] = []
    for intent in intents:
        if intent.target_id not in world.comet_ids:
            out.append(intent)
            continue
        if intent.aim_angle is not None:
            out.append(intent)
            continue
        src = world.planets_by_id.get(intent.src_id)
        target = world.planets_by_id.get(intent.target_id)
        path_info = paths_by_id.get(intent.target_id)
        if src is None or target is None or path_info is None:
            out.append(intent)
            continue
        path, path_index = path_info
        v = fleet_speed(intent.ships)
        d = math.hypot(target.x - src.x, target.y - src.y)
        eta = math.ceil(d / v) if v > 0 else 0
        future_index = path_index + eta
        if future_index >= len(path):
            # Comet exits before the fleet arrives — drop rather than waste ships.
            continue
        fx, fy = path[future_index]
        intent.aim_angle = math.atan2(fy - src.y, fx - src.x)
        out.append(intent)
    return out


# ---------------------------------------------------------------------------
# lead_aim — orbit-aware lead, ports v1's _aim_angle exactly
# ---------------------------------------------------------------------------


def lead_aim(intents: list[Intent], world: World) -> list[Intent]:
    """Populate `aim_angle` for each intent.

    For orbiting non-comet targets, performs one fixed-point iteration over
    `(arrival_time, predicted_position)` — the same algorithm v1 used in
    its embedded `_aim_angle`. For static planets and comets, falls through
    to atan2 of the current target position. Comet path-indexed leading is
    `comet_aim`'s job (3.5.C); this mechanism intentionally aims comets at
    current position so `comet_aim` can override.

    Intents that already have `aim_angle` set (e.g. by an earlier
    mechanism) are left untouched.
    """
    for intent in intents:
        if intent.aim_angle is not None:
            continue
        src = world.planets_by_id.get(intent.src_id)
        target = world.planets_by_id.get(intent.target_id)
        if src is None or target is None:
            continue

        target_xy = (target.x, target.y)
        target_tuple = [
            target.id, target.owner, target.x, target.y,
            target.radius, target.ships, target.production,
        ]
        is_orbit = (
            is_orbiting(target_tuple)
            and target.id not in world.comet_ids
        )
        if is_orbit and world.omega != 0.0:
            v = fleet_speed(intent.ships)
            # Fleet spawns just outside source (src.radius + 0.1) and
            # captures when it crosses into target.radius. Subtract both
            # from center-to-center distance to get actual flight distance.
            # Without this, ETA overestimates and lead is too far ahead —
            # systematic miss in the orbit-forward direction.
            r_offset = src.radius + target.radius + 0.1
            tx, ty = target.x, target.y
            for _ in range(2):
                d = math.hypot(tx - src.x, ty - src.y)
                flight_d = max(0.0, d - r_offset)
                t = flight_d / v
                tx, ty = predict_relative(target_tuple, world.omega, t)
            target_xy = (tx, ty)
        intent.aim_angle = math.atan2(target_xy[1] - src.y, target_xy[0] - src.x)
    return intents


# ---------------------------------------------------------------------------
# Canonical pipeline
# ---------------------------------------------------------------------------

# Pipeline order rationale:
#   validate      — drop unsafe intents up-front so nothing downstream
#                   computes against bad data.
#   arrival_size  — bump fleet size for enemy targets BEFORE lead_aim/comet_aim
#                   because lead time (and thus the projected position) depends
#                   on fleet size via fleet_speed.
#   lead_aim      — populates aim_angle for everything else (orbiting non-comets
#                   get the orbit-fixed-point lead; statics get plain atan2;
#                   comets get current-position atan2 — see note below).
#   sun_avoid (3.5.D) — last; needs the angle set by lead_aim/comet_aim.
#
# ---------------------------------------------------------------------------
# sun_avoid — drop intents whose straight-line path crosses the sun
# ---------------------------------------------------------------------------


def sun_avoid(intents: list[Intent], world: World) -> list[Intent]:
    """Drop intents whose actual fleet path would intersect the sun.

    2026-05-11 rewrite: now uses `lib.trajectory.predict_fleet_fate` to
    ray-cast the FULL fleet trajectory (not just the segment up to the
    predicted target arrival point). Previous endpoint-only check missed
    sun collisions in the trajectory's overshoot tail when the lead
    prediction misses (orbital drift, tangent shot). Live-replay
    evidence: 3.2% of our fleets died in the sun under the old guard.

    Drop-only — re-routing via a waypoint planet is a v3 mission concern.
    """
    out: list[Intent] = []
    for intent in intents:
        if intent.aim_angle is None:
            out.append(intent)
            continue
        src = world.planets_by_id.get(intent.src_id)
        target = world.planets_by_id.get(intent.target_id)
        if src is None or target is None:
            out.append(intent)
            continue
        fate = predict_fleet_fate(src, target, intent.aim_angle, intent.ships, world)
        if fate.outcome == "sun":
            continue
        out.append(intent)
    return out


# ---------------------------------------------------------------------------
# lead_aim_v2 — 5-iter fixed-point + search_safe_intercept fallback
# ---------------------------------------------------------------------------


def lead_aim_v2(intents: list[Intent], world: World) -> list[Intent]:
    """Populate `aim_angle` AND `arrival_xy` for each intent via the
    public-kernel pattern: 5-iter fixed-point + safe-intercept fallback.

    Differences from the legacy `lead_aim`:
    - 5 iterations (was 2) with explicit XY convergence check.
    - `search_safe_intercept` fallback when the fixed-point doesn't
      converge (orbital targets at long range, where eta oscillates).
    - Populates `intent.arrival_xy` so `sun_avoid`, `path_clears_other_planets`,
      and `oob_guard` downstream can reason about the actual fleet endpoint.
    - For static targets and comets, falls through to atan2 of current
      target position (same as legacy lead_aim; `comet_aim` overrides
      comets when enabled).

    Intents that already have `aim_angle` set are left untouched
    (mechanism ordering: a future planner-set aim shouldn't be clobbered).
    """
    for intent in intents:
        if intent.aim_angle is not None:
            continue
        src = world.planets_by_id.get(intent.src_id)
        target = world.planets_by_id.get(intent.target_id)
        if src is None or target is None:
            continue

        target_tuple = [
            target.id, target.owner, target.x, target.y,
            target.radius, target.ships, target.production,
        ]
        is_orbit = (
            is_orbiting(target_tuple)
            and target.id not in world.comet_ids
        )

        if is_orbit and world.omega != 0.0:
            result = aim_orbiting(
                (src.x, src.y), src.radius,
                target_tuple, target.radius,
                intent.ships, world.omega,
            )
            if result is None:
                # No valid intercept — let realize() drop the intent
                # via the aim_angle=None gate.
                continue
            intent.aim_angle, intent.arrival_xy, _eta = result
        else:
            # Static or comet → aim at current; record arrival_xy for
            # downstream sun/OOB/path checks even though there's no lead.
            intent.aim_angle = math.atan2(target.y - src.y, target.x - src.x)
            intent.arrival_xy = (target.x, target.y)
    return intents


# ---------------------------------------------------------------------------
# path_clears_other_planets — drop intents swept by a non-target planet
# ---------------------------------------------------------------------------


def path_clears_other_planets(intents: list[Intent], world: World) -> list[Intent]:
    """Drop intents whose flight path collides with a non-target planet.

    2026-05-11 rewrite: delegates to `lib.trajectory.predict_fleet_fate`.
    Previous impl simulated only up to the predicted target arrival
    step (~total_dist / speed); the overshoot tail wasn't checked. Now
    we walk the full trajectory.

    Capture-probe (2026-05-10) showed 10.7% non-target-planet collisions
    as the biggest physics-loss bucket. Live-replay (2026-05-11) showed
    the residual was still significant because of the truncated-horizon
    bug. Full-trajectory ray-cast closes that gap.
    """
    out: list[Intent] = []
    for intent in intents:
        if intent.aim_angle is None:
            out.append(intent)
            continue
        src = world.planets_by_id.get(intent.src_id)
        target = world.planets_by_id.get(intent.target_id)
        if src is None or target is None:
            out.append(intent)
            continue
        fate = predict_fleet_fate(src, target, intent.aim_angle, intent.ships, world)
        if fate.outcome == "planet":
            continue
        out.append(intent)
    return out


# ---------------------------------------------------------------------------
# oob_guard — drop intents whose projected endpoint exits the board
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# arrival_ledger — skip intents whose target will already be ours at arrival
# ---------------------------------------------------------------------------


def arrival_ledger(intents: list[Intent], world: World) -> list[Intent]:
    """Drop intents we don't need: target will be ours with enough ships
    at our arrival step.

    Builds a `WorldModel` snapshot (in-flight fleet arrival ledger +
    per-planet timeline) for this turn. For each intent:
    - Estimate arrival step via straight-line dist / fleet_speed.
    - Look up `(predicted_owner, predicted_ships)` at that step.
    - If predicted_owner == us AND predicted_ships >= intent.ships,
      drop the intent — adding another fleet would double-commit.

    Stronger variants (intercept enemy arrivals, gang-up timing) live
    in v3 mission classes; this is the minimum-viable v2 use case.

    Cost: O(planets * horizon + fleets * planets) per turn for the
    WorldModel build (~5 ms on a 40-planet board). Cached for the
    duration of one mechanism call.
    """
    if not intents:
        return intents
    wm = WorldModel.from_world(world)
    out: list[Intent] = []
    for intent in intents:
        src = world.planets_by_id.get(intent.src_id)
        target = world.planets_by_id.get(intent.target_id)
        if src is None or target is None:
            out.append(intent)
            continue
        # ETA: straight-line center-to-center / fleet_speed(intent.ships).
        # Rough — doesn't account for orbital motion of target. Adequate
        # for the "don't double-commit" use case.
        d = math.hypot(target.x - src.x, target.y - src.y)
        v = fleet_speed(intent.ships)
        eta = int(math.ceil(d / max(v, 1e-6))) if v > 0 else 0
        pred_owner = wm.owner_at(target.id, eta)
        pred_ships = wm.ships_at(target.id, eta)
        if (
            pred_owner == world.my_id
            and pred_ships is not None
            and pred_ships >= intent.ships
        ):
            # Already going to be ours with surplus garrison; ship would
            # be wasted on a target we're about to own anyway.
            continue
        out.append(intent)
    return out


def oob_guard(intents: list[Intent], world: World) -> list[Intent]:
    """Drop intents whose fleet path would exit the [0, BOARD_SIZE] box
    before colliding with anything.

    2026-05-11 rewrite: delegates to `lib.trajectory.predict_fleet_fate`.
    Previous impl checked only the predicted endpoint; if the lead-
    prediction missed the target the fleet kept flying past it through
    empty space until OOB, but the guard didn't see that overshoot tail.

    Live-replay evidence (audit/live-episodes/52532938/): 7.5% of our
    fleets flew OOB under the old guard, including one 7-ship fleet
    that travelled 79 units through empty space before exiting the
    board (the predicted target had moved by the time we arrived).
    """
    out: list[Intent] = []
    for intent in intents:
        if intent.aim_angle is None:
            out.append(intent)
            continue
        src = world.planets_by_id.get(intent.src_id)
        target = world.planets_by_id.get(intent.target_id)
        if src is None or target is None:
            out.append(intent)
            continue
        fate = predict_fleet_fate(src, target, intent.aim_angle, intent.ships, world)
        # Drop both `oob` and `timeout` outcomes: a 200-step ray-cast
        # without collision means the fleet doesn't reach anything
        # useful — same effective waste as flying OOB.
        if fate.outcome in ("oob", "timeout"):
            continue
        out.append(intent)
    return out


# 2026-05-10 PM physics upgrade (capture-probe + Roman teardown):
# - `lead_aim_v2` replaces `lead_aim` in DEFAULT_MECHANISMS. 5-iter
#   fixed-point + `search_safe_intercept` fallback (lib/aim.py). Populates
#   `intent.arrival_xy` so downstream checks reason about the actual
#   flight endpoint.
# - `sun_avoid` re-enabled with the punch-#7 fix: uses `intent.arrival_xy`
#   if set (lead-predicted arrival) instead of `target.xy`. Previous
#   regressions are addressed because the check now matches the actual
#   fleet trajectory.
# - `path_clears_other_planets` added: addresses the 10.7% collided_other
#   bucket from the capture probe. Replays the env's swept-pair check
#   against every non-target planet's projected orbital chord.
# - `oob_guard` added: addresses the 7.6% OOB bucket. Drops intents whose
#   projected endpoint exits the board.
# - `comet_aim` remains EXCLUDED pending a comet-gated re-enable
#   (research-note §G.14: gate on `production * expected_lifetime > ships`).
DEFAULT_MECHANISMS = [
    gang_up_size,                 # v3.6: no-op when GANG_UP_ENABLED=0
    validate,
    arrival_size,
    lead_aim_v2,
    sun_avoid,
    path_clears_other_planets,
    oob_guard,
]
# `arrival_ledger` is implemented but EXCLUDED from DEFAULT_MECHANISMS.
# Local A/B showed it regressed WR from 56% to 50% (Block C audit) because
# per-source greedy strategies don't re-pick after the mechanism drops an
# intent: the source planet ends the turn with no action. The mechanism's
# real value materialises when paired with the v3 planner (Block D), which
# can re-allocate the freed ships to a different target/mission. Keep here
# for direct use from the planner; do NOT add to DEFAULT until then.

# Frozen pre-upgrade stack (validate + arrival_size + 2-iter lead_aim only).
# Used by `agents/simple/roi_baseline.py` for A/B against the upgraded
# DEFAULT_MECHANISMS without round-tripping through a bundled submission.
DEFAULT_MECHANISMS_PRE_PHYSICS = [validate, arrival_size, lead_aim]

# Pinned subset for the v1 parity gate — must match pre-refactor v1
# behaviour exactly. Don't add new mechanisms here without bumping the
# pre-refactor snapshot.
PARITY_MECHANISMS = [validate, lead_aim]

__all__ = [
    "DEFAULT_MECHANISMS",
    "DEFAULT_MECHANISMS_PRE_PHYSICS",
    "PARITY_MECHANISMS",
    "validate",
    "arrival_size",
    "gang_up_size",
    "comet_aim",
    "lead_aim",
    "lead_aim_v2",
    "sun_avoid",
    "path_clears_other_planets",
    "oob_guard",
    "arrival_ledger",
]
