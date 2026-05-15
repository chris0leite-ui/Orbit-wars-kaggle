"""iter — fast-iteration fork of v7_pv (= v7_0_drop_one + PV_GAMMA=0.99).

Day-zero behaviour: functionally equivalent to v7_pv (ladder mu=1064.4).
Edit the knobs below to A/B variants; add code under the PRE_FILTER /
POST_PROCESS hooks to fix specific bugs observed in live replays.

Eval cycle:
    python fast.py eval iter --vs-panel default --max-seeds 32       # 2P
    python -m scripts.ffa_panel --focals agents/iter/main.py --seeds 32   # 4P

Bundle + parity:
    python scripts/bundle_agent.py agents/iter
    pytest tests/test_iter_agent.py tests/test_bundle.py -q

See agents/iter/README.md for the four patch surfaces and submission gate.
"""

from __future__ import annotations

# ============================================================================
# ITER KNOBS — edit these for a knob sweep, one line per variant.
# ============================================================================
K = 10                          # lookahead horizon (8 / 10 / 12 / 15)
WALLCLOCK_MS = 700.0            # per-turn budget (ms); matches iter_v1. Worst-case wallclock =
                                # WALLCLOCK_MS + K_CAP×20ms = 700+280 = 980 ms, safely under Kaggle's 1000.
ENUMERATOR_MODE = "drop_one"    # see lib.v7_search proposers
OPP_TIERS = (1,)                # opp-model tier(s); >1 entry => MAXIMIN
PV_GAMMA = 0.99                 # 1.0 = v7_0_drop_one; 0.99 = v7_pv equivalent
VALUE_FN = "composite"          # "default" | "composite" | "defensibility" | "composite_plus_defensibility"
                                # | "territory" | "composite_plus_territory"
DEFENSIBILITY_ALPHA = 0.2       # SMALL coefficient — V2 α=1.0 over-penalised; V3 uses defens as tiebreaker only
TERRITORY_WEIGHT = 0.01         # production×hold sums to ~5k-10k mid-game; 0.01 keeps the term ≈ ±50, comparable to delta
K_4P = 8                        # 4P-branch lookahead (choose_4p default); kept separate from K (2P)

# --- Adaptive K (Option A — 2026-05-15) -------------------------------------
K_CAP = 14                      # ceiling on effective K. Set by wallclock math: WALLCLOCK_MS=700 +
                                # K_CAP×20ms must stay under Kaggle's 1000 ms hard cap. K_CAP=14 ⇒
                                # worst-case 980 ms. Two-phase scoring would unlock K_CAP=20+ — parked.
K_BUFFER = 2                    # extra steps past max in-flight ETA, for post-arrival evaluation
RELEVANCE_PROD_FRACTION = 0.5   # treat planets with production >= max_production × this as "high-value"
                                # and count fleets targeting them in K_eff. 0.5 = top half by prod.

# --- Comet anti-panic (Bug 2 POST-PROCESS) ----------------------------------
COMET_MAX_LAUNCHES_PER_TURN = 999  # cap effectively disabled; ablation showed it costs 8-9 pp
                                    # vs v7_0/v4_planner (chooser's multi-source choice was usually right)

# --- Comet evacuation (Bug 3 PRE-FILTER) ------------------------------------
COMET_EVAC_THRESHOLD = 5        # if our comet has < N steps remaining, evac ships off it
COMET_EVAC_RESERVE = 1          # min garrison left on the comet after evac
# ============================================================================

# Dev-mode: override lib.scoring.PV_GAMMA BEFORE v7_search imports propagate
# the `from lib.scoring import PV_GAMMA` bindings into snipe/reinforce.
# Bundled form: lib.scoring is not a separate module (concatenated above),
# so this import raises ImportError and we rely on the module-scope
# PV_GAMMA rebind above — every callsite looks PV_GAMMA up by name at
# call time (verified across snipe.py + reinforce.py).
try:
    import lib.scoring as _scoring
    _scoring.PV_GAMMA = PV_GAMMA
except ImportError:
    pass

from lib.v7_search import choose, choose_4p
from lib.intent import World
from lib.world_model import fleet_target_planet, comet_remaining_lifetime, fleet_speed
import math


def _resolve_value_fn(name):
    if name == "default":
        return None  # lib.v7_search.score_candidate defaults to delta_us_minus_them
    if name == "composite":
        from lib.value_heads import composite_capture_value
        return composite_capture_value
    if name == "defensibility":
        from lib.value_heads import defensibility_value
        return lambda obs, mid: defensibility_value(obs, mid, weight=DEFENSIBILITY_ALPHA)
    if name == "composite_plus_defensibility":
        from lib.value_heads import composite_plus_defensibility
        return lambda obs, mid: composite_plus_defensibility(
            obs, mid, defensibility_weight=DEFENSIBILITY_ALPHA
        )
    if name == "territory":
        from lib.value_heads import territory_value
        return lambda obs, mid: territory_value(obs, mid, weight=TERRITORY_WEIGHT)
    if name == "composite_plus_territory":
        from lib.value_heads import composite_plus_territory
        return lambda obs, mid: composite_plus_territory(
            obs, mid, territory_weight=TERRITORY_WEIGHT
        )
    raise ValueError(f"unknown VALUE_FN: {name!r}")


def _detect_num_seats(world) -> int:
    """Infer seat count from owner IDs on planets + in-flight fleets.

    Inline reimplementation of `lib.v7_search._infer_num_seats` so we don't
    reach into a private helper. 2P if max non-neutral owner ID is ≤ 1;
    4P if max is ≥ 2.
    """
    max_id = -1
    for p in world.planets_by_id.values():
        if p.owner >= 0 and p.owner > max_id:
            max_id = p.owner
    raw = world.obs_raw
    fleets = raw.get("fleets", []) if isinstance(raw, dict) else getattr(raw, "fleets", [])
    for f in fleets:
        owner = int(f[1])
        if owner >= 0 and owner > max_id:
            max_id = owner
    return 4 if max_id >= 2 else 2


def _fleets_raw_from_world(world):
    raw = world.obs_raw
    return raw.get("fleets", []) if isinstance(raw, dict) else getattr(raw, "fleets", [])


def _relevant_target_ids(world, prod_fraction: float) -> set:
    """Planet ids whose state changes matter to our short-term decisions.

    Includes (a) every planet we own (defensibility), (b) every planet with
    production >= prod_fraction × max_production (high-value to capture or
    deny). Excludes obscure low-prod neutrals far from action.
    """
    my_id = world.my_id
    planets = list(world.planets_by_id.values())
    if not planets:
        return set()
    max_prod = max(p.production for p in planets) or 1
    threshold = float(max_prod) * float(prod_fraction)
    out = set()
    for p in planets:
        if p.owner == my_id or float(p.production) >= threshold:
            out.add(int(p.id))
    return out


def _max_inflight_eta(world) -> int:
    """Max ETA of in-flight fleets whose RAY-CAST target is "relevant".

    Relevance filter: we extend K_eff only when there's a fleet inbound to
    one of OUR planets (defensibility) or to a high-production planet
    (capture/denial). Long fleets headed for obscure corners don't inflate
    K — so we can afford a higher K_CAP without blowing wallclock.

    Cost: O(N_fleets * N_planets) for the ray-cast loop + a single relevance
    set construction. Both cheap.
    """
    from kaggle_environments.envs.orbit_wars.orbit_wars import Fleet
    fleets_raw = _fleets_raw_from_world(world)
    if not fleets_raw:
        return 0
    planets_list = list(world.planets_by_id.values())
    if not planets_list:
        return 0
    relevant = _relevant_target_ids(world, RELEVANCE_PROD_FRACTION)
    max_eta = 0
    for f_raw in fleets_raw:
        try:
            f = Fleet(*f_raw)
        except TypeError:
            continue
        tgt, eta = fleet_target_planet(f, planets_list)
        if tgt is None or eta is None:
            continue
        if int(tgt.id) not in relevant:
            continue
        if int(eta) > max_eta:
            max_eta = int(eta)
    return max_eta


def _launch_eta(src_planet, angle_rad: float, ships: int, planets_list) -> tuple:
    """Predicted (target_planet, eta) for a NEW launch we're about to make.

    The env ray-casts every launch along its angle; we mirror that to map
    a chosen launch back to a target. Builds a synthetic Fleet at the
    source planet's centre + chosen angle + ship count.
    """
    from kaggle_environments.envs.orbit_wars.orbit_wars import Fleet
    spd = fleet_speed(ships)
    if spd <= 0:
        return None, None
    # Fleet schema (per env): id, owner, x, y, angle, speed, ships, ...
    # We only need fields ray-cast reads: x, y, angle, ships.
    # Build a partial Fleet with the minimum fields the helper consumes.
    f = Fleet(-1, int(src_planet.owner), float(src_planet.x), float(src_planet.y),
              float(angle_rad), float(spd), int(ships))
    return fleet_target_planet(f, planets_list)


def _cap_comet_launches(action, world, cap: int):
    """Strip excess launches targeting the SAME comet (Bug 2 POST-PROCESS).

    Action entries are `[src_id, angle_rad, ships]`. For each launch, ray-
    cast to find its predicted target via `_launch_eta`. Group by target
    planet id; for targets in `world.comet_ids` keep at most `cap` entries
    (prefer shortest ETA = first to arrive). Non-comet targets pass through.

    Hard fallback: if `cap >= len(action)` or no comets present, no-op.
    """
    if not action or cap <= 0:
        return action
    comet_ids = world.comet_ids
    if not comet_ids:
        return action
    planets_list = list(world.planets_by_id.values())
    by_comet: dict = {}      # target_id -> list of (eta, idx)
    other_idx = []
    for i, entry in enumerate(action):
        try:
            src_id, angle, ships = int(entry[0]), float(entry[1]), int(entry[2])
        except (ValueError, TypeError, IndexError):
            other_idx.append(i)
            continue
        src = world.planets_by_id.get(src_id)
        if src is None:
            other_idx.append(i)
            continue
        tgt, eta = _launch_eta(src, angle, ships, planets_list)
        if tgt is None or tgt.id not in comet_ids:
            other_idx.append(i)
            continue
        by_comet.setdefault(tgt.id, []).append((int(eta), i))
    if not by_comet:
        return action
    keep_idx = set(other_idx)
    for tid, entries in by_comet.items():
        entries.sort(key=lambda x: x[0])  # shortest ETA first
        keep_idx.update(idx for _, idx in entries[:cap])
    return [a for i, a in enumerate(action) if i in keep_idx]


def _comet_evacuation_launches(world, my_id: int):
    """Emit launches FROM our short-lifetime comets to a safe destination.

    For each planet we own that is also a comet with
    `remaining_lifetime <= COMET_EVAC_THRESHOLD`: emit a launch carrying
    `ships - COMET_EVAC_RESERVE` ships at the angle pointing to the
    nearest non-comet planet. Skip if no non-comet target exists or if
    the comet has too few ships.
    """
    launches = []
    sources = [
        p for p in world.planets_by_id.values()
        if p.owner == my_id and p.id in world.comet_ids
    ]
    if not sources:
        return launches
    non_comet_targets = [
        p for p in world.planets_by_id.values()
        if p.id not in world.comet_ids
    ]
    if not non_comet_targets:
        return launches
    for c in sources:
        rem = comet_remaining_lifetime(c.id, world)
        if rem is None or rem > COMET_EVAC_THRESHOLD:
            continue
        if int(c.ships) <= COMET_EVAC_RESERVE:
            continue
        # Nearest non-comet planet (any owner — destination is "off the comet").
        best = None
        best_d2 = None
        for t in non_comet_targets:
            dx = t.x - c.x
            dy = t.y - c.y
            d2 = dx * dx + dy * dy
            if best_d2 is None or d2 < best_d2:
                best_d2 = d2
                best = t
        if best is None:
            continue
        angle = math.atan2(best.y - c.y, best.x - c.x)
        ships_out = int(c.ships) - COMET_EVAC_RESERVE
        launches.append([int(c.id), float(angle), int(ships_out)])
    return launches


def agent(obs, configuration=None):
    world = World.from_obs(obs)

    # ------------------------------------------------------------------
    # PRE-FILTER HOOK — Bug 3: comet evacuation
    # Compute evacuation launches from owned comets about to leave the
    # board. Merged into the action AFTER the chooser runs, but only for
    # source planets the chooser didn't already use (avoid double-spend).
    # ------------------------------------------------------------------
    evac_launches = _comet_evacuation_launches(world, world.my_id)

    # ------------------------------------------------------------------
    # Adaptive K (Option A): scale lookahead to cover the longest in-flight
    # fleet. The wallclock watchdog inside choose() caps total cost.
    # ------------------------------------------------------------------
    max_eta = _max_inflight_eta(world)
    K_eff = min(K_CAP, max(K, max_eta + K_BUFFER))
    K_eff_4p = min(K_CAP, max(K_4P, max_eta + K_BUFFER))

    # Dispatch: 2P uses iter_v1's validated choose(enumerator_mode="drop_one",
    # opp_tiers=[1]) path. 4P uses choose_4p() instead of falling back to the
    # v3.5.1 incumbent (the pre-fix behaviour). choose_maximin is NOT used in
    # 2P because v7.1+ maximin variants regressed historically; we preserve
    # iter_v1's 2P behaviour exactly while adding 4P competence.
    n_seats = _detect_num_seats(world)
    value_fn = _resolve_value_fn(VALUE_FN)
    if n_seats == 4:
        action = choose_4p(
            obs, configuration,
            K=K_eff_4p,
            wallclock_ms=WALLCLOCK_MS,
            include_recapture=True,
            value_fn=value_fn,
        )
    else:
        action = choose(
            obs, configuration,
            enumerator_mode=ENUMERATOR_MODE,
            K=K_eff,
            wallclock_ms=WALLCLOCK_MS,
            opp_tiers=list(OPP_TIERS),
            value_fn=value_fn,
        )

    # ------------------------------------------------------------------
    # POST-PROCESS HOOK — Bug 2: cap launches per comet target
    # Multi-source pile-on at a single comet is the panic; cap at
    # COMET_MAX_LAUNCHES_PER_TURN (shortest ETA wins the slots).
    # ------------------------------------------------------------------
    action = _cap_comet_launches(action, world, COMET_MAX_LAUNCHES_PER_TURN)

    # ------------------------------------------------------------------
    # Merge evacuation launches (Bug 3) — only for sources the chooser
    # didn't already launch from. Each `entry[0]` is the src planet id.
    # ------------------------------------------------------------------
    if evac_launches:
        chosen_sources = {int(entry[0]) for entry in action if entry}
        for ev in evac_launches:
            if int(ev[0]) not in chosen_sources:
                action.append(ev)

    return action
