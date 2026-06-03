"""Post-emit launch-discipline validator (PI rules, 2026-05-29).

Two rules the champion must GUARANTEE, not merely encourage:

  Rule A — neutral discipline. Never send ships to a NEUTRAL planet
    unless they capture it. Capture by multiple fleets arriving the SAME
    tick from different planets is allowed; staggered pokes are not
    (combat sums only same-tick arrivals — see lib/combat.resolve_arrivals).

  Rule B — opponent predictability ceiling. Only commit to capturing an
    OPPONENT planet if the fleet arrives within K turns (K = the horizon
    of predictability, default 10). Drop captures arriving later than K.

  Universal K ceiling (PI 2026-05-30) — K bounds EVERY launch, not just
    opponent captures: any launch whose fleet arrives after K turns is
    dropped, including neutral captures and reinforcements of our own
    planets (and comet-sourced launches). Beyond K the board state is
    unpredictable, so a far launch routinely lands at a flipped/contested
    planet and loses its (often small) fleet.

This runs as the LAST pass over the fully-assembled move list in
`agents/baseline/main.agent()`, AFTER the chooser, the commit ledger,
threat reinforcements, the three rear-drain helpers and sniper strikes —
the single chokepoint every launch funnels through. A proposer-side prune
(`proposer.propose`) handles efficiency but cannot GUARANTEE the rules,
because drain_*/sniper inject launches past the proposer.

Default OFF (byte-for-byte legacy). Opt-in via `BASELINE_LAUNCH_RULES=1`;
K via `BASELINE_CAPTURE_HORIZON_K` (default 10).

Reuses tested primitives (Rule 47 — no new physics):
  lib.trajectory.predict_fleet_fate    — first planet struck + arrival tick
  lib.world_model.predict_garrison_at  — same-tick combat outcome at arrival
"""

from __future__ import annotations

import os
from types import SimpleNamespace

from lib.trajectory import predict_fleet_fate
from lib.world_model import (
    opp_contest_tick,
    predict_arrival_contest,
    predict_garrison_at,
)

DEFAULT_CAPTURE_HORIZON_K = 10

# Value-driven horizon (2026-06-03): the universal-K ceiling spends the LEAST
# horizon on the most valuable, most-contested planets — so winnable far
# captures get dropped (replay pre-check: ~88 race-win captures dropped in a
# single 2P game). This lever ADMITS captures past K when we win the race
# (race_win) or the target cannot be contested at all (bankable), capped at
# DEFAULT_VALUE_HORIZON_MAX. It only ever RELAXES the ceiling (monotone): a
# launch K already keeps stays kept; nothing K admits is newly dropped. The
# falsified hard race_loss->drop gate is NOT here — race_loss simply keeps the
# existing K behaviour. Default OFF (byte-identical champion).
DEFAULT_VALUE_HORIZON_MAX = 40

# State-driven-K ceiling: the farthest-out arrival a launch may target when
# the lever admits an uncontested planet (design §3 Lever A). K is clamped
# to [floor, this] so the lever can only ever RAISE K above the champion's
# floor for safe targets — never lower it — keeping the change monotone.
DEFAULT_STATE_K_CEIL = 30

# Sentinel target whose id never matches a real planet, so
# predict_fleet_fate reports the FIRST planet actually struck as
# outcome "planet" with hit_planet_id set (its "target" branch only
# fires when pid == target.id). The fate ray-cast reads target.id only.
_SENTINEL_TARGET = SimpleNamespace(id=-999999)


def launch_rules_enabled() -> bool:
    return os.environ.get("BASELINE_LAUNCH_RULES", "0").strip().lower() in (
        "1", "true", "on", "yes",
    )


def _capture_horizon_floor() -> int:
    raw = os.environ.get("BASELINE_CAPTURE_HORIZON_K", "").strip()
    if not raw:
        return DEFAULT_CAPTURE_HORIZON_K
    try:
        return int(raw)
    except ValueError:
        return DEFAULT_CAPTURE_HORIZON_K


def _adaptive_k_enabled() -> bool:
    return os.environ.get("BASELINE_ADAPTIVE_K", "0").strip().lower() in (
        "1", "true", "on", "yes",
    )


def state_driven_k_enabled() -> bool:
    return os.environ.get("BASELINE_STATE_DRIVEN_K", "0").strip().lower() in (
        "1", "true", "on", "yes",
    )


def _state_k_orbital_lead_enabled() -> bool:
    """Orbital-lead horizon model for the state-driven contest tick.

    When on, the per-target contest tick aims the enemy at the target's
    FUTURE (intercept) position instead of its current one — fixing the
    shipped state-K's mis-estimate for orbiting targets — while keeping the
    conservative launch-now timing. Default OFF (byte-identical state-K)."""
    return os.environ.get("BASELINE_STATE_K_ORBITAL_LEAD", "0").strip().lower() in (
        "1", "true", "on", "yes",
    )


def _value_horizon_enabled() -> bool:
    """Value-driven horizon: admit race_win / bankable captures past the K
    ceiling (up to DEFAULT_VALUE_HORIZON_MAX). Only relaxes K; never tightens
    it. Default OFF (byte-identical champion)."""
    return os.environ.get("BASELINE_VALUE_HORIZON", "0").strip().lower() in (
        "1", "true", "on", "yes",
    )


def _value_horizon_max() -> int:
    return _env_int("BASELINE_VALUE_HORIZON_MAX", DEFAULT_VALUE_HORIZON_MAX)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def capture_horizon_k(step=None, *, tgt_id=None, world=None, model=None,
                      me=None) -> int:
    """Predictability ceiling K (launches arriving after K are dropped).

    Static (default, byte-identical champion): returns the floor (env
    ``BASELINE_CAPTURE_HORIZON_K`` or ``DEFAULT_CAPTURE_HORIZON_K`` = 10),
    ignoring every argument.

    State-driven (``BASELINE_STATE_DRIVEN_K=1``): K becomes the predictability
    of the *specific target* rather than a clock —
    ``K_target = clamp(floor, ceil, opp_earliest_contest_tick)``. An
    uncontested target (no enemy can reach it) → ceil (commit long); a target
    the opponent can contest at tick T → K=T (don't commit a fleet that lands
    after the board there changes). Clamped to ``[floor, ceil]`` so the lever
    can only RAISE K above the champion's floor for safe/far targets, never
    lower it — a monotone extension. Requires ``tgt_id`` + ``world`` +
    ``model`` + ``me``; a target-less call (efficiency/global sites) returns
    the ceil so it never pre-drops a launch the per-target gate would admit.
    Ceil via ``BASELINE_STATE_K_CEIL`` (default 30). This is the principled
    form of the adaptive step-schedule below — design §3 Lever A,
    knowledge-base/concepts/contest-aware-conversion-design.md.

    Adaptive step-schedule (``BASELINE_ADAPTIVE_K=1``): K is LARGE early and
    decays linearly to the floor by ``T_SETTLE``. The opening is predictable
    (few in-flight fleets, planets at known positions) so a far launch is
    safe there; as the board fills, K returns to the champion's disciplined
    floor. Empirically the static floor=10 hides ~75% of the opening
    expansion map (median neutral ETA 22) — see
    audit/2026-06-01-adaptive-horizon-k-investigation.md.

      K(step) = max(floor, round(K_OPEN - (K_OPEN-floor)*step/T_SETTLE))

    Tunables (for the A/B sweep): ``BASELINE_ADAPTIVE_K_OPEN`` (default 20),
    ``BASELINE_ADAPTIVE_K_TSETTLE`` (default 30). ``step`` is the current
    game step (``world.step``); when ``None`` or adaptive-off, the floor is
    returned (preserves every existing caller + the static champion).
    """
    floor = _capture_horizon_floor()

    if state_driven_k_enabled():
        ceil = _env_int("BASELINE_STATE_K_CEIL", DEFAULT_STATE_K_CEIL)
        if ceil < floor:
            ceil = floor
        if tgt_id is None or world is None or model is None or me is None:
            return ceil  # global/efficiency call — permissive; gate is per-target
        opp_tick = opp_contest_tick(
            model, world, int(tgt_id), int(me),
            lead_now=_state_k_orbital_lead_enabled(),
        )
        if opp_tick is None:
            return ceil  # uncontested target → safe to commit long
        return max(floor, min(ceil, int(opp_tick)))

    if step is None or not _adaptive_k_enabled():
        return floor
    k_open = _env_int("BASELINE_ADAPTIVE_K_OPEN", 20)
    t_settle = _env_int("BASELINE_ADAPTIVE_K_TSETTLE", 30)
    if t_settle <= 0 or int(step) >= t_settle or k_open <= floor:
        return floor
    decayed = k_open - (k_open - floor) * int(step) / float(t_settle)
    return max(floor, int(round(decayed)))


def resolve_launch_target(src, angle, ships, world):
    """Resolve a bare ``[src, angle, ships]`` launch to its true destination.

    Returns ``(hit_pid | None, step, outcome)``: the first planet the
    fleet's straight-line trajectory strikes and the 1-based arrival
    tick. ``outcome`` is one of ``{"target", "planet", "sun", "oob",
    "timeout"}`` per ``predict_fleet_fate``; ``hit_pid`` is ``None`` when
    the fleet dies in the sun / out of bounds / never collides.
    """
    fate = predict_fleet_fate(
        src, _SENTINEL_TARGET, float(angle), int(ships), world,
    )
    if fate.outcome in ("target", "planet"):
        return fate.hit_planet_id, int(fate.step), fate.outcome
    return None, int(fate.step), fate.outcome


def enforce_launch_rules(moves, planets, me, world, model, k=None):
    """Drop launches violating Rule A (neutral) or Rule B (opponent-K).

    ``moves`` is a list of ``[src_id, angle, ships]``. Returns a filtered
    list preserving input order. No-op (returns ``moves`` unchanged) when
    the gate is off or there are no moves.

    Out of scope (kept unchanged): fleets whose trajectory ends in sun /
    out-of-bounds / timeout (a different waste class the chooser's
    trajectory filter already prunes). Reinforcements of our own planets
    are kept WITHIN the K horizon but dropped beyond it (universal ceiling).
    """
    if not launch_rules_enabled() or not moves:
        return moves
    world_step = getattr(world, "step", None)
    # When K is auto-computed (k is None) and the state-driven lever is on,
    # the ceiling K is enforced PER TARGET in Pass 3 below; an explicit k
    # override (tests / callers) is respected as a fixed ceiling.
    per_target_k = (k is None) and state_driven_k_enabled()
    if k is None:
        k = capture_horizon_k(world_step)
    me = int(me)

    by_id = world.planets_by_id

    # Pass 1 — resolve every move's destination + arrival tick once.
    resolved = []  # (move, hit_pid|None, step|None, owner|None)
    for mv in moves:
        src = by_id.get(int(mv[0]))
        if src is None:
            resolved.append((mv, None, None, None))
            continue
        hit_pid, step, _outcome = resolve_launch_target(
            src, mv[1], mv[2], world,
        )
        if hit_pid is None:
            resolved.append((mv, None, None, None))
            continue
        tgt = by_id.get(int(hit_pid))
        owner = int(tgt.owner) if tgt is not None else None
        resolved.append((mv, int(hit_pid), int(step), owner))

    # Pass 2 — Rule A is evaluated per same-tick neutral group so a
    # genuine coalition (>= 2 fleets, same planet, same arrival tick) is
    # judged together: keep the group iff our combined arrival captures
    # the planet at that tick. predict_garrison_at sums same-owner
    # same-tick arrivals via lib.combat and walks the in-flight ledger,
    # so enemy fleets already inbound to the neutral are accounted for.
    neutral_groups: dict[tuple[int, int], list[int]] = {}
    for idx, (_mv, hit_pid, step, owner) in enumerate(resolved):
        if owner == -1:
            neutral_groups.setdefault((hit_pid, step), []).append(idx)

    neutral_keep: dict[tuple[int, int], bool] = {}
    for (pid, step), idxs in neutral_groups.items():
        tgt = by_id.get(pid)
        if tgt is None:
            neutral_keep[(pid, step)] = False
            continue
        base_arrivals = (
            list(model.ledger.get(pid, [])) if model is not None else []
        )
        my_legs = [(step, me, int(resolved[i][0][2])) for i in idxs]
        try:
            pred_owner, _ = predict_garrison_at(
                tgt, step, base_arrivals + my_legs,
            )
        except Exception:
            pred_owner = None
        neutral_keep[(pid, step)] = (pred_owner == me)

    # Pass 3 — emit the survivors in original order.
    out = []
    for mv, hit_pid, step, owner in resolved:
        if hit_pid is None or owner is None:
            out.append(mv)            # sun / oob / timeout / unknown src
            continue
        # Universal predictability ceiling (PI 2026-05-30): NOTHING travels
        # beyond K. Applies to opponent captures, neutral captures, AND
        # reinforcement of our own planets (incl. comet-sourced) — a fleet
        # that arrives after the horizon is betting on an unpredictable board
        # and routinely lands at a flipped/contested planet and loses.
        # State-driven lever: K is the predictability of THIS target.
        k_eff = k
        if per_target_k:
            k_eff = capture_horizon_k(
                world_step, tgt_id=hit_pid, world=world, model=model, me=me,
            )
        if step > k_eff:
            # Value-driven admission: a launch past the ceiling is kept only if
            # it is a winnable race (race_win) or uncontestable (bankable) for a
            # CAPTURE (opponent/neutral, not own-planet reinforcement), within
            # the value-horizon cap. Monotone: this can only ADD survivors.
            if not (_value_horizon_enabled()
                    and model is not None
                    and owner != me
                    and int(step) <= _value_horizon_max()):
                continue
            try:
                ac = predict_arrival_contest(
                    model, world, int(hit_pid), int(step), me,
                )
            except Exception:
                continue
            if ac.race_class not in ("race_win", "bankable"):
                continue
            # fall through to the owner-based emit (neutral still needs its
            # same-tick capture check via neutral_keep below).
        if owner == me:
            out.append(mv)            # reinforcement within horizon
            continue
        if owner == -1:               # Rule A (within horizon)
            if neutral_keep.get((hit_pid, step), False):
                out.append(mv)
            continue                  # non-capturing neutral launch — drop
        # opponent within horizon — keep
        out.append(mv)
    return out
