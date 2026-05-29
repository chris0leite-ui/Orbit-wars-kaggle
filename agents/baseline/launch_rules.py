"""Post-emit launch-discipline validator (PI rules, 2026-05-29).

Two rules the champion must GUARANTEE, not merely encourage:

  Rule A — neutral discipline. Never send ships to a NEUTRAL planet
    unless they capture it. Capture by multiple fleets arriving the SAME
    tick from different planets is allowed; staggered pokes are not
    (combat sums only same-tick arrivals — see lib/combat.resolve_arrivals).

  Rule B — opponent predictability ceiling. Only commit to capturing an
    OPPONENT planet if the fleet arrives within K turns (K = the horizon
    of predictability, default 10). Drop captures arriving later than K.

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
from lib.world_model import predict_garrison_at

DEFAULT_CAPTURE_HORIZON_K = 10

# Sentinel target whose id never matches a real planet, so
# predict_fleet_fate reports the FIRST planet actually struck as
# outcome "planet" with hit_planet_id set (its "target" branch only
# fires when pid == target.id). The fate ray-cast reads target.id only.
_SENTINEL_TARGET = SimpleNamespace(id=-999999)


def launch_rules_enabled() -> bool:
    return os.environ.get("BASELINE_LAUNCH_RULES", "0").strip().lower() in (
        "1", "true", "on", "yes",
    )


def capture_horizon_k() -> int:
    raw = os.environ.get("BASELINE_CAPTURE_HORIZON_K", "").strip()
    if not raw:
        return DEFAULT_CAPTURE_HORIZON_K
    try:
        return int(raw)
    except ValueError:
        return DEFAULT_CAPTURE_HORIZON_K


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

    Out of scope (kept unchanged): reinforcements of our own planets, and
    fleets whose trajectory ends in sun / out-of-bounds / timeout (a
    different waste class the chooser's trajectory filter already prunes).
    """
    if not launch_rules_enabled() or not moves:
        return moves
    if k is None:
        k = capture_horizon_k()
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
        if owner == me:
            out.append(mv)            # reinforcement — exempt
            continue
        if owner == -1:               # Rule A
            if neutral_keep.get((hit_pid, step), False):
                out.append(mv)
            continue                  # non-capturing neutral launch — drop
        # owner is an opponent — Rule B ceiling.
        if step <= k:
            out.append(mv)
        # arrival beyond the predictability horizon — drop
    return out
