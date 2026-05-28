"""Macro mission planner — 2P state machine: EXPAND / STOCKPILE / STRIKE / DEFEND.

Layered ON TOP of the per-move chooser (PV_ETA + LEAF_PV_2P stack), the
macro plans at the symmetric-group / flank level. It commits the agent
to one forward lateral, bundles forces, and parks defenders against
arriving opponents.

Geometric premise (closed-form, no per-game detection needed):

  - Engine guarantees `omega` is sampled from `uniform(0.025, 0.05)` —
    ALWAYS POSITIVE. The "forward" direction in the orbital ring is
    fixed across all games (`home_angle + pi/2`).
  - In 2P, opponent's home is the 180-rotated counterpart of ours
    (`lib.mirror.diagonal_opponent`). A straight chord home-to-opp's-home
    crosses the sun. The two laterals in the same symmetric group both
    give sun-free chords to opp's home.
  - Picking the FORWARD lateral (the one whose polar angle is
    `home_angle + pi/2`, modulo the ring) means opp picks their own
    forward lateral by symmetry, and the two choices land on
    DIFFERENT physical planets — uncontested expansion on both sides.

State semantics:

  EXPAND      — we don't own the chosen forward lateral. Bundle ships
                from home; emit ONE launch when we can afford the
                capture (target.ships + 1 + margin). Otherwise no
                emit; the home accumulates.
  STOCKPILE   — we own the chosen lateral. Stockpile is below STRIKE
                threshold. Block the chooser from draining the lateral
                via `hold_src`; let production accumulate.
  STRIKE      — stockpile sufficient to overwhelm opp's weakest
                planet (with margin). Emit ONE bundled launch.
  DEFEND      — predicted owner flip on our home within DEFEND_HORIZON.
                Overrides all other states. No emit, no hold; current
                chooser stack handles defense.
  DISABLED    — `num_seats != 2`, no orbital home group, or
                home/opp/lateral identification fails. No-op.

Entry point: `determine_macro_state(world, model, me, num_seats, omega,
initial_planets)` returns a `MacroState` dataclass. Caller (the agent in
`agents/baseline/main.py`) converts `MacroState.emit` to a `[src_id,
angle, ships]` move via `proposer.aim_and_eta`, and merges `hold_src`
into the chooser's `reserved_srcs` set.

This module is import-pure: it depends only on `lib/` (mirror, geometry).
No agent-internal imports. That makes it cleanly unit-testable.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Iterable, Optional

from lib.geometry import BOARD_SIZE, CENTER
from lib.mirror import build_bijection, detect_num_players, diagonal_opponent


# ---------------------------------------------------------------------------
# Calibrations — all env-var-tunable. Defaults are conservative first-pass
# values; A/B-sweep before any submit.
# ---------------------------------------------------------------------------

def _envf(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _envi(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


EXPAND_MARGIN = _envi("BASELINE_MACRO_EXPAND_MARGIN", 2)
# Minimum ships kept at home during EXPAND. The opening is when ladder
# leaders drain garrisons most aggressively (top-10 median first launch
# step 4.1 vs midpack 10.5, per knowledge-base/concepts/top-performer-
# strategies.md). DEFEND state catches concrete threats via owner_at
# prediction; the home_min sentry is a per-step safety floor, not a
# strategic reserve. Default 0 = full opening aggression.
EXPAND_HOME_MIN = _envi("BASELINE_MACRO_EXPAND_HOME_MIN", 0)
DEFEND_HORIZON = _envi("BASELINE_MACRO_DEFEND_HORIZON", 20)
STRIKE_RESERVE = _envi("BASELINE_MACRO_STRIKE_RESERVE", 20)
STRIKE_MARGIN = _envf("BASELINE_MACRO_STRIKE_MARGIN", 1.15)


# ---------------------------------------------------------------------------
# Dataclass — what the agent consumes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MacroEmit:
    """A planned launch the agent should emit this turn.

    The agent converts (src_id, tgt_id, ships) to a [src_id, angle, ships]
    move via `proposer.aim_and_eta`, which handles orbital lead-aim and
    comet path-aim for free.
    """
    src_id: int
    tgt_id: int
    ships: int


@dataclass(frozen=True)
class MacroState:
    phase: str                                # one of EXPAND/STOCKPILE/STRIKE/DEFEND/DISABLED
    home_id: Optional[int] = None
    chosen_lateral_id: Optional[int] = None
    opp_home_id: Optional[int] = None
    hold_src: Optional[int] = None            # source the chooser must NOT launch from
    emit: Optional[MacroEmit] = None          # at most one launch per turn
    reason: str = ""                           # tracing


# ---------------------------------------------------------------------------
# Geometric helpers
# ---------------------------------------------------------------------------


def _polar_angle(x: float, y: float) -> float:
    """Polar angle of (x, y) from the board centre, in [0, 2*pi)."""
    return math.atan2(y - CENTER, x - CENTER) % (2 * math.pi)


def _angular_distance(a: float, b: float) -> float:
    """Shortest angular distance between two angles."""
    d = (a - b) % (2 * math.pi)
    return min(d, 2 * math.pi - d)


def _pick_forward_lateral(laterals, home):
    """Return the lateral whose polar angle is +pi/2 ahead of home.

    `omega > 0` in all games, so `home_angle + pi/2` (mod 2*pi) is the
    forward direction in rotation. Among the two laterals in the home
    symmetric group, the forward one is the angular-nearest match.
    """
    home_angle = _polar_angle(float(home.x), float(home.y))
    forward_angle = (home_angle + math.pi / 2) % (2 * math.pi)
    return min(
        laterals,
        key=lambda p: _angular_distance(_polar_angle(float(p.x), float(p.y)),
                                         forward_angle),
    )


def _home_group_ids(initial_planets, home_id: int, bij: dict) -> set[int]:
    """Identify the four-planet symmetric group containing `home`.

    The env (`kaggle_environments/envs/orbit_wars/orbit_wars.py`) places
    planets via 90-degree rotational symmetry, but allocates ids in
    contiguous blocks of 4 per symmetric group. So the group of any
    planet is `(base, base+1, base+2, base+3)` where `base = id // 4
    * 4`. We additionally verify each candidate id exists in the
    initial-planets list to guard against truncated obs.
    """
    base = (int(home_id) // 4) * 4
    by_id = {int(p[0]) for p in initial_planets}
    return {pid for pid in (base, base + 1, base + 2, base + 3) if pid in by_id}


def _identify_home(world, me: int):
    """Return our home planet — the one we own with the smallest id at step 0.

    Heuristic: at game start, each player owns exactly one planet (their
    home). We pick the smallest-id owned planet, which is stable across
    turns: the home planet keeps its id until captured.
    """
    owned = [
        p for p in world.planets_by_id.values()
        if int(p.owner) == int(me)
    ]
    if not owned:
        return None
    return min(owned, key=lambda p: int(p.id))


def _identify_opp_home(world, opp_id: int, my_home, bij: dict):
    """Return opp's home planet using the 180-deg mirror bijection.

    Fallback: opp's smallest-id owned planet at step 0. The bijection
    method is preferred because it's robust even after the opp's home
    has been captured (its 180-rotated counterpart is still our home).
    """
    if my_home is not None and int(my_home.id) in bij:
        opp_home_id = bij[int(my_home.id)]
        p = world.planets_by_id.get(opp_home_id)
        if p is not None:
            return p
    # Fallback: smallest-id planet owned by opp.
    owned = [
        p for p in world.planets_by_id.values()
        if int(p.owner) == int(opp_id)
    ]
    if not owned:
        return None
    return min(owned, key=lambda p: int(p.id))


# ---------------------------------------------------------------------------
# State logic
# ---------------------------------------------------------------------------


def _will_home_flip(model, my_home_id: int, me: int, horizon: int) -> bool:
    """Predict whether our home flips to an enemy owner within `horizon`."""
    for t in range(1, horizon + 1):
        o = model.owner_at(my_home_id, t)
        if o is not None and int(o) != int(me) and int(o) != -1:
            return True
    return False


def _pick_strike_target(world, model, me: int, lateral, opp_home):
    """Pick opp's weakest reachable planet from our captured lateral.

    First cut: opp_home (deterministic, known). Future iteration: scan
    opp-owned planets, exclude sun-crossing chords, pick lowest predicted
    garrison at arrival. For now we route every STRIKE at opp_home.
    """
    return opp_home


def _strike_threshold(lateral, target, opp_home_production: int = 0) -> int:
    """Ships required on lateral to STRIKE `target` with margin.

    Predicted target garrison at our arrival = target.ships + target.prod
    * eta. We pad by `STRIKE_MARGIN` to absorb the trajectory chooser's
    one-tick combat prediction uncertainty + opp launches we can't see.

    eta estimated from lateral->target straight-line distance at speed 5
    (typical for a 100+ ship bundle).
    """
    dx = float(target.x) - float(lateral.x)
    dy = float(target.y) - float(lateral.y)
    dist = math.hypot(dx, dy)
    eta = max(1, int(math.ceil(dist / 5.0)))
    predicted_garrison = int(target.ships) + int(target.production) * eta
    return int(math.ceil(predicted_garrison * STRIKE_MARGIN)) + STRIKE_RESERVE


def determine_macro_state(
    world,
    model,
    me: int,
    num_seats: int,
    omega: float,
    initial_planets,
) -> MacroState:
    """Top-level macro decision. See module docstring for state semantics."""
    # Gate 1: 2P only. 4P geometry doesn't reduce to a clean diagonal.
    if int(num_seats) != 2:
        return MacroState(phase="DISABLED", reason="num_seats!=2")

    home = _identify_home(world, me)
    if home is None:
        return MacroState(phase="DISABLED", reason="no_home")

    # Build mirror bijection from initial planets; needed for opp_home id.
    try:
        bij = build_bijection(initial_planets)
    except Exception:
        bij = {}

    opp_id = diagonal_opponent(int(me), 2)
    opp_home = _identify_opp_home(world, opp_id, home, bij)
    if opp_home is None:
        return MacroState(phase="DISABLED", home_id=int(home.id),
                          reason="no_opp_home")

    # Identify the four-planet symmetric group containing home; pick the
    # two laterals (not home, not opp_home).
    group_ids = _home_group_ids(initial_planets, int(home.id), bij)
    if not group_ids:
        return MacroState(phase="DISABLED", home_id=int(home.id),
                          opp_home_id=int(opp_home.id),
                          reason="no_home_group")
    laterals = [
        world.planets_by_id[pid]
        for pid in group_ids
        if pid != int(home.id) and pid != int(opp_home.id)
           and pid in world.planets_by_id
    ]
    if len(laterals) != 2:
        return MacroState(phase="DISABLED", home_id=int(home.id),
                          opp_home_id=int(opp_home.id),
                          reason=f"bad_laterals_count={len(laterals)}")

    chosen = _pick_forward_lateral(laterals, home)

    # DEFEND gate: overrides every other state if home is about to flip.
    if _will_home_flip(model, int(home.id), int(me), DEFEND_HORIZON):
        return MacroState(
            phase="DEFEND",
            home_id=int(home.id),
            chosen_lateral_id=int(chosen.id),
            opp_home_id=int(opp_home.id),
            reason="home_flip_predicted",
        )

    # EXPAND: we don't own the chosen lateral yet. Use EXPAND_HOME_MIN
    # (opening-aggressive, default 0) — DEFEND state catches concrete
    # incoming threats; the per-step home_min reserve is a safety floor.
    if int(chosen.owner) != int(me):
        spare = int(home.ships) - EXPAND_HOME_MIN
        # Size for PREDICTED garrison at arrival, not current. With
        # production accumulating during ~10-turn travel, current-garrison
        # sizing systematically undershoots (lateral grows by prod * eta).
        # Use the same straight-line eta heuristic as _strike_threshold.
        dx = float(chosen.x) - float(home.x)
        dy = float(chosen.y) - float(home.y)
        eta_est = max(1, int(math.ceil(math.hypot(dx, dy) / 5.0)))
        predicted_garrison = int(chosen.ships) + int(chosen.production) * eta_est
        ships_needed = predicted_garrison + 1 + EXPAND_MARGIN
        if spare >= ships_needed:
            # Bundle as much as we safely can while keeping home above min.
            # Cap at 2x the strictly-needed count to avoid over-allocating
            # if home has accumulated a huge garrison.
            send = min(spare, max(ships_needed, ships_needed * 2))
            emit = MacroEmit(src_id=int(home.id), tgt_id=int(chosen.id),
                             ships=int(send))
            return MacroState(
                phase="EXPAND",
                home_id=int(home.id),
                chosen_lateral_id=int(chosen.id),
                opp_home_id=int(opp_home.id),
                emit=emit,
                reason="expand_emit",
            )
        return MacroState(
            phase="EXPAND",
            home_id=int(home.id),
            chosen_lateral_id=int(chosen.id),
            opp_home_id=int(opp_home.id),
            reason="expand_accumulating",
        )

    # We own the chosen lateral. STOCKPILE or STRIKE.
    target = _pick_strike_target(world, model, int(me), chosen, opp_home)
    if target is None or int(target.owner) == int(me):
        # No strike target (opp eliminated or we already own it). Hold.
        return MacroState(
            phase="STOCKPILE",
            home_id=int(home.id),
            chosen_lateral_id=int(chosen.id),
            opp_home_id=int(opp_home.id),
            hold_src=int(chosen.id),
            reason="no_strike_target",
        )

    threshold = _strike_threshold(chosen, target)
    if int(chosen.ships) >= threshold:
        send = max(1, int(chosen.ships) - STRIKE_RESERVE)
        emit = MacroEmit(src_id=int(chosen.id), tgt_id=int(target.id),
                         ships=int(send))
        return MacroState(
            phase="STRIKE",
            home_id=int(home.id),
            chosen_lateral_id=int(chosen.id),
            opp_home_id=int(opp_home.id),
            emit=emit,
            reason=f"strike_emit_threshold={threshold}",
        )

    return MacroState(
        phase="STOCKPILE",
        home_id=int(home.id),
        chosen_lateral_id=int(chosen.id),
        opp_home_id=int(opp_home.id),
        hold_src=int(chosen.id),
        reason=f"stockpile_threshold={threshold}_ships={int(chosen.ships)}",
    )


__all__ = [
    "MacroEmit",
    "MacroState",
    "determine_macro_state",
]
