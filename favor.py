"""favor.py — world-favorability evaluator for Orbit Wars.

Returns a single float `favor(obs, player)` answering the question
"how favorable is this world state for `player`?" — independent of any
action being considered.

Design contract
---------------
Pure function: same observation + same player → same favor. No I/O,
no global state. Cheap (O(planets² + fleets) per call). Symmetric:
for a 2-player symmetric state, `favor(s, 0) + favor(s, 1) ≈ 0`.

Validation: `validate_favor.py --replay` re-scores 192 mid-game
snapshots; sign(favor_diff) predicts the eventual winner. v1 (F1 + F2
only) reached AUC = 0.945. Iterate on saved states (1 s per cycle)
before any local A/B (15 min per cycle).

Feature ledger
--------------
v1 (initial): F1 + F2.
v2 (this version): F1 + F2*, where F2* discounts each planet's
    production by its expected hold-time. Closes the "capture with
    1 ship, opponent recaptures" failure mode that 1- and 10-step
    greedy(favor) couldn't see vs v7_0 (0 / 60 across 4 variants).

  F1 = my_ships − strongest_opponent_ships
       (garrison + in-transit fleets)

  F2* = Σ over my planets of  production × expected_hold_time
        − Σ over strongest-opp planets of  production × expected_hold_time

  expected_hold_time(P) = min(turns_remaining,
        min over enemy planets Q of arrival_turn(Q→P)
            if Q.ships > my_garrison_at_arrival(P, Q))

  comets are additionally capped by remaining_lifetime.

Backlog: F4 reachability, F5 flexibility, F6 production-uncommitted,
F7 expansion_reach, F8 spatial_spread. Add iteratively, gated on AUC
lift on the saved-state set.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

# Planet field indices:  [id, owner, x, y, radius, ships, production]
P_ID, P_OWNER, P_X, P_Y, P_R, P_SHIPS, P_PROD = range(7)

# Fleet field indices:   [id, owner, x, y, angle, from_planet_id, ships]
F_ID, F_OWNER, F_X, F_Y, F_ANGLE, F_FROM, F_SHIPS = range(7)

EPISODE_STEPS = 500
MAX_SPEED = 6.0


def _speed(ships: int) -> float:
    """Fleet speed as a function of size (per data/README.md comp spec).
    Mirrored in main.py:_speed; kept here so favor.py has no main deps.
    """
    if ships <= 1:
        return 1.0
    return 1.0 + (MAX_SPEED - 1.0) * (math.log(ships) / math.log(1000.0)) ** 1.5


@dataclass(frozen=True)
class FavorConfig:
    """Hand-tunable weights. v0: F1 + F2 only (gamma/delta unused)."""

    alpha: float = 1.0   # weight on ship lead              (F1)
    beta: float = 1.0    # weight on production lead × horizon  (F2)
    gamma: float = 2.0   # defensibility penalty            (F3 — unused in v0)
    delta: float = 0.5   # reachability bonus               (F4 — unused in v0)


# ---------------------------------------------------------------------------
# Observation accessors — work for both dict and Struct-style obs
# ---------------------------------------------------------------------------


def _get(obs: Any, key: str, default: Any) -> Any:
    if isinstance(obs, dict):
        return obs.get(key, default)
    return getattr(obs, key, default)


def _player_ids(obs: Any) -> set[int]:
    """Distinct non-neutral owners present in planets or fleets."""
    owners: set[int] = set()
    for p in _get(obs, "planets", []):
        o = int(p[P_OWNER])
        if o >= 0:
            owners.add(o)
    for f in _get(obs, "fleets", []):
        o = int(f[F_OWNER])
        if o >= 0:
            owners.add(o)
    return owners


def _comet_remaining_lifetime(planet_id: int, obs: Any) -> int | None:
    """Turns of board-life remaining for the given comet. None if not a comet."""
    for group in _get(obs, "comets", []) or []:
        ids = group.get("planet_ids") if isinstance(group, dict) else getattr(group, "planet_ids", [])
        if planet_id not in ids:
            continue
        path_index = (
            group.get("path_index")
            if isinstance(group, dict)
            else getattr(group, "path_index", 0)
        )
        paths = group.get("paths") if isinstance(group, dict) else getattr(group, "paths", [])
        if not paths:
            return 0
        # All 4 comets in a group share path length; index 0 is canonical.
        return max(0, len(paths[0]) - int(path_index))
    return None


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------


def _ships_total(obs: Any, player: int) -> int:
    total = 0
    for p in _get(obs, "planets", []):
        if int(p[P_OWNER]) == player:
            total += int(p[P_SHIPS])
    for f in _get(obs, "fleets", []):
        if int(f[F_OWNER]) == player:
            total += int(f[F_SHIPS])
    return total


def _expected_hold_time(
    p_planet: list,
    enemy_planets: list[list],
    turns_remaining: int,
) -> int:
    """Estimated turns the planet stays in current owner's hands before
    being recaptured.

    Closed-form: for each enemy planet Q, ask "if Q launches its full
    garrison toward P right now, when does it arrive and will it
    succeed?". Take the earliest successful attack.

    Approximations (v0):
      - Q launches its current ships immediately (no waiting/saving)
      - my_garrison_at_arrival = P.ships + P.production × arrival_turns
      - no reinforcement from other owned planets modeled
      - orbital motion ignored; current positions used
      - sun-crossing not penalised (slight under-estimate of safety)

    If no enemy can capture P with their current ships, hold time =
    turns_remaining (full horizon).
    """
    if turns_remaining <= 0:
        return 0
    px, py = float(p_planet[P_X]), float(p_planet[P_Y])
    p_r = float(p_planet[P_R])
    p_ships = int(p_planet[P_SHIPS])
    p_prod = int(p_planet[P_PROD])
    hold = turns_remaining
    for q in enemy_planets:
        q_ships = int(q[P_SHIPS])
        if q_ships <= 0:
            continue
        qx, qy = float(q[P_X]), float(q[P_Y])
        q_r = float(q[P_R])
        dist = max(0.0, math.hypot(px - qx, py - qy) - p_r - q_r)
        arrival = max(1, math.ceil(dist / _speed(q_ships)))
        if arrival >= hold:
            continue                              # already beaten by another threat
        my_garrison_at_arrival = p_ships + p_prod * arrival
        if q_ships > my_garrison_at_arrival:
            hold = arrival
    return hold


def _production_value(obs: Any, player: int, turns_remaining: int) -> float:
    """Sum of (production × expected_hold_time) for `player`'s planets.

    Comets are additionally capped by remaining_lifetime — a comet
    that will leave the board in 3 turns can't generate more
    production than 3 × prod no matter how defensible it is.

    Replaces v1's `_production_total`, which used the full
    turns_remaining horizon for every planet regardless of how
    defendable it was. That v1 simplification over-credited
    undefendable captures by up to 80×; closing that gap is the whole
    point of v2.
    """
    if turns_remaining <= 0:
        return 0.0
    comet_ids = set(_get(obs, "comet_planet_ids", []) or [])
    raw_planets = _get(obs, "planets", [])
    enemy_planets = [
        p for p in raw_planets
        if int(p[P_OWNER]) >= 0 and int(p[P_OWNER]) != player
    ]
    total = 0.0
    for p in raw_planets:
        if int(p[P_OWNER]) != player:
            continue
        pid = int(p[P_ID])
        prod = float(p[P_PROD])
        hold = _expected_hold_time(p, enemy_planets, turns_remaining)
        if pid in comet_ids:
            life = _comet_remaining_lifetime(pid, obs) or 0
            hold = min(hold, life)
        total += prod * hold
    return total


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def favor(obs: Any, player: int, config: FavorConfig | None = None) -> float:
    """World-favorability score for `player`. See module docstring."""
    cfg = config or FavorConfig()
    step = int(_get(obs, "step", 0))
    turns_remaining = max(0, EPISODE_STEPS - step)

    # F1 — ship lead vs strongest opponent
    my_ships = _ships_total(obs, player)
    opp_ships_max = 0
    for owner in _player_ids(obs):
        if owner == player:
            continue
        s = _ships_total(obs, owner)
        if s > opp_ships_max:
            opp_ships_max = s
    ship_lead = my_ships - opp_ships_max

    # F2* — production-value lead (each planet's prod × expected hold time).
    my_prod_val = _production_value(obs, player, turns_remaining)
    opp_prod_val_max = 0.0
    for owner in _player_ids(obs):
        if owner == player:
            continue
        pv = _production_value(obs, owner, turns_remaining)
        if pv > opp_prod_val_max:
            opp_prod_val_max = pv
    prod_lead = my_prod_val - opp_prod_val_max

    return cfg.alpha * ship_lead + cfg.beta * prod_lead


def favor_breakdown(
    obs: Any, player: int, config: FavorConfig | None = None
) -> dict[str, float]:
    """Same as favor(), but returns each feature contribution separately.

    Useful for debugging: which feature dominates a given state's score?
    """
    cfg = config or FavorConfig()
    step = int(_get(obs, "step", 0))
    turns_remaining = max(0, EPISODE_STEPS - step)

    my_ships = _ships_total(obs, player)
    opp_ships_max = max(
        (_ships_total(obs, o) for o in _player_ids(obs) if o != player),
        default=0,
    )
    f1 = my_ships - opp_ships_max

    my_prod_val = _production_value(obs, player, turns_remaining)
    opp_prod_val_max = max(
        (_production_value(obs, o, turns_remaining) for o in _player_ids(obs) if o != player),
        default=0.0,
    )
    f2 = my_prod_val - opp_prod_val_max

    return {
        "F1_ship_lead": cfg.alpha * f1,
        "F2_prod_value_lead": cfg.beta * f2,
        "total": cfg.alpha * f1 + cfg.beta * f2,
        "step": step,
        "turns_remaining": turns_remaining,
        "my_ships": my_ships,
        "opp_ships_max": opp_ships_max,
        "my_prod_value": my_prod_val,
        "opp_prod_value_max": opp_prod_val_max,
    }
