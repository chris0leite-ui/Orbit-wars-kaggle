"""favor.py — world-favorability evaluator for Orbit Wars.

Returns a single float `favor(obs, player)` answering the question
"how favorable is this world state for `player`?" — independent of any
action being considered.

Design contract
---------------
Pure function: same observation + same player → same favor. No I/O,
no global state. Cheap (O(planets² + fleets) per call). Symmetric:
for a 2-player symmetric state, `favor(s, 0) + favor(s, 1) ≈ 0`.

The point of decoupling this from action selection (next session) is so
we can iterate the formula in milliseconds instead of running 24-game
A/Bs every change. Validation lives in `validate_favor.py`: sample
mid-game states, check that `sign(favor_diff)` predicts the eventual
winner. If AUC ≥ 0.75 we wire it into `main.py`.

Starting feature set (v0)
-------------------------
Linear sum of two features. Add more (F3 defensibility, F4
reachability, F5+ from the PI-suggested backlog) only if a feature
moves AUC on the saved-state set. See `docs/strategies/favor.md` (TBD)
for the full feature ledger.

  F1 = my_ships − strongest_opponent_ships
       (garrison + in-transit fleets)
  F2 = (my_prod − strongest_opp_prod) × turns_remaining
       (comets discounted by remaining-lifetime / turns_remaining)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Planet field indices:  [id, owner, x, y, radius, ships, production]
P_ID, P_OWNER, P_X, P_Y, P_R, P_SHIPS, P_PROD = range(7)

# Fleet field indices:   [id, owner, x, y, angle, from_planet_id, ships]
F_ID, F_OWNER, F_X, F_Y, F_ANGLE, F_FROM, F_SHIPS = range(7)

EPISODE_STEPS = 500


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


def _production_total(obs: Any, player: int, turns_remaining: int) -> float:
    """Sum of production with comet remaining-lifetime discount.

    Non-comet planets contribute their full production.
    Comets contribute `prod × min(remaining_lifetime, turns_remaining)
    / turns_remaining` — so a comet about to leave the board is worth a
    fraction of a stable planet of the same production.
    """
    if turns_remaining <= 0:
        return 0.0
    comet_ids = set(_get(obs, "comet_planet_ids", []) or [])
    total = 0.0
    for p in _get(obs, "planets", []):
        if int(p[P_OWNER]) != player:
            continue
        pid = int(p[P_ID])
        prod = float(p[P_PROD])
        if pid in comet_ids:
            life = _comet_remaining_lifetime(pid, obs) or 0
            life_capped = min(life, turns_remaining)
            total += prod * (life_capped / turns_remaining)
        else:
            total += prod
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

    # F2 — production lead × turns_remaining
    my_prod = _production_total(obs, player, turns_remaining)
    opp_prod_max = 0.0
    for owner in _player_ids(obs):
        if owner == player:
            continue
        pp = _production_total(obs, owner, turns_remaining)
        if pp > opp_prod_max:
            opp_prod_max = pp
    prod_lead = (my_prod - opp_prod_max) * turns_remaining

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

    my_prod = _production_total(obs, player, turns_remaining)
    opp_prod_max = max(
        (_production_total(obs, o, turns_remaining) for o in _player_ids(obs) if o != player),
        default=0.0,
    )
    f2 = (my_prod - opp_prod_max) * turns_remaining

    return {
        "F1_ship_lead": cfg.alpha * f1,
        "F2_prod_lead_x_horizon": cfg.beta * f2,
        "total": cfg.alpha * f1 + cfg.beta * f2,
        "step": step,
        "turns_remaining": turns_remaining,
        "my_ships": my_ships,
        "opp_ships_max": opp_ships_max,
        "my_prod": my_prod,
        "opp_prod_max": opp_prod_max,
    }
