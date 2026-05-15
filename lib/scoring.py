"""Shared scoring primitives for ROI-family strategies.

These helpers project fleet arrival deterministically against the current
obs snapshot. Variants under `agents/simple/roi_*.py` build their score
function on top of these primitives — the goal is that any strategy that
wants "what does this target look like at arrival" gets one consistent
answer rather than re-implementing the projection per file.

Caveat: the projection is single-pass and uses the current `target.ships`
to seed fleet speed. Strategies that send larger fleets (post-
`arrival_size` inflation) actually arrive a hair sooner, so `eta_proxy`
is a slight over-estimate. That's deliberately conservative: scoring
penalises far targets a touch more than the truth, biasing selection
toward closer / safer captures.

`T_TOTAL` defaults to the Orbit Wars episode-step limit (500 per the
data/README spec). If a future game has a different limit, pass it
explicitly.
"""

from __future__ import annotations

import math
import os

from kaggle_environments.envs.orbit_wars.orbit_wars import Planet

from lib.fleet import speed as fleet_speed
from lib.geometry import dist

T_TOTAL_DEFAULT: int = 500


def eta_proxy(mine: Planet, target: Planet) -> int:
    """Conservative integer-turn ETA for a launch from `mine` to `target`.

    Uses `fleet_speed(target.ships + 1)` as the speed proxy — this is the
    speed of the minimum-cover fleet a strategy would send before
    `arrival_size` inflates it. Returns 0 for zero-distance pairs.
    """
    d = dist((mine.x, mine.y), (target.x, target.y))
    if d == 0.0:
        return 0
    v = fleet_speed(target.ships + 1)
    return int(math.ceil(d / v))


def projected_garrison(target: Planet, eta: int) -> int:
    """Predicted target garrison at arrival.

    Neutral targets (`owner == -1`) don't produce; their garrison stays
    flat. Owned targets (ours or enemy) grow by `production * eta`.
    """
    if target.owner == -1:
        return target.ships
    return target.ships + target.production * eta


def s_needed(target: Planet, eta: int) -> int:
    """Minimum fleet size to capture `target` at arrival.

    Mirrors `lib.mechanism.arrival_size`'s formula so strategy-side gates
    use the same number the mechanism layer will end up sizing fleets
    against. The +1 ensures strict win on the combat resolver.
    """
    return projected_garrison(target, eta) + 1


def horizon(step: int, eta: int, t_total: int = T_TOTAL_DEFAULT) -> int:
    """Remaining turns the captured planet can produce for us.

    `H = max(0, T_total - step - eta)`. Late-game this collapses toward 0
    so production-weighted scores naturally pivot to cheap snipes /
    denial in the closing turns.
    """
    return max(0, t_total - step - eta)


# Discount factor for present-value horizon valuation. With γ < 1, future
# production is discounted at γ per turn from the arrival step. At γ = 1.0
# (the default) the function reduces to the linear horizon above; that
# preserves the pre-PV scoring shape so existing snipe/reinforce tests pass
# unchanged. A/B candidates set γ < 1 (typically 0.99 per discussion-thread
# TID 699003) via `scripts/ab_variants.py --variant pv PV_GAMMA=0.99`.
#
# Env-var override (PV_GAMMA): an agent that wants PV-aware proposers can
# `os.environ.setdefault("PV_GAMMA", "0.99")` BEFORE this module is
# imported. v7_pv (live μ=1064.4) is v7_0_drop_one + this single config.
PV_GAMMA = float(os.environ.get("PV_GAMMA", "1.0"))


# Sensitivity coefficient for the 3-NN allegiance danger field (H17 /
# TID 699003). Multiplicative on snipe + reinforce score:
#     score *= max(MIN_DANGER3_MULT, 1.0 + DANGER3_KAPPA · danger_3nn(target))
# At κ=0 the field has no effect — preserves the snipe/reinforce score
# numerics for the existing parity tests. Typical A/B candidate values
# are 0.1-0.3 (each ally-neighbour boosts score by 10-30 %, each enemy
# discounts it by the same). `MIN_DANGER3_MULT` clamps the multiplier
# above zero so a 3-enemy neighbourhood at κ ≥ 1/3 doesn't zero the score.
DANGER3_KAPPA = 0.0
MIN_DANGER3_MULT = 0.05


def pv_horizon(
    step: int, eta: int, gamma: float = PV_GAMMA,
    t_total: int = T_TOTAL_DEFAULT,
) -> float:
    """Present-value of a unit production stream starting at `step + eta`.

    With γ < 1: `γ^eta · Σ_{k=0}^{h-1} γ^k = γ^eta · (1-γ^h)/(1-γ)` where
    `h = t_total - step - eta`. The early arrival turns count for full
    γ^eta weight; far-future production is exponentially discounted. At
    γ = 1.0 the formula degenerates to the linear horizon (`h` turns of
    equal weight), matching `horizon()` above modulo floating-point cast.
    Returns 0 when no production turns remain.
    """
    h = t_total - step - eta
    if h <= 0:
        return 0.0
    if gamma >= 1.0:
        return float(h)
    return (gamma ** eta) * (1.0 - gamma ** h) / (1.0 - gamma)


def margin_multiplier(target: Planet, my_id: int) -> int:
    """Owner-flip multiplier for margin-based scoring.

    Captures contribute to the margin (my_ships - their_ships):
    - Self (already ours): 0 — reinforce moves don't change margin.
    - Neutral: 1 — we gain P/turn going forward.
    - Enemy: 2 — we gain P/turn AND deny them P/turn (zero-sum).
    """
    if target.owner == my_id:
        return 0
    if target.owner == -1:
        return 1
    return 2


def expected_hold(
    target_id: int, eta: int, world, model,
    t_total: int = T_TOTAL_DEFAULT,
) -> int:
    """Predicted turns we'll own `target_id` after capturing at our
    arrival turn `now + eta`. Capped above by remaining-game-end and
    below at 0.

    Computed as `min(remaining_game, threat_eta − eta)` where
    `threat_eta = WorldModel.time_to_enemy_threat(target, my_id)`.
    `None` from the helper (no enemy threat) → saturate at the
    remaining-game-end cap.

    Used by HAV-1: drives the `pv_horizon` `t_total` per-target,
    instead of the flat `EPISODE_STEPS − step − eta` form. Targets
    deep in enemy space get short `expected_hold` (the enemy retakes
    them quickly); targets in our cluster saturate.
    """
    step_now = int(world.step)
    remaining = max(0, t_total - step_now - eta)
    if remaining == 0:
        return 0
    threat = model.time_to_enemy_threat(target_id, world.my_id, world)
    if threat is None:
        # No enemy can plausibly reach — saturate at remaining game.
        return remaining
    # The threat fleet arrives at `threat` from now; we arrive at
    # `eta`; we hold from arrival to threat-arrival.
    hold = max(0, int(threat) - int(eta))
    return min(remaining, hold)


__all__ = [
    "T_TOTAL_DEFAULT",
    "PV_GAMMA",
    "eta_proxy",
    "projected_garrison",
    "s_needed",
    "horizon",
    "pv_horizon",
    "expected_hold",
    "margin_multiplier",
]
