"""Opponent reach estimator for the reach-frontier chooser.

Wraps `lib.world_model.WorldModel.time_to_enemy_threat`, which already
handles (a) in-flight enemy fleets currently inbound, (b) potential
launches from every currently-stationary enemy-owned planet, and (c)
the 5-iteration orbital fixed-point for rendezvous prediction. To match
the doctrine's amortised reach cost ρ = arrival + ships/p̃_s, we add a
source-recovery term keyed off the opponent's strongest source
production (conservative direction: opp recovers faster → our hold is
shorter → we play it safe).

Known biases (documented in README §"v1 biases"):
- Symmetric-strength assumption (every opp plays best reach).
- Opp source-recovery cost uses strongest opp production, not the
  actual launching source's production (over-estimates opp speed).
- No 4P collaboration modelling (each opp treated independently).
"""

from __future__ import annotations


def _max_opp_production(world, me: int) -> float:
    """Strongest production across non-me, non-neutral planets.

    Used as the denominator for opp's source-recovery cost. Clamped at
    1.0 to avoid div-by-zero in edge cases where opp has no orbital
    sources (only outer / production-0 planets).
    """
    best = 0.0
    for p in world.planets_by_id.values():
        owner = int(p.owner)
        if owner == me or owner == -1:
            continue
        prod = float(getattr(p, "production", 0.0))
        if prod > best:
            best = prod
    return max(1.0, best)


def estimate_opp_reach(world, me: int, world_model) -> dict[int, float]:
    """Min reach cost ρ_opp(p) per target, across all opponents.

    Returns `target_id -> ρ_opp` (float). `inf` for targets no opponent
    can plausibly threaten within the WorldModel horizon.
    """
    me_id = int(me)
    opp_prod = _max_opp_production(world, me_id)
    out: dict[int, float] = {}
    for p in world.planets_by_id.values():
        eta = world_model.time_to_enemy_threat(int(p.id), me_id, world)
        if eta is None:
            out[int(p.id)] = float("inf")
            continue
        # Doctrine-symmetric recovery: ships-to-capture / opp production.
        # Estimate ships needed as current garrison + 1 (minimum to flip).
        # Opp's source-recovery time then = ships_needed / opp_prod.
        ships_needed = max(1.0, float(getattr(p, "ships", 0)) + 1.0)
        out[int(p.id)] = float(eta) + ships_needed / opp_prod
    return out
