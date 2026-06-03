"""Production-time-integral value of a capture.

The game is a production-time integral: a captured planet is worth its
production rate times the number of turns we'll hold it. So:

    value = owner_mult * weakest_mult * production * hold_time

  * ``hold_time``    = min(remaining_game, opp_reach - eta), or remaining_game
                       when no enemy can reach. Earlier captures (smaller eta)
                       get a larger hold_time -> early captures compound, with
                       no special-case rule.
  * ``owner_mult``   = enemy_capture_weight (2.0) for an enemy planet (gain +
                       denial double-counts), neutral_capture_weight (1.0) for
                       a neutral, 0 for one already ours.
  * ``weakest_mult`` = 1 + weakest_opp_bias on the weakest opponent's planets
                       in a 4P game (attack-weakest), 1.0 otherwise.

This is risk-neutral (expected-gap) valuation — sound when even or ahead. A
variance tilt when behind is a deliberately deferred refinement.
"""

from __future__ import annotations


def _weakest_seat(view):
    if not view.opp_strength:
        return None
    return min(view.opp_strength, key=view.opp_strength.get)


def planet_value(view, tgt, my_hold, eta, opp_reach, cfg) -> float:
    """Differential value of capturing ``tgt``: what I GAIN plus what I DENY the
    opponent, in production x time. ``value = production * (my_hold + denial)``.

    The game is won by the differential ``my_ships - opp_ships``, not by our own
    production — so a capture is worth my gain over the hold PLUS the production
    it takes away from what would otherwise be the opponent's control:

      - ENEMY tgt: they own it now, so every turn I hold displaces them ->
        ``denial = my_hold`` (the principled, time-exact version of an enemy
        double-count). Holdable enemy planets are top value -> suppression.
      - CONTESTED neutral (``opp_reach`` not None): they'd take it at
        ``opp_reach`` regardless, so I only deny them for the turns I hold PAST
        that -> ``denial = max(0, min(my_hold, remaining) - max(0, opp_reach -
        eta))``. Nonzero only if I out-HOLD their reach (bring enough force);
        a brief pressure-grab denies ~nothing (they take it at opp_reach anyway).
      - SAFE deep neutral (``opp_reach is None``): ``denial = 0`` -> pure
        self-growth, ranked last.

    ``denial_weight`` (cfg) scales the denial term (1.0 = the true differential).
    """
    owner = int(tgt.owner)
    if owner == view.me or my_hold <= 0:
        return 0.0

    remaining = max(0, cfg.game_horizon - view.step - int(eta))
    if owner == -1:
        if opp_reach is None:
            denial = 0.0
        else:
            denial = float(max(0, min(int(my_hold), remaining) - max(0, int(opp_reach) - int(eta))))
    else:
        denial = float(my_hold)

    weakest_mult = 1.0
    if view.num_seats == 4 and owner != -1:
        weakest = _weakest_seat(view)
        if weakest is not None and owner == weakest:
            weakest_mult = 1.0 + cfg.weakest_opp_bias

    return weakest_mult * float(tgt.production) * (float(my_hold) + cfg.denial_weight * denial)


def preserve_value(view, planet, fall_turn, cfg) -> float:
    """Differential value of reinforcing one of my planets the timeline says
    falls at ``fall_turn``. Losing it to the opponent is a DOUBLE swing (they
    gain the stream, I lose it), so preventing the fall is worth ``production *
    (remaining - fall_turn) * (1 + denial_weight)`` — defense competes with
    offense on the same differential scale.
    """
    keep = max(0, cfg.game_horizon - view.step - int(fall_turn))
    return float(planet.production) * float(keep) * (1.0 + cfg.denial_weight)
