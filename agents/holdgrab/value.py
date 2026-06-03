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


def planet_value(view, tgt, hold_ticks, cfg) -> float:
    """Production-time-integral value of capturing ``tgt`` and holding it for
    ``hold_ticks`` turns: ``owner_mult * weakest_mult * production * hold_ticks``.

    ``hold_ticks`` is the caller's estimate of how long we keep it — the full
    remaining game for a hold-guaranteed capture, or a short denial window for
    a pressure capture we can't fully hold. Capturing an enemy planet earns the
    enemy_capture_weight (gain + denial double-count); a 4P weakest-target gets
    an extra bias.
    """
    owner = int(tgt.owner)
    if owner == view.me or hold_ticks <= 0:
        return 0.0

    owner_mult = cfg.enemy_capture_weight if owner != -1 else cfg.neutral_capture_weight

    weakest_mult = 1.0
    if view.num_seats == 4 and owner != -1:
        weakest = _weakest_seat(view)
        if weakest is not None and owner == weakest:
            weakest_mult = 1.0 + cfg.weakest_opp_bias

    return owner_mult * weakest_mult * float(tgt.production) * float(hold_ticks)


def preserve_value(view, planet, fall_turn, cfg) -> float:
    """Production *preserved* by reinforcing one of my planets that the timeline
    says falls at ``fall_turn`` — the stream from the fall to game-end that we'd
    otherwise lose. Same ``production x time`` units as a capture's *gained*
    value, so defense and offense compete on one scale (owner-weight 1.0: it's
    our own production, no denial double-count).
    """
    keep = max(0, cfg.game_horizon - view.step - int(fall_turn))
    return float(planet.production) * float(keep)
