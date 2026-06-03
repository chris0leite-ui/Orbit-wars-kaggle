"""Worst-case physical reach — the ONLY opponent-aware module.

holdgrab never models an opponent's policy. It treats the opponent purely
as physics: "how soon, and with how many ships, could the nearest enemy
force reach this planet?". Two quantities the chooser asks for:

  * ``opp_reach_tick``    — earliest tick any enemy can have a fleet here
                            (reuses ``WorldModel.time_to_enemy_threat``,
                            which already folds in-flight fleets + potential
                            launches + orbital lead).
  * follow-on force       — how many enemy ships could land here. The posture
                            is ASYMMETRIC (research synthesis): OPTIMISTIC on
                            offense (the single nearest enemy source, so we
                            don't go passive imagining the whole map ganging
                            up on one capture), WORST-CASE on defense (the
                            strongest single enemy seat's full reachable muster
                            plus everything already in flight, so we never
                            leave a planet we'd lose).

Combat here is the Lanchester *linear* law (survivors = attacker - defender),
so "ships that can land" is the immediate strike at each source's CURRENT
garrison; muster-growth over many turns is a later refinement that per-turn
re-solve + our own post-capture production already cover.
"""

from __future__ import annotations

import math

from lib.fleet import speed as fleet_speed
from lib.world_model import _position_at


def opp_reach_tick(view, tgt_id, arrival_eta):
    """Earliest tick an enemy can have a fleet at ``tgt_id`` (``None`` if never).

    ``arrival_eta`` > 0 predicts target + enemy positions at our arrival, so
    an orbiting target that rotates into enemy space by the time we land is
    scored against its arrival-time neighbourhood, not its current one.
    """
    return view.model.time_to_enemy_threat(
        int(tgt_id), int(view.me), view.world, arrival_eta=int(arrival_eta),
    )


def _enemy_landable(view, target, arrival_eta, window):
    """``[(seat, ships, arrival_tick), ...]`` for every enemy source whose
    fleet could reach ``target``'s position-at-arrival within ``window`` ticks
    of our arrival, counting the source's CURRENT garrison.

    ``target`` may be one of our planets (defense) or a capture target
    (offense) — only its position is used.
    """
    omega = view.omega
    tx, ty = _position_at(target, omega, int(arrival_eta))
    out = []
    for p in view.enemy_planets:
        if int(p.id) == int(target.id) or float(p.ships) <= 0:
            continue
        px, py = _position_at(p, omega, int(arrival_eta))
        d = math.hypot(tx - px, ty - py)
        v = fleet_speed(int(p.ships))
        if v <= 0:
            continue
        eta_travel = int(math.ceil(d / v))
        if eta_travel > window:
            continue
        out.append((int(p.owner), float(p.ships), int(arrival_eta) + eta_travel))
    return out


def contest_force(view, target, arrival_eta, window):
    """The enemy's full reachable counter to a planet we'd own at
    ``arrival_eta`` — bocsimacko's "full attack", bounded to a reaction
    ``window``: the strongest single enemy SEAT's summed ships that can reach
    ``target`` within ``window`` ticks of our arrival, plus enemy ships already
    in flight aimed here arriving after us.

    Bounding to a window (rather than every hypothetical muster forever) is
    what lets us attack when we hold a force lead instead of turtling. Used to
    both SIZE a capture (bring enough to beat this) and GATE it (only commit
    if we can afford to beat it) — the force-based full-attack-future test.
    """
    by_seat: dict[int, float] = {}
    for (seat, ships, _tick) in _enemy_landable(view, target, arrival_eta, window):
        by_seat[seat] = by_seat.get(seat, 0.0) + ships
    worst = max(by_seat.values()) if by_seat else 0.0

    inflight = 0.0
    for (eta, owner, ships) in view.model.ledger.get(int(target.id), []):
        if int(owner) != int(view.me) and int(owner) != -1 and int(eta) > int(arrival_eta):
            inflight += float(ships)
    return worst + inflight


def defense_follow_on(view, my_planet, window):
    """Ships to reserve on one of MY planets to survive COMMITTED, in-flight
    enemy attacks — bocsimacko's "surplus": what is safe to send is bounded
    by fleets already in space, NOT by every hypothetical future launch.

    Reserving against hypothetical musters (every nearby enemy planet could
    launch) is unbounded as the opponent grows and turtles us to death — the
    rf over-pessimism spiral. Hypothetical launches are handled where they
    belong: the offense gate refuses to grab contested planets, and per-turn
    re-solve lets us pull back the instant an enemy actually commits a fleet.

    Floor = max over each committed enemy wave of ``wave_ships - production *
    wave_eta`` (the planet produces while the wave flies, so it only needs to
    reserve the shortfall). Same-tick enemy fleets sum; sequential waves are
    each met by the refilled garrison, so we take the per-wave max, not the
    cumulative sum.
    """
    me = int(view.me)
    prod = float(my_planet.production)
    by_eta: dict[int, float] = {}
    for (eta, owner, ships) in view.model.ledger.get(int(my_planet.id), []):
        if int(owner) != me and int(owner) != -1 and int(eta) <= window:
            by_eta[int(eta)] = by_eta.get(int(eta), 0.0) + float(ships)
    floor = 0.0
    for eta, ships in by_eta.items():
        shortfall = ships - prod * float(eta)
        if shortfall > floor:
            floor = shortfall
    return max(0.0, floor)
