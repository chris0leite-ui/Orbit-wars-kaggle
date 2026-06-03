"""Per-turn parsed view of the board.

One ``build_turn_view`` call parses the obs into a ``World`` + ``WorldModel``
(each built once) plus the classified planet sets the chooser needs. The
opponent enters this view ONLY as physical strength (for the 4P
weakest-target bias) — never as a modelled policy. Everything opponent-
*reach* related lives in ``threat.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

from lib.intent import World
from lib.world_model import WorldModel


def _as_dict(obs) -> dict:
    """Coerce obs (dict on the live ladder, Struct in-memory) to a plain dict."""
    if isinstance(obs, dict):
        return obs
    return {
        "player": getattr(obs, "player", 0),
        "step": getattr(obs, "step", 0),
        "planets": list(getattr(obs, "planets", []) or []),
        "fleets": list(getattr(obs, "fleets", []) or []),
        "comets": list(getattr(obs, "comets", []) or []),
        "comet_planet_ids": list(getattr(obs, "comet_planet_ids", []) or []),
        "angular_velocity": float(getattr(obs, "angular_velocity", 0.0)),
    }


def _num_seats(planets) -> int:
    """2 unless any planet is owned by seat >= 2 (then a 4P free-for-all)."""
    max_owner = -1
    for p in planets:
        if int(p.owner) > max_owner:
            max_owner = int(p.owner)
    return 4 if max_owner >= 2 else 2


@dataclass
class TurnView:
    world: object
    model: object
    me: int
    step: int
    omega: float
    my_sources: list      # owner == me and ships > 0  (can launch this turn)
    my_planets: list      # owner == me               (need defending)
    targets: list         # owner != me               (enemy + neutral)
    enemy_planets: list   # owner not in (me, -1)
    opp_strength: dict     # seat -> ships + production + in-flight ships
    num_seats: int


def build_turn_view(obs, cfg) -> TurnView:
    obs_d = _as_dict(obs)
    world = World.from_obs(obs_d)
    me = int(world.my_id)
    step = int(world.step)
    model = WorldModel.from_world(world)

    planets = list(world.planets_by_id.values())
    my_planets = [p for p in planets if int(p.owner) == me]
    my_sources = [p for p in my_planets if int(p.ships) > 0]
    targets = [p for p in planets if int(p.owner) != me]
    enemy_planets = [p for p in planets if int(p.owner) not in (me, -1)]

    # Opponent strength for the 4P weakest-target bias: garrison + one turn
    # of production per enemy planet, plus their in-flight ships. A cheap
    # "how big is this player" proxy, NOT a behaviour model.
    opp_strength: dict[int, float] = {}
    for p in enemy_planets:
        seat = int(p.owner)
        opp_strength[seat] = opp_strength.get(seat, 0.0) + float(p.ships) + float(p.production)
    for arrivals in model.ledger.values():
        for (_eta, owner, ships) in arrivals:
            seat = int(owner)
            if seat != me and seat != -1:
                opp_strength[seat] = opp_strength.get(seat, 0.0) + float(ships)

    return TurnView(
        world=world,
        model=model,
        me=me,
        step=step,
        omega=float(world.omega),
        my_sources=my_sources,
        my_planets=my_planets,
        targets=targets,
        enemy_planets=enemy_planets,
        opp_strength=opp_strength,
        num_seats=_num_seats(planets),
    )
