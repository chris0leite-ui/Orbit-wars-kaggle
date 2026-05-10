"""Intent + World + realize() — the substrate for the strategy/mechanism split.

A **strategy** emits `list[Intent]` ("attack planet T from source S with ~K
ships"). The **mechanism layer** (lib/mechanism.py) transforms that list
through a fixed pipeline (validate → arrival_size → lead_aim → comet_aim →
sun_avoid) before `realize()` emits env-format actions.

This separation lets every future strategy inherit the obvious-rule wins
(production-aware sizing, sun-avoidance, comet-path leading) uniformly,
and lets us run ablation tournaments where the same strategy plays with
different mechanism subsets to measure each mechanism's contribution.
"""

from __future__ import annotations

from dataclasses import dataclass

from kaggle_environments.envs.orbit_wars.orbit_wars import Planet


@dataclass
class Intent:
    """A strategy's request for a single fleet launch.

    `ships` is the strategy's *desired* size; `arrival_size` (the mechanism)
    may revise it upward to account for production growth during flight.
    `aim_angle` starts None and is populated by `lead_aim` / `comet_aim` /
    `lead_aim_v2`. `arrival_xy` is populated by `lead_aim_v2` (and by
    `comet_aim` when re-enabled) so downstream mechanisms (`sun_avoid`,
    `path_clears_other_planets`, `oob_guard`) can check the actual fleet
    path endpoint rather than the target's current position. Mechanisms
    may also drop intents (e.g. `validate`, `sun_avoid` when no detour
    exists).
    """
    src_id: int
    target_id: int
    ships: int
    aim_angle: float | None = None
    arrival_xy: tuple[float, float] | None = None
    note: str = ""


@dataclass
class World:
    """Frozen-once-per-turn view over an obs.

    Built once at the top of `realize()` and passed to every mechanism so
    each one is a pure function of `(intents, world)` — easy to test, easy
    to reorder. `obs_raw` is kept for mechanisms that need fields not yet
    materialised here (e.g. comet paths in `comet_aim`).
    """
    my_id: int
    planets_by_id: dict[int, "Planet"]
    omega: float
    comet_ids: frozenset[int]
    step: int
    obs_raw: object

    @classmethod
    def from_obs(cls, obs) -> "World":
        my_id = obs.get("player", 0) if isinstance(obs, dict) else obs.player
        raw_planets = obs.get("planets", []) if isinstance(obs, dict) else obs.planets
        omega = (
            float(obs.get("angular_velocity", 0.0))
            if isinstance(obs, dict)
            else float(getattr(obs, "angular_velocity", 0.0))
        )
        raw_comet_ids = (
            obs.get("comet_planet_ids", [])
            if isinstance(obs, dict)
            else getattr(obs, "comet_planet_ids", [])
        )
        step = (
            int(obs.get("step", 0))
            if isinstance(obs, dict)
            else int(getattr(obs, "step", 0))
        )
        planets_by_id = {p[0]: Planet(*p) for p in raw_planets}
        comet_ids = frozenset(int(c) for c in raw_comet_ids) if raw_comet_ids else frozenset()
        return cls(
            my_id=my_id,
            planets_by_id=planets_by_id,
            omega=omega,
            comet_ids=comet_ids,
            step=step,
            obs_raw=obs,
        )


def realize(intents, obs, *, mechanisms) -> list[list]:
    """Apply the mechanism pipeline and emit env-format actions.

    Final emission to `[src_id, aim_angle, ships]` lists is hard-coded —
    NOT a user-pluggable mechanism. Intents missing `aim_angle` or with
    `ships <= 0` after the pipeline are silently dropped (treated as
    validation failures).
    """
    world = World.from_obs(obs)
    for m in mechanisms:
        intents = m(intents, world)
    return [
        [i.src_id, i.aim_angle, i.ships]
        for i in intents
        if i.ships > 0 and i.aim_angle is not None
    ]
