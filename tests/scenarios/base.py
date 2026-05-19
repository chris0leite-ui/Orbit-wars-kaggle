"""Scenario substrate — observation-grounded gate for the ROI rebuild.

Each `Scenario` constructs a synthetic obs, drives the agent under test
(single-turn or K-turn multi-turn rollout under `lite_greedy_policy`),
and validates an outcome predicate. Failing a scenario means the agent
exhibits one of PI's named failure modes (a-e) on a constructed example.

Reuses `_planet`/`_obs` shapes from `tests/test_bundle_oracles.py:51-104`
but parameterises the agent dispatch so the same scenario runs against
`agents.baseline.main`, `agents.bundle.main`, and the forthcoming
`agents.trajectory_roi.main`.

Phase 1b scope: substrate + DI1 + G1.
Phase 1c will add S1-S3, SM1, R1, D1.
"""

from __future__ import annotations

import importlib
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

from lib import fast_sim
from lib.opp_model import lite_greedy_policy
from lib.trajectory import predict_fleet_fate
from lib.trajectory_layer import World


# ---- obs construction helpers (mirror tests/test_bundle_oracles.py) -------


def _planet(pid: int, owner: int, x: float, y: float,
            ships: int = 10, production: int = 1,
            radius: float = 1.5) -> list:
    return [int(pid), int(owner), float(x), float(y),
            float(radius), int(ships), int(production)]


def _obs(planets: list[list], fleets: list[list] | None = None,
        step: int = 0, player: int = 0,
        angular_velocity: float = 0.0) -> dict:
    return {
        "player": int(player),
        "step": int(step),
        "planets": planets,
        "fleets": fleets or [],
        "comets": [],
        "comet_planet_ids": [],
        "angular_velocity": float(angular_velocity),
        "initial_planets": [list(p) for p in planets],
    }


def _targets_of_emits(obs: dict, moves: list[list]) -> list[int]:
    """Raycast each emit and return planet IDs of first-hit targets."""
    pmap = {p[0]: p for p in obs["planets"]}
    out: list[int] = []
    for m in moves:
        src = pmap.get(int(m[0]))
        if src is None:
            continue
        sx, sy = src[2], src[3]
        ang = float(m[1])
        dx, dy = math.cos(ang), math.sin(ang)
        best_id, best_d = None, float("inf")
        for p in obs["planets"]:
            if p[0] == src[0]:
                continue
            ex, ey, pr = p[2] - sx, p[3] - sy, p[4]
            proj = ex * dx + ey * dy
            if proj < 0:
                continue
            perp_sq = ex * ex + ey * ey - proj * proj
            if perp_sq >= pr * pr:
                continue
            hit = proj - math.sqrt(pr * pr - perp_sq)
            if hit < best_d:
                best_d, best_id = hit, p[0]
        if best_id is not None:
            out.append(best_id)
    return out


# ---- agent dispatch ------------------------------------------------------


def _run_agent(agent_module: str, obs: dict, configuration: Any = None) -> list:
    """Import `agent_module` and call its `agent(obs, configuration)`."""
    mod = importlib.import_module(agent_module)
    return mod.agent(obs, configuration)


def _obs_from_snapshot(snap: fast_sim.Snapshot, seat: int = 0) -> dict:
    """Materialise a dict-form obs from a fast_sim Snapshot's seat 0/1."""
    s_obs = snap.state[seat].observation
    return {
        "player": int(getattr(s_obs, "player", seat)),
        "step": int(getattr(s_obs, "step", 0)),
        "planets": [list(p) for p in s_obs.planets],
        "fleets": [list(f) for f in (s_obs.fleets or [])],
        "comets": list(getattr(s_obs, "comets", [])),
        "comet_planet_ids": list(getattr(s_obs, "comet_planet_ids", [])),
        "angular_velocity": float(getattr(s_obs, "angular_velocity", 0.0)),
        "initial_planets": [list(p) for p in getattr(s_obs, "initial_planets", s_obs.planets)],
    }


def _rollout(agent_module: str, initial_obs: dict, K: int,
             ) -> tuple[list[list], list[dict]]:
    """Drive `agent_module` for K turns vs `lite_greedy_policy` opp.

    Returns (emit_log, world_log):
      - emit_log[t] = agent's emit list for OUR seat at the start of turn t.
        Length K.
      - world_log[t] = obs for OUR seat at the START of turn t (turn 0 is
        the initial obs). Plus a final entry capturing the AFTER-K-STEPS
        state. Length K+1.
    """
    snap = fast_sim.from_obs(initial_obs, configuration=None)
    emit_log: list[list] = []
    world_log: list[dict] = []
    for _ in range(K):
        obs_us = _obs_from_snapshot(snap, seat=0)
        world_log.append(obs_us)
        emit_us = _run_agent(agent_module, obs_us)
        obs_opp = _obs_from_snapshot(snap, seat=1)
        emit_opp = lite_greedy_policy(obs_opp)
        emit_log.append(emit_us)
        snap = fast_sim.step(snap, [emit_us, emit_opp])
        if snap.fake_env.done:
            break
    # Append the final post-rollout state so validate() can read end-of-
    # rollout planet ownership / ship counts.
    world_log.append(_obs_from_snapshot(snap, seat=0))
    return emit_log, world_log


# ---- Scenario ABC --------------------------------------------------------


@dataclass
class ValidationResult:
    passed: bool
    explanation: str


class Scenario(ABC):
    """Base class for observation-grounded scenarios.

    Subclasses set `name`, `rationale`, `source`, `flavour` as plain
    class attributes; override `setup()` and `validate()`. Optionally
    override `self_check()` to add layout invariants (e.g. "the
    prohibited launch is physically reachable, otherwise this test is
    vacuous").

    Not a dataclass: subclasses set class attributes directly and we
    don't want `@dataclass` inheritance complicating that.
    """

    name: str = ""
    rationale: str = ""
    source: str = ""
    flavour: Literal["single-turn", "multi-turn"] = "single-turn"
    rollout_K: int = 10  # only consulted for multi-turn flavour

    @abstractmethod
    def setup(self) -> dict:
        """Return the initial obs dict."""

    @abstractmethod
    def validate(self, emit_log: list[list],
                 world_log: list[dict]) -> ValidationResult:
        """Decide PASS / FAIL given the per-turn agent emits + obs."""

    def self_check(self) -> ValidationResult:
        """Default: round-trip via World.from_obs.

        Subclasses with MUST-launch / MUST-NOT-launch invariants
        override this and stack assertions on top via super().
        """
        obs = self.setup()
        try:
            world = World.from_obs(obs)
        except Exception as exc:  # noqa: BLE001
            return ValidationResult(False, f"World.from_obs failed: {exc!r}")
        if not world.planets:
            return ValidationResult(False, "World has no planets after from_obs")
        return ValidationResult(True, "self-check ok")

    def run(self, agent_module: str) -> ValidationResult:
        """Drive the scenario and return the validation result."""
        chk = self.self_check()
        if not chk.passed:
            return ValidationResult(False, f"self-check failed: {chk.explanation}")
        obs = self.setup()
        if self.flavour == "single-turn":
            emit = _run_agent(agent_module, obs)
            return self.validate([emit], [obs])
        # multi-turn
        emit_log, world_log = _rollout(agent_module, obs, self.rollout_K)
        return self.validate(emit_log, world_log)


# ---- public registry -----------------------------------------------------


_REGISTERED: list[type[Scenario]] = []


def register(cls: type[Scenario]) -> type[Scenario]:
    """Class decorator to add a Scenario subclass to the suite."""
    _REGISTERED.append(cls)
    return cls


def all_scenarios() -> list[Scenario]:
    return [cls() for cls in _REGISTERED]
