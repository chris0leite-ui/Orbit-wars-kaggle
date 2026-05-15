"""Action representations for the foundation layer.

Two forms:
- `ActionSpec` — ergonomic Python form for one launch (a single
  fleet leaving a planet on a particular turn).
- `ActionTensor` — padded array form for JAX vmap'd evaluation.

The padded form has shape `(C, T, A, L)`:
    C = candidates (parallel proposals to score)
    T = horizon turns (per-turn launch schedule)
    A = agents (MAX_AGENTS = 4)
    L = launches per agent per turn (MAX_LAUNCH_PER_AGENT = 20)

This is a direct generalisation of the existing 2D
`(C, MAX_LAUNCH_PER_AGENT)` triple in
`lib.game.jax.jax_brute_search.candidate_emits_to_tensors`. The added
`T` axis encodes the launch turn; the added `A` axis encodes which
seat fires the launch (so we can carry opp-as-strategy actions in the
same tensor used by the evaluator).

Empty slots:
    pids[c, t, a, l]   = -1
    ships[c, t, a, l]  = 0
    angles[c, t, a, l] = 0.0  (ignored when ships == 0)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from lib.game.jax.jax_types import MAX_AGENTS, MAX_LAUNCH_PER_AGENT


@dataclass(frozen=True)
class ActionSpec:
    """One launch by one agent on one turn.

    `launch_turn=0` means "this turn"; positive values delay the launch
    by N turns into a hypothetical future (used by the Predictor's
    `arrival_ledger(hypothetical=...)` argument in Step 6).
    """

    from_planet_id: int
    dir_angle: float
    ships: int
    launch_turn: int = 0
    agent_id: int = 0


class ActionTensor(NamedTuple):
    """Padded action arrays for JAX vmap'd evaluation.

    Shapes (all `(C, T, A, L)`):
        pids   : int32   — source planet id, -1 = no-op
        angles : float32 — launch direction in radians
        ships  : int32   — fleet size, 0 = no-op
    """

    pids: np.ndarray
    angles: np.ndarray
    ships: np.ndarray

    @property
    def shape(self) -> tuple[int, int, int, int]:
        """Returns `(C, T, A, L)`. All three arrays share this shape."""
        return self.pids.shape


def specs_to_tensor(
    candidates: list[list[ActionSpec]],
    horizon: int,
    num_agents: int = MAX_AGENTS,
    max_launch: int = MAX_LAUNCH_PER_AGENT,
) -> ActionTensor:
    """Pack a list of candidate action sequences into an `ActionTensor`.

    `candidates[c]` is a list of `ActionSpec` for candidate c. Each
    spec is assigned to slot `(launch_turn, agent_id, slot)` where
    `slot` is auto-allocated in input order. Raises `ValueError` if a
    spec exceeds the bounds.
    """
    C = len(candidates)
    if C == 0:
        raise ValueError("specs_to_tensor: need at least one candidate")
    if horizon <= 0:
        raise ValueError(f"specs_to_tensor: horizon must be positive, got {horizon}")

    pids = -np.ones((C, horizon, num_agents, max_launch), dtype=np.int32)
    angles = np.zeros((C, horizon, num_agents, max_launch), dtype=np.float32)
    ships = np.zeros((C, horizon, num_agents, max_launch), dtype=np.int32)

    for c, specs in enumerate(candidates):
        next_slot: dict[tuple[int, int], int] = {}
        for spec in specs:
            t = spec.launch_turn
            a = spec.agent_id
            if not (0 <= t < horizon):
                raise ValueError(
                    f"candidate {c}: launch_turn={t} out of [0, {horizon})"
                )
            if not (0 <= a < num_agents):
                raise ValueError(
                    f"candidate {c}: agent_id={a} out of [0, {num_agents})"
                )
            slot = next_slot.get((t, a), 0)
            if slot >= max_launch:
                raise ValueError(
                    f"candidate {c}: more than {max_launch} launches at "
                    f"(turn={t}, agent={a})"
                )
            pids[c, t, a, slot] = spec.from_planet_id
            angles[c, t, a, slot] = spec.dir_angle
            ships[c, t, a, slot] = spec.ships
            next_slot[(t, a)] = slot + 1

    return ActionTensor(pids=pids, angles=angles, ships=ships)


def tensor_to_specs(tensor: ActionTensor) -> list[list[ActionSpec]]:
    """Unpack an `ActionTensor` back into a list of `ActionSpec` lists.

    Inverse of `specs_to_tensor` (modulo slot ordering — output is
    sorted by `(launch_turn, agent_id, slot)`; empty slots are
    dropped).
    """
    pids = np.asarray(tensor.pids)
    angles = np.asarray(tensor.angles)
    ships = np.asarray(tensor.ships)
    C, T, A, L = pids.shape
    out: list[list[ActionSpec]] = []
    for c in range(C):
        specs: list[ActionSpec] = []
        for t in range(T):
            for a in range(A):
                for slot in range(L):
                    if pids[c, t, a, slot] < 0 or ships[c, t, a, slot] <= 0:
                        continue
                    specs.append(
                        ActionSpec(
                            from_planet_id=int(pids[c, t, a, slot]),
                            dir_angle=float(angles[c, t, a, slot]),
                            ships=int(ships[c, t, a, slot]),
                            launch_turn=t,
                            agent_id=a,
                        )
                    )
        out.append(specs)
    return out
