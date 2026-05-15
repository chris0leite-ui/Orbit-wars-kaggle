"""Wrap `lib/missions/opp_archetypes.py` for Phase C's multi-turn JAX rollout.

The existing archetype functions consume env-format `obs` (dict) and
return env-format `[[src, angle, ships], ...]` actions. The Phase C
joint scorer needs per-turn padded arrays of shape
`(H, MAX_LAUNCH_PER_AGENT)`. This module bridges the two.

Phase C ships a 3-archetype panel by default (PI-ratified choice;
see the plan): `no_launch`, `v351`, `counter_snipe`. The other two
(`counter_reinforce`, `cross_attack`) correlate >=0.6 with members of
the kept set and add cost without changing the min-regret outcome
materially.

Each archetype acts on turn 0 only; turns `t >= 1` are zero-padded
no-ops. This bounds the joint-action space (no opp-of-opp recursion)
and matches the depth-2 precedent in `lib.v7_search`.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from lib.foundation.obs_to_state import _read_field
from lib.game.jax.jax_types import MAX_LAUNCH_PER_AGENT
from lib.intent import World
from lib.missions.opp_archetypes import (
    archetype_counter_snipe,
    archetype_no_launch,
    archetype_v351,
    opp_pov_obs,
)
from lib.world_model import WorldModel


# PI-ratified default panel.
DEFAULT_ARCHETYPE_NAMES: tuple[str, ...] = (
    "no_launch", "v351", "counter_snipe",
)


def _env_action_to_padded(
    env_action: list[list], H: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pack env-format `[[src, angle, ships], ...]` into per-turn arrays.

    Returns `(pids_h, angles_h, ships_h)` shape `(H, MAX_LAUNCH_PER_AGENT)`
    each. Turn 0 carries the archetype's launches (truncated to
    `MAX_LAUNCH_PER_AGENT`); turns `t >= 1` are no-ops (pids=-1, ships=0).
    """
    pids_h = -np.ones((H, MAX_LAUNCH_PER_AGENT), dtype=np.int32)
    angles_h = np.zeros((H, MAX_LAUNCH_PER_AGENT), dtype=np.float32)
    ships_h = np.zeros((H, MAX_LAUNCH_PER_AGENT), dtype=np.int32)
    for k, entry in enumerate(env_action[:MAX_LAUNCH_PER_AGENT]):
        pids_h[0, k] = int(entry[0])
        angles_h[0, k] = float(entry[1])
        ships_h[0, k] = int(entry[2])
    return pids_h, angles_h, ships_h


def build_opp_archetype_panel(
    raw_obs: Any,
    my_id: int,
    H: int,
    archetype_names: tuple[str, ...] = DEFAULT_ARCHETYPE_NAMES,
) -> list[tuple[str, np.ndarray, np.ndarray, np.ndarray]]:
    """Build the K_opp archetype panel for Phase C scoring.

    Inputs:
        raw_obs: the original Kaggle obs (dict or Struct). Used to
            derive opp's POV via `opp_pov_obs(raw_obs, opp_id)`.
        my_id: our seat (2P-only; opp_id = 1 - my_id).
        H: rollout horizon.
        archetype_names: subset of `{"no_launch","v351","counter_snipe",
            "counter_reinforce","cross_attack"}`. Default = PI panel.

    Returns a list of `(name, pids_h, angles_h, ships_h)` tuples. Each
    array is shape `(H, MAX_LAUNCH_PER_AGENT)`. Identical env actions
    across archetypes are dedup'd by row-equality (matches the existing
    `build_opp_archetypes` behaviour).
    """
    if my_id not in (0, 1):
        raise ValueError(f"build_opp_archetype_panel is 2P-only (my_id={my_id})")
    opp_id = 1 - my_id

    # Some archetypes ('counter_reinforce') would need our_intents; the
    # default panel doesn't include it, so we skip the our_intents lift.
    opp_obs = opp_pov_obs(raw_obs, opp_id)

    # Build opp POV state ONCE (shared across archetypes that need it).
    try:
        opp_world = World.from_obs(opp_obs)
    except Exception:
        # If POV obs is malformed, fall back to no-launch only.
        nl_pids, nl_ang, nl_ships = _env_action_to_padded([], H)
        return [("no_launch", nl_pids, nl_ang, nl_ships)]
    if not opp_world.planets_by_id:
        nl_pids, nl_ang, nl_ships = _env_action_to_padded([], H)
        return [("no_launch", nl_pids, nl_ang, nl_ships)]
    opp_model = WorldModel.from_world(opp_world)

    raw_actions: list[tuple[str, list[list]]] = []
    for name in archetype_names:
        if name == "no_launch":
            raw_actions.append((name, archetype_no_launch()))
        elif name == "v351":
            raw_actions.append((name, archetype_v351(opp_world, opp_model, opp_obs)))
        elif name == "counter_snipe":
            raw_actions.append((name, archetype_counter_snipe(opp_world, opp_obs)))
        else:
            raise ValueError(f"unknown archetype: {name!r}")

    # Dedup by row-equality; preserve first-occurrence order.
    seen: list[tuple[str, list[list]]] = []
    seen_actions: list[list[list]] = []
    for name, action in raw_actions:
        if action in seen_actions:
            continue
        seen_actions.append(action)
        seen.append((name, action))

    out: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []
    for name, action in seen:
        pids_h, angles_h, ships_h = _env_action_to_padded(action, H)
        out.append((name, pids_h, angles_h, ships_h))
    return out
