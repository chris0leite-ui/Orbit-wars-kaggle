"""Multi-turn atomic-launch enumeration for Phase C.

Extends Phase A's `enumerate_atomic_launches` (single-turn) by generating
atoms whose `launch_turn` ranges across a horizon `H`. For each
`launch_turn = t > 0`, source and target positions are rotated forward
by `omega * t` via `lib.orbit.predict_relative`; `lib.aim.aim_orbiting`
then computes the orbital intercept aim from the *future* src position
to the *future* target trajectory.

Why we need it: the Phase A beam picks single-turn plans (every launch
fires this turn). Multi-turn plans like "wave-of-3 capture on turn 0
THEN reinforce-from-same-source on turn 1" are invisible to that beam
because the (src, t=0) and (src, t=1) launches are not enumerated as
distinct atoms.

The cost is one extra `aim_orbiting` call per (src, target, fraction, t)
combination. For H=2, P_owned=5, P_total=24 we get ~5×24×2×2 ≈ 480
atoms — within the Tier-1 batch budget after eta-filtering.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from lib.aim import aim_orbiting
from lib.foundation.actions import ActionSpec
from lib.game.jax.jax_types import GameState
from lib.orbit import predict_relative


def enumerate_multi_turn_atoms(
    state: GameState,
    my_id: int,
    *,
    horizon: int = 2,
    ship_fractions: Tuple[float, ...] = (0.5, 1.0),
    max_eta: int = 80,
) -> list[ActionSpec]:
    """Strategy-agnostic multi-turn atomic-launch enumeration.

    For each `(src, target, fraction, launch_turn)` where:
        - `src.owner == my_id` and `src.ships > 1` at turn 0,
        - `target` is alive,
        - `launch_turn in range(horizon)`,
        - `fraction in ship_fractions`,
    rotate `src` and `target` positions forward by `omega * launch_turn`,
    compute orbit-aware intercept via `aim_orbiting`, drop if no valid
    intercept OR eta > `max_eta`.

    Returns a list of `ActionSpec` with `launch_turn` set per-atom.
    Source ships are conservatively read from CURRENT garrison (we do
    NOT add per-turn production over `launch_turn` turns; the JAX
    rollout's `jax_step` zero-truncates the launch if garrison is
    insufficient at fire-time).

    When `horizon == 1`, behaviour reduces to Phase A's
    `enumerate_atomic_launches`: only `launch_turn = 0` atoms emit.
    """
    out: list[ActionSpec] = []

    alive = np.asarray(state.planets_alive)
    ids = np.asarray(state.planets_id)
    owner = np.asarray(state.planets_owner)
    ships = np.asarray(state.planets_ships)
    x = np.asarray(state.planets_x)
    y = np.asarray(state.planets_y)
    radius = np.asarray(state.planets_radius)
    prod = np.asarray(state.planets_prod)
    omega = float(state.angular_velocity)

    P = len(alive)
    my_planets = [
        i for i in range(P)
        if bool(alive[i]) and int(owner[i]) == my_id and int(ships[i]) > 1
    ]
    all_targets = [
        i for i in range(P) if bool(alive[i]) and int(ids[i]) >= 0
    ]

    for t in range(horizon):
        for src_i in my_planets:
            src_id = int(ids[src_i])
            # planet tuple format for predict_relative: [id, owner, x, y, radius, ships, prod].
            src_planet = (
                src_id, int(owner[src_i]),
                float(x[src_i]), float(y[src_i]),
                float(radius[src_i]),
                int(ships[src_i]), int(prod[src_i]),
            )
            if t == 0:
                src_pos = (float(x[src_i]), float(y[src_i]))
            else:
                src_pos = predict_relative(src_planet, omega, float(t))
            src_radius = float(radius[src_i])
            src_ships_now = int(ships[src_i])

            for tgt_i in all_targets:
                if tgt_i == src_i:
                    continue
                tgt_planet = (
                    int(ids[tgt_i]), int(owner[tgt_i]),
                    float(x[tgt_i]), float(y[tgt_i]),
                    float(radius[tgt_i]),
                    int(ships[tgt_i]), int(prod[tgt_i]),
                )
                if t == 0:
                    tgt_x_at_t, tgt_y_at_t = float(x[tgt_i]), float(y[tgt_i])
                else:
                    tgt_x_at_t, tgt_y_at_t = predict_relative(
                        tgt_planet, omega, float(t),
                    )
                tgt_tuple = (
                    int(ids[tgt_i]), int(owner[tgt_i]),
                    tgt_x_at_t, tgt_y_at_t,
                    float(radius[tgt_i]),
                    int(ships[tgt_i]), int(prod[tgt_i]),
                )
                tgt_radius = float(radius[tgt_i])

                for fraction in ship_fractions:
                    fleet_ships = max(1, int(src_ships_now * fraction))
                    if fleet_ships > src_ships_now:
                        continue
                    aim = aim_orbiting(
                        src_pos, src_radius, tgt_tuple, tgt_radius,
                        fleet_ships, omega,
                    )
                    if aim is None:
                        continue
                    aim_angle, _arrival, eta = aim
                    if eta is None or eta > max_eta:
                        continue
                    out.append(ActionSpec(
                        from_planet_id=src_id,
                        dir_angle=float(aim_angle),
                        ships=fleet_ships,
                        launch_turn=t,
                        agent_id=my_id,
                    ))

    return out
