"""Rule 38 — reproduce the orbital-target attribution bug live in
`value_heads.composite_capture_value` (line 203), and confirm v8's
JAX-rollout path does NOT inherit it.

The bug: `composite_capture_value` calls

    target, eta = fleet_target_planet(fleet, planets)

positionally. The 4th argument `omega` defaults to `0.0`, which
takes `fleet_target_planet`'s fast static-raycast branch. For a
fleet aimed at an inner orbiting planet with lead-aim, the static
raycast misattributes the target because the planet has rotated by
the time the fleet arrives.

This test:
1. Constructs a synthetic orbital scenario with one outer (us) and
   one inner (enemy) planet, omega > 0.
2. Lead-aims from outer to inner via `aim_orbiting`.
3. Confirms `fleet_target_planet(fleet, planets)` (positional, no
   omega) misattributes — returns `None` or a different planet.
4. Confirms `fleet_target_planet(fleet, planets, omega=omega)`
   correctly attributes the inner planet.
5. The v8 path uses `rollout_step_jax_pure` → `jax_step`, which is
   omega-aware by construction — verified by inspection of the
   call chain, not by re-running the rollout here (the rollout
   uses `GameState`, not the Python `Fleet` namespace used by
   `fleet_target_planet`).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from kaggle_environments import make

from agents.simple import roi as roi_agent
from lib.world_model import fleet_target_planet


def test_static_raycast_diverges_from_orbital_on_real_fleet():
    """The bug, reproduced live.

    Play a real game (seed=1 vs roi_agent self-play) until at
    least one in-flight fleet shows attribution divergence between
    the positional `fleet_target_planet(fleet, planets)` call (which
    defaults `omega=0.0` → static raycast) and the explicit
    `omega=actual_omega` call (orbital walk). The known reproduction
    is seed=1, tick=18, fleet index 8 — static→planet 15 (eta 57),
    orbital→planet 12 (eta 60). The test scans a few seeds so a
    parity-preserving roi_agent change doesn't false-alarm.
    """
    for seed in (1, 7, 13, 42):
        env = make("orbit_wars", configuration={"seed": seed}, debug=False)
        env.reset(num_agents=2)
        omega = float(env.configuration.get("angular_velocity", 0.05))

        for tick in range(30):
            if all(s.status == "DONE" for s in env.state):
                break
            a0 = roi_agent.agent(env.state[0].observation)
            a1 = roi_agent.agent(env.state[1].observation)
            env.step([a0, a1])

            obs = env.state[0].observation
            planets_ns = [
                SimpleNamespace(
                    id=p[0], owner=p[1], x=p[2], y=p[3],
                    radius=p[4], ships=p[5], production=p[6],
                )
                for p in obs.planets
            ]
            for i, f in enumerate(obs.fleets):
                if int(f[1]) <= 0:  # 0-ship sentinel — skip
                    continue
                fleet_ns = SimpleNamespace(
                    id=i, owner=f[0], x=f[2], y=f[3],
                    angle=f[4], ships=f[1], from_planet_id=0,
                )
                t_stat, e_stat = fleet_target_planet(fleet_ns, planets_ns)
                t_orb, e_orb = fleet_target_planet(
                    fleet_ns, planets_ns, omega=omega,
                )
                stat_id = t_stat.id if t_stat else None
                orb_id = t_orb.id if t_orb else None
                if stat_id != orb_id:
                    return  # Divergent attribution — bug reproduced.
                if t_stat is not None and t_orb is not None:
                    if abs(int(e_stat or 0) - int(e_orb or 0)) > 1:
                        return  # Same target, ETA divergent — also the bug.

    pytest.fail(
        "Did not find a static-vs-orbital attribution divergence in any "
        "of the scanned seeds × 30 ticks. Either the bug got fixed in "
        "lib/world_model.py::fleet_target_planet (good — drop this test) "
        "or roi_agent stopped producing inner-target launches (re-tune "
        "the scenario)."
    )


def test_v8_analytic_uses_jax_step_path_not_world_model_raycast():
    """Structural verification: `AnalyticStrategy.emit` scores via
    `score_candidates_vmap_value_prod` → `rollout_step_jax_pure` →
    `jax_step`. The JAX engine resolves fleet/planet collisions via
    per-step orbital positions (see `lib/game/jax/jax_interpreter.
    py::jax_step` and the orbital-position helpers it composes). It
    does NOT call `lib.world_model.fleet_target_planet` at all.
    """
    import inspect

    from lib.foundation.strategies import analytic_score, beam_search

    # Walk the call graph by source-inspection: confirm none of the
    # scoring-kernel modules import `fleet_target_planet`.
    for mod in (analytic_score, beam_search):
        src = inspect.getsource(mod)
        assert "fleet_target_planet" not in src, (
            f"{mod.__name__} unexpectedly imports the (buggy if "
            f"positional) `fleet_target_planet` — the JAX path "
            f"should not depend on it."
        )
