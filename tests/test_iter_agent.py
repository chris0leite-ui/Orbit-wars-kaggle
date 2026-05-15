"""Tests for the iter fast-iteration scaffold.

The iter agent (`agents/iter/main.py`) is a thin fork of v7_pv. Two
invariants we guard:

1. The module exports a callable `agent` and completes a 1-game smoke
   against a stock kaggle_environments opponent without crashing.
2. Importing `agents.iter.main` enforces `lib.scoring.PV_GAMMA == 0.99`,
   which is what makes day-zero iter functionally equivalent to v7_pv.
   v7_pv was never a committed source file (it was a bundled artifact
   produced with `scripts/ab_variants.py --variant pv PV_GAMMA=0.99`),
   so the parity contract lives in this PV_GAMMA invariant plus
   `tests/test_jax_pv_horizon_parity.py` which verifies the PV math.
"""

from __future__ import annotations

import importlib
import math
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


@pytest.fixture()
def iter_agent_module():
    # Force a fresh import: PV_GAMMA override only fires on first load.
    for name in list(sys.modules):
        if name.startswith("agents.iter") or name == "lib.scoring":
            del sys.modules[name]
    mod = importlib.import_module("agents.iter.main")
    return mod


def test_iter_agent_callable(iter_agent_module):
    assert callable(iter_agent_module.agent)


def test_iter_pv_gamma_is_enforced(iter_agent_module):
    import lib.scoring as scoring
    assert scoring.PV_GAMMA == 0.99, (
        f"iter must enforce PV_GAMMA=0.99 (v7_pv parity); got {scoring.PV_GAMMA}"
    )
    # Also assert the agent module's own constant matches.
    assert iter_agent_module.PV_GAMMA == 0.99


def test_iter_knob_constants_present(iter_agent_module):
    # If any knob name disappears, downstream sweep scripts will need
    # updating — surface it here instead of via silent agent regression.
    for knob in ("K", "WALLCLOCK_MS", "ENUMERATOR_MODE", "OPP_TIERS",
                 "PV_GAMMA", "VALUE_FN", "TERRITORY_WEIGHT", "K_4P",
                 "K_CAP", "K_BUFFER", "RELEVANCE_PROD_FRACTION",
                 "COMET_MAX_LAUNCHES_PER_TURN",
                 "COMET_EVAC_THRESHOLD", "COMET_EVAC_RESERVE",
                 "TWO_PHASE", "PHASE1_HORIZON", "PHASE2_TOP_K", "K_DEEP",
                 "LATEST_LAUNCH_ENABLED", "LATEST_LAUNCH_BUFFER_TURNS",
                 "LATEST_LAUNCH_MIN_FLEET"):
        assert hasattr(iter_agent_module, knob), f"missing knob: {knob}"


def test_max_inflight_eta_zero_when_no_fleets(iter_agent_module):
    # Fresh env has no in-flight fleets ⇒ max_eta=0 ⇒ K_eff falls back to K.
    from kaggle_environments import make
    from lib.intent import World
    env = make("orbit_wars", debug=False)
    env.reset(2)
    obs = env.state[0]["observation"]
    world = World.from_obs(obs)
    assert iter_agent_module._max_inflight_eta(world) == 0


def test_max_inflight_eta_positive_mid_game(iter_agent_module):
    # After a few turns of self-play there should be at least one fleet
    # in flight ⇒ max_eta > 0.
    from kaggle_environments import make
    from lib.intent import World
    env = make("orbit_wars", debug=False)
    env.reset(2)
    for _ in range(40):
        if env.done:
            break
        env.step([iter_agent_module.agent(env.state[0]["observation"], env.configuration),
                  None])
    obs = env.state[0]["observation"]
    world = World.from_obs(obs)
    # Either the game is over already (very rare in 40 turns) or there's a
    # fleet inbound somewhere. Both states are valid for this assertion.
    if env.done:
        return
    eta = iter_agent_module._max_inflight_eta(world)
    # Allow 0 only if obs.fleets is literally empty at this step.
    fleets_raw = (obs.get("fleets", []) if isinstance(obs, dict)
                  else getattr(obs, "fleets", []))
    if fleets_raw:
        assert eta > 0, f"fleets={len(fleets_raw)} but max_eta={eta}"


def test_cap_comet_launches_keeps_only_n_per_comet(iter_agent_module, monkeypatch):
    # Synthetic: 4 launches all target the same comet (id=99). Cap=2 ⇒
    # we should keep the 2 with shortest ETA. Monkeypatch _launch_eta so
    # the test doesn't depend on actual ray-cast geometry.

    class FakePlanet:
        def __init__(self, pid, owner=-1, ships=10, x=0.0, y=0.0):
            self.id = pid
            self.owner = owner
            self.ships = ships
            self.x = x
            self.y = y

    class FakeWorld:
        my_id = 0
        comet_ids = frozenset({99})
        planets_by_id = {
            99: FakePlanet(99, owner=1, ships=15, x=10, y=0),
            0: FakePlanet(0, owner=0, ships=20, x=0, y=0),
            1: FakePlanet(1, owner=0, ships=20, x=2, y=0),
            2: FakePlanet(2, owner=0, ships=20, x=4, y=0),
            3: FakePlanet(3, owner=0, ships=20, x=6, y=0),
        }

    def fake_launch_eta(src, angle, ships, planets_list):
        # All launches "hit" the comet; ETA = src.id, so srcs 0 and 1 win.
        comet = next(p for p in planets_list if p.id == 99)
        return comet, src.id

    monkeypatch.setattr(iter_agent_module, "_launch_eta", fake_launch_eta)
    action = [[i, 0.0, 5] for i in range(4)]
    capped = iter_agent_module._cap_comet_launches(action, FakeWorld(), 2)
    assert len(capped) == 2, f"expected 2 launches, got {len(capped)}: {capped}"
    kept_srcs = {e[0] for e in capped}
    assert kept_srcs == {0, 1}, f"expected sources {{0,1}} (shortest ETA), got {kept_srcs}"


def test_cap_comet_launches_passes_non_comet_through(iter_agent_module, monkeypatch):
    # Launches that don't target a comet should pass through unchanged.
    class FakePlanet:
        def __init__(self, pid, owner=-1, x=0.0, y=0.0, ships=10):
            self.id = pid; self.owner = owner; self.x = x; self.y = y; self.ships = ships

    class FakeWorld:
        my_id = 0
        comet_ids = frozenset({99})  # no entries in action target this
        planets_by_id = {
            99: FakePlanet(99, owner=1, x=10, y=0),
            5: FakePlanet(5, owner=1, x=5, y=0),
            0: FakePlanet(0, owner=0, x=0, y=0),
            1: FakePlanet(1, owner=0, x=1, y=0),
        }

    def fake_launch_eta(src, angle, ships, planets_list):
        # Both launches target planet id=5 (non-comet).
        non_comet = next(p for p in planets_list if p.id == 5)
        return non_comet, 3
    monkeypatch.setattr(iter_agent_module, "_launch_eta", fake_launch_eta)
    action = [[0, 0.0, 5], [1, 0.0, 5]]
    capped = iter_agent_module._cap_comet_launches(action, FakeWorld(), 2)
    assert capped == action, f"non-comet launches should pass through; got {capped}"


def test_comet_evacuation_emits_when_lifetime_short(iter_agent_module):
    # We can't force ownership of a comet from a fresh env easily.
    # Smoke test: the helper returns a list (possibly empty) without
    # crashing on a normal mid-game state. The action-variance test in
    # 4P+2P smokes already exercises the merge path at runtime.
    from kaggle_environments import make
    from lib.intent import World
    env = make("orbit_wars", debug=False)
    env.reset(2)
    for _ in range(40):
        if env.done:
            break
        env.step([iter_agent_module.agent(env.state[0]["observation"], env.configuration),
                  None])
    obs = env.state[0]["observation"]
    world = World.from_obs(obs)
    out = iter_agent_module._comet_evacuation_launches(world, 0)
    assert isinstance(out, list)
    for entry in out:
        assert len(entry) == 3, f"evac launch wrong shape: {entry}"
        src_id, angle, ships = entry
        # Ship count is meaningful and the source is one of our comets.
        assert ships >= 1
        src = world.planets_by_id.get(int(src_id))
        assert src is not None and src.owner == 0
        assert src.id in world.comet_ids


def test_iter_smoke_one_game_completes(iter_agent_module):
    # 1-game self-play smoke (iter vs the kaggle_environments `random`
    # builtin). Cheap — should run in <30 s on local CPU.
    from kaggle_environments import make
    env = make("orbit_wars", debug=False)
    env.run([iter_agent_module.agent, "random"])
    assert env.done, "env did not reach DONE"
    statuses = [s["status"] for s in env.steps[-1]]
    assert all(s in {"DONE", "INACTIVE"} for s in statuses), (
        f"unexpected terminal statuses: {statuses}"
    )


def test_iter_smoke_4p_game_completes(iter_agent_module):
    # 1-game 4P smoke. Proves iter actually runs in 4P after the choose_with_4p
    # dispatch swap — pre-swap iter would silently fall back to the v3.5.1
    # incumbent. Action-variance check ensures we're not stuck on a single
    # fixed incumbent action every turn.
    from kaggle_environments import make
    env = make("orbit_wars", debug=False, configuration={"actTimeout": 2.5})
    # Players-4 is the default 4P mode for orbit_wars; passing 4 explicit
    # agents auto-configures the 4-seat board.
    env.run([iter_agent_module.agent, "random", "random", "random"])
    assert env.done, "4P env did not reach DONE"
    # Action-variance check: collect iter's actions across the first 20
    # turns and confirm at least 2 distinct action shapes. If we always
    # returned the same action, we're probably stuck on a fallback.
    actions_p0 = []
    for step_record in env.steps[:20]:
        a = step_record[0].get("action")
        if a is not None:
            actions_p0.append(tuple(tuple(x) for x in a) if a else ())
    distinct = set(actions_p0)
    assert len(distinct) >= 2, (
        f"iter returned only {len(distinct)} distinct action(s) across 20 turns; "
        f"likely fell to incumbent fallback. Actions: {actions_p0[:5]}"
    )


def test_phase1_analytical_returns_finite_score(iter_agent_module):
    # Smoke: Phase 1 analytical scoring runs on a real mid-game obs without
    # crashing and returns a finite float. Both empty action (incumbent
    # equivalent of "do nothing") and a non-trivial action must work.
    from kaggle_environments import make
    from lib.intent import World
    env = make("orbit_wars", debug=False)
    env.reset(2)
    for _ in range(30):
        if env.done:
            break
        env.step([iter_agent_module.agent(env.state[0]["observation"], env.configuration),
                  None])
    obs = env.state[0]["observation"]
    world = World.from_obs(obs)
    s_empty = iter_agent_module._score_phase1_analytical(world, [], world.my_id, 50)
    assert math.isfinite(s_empty), f"empty-action score not finite: {s_empty}"
    # A 1-launch action from our first owned planet at a default angle.
    mine = [p for p in world.planets_by_id.values() if p.owner == world.my_id]
    if mine:
        s_action = iter_agent_module._score_phase1_analytical(
            world, [[int(mine[0].id), 0.0, 5]], world.my_id, 50
        )
        assert math.isfinite(s_action), f"action score not finite: {s_action}"


def test_shrink_to_min_viable_no_op_on_safe_target(iter_agent_module):
    # If no enemy threat → time_to_enemy_threat returns None → no shrinking.
    from types import SimpleNamespace
    from lib.intent import World

    me = SimpleNamespace(id=0, owner=0, x=0.0, y=0.0, radius=1.5, ships=50, production=2)
    enemy = SimpleNamespace(id=1, owner=1, x=20.0, y=0.0, radius=1.5, ships=10, production=2)
    obs = {
        "player": 0,
        "planets": [(p.id, p.owner, p.x, p.y, p.radius, p.ships, p.production)
                    for p in (me, enemy)],
        "fleets": [], "angular_velocity": 0.0, "comet_planet_ids": [], "step": 0,
    }
    world = World.from_obs(obs)
    from lib.world_model import WorldModel
    model = WorldModel.from_world(world)
    action = [[0, 0.0, 20]]   # send 20 ships from our planet
    out = iter_agent_module._shrink_to_min_viable(action, world, model)
    # No threat to enemy planet (we ARE the only threat) → time_to_enemy_threat
    # returns None for the enemy planet's perspective → no shrink.
    assert out == action, f"safe target should pass through; got {out}"


def test_territory_value_runs_under_5ms(iter_agent_module):
    # Bench gate: territory head must not blow the per-call budget. value_fn
    # is invoked once per candidate at rollout leaf; ~50 candidates per turn
    # × 5 ms each = 250 ms, leaving 450 ms headroom inside the 700 ms knob.
    import time
    from kaggle_environments import make
    from lib.value_heads import territory_value

    env = make("orbit_wars", debug=False)
    env.reset(2)
    # Roll forward 30 turns with iter on P0 so the obs has real ownership /
    # in-flight fleet state — neutral starting obs returns 0 trivially.
    for _ in range(30):
        if env.done:
            break
        env.step([iter_agent_module.agent(env.state[0]["observation"], env.configuration),
                  None])
    obs = env.state[0]["observation"]

    t0 = time.perf_counter()
    for _ in range(20):
        territory_value(obs, 0, weight=0.01)
    avg_ms = (time.perf_counter() - t0) * 1000.0 / 20
    assert avg_ms < 5.0, f"territory_value too slow: {avg_ms:.2f} ms/call"
