"""Pin tests for `agents/lagrange_simple/` — the simplest Lagrangian agent.

Two layers:
  1. Score (Candidate enumeration with precision physics filters).
  2. Dual (3-sweep Lagrangian with shadow prices, feasibility fix-up).

Plus one end-to-end smoke test: agent plays a full game without raising.
"""
from __future__ import annotations

import math

import pytest

from kaggle_environments import make
from kaggle_environments.envs.orbit_wars.orbit_wars import Planet

from lib.intent import World
from lib.world_model import WorldModel

from agents.lagrange_simple.dual import _inner_solve, solve
from agents.lagrange_simple.main import agent
from agents.lagrange_simple.score import (
    Candidate,
    EPISODE_STEPS,
    MAX_LAUNCH_TICK,
    _source_defensive_ok,
    enumerate_candidates,
)


def _ctx_from_seed(seed: int, prerun_steps: int = 0):
    env = make("orbit_wars", configuration={"seed": int(seed)}, debug=False)
    env.reset()
    for _ in range(prerun_steps):
        obs0 = env.steps[-1][0]["observation"]
        obs1 = env.steps[-1][1]["observation"]
        env.step([agent(obs0, env.configuration),
                  agent(obs1, env.configuration)])
    obs = env.steps[-1][0]["observation"]
    if not isinstance(obs, dict):
        obs = {k: getattr(obs, k) for k in dir(obs) if not k.startswith("_")}
    return obs, env


def _c(src, tgt, ships, value, *, wait=0, eta=10):
    return Candidate(
        src_id=src, tgt_id=tgt, launch_tick=wait, angle=0.0,
        ships=ships, eta=eta, arrival_step=wait + eta, value=value,
    )


# ---------------------------------------------------------------------------
# Dual solver — math invariants.
# ---------------------------------------------------------------------------


def test_dual_empty_input_returns_empty():
    assert solve([], {}) == []


def test_dual_single_candidate_under_budget_picks_it():
    cands = [_c(src=0, tgt=1, ships=5, value=100.0)]
    picked = solve(cands, {0: 10})
    assert len(picked) == 1
    assert picked[0].tgt_id == 1


def test_dual_picks_higher_value_when_per_target_argmax():
    """Two candidates for the same target: argmax picks the higher V."""
    cands = [
        _c(src=0, tgt=5, ships=5, value=100.0),
        _c(src=1, tgt=5, ships=5, value=200.0),
    ]
    picked = solve(cands, {0: 10, 1: 10})
    assert len(picked) == 1
    assert picked[0].src_id == 1  # the higher-value one


def test_dual_respects_per_source_budget():
    """A single source with budget=5 but two candidates needing 5+5=10 ships
    must pick at most one of them (feasibility fix-up enforces it)."""
    cands = [
        _c(src=0, tgt=1, ships=5, value=100.0),
        _c(src=0, tgt=2, ships=5, value=100.0),
    ]
    picked = solve(cands, {0: 5})
    total_ships = sum(c.ships for c in picked)
    assert total_ships <= 5


def test_dual_shadow_price_redirects_to_cheaper_source():
    """Two sources both have a candidate for the same two targets. Sub-source
    is over-committed. After λ_s rises, the LATER sweep should re-route to
    the under-committed source.

    Here: src=0 has budget 5, src=1 has budget 20. Three targets, each with
    one cand from each source, ships=5, all V=100. With λ all-0 the picker
    grabs the first 3 wins (src=0 over-commits 15 ships). λ rises on src=0;
    next sweep, src=1 wins all three. Feasibility met.
    """
    cands = []
    for tid in (1, 2, 3):
        cands.append(_c(src=0, tgt=tid, ships=5, value=100.0))
        cands.append(_c(src=1, tgt=tid, ships=5, value=100.0))
    picked = solve(cands, {0: 5, 1: 20}, sweeps=3, step=1.0)
    # Every target captured (3 picks).
    assert len({c.tgt_id for c in picked}) == 3
    # All three captures should come from src=1 (the cheap source).
    assert all(c.src_id == 1 for c in picked), (
        f"shadow prices didn't redirect: {[(c.src_id, c.tgt_id) for c in picked]}"
    )


def test_dual_feasibility_fixup_keeps_best_per_source():
    """When λ-relaxed solution still over-commits a source after 3 sweeps
    (e.g. step too small), feasibility fix-up drops worst-(value/ships)
    candidates from over-budget sources until feasible."""
    # Both at wait=0 → cumulative=10 ships at t=0 ≤ R_s+P_s·0=5 violated.
    cands = [
        _c(src=0, tgt=1, ships=5, value=10.0, wait=0),    # bad value/ship = 2
        _c(src=0, tgt=2, ships=5, value=100.0, wait=0),   # great value/ship = 20
    ]
    picked = solve(cands, {0: 5}, {0: 0}, sweeps=1, step=0.0)  # P=0, R=5
    # After fix-up: keep the V=100 one, drop V=10.
    assert len(picked) == 1
    assert picked[0].value == 100.0


def test_dual_time_indexed_allows_late_launch_above_current_ships():
    """A candidate at wait=5 from a source with R=5, P=2 should be feasible
    even though ships=10 > R: at t=5 the source has R+P*5=15 ships."""
    cands = [_c(src=0, tgt=1, ships=10, value=100.0, wait=5)]
    picked = solve(cands, {0: 5}, {0: 2})
    assert len(picked) == 1


def test_dual_time_indexed_rejects_combined_overdraw_at_early_tick():
    """Two picks at wait=0 (both fire NOW) from src with R=5 P=10:
    cumulative-at-t=0 = sum(ships) must ≤ 5. Should drop one."""
    cands = [
        _c(src=0, tgt=1, ships=4, value=100.0, wait=0),
        _c(src=0, tgt=2, ships=4, value=80.0, wait=0),
    ]
    picked = solve(cands, {0: 5}, {0: 10}, sweeps=1, step=0.0)
    # Cumulative at t=0 is 4+4=8 > 5+10·0=5 → fix-up drops worst.
    assert len(picked) == 1
    assert picked[0].value == 100.0  # higher V/ship kept


# ---------------------------------------------------------------------------
# Score — Candidate enumeration on a real game state.
# ---------------------------------------------------------------------------


def test_enumerate_returns_nothing_on_empty_obs():
    """Edge: empty obs → no candidates (no crash)."""
    env = make("orbit_wars", configuration={"seed": 42}, debug=False)
    env.reset()
    obs = env.steps[0][0]["observation"]
    if not isinstance(obs, dict):
        obs = {k: getattr(obs, k) for k in dir(obs) if not k.startswith("_")}
    world = World.from_obs(obs)
    model = WorldModel.from_world(world)
    # my_id with no planets → empty
    bogus_me = 99
    res = enumerate_candidates(world, model, bogus_me, 0.0, set())
    assert res == []


def test_enumerate_candidate_invariants():
    """On a real mid-game state, every enumerated candidate must satisfy:
      - launch_tick ∈ [0, MAX_LAUNCH_TICK]
      - ships ≥ 1
      - arrival_step == launch_tick + eta
      - value > 0 (production stream still has remaining episode left)
      - ships ≤ time-indexed source budget (R_s + P_s · launch_tick)
    """
    obs, _env = _ctx_from_seed(42, prerun_steps=20)
    me = int(obs.get("player", 0))
    planets = [Planet(*p) for p in obs["planets"]]
    if not any(int(p.owner) == me for p in planets):
        pytest.skip("seat 0 lost at step 20")
    src_ships = {int(p.id): int(p.ships) for p in planets if int(p.owner) == me}
    src_prods = {int(p.id): int(p.production) for p in planets if int(p.owner) == me}

    world = World.from_obs(obs)
    model = WorldModel.from_world(world)
    omega = float(obs.get("angular_velocity", 0.0) or 0.0)
    comet_ids = set(int(c) for c in (obs.get("comet_planet_ids", []) or []))

    cands = enumerate_candidates(world, model, me, omega, comet_ids)
    for c in cands:
        assert 0 <= c.launch_tick <= MAX_LAUNCH_TICK
        assert c.ships >= 1
        assert c.arrival_step == c.launch_tick + c.eta
        assert c.value > 0.0
        budget_at_launch = (
            src_ships.get(c.src_id, 0)
            + src_prods.get(c.src_id, 0) * c.launch_tick
        )
        assert c.ships <= budget_at_launch, (
            f"candidate {c} exceeds time-indexed budget "
            f"({src_ships.get(c.src_id, 0)} + "
            f"{src_prods.get(c.src_id, 0)}*{c.launch_tick} = {budget_at_launch})"
        )


# ---------------------------------------------------------------------------
# End-to-end: agent plays a full game without raising.
# ---------------------------------------------------------------------------


def test_agent_plays_full_game_vs_random():
    """Smoke: lagrange_simple plays a full game against `random` baseline.

    Must not raise; must produce a valid terminal status.
    """
    env = make("orbit_wars", configuration={"seed": 42}, debug=False)
    env.reset()
    result = env.run([agent, "random"])
    final = result[-1]
    assert final[0]["status"] == "DONE"
    assert final[1]["status"] == "DONE"


def test_agent_returns_legal_moves_format():
    """Every emitted move must be a [src_id:int, angle:float, ships:int]
    triple with ships ≤ source.ships at that turn."""
    env = make("orbit_wars", configuration={"seed": 42}, debug=False)
    env.reset()
    obs = env.steps[0][0]["observation"]
    moves = agent(obs, env.configuration)
    if not moves:
        return  # vacuously OK
    planets = [Planet(*p) for p in obs["planets"]]
    me = int(obs["player"]) if isinstance(obs, dict) else int(obs.player)
    ships_by_id = {int(p.id): int(p.ships) for p in planets if int(p.owner) == me}
    for m in moves:
        assert isinstance(m, list) and len(m) == 3
        src_id, angle, ships = m
        assert isinstance(src_id, int)
        assert isinstance(angle, float)
        assert isinstance(ships, int) and ships >= 1
        assert src_id in ships_by_id
        assert ships <= ships_by_id[src_id]


# ---------------------------------------------------------------------------
# Regression: planet id 0 must be a viable capture target (Python `0 or -1`
# truthiness bug, 2026-05-23). When `fate.hit_planet_id == 0`, the old
# `int(fate.hit_planet_id or -1)` evaluated to -1, so we silently dropped
# every shot at planet id 0. This bug caused the seed-32966 random-elim
# gate failure (game ran 500 steps with opp's planet 0 untouched).
# ---------------------------------------------------------------------------


def test_source_defensive_ok_rejects_when_opp_counter_captures():
    """A src with 5 ships facing an incoming opp arrival of 10 ships at
    eta=3 cannot survive a launch of all 5 ships at launch_tick=0:
    after launch ships=0, by eta=3 production=0*3=0 (assume p.production=0)
    or 6 (if production=2), but opp's 10 ships wipe a 0-6-ship garrison
    and capture. Defensive check should return False."""
    from types import SimpleNamespace
    src = SimpleNamespace(id=7, owner=0, ships=5, production=0)
    src_arrivals = [(3, 1, 10)]  # opp arrives in 3 ticks with 10 ships
    assert _source_defensive_ok(src, 5, 0, src_arrivals, horizon=10) is False


def test_source_defensive_ok_passes_when_we_can_hold():
    """A src with 20 ships facing an opp arrival of 5 ships at eta=3
    after launching 5 ships at tick=0: ships=15 at tick=0, +0 prod per
    tick, then 5 opp ships hit; 15 - 5 = 10 ours stays positive."""
    from types import SimpleNamespace
    src = SimpleNamespace(id=7, owner=0, ships=20, production=0)
    src_arrivals = [(3, 1, 5)]
    assert _source_defensive_ok(src, 5, 0, src_arrivals, horizon=10) is True


def test_source_defensive_ok_no_opp_arrivals_always_passes():
    """Trivial: no incoming opp arrivals means launch can never cause a flip."""
    from types import SimpleNamespace
    src = SimpleNamespace(id=7, owner=0, ships=10, production=0)
    assert _source_defensive_ok(src, 10, 0, [], horizon=10) is True


def test_planet_id_zero_is_not_silently_dropped():
    """A candidate that successfully hits planet id 0 must NOT be filtered
    by the `hit_planet_id == 0` truthiness gotcha.

    Construct: a state where the only opp is planet id 0 and at least one
    of our planets has a clear shot (fate.outcome=='target', hit_id==0).
    Verify enumerate_candidates returns ≥1 candidate with tgt_id==0.

    Reproducible via the original failure: seed 32966 step 150 vs random.
    """
    import math
    import random as pyrandom
    rng = pyrandom.Random(42)

    def random_p0(obs, config):
        obs_d = obs if isinstance(obs, dict) else {
            k: getattr(obs, k) for k in dir(obs) if not k.startswith("_")
        }
        me = obs_d.get("player", 0)
        planets = obs_d.get("planets", [])
        my_p = [p for p in planets if int(p[1]) == int(me)]
        if not my_p:
            return []
        src = rng.choice(my_p)
        return [[int(src[0]),
                 float(rng.uniform(-math.pi, math.pi)),
                 int(rng.randint(1, max(1, int(src[5]))))]]

    env = make("orbit_wars", configuration={"seed": 32966}, debug=False)
    env.reset()
    # Step to where the original bug manifested.
    for _ in range(150):
        obs0 = env.steps[-1][0]["observation"]
        obs1 = env.steps[-1][1]["observation"]
        env.step([random_p0(obs0, env.configuration),
                  agent(obs1, env.configuration)])
    obs = env.steps[-1][1]["observation"]
    obs_d = obs if isinstance(obs, dict) else {
        k: getattr(obs, k) for k in dir(obs) if not k.startswith("_")
    }
    me = 1
    planets = [Planet(*p) for p in obs_d["planets"]]
    opp_planets = [p for p in planets if int(p.owner) != me and int(p.owner) >= 0]
    if not opp_planets or not any(int(p.id) == 0 for p in opp_planets):
        pytest.skip("seed 32966 step 150 doesn't have opp planet id 0 anymore")

    world = World.from_obs(obs_d)
    model = WorldModel.from_world(world)
    omega = float(obs_d.get("angular_velocity", 0.0) or 0.0)
    cands = enumerate_candidates(world, model, me, omega, set())
    cands_to_0 = [c for c in cands if c.tgt_id == 0]
    assert cands_to_0, (
        "regression: zero candidates target opp planet id 0 — the "
        "`fate.hit_planet_id or -1` Python-truthiness gotcha is back."
    )
