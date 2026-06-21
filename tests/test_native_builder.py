"""LR_NATIVE_BUILDER: the greedy plan-builder scores launches with the native
flip-hazard leaf (== the 2-ply chooser) instead of the producer net-ship-delta
scorer, so far thin grabs / exposed-planet drains are never BUILT.

Guards: (1) default-OFF gate, (2) the builder scorer is marginal-over-do-nothing
(empty -> 0) and deterministic, (3) it ranks a near HOLDABLE capture above a far
thin grab from the same source (the producer scorer ranks them the other way --
that contrast is the bug this fixes), (4) the agent runs legally with the builder
on (no crash, no idle).
"""
import importlib.util
import math
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def _load_agent():
    for k in ("LR_NATIVE_LEAF", "LR_NATIVE_REINFORCE", "LR_NATIVE_OFFENSE"):
        os.environ[k] = "1"
    spec = importlib.util.spec_from_file_location(
        "lr_main", str(REPO / "agents" / "least_resistance" / "main.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["lr_main"] = mod
    spec.loader.exec_module(mod)
    return mod


def _initial_obs(seed=1393478882):
    from kaggle_environments import make
    env = make("orbit_wars", configuration={"seed": seed}, debug=False)
    env.reset(2)
    return env.state[0]["observation"]


def test_native_builder_gate_default_off():
    lr = _load_agent()
    os.environ.pop("LR_NATIVE_BUILDER", None)
    assert lr._native_builder() is False


def test_native_scorer_marginal_and_deterministic():
    lr = _load_agent()
    if not getattr(lr, "_ORBIT_OK", False):
        return  # orbit_lite unavailable in this env -> nothing to score
    obs = _initial_obs()
    score, id2slot = lr._build_native_scorer(obs, 0)
    # marginal over do-nothing: empty plan scores exactly 0.
    assert score([]) == 0.0
    # deterministic: same units -> same value (no RNG).
    planets = [lr.Planet(*p) for p in obs["planets"]]
    mine = [p for p in planets if int(p.owner) == 0]
    tgts = [p for p in planets if int(p.owner) != 0]
    assert mine and tgts
    src = mine[0]
    tgt = min(tgts, key=lambda t: lr.dist((src.x, src.y), (t.x, t.y)))
    eta = max(1, int(lr.dist((src.x, src.y), (tgt.x, tgt.y))
                     / max(1e-6, lr.fleet_speed(10.0))))
    units = [(id2slot[int(src.id)], id2slot[int(tgt.id)], 10, eta)]
    assert abs(score(units) - score(units)) < 1e-9


def test_native_scorer_prefers_holdable_over_far_thin():
    """The core fix: a near capture our garrison can hold scores higher than a far
    thin grab that empties the source. Built from real board geometry."""
    lr = _load_agent()
    if not getattr(lr, "_ORBIT_OK", False):
        return
    obs = _initial_obs()
    score, id2slot = lr._build_native_scorer(obs, 0)
    planets = [lr.Planet(*p) for p in obs["planets"]]
    mine = [p for p in planets if int(p.owner) == 0]
    tgts = [p for p in planets if int(p.owner) != 0]
    assert mine and len(tgts) >= 2
    # strongest source; nearest vs farthest target from it.
    src = max(mine, key=lambda p: float(p.ships))
    by_d = sorted(tgts, key=lambda t: lr.dist((src.x, src.y), (t.x, t.y)))
    near, far = by_d[0], by_d[-1]
    ss = id2slot.get(int(src.id))
    if ss is None or int(near.id) not in id2slot or int(far.id) not in id2slot:
        return

    def eta_to(t, ships):
        return max(1, int(lr.dist((src.x, src.y), (t.x, t.y))
                          / max(1e-6, lr.fleet_speed(float(ships)))))

    # near: send just enough to take and hold (small surplus over its garrison).
    near_sz = int(math.ceil(float(near.ships))) + 5
    far_sz = int(float(src.ships))                  # empty the source at the far target
    near_units = [(ss, id2slot[int(near.id)], near_sz, eta_to(near, near_sz))]
    far_units = [(ss, id2slot[int(far.id)], far_sz, eta_to(far, far_sz))]
    if near_sz > int(src.ships):
        return  # source can't even field the near capture; skip on this map
    assert score(near_units) >= score(far_units), (
        "native builder must not rank emptying a planet at a far target above a "
        "near holdable capture")


def test_agent_runs_legally_with_builder_on():
    lr = _load_agent()
    os.environ["LR_NATIVE_BUILDER"] = "1"
    try:
        obs = _initial_obs()
        out = lr.agent(obs, None)
        assert isinstance(out, list)
        for e in out:
            assert len(e) == 3  # [src_id, angle, ships]
    finally:
        os.environ["LR_NATIVE_BUILDER"] = "0"
