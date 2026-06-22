"""LR_RESERVE_4P: a 4P-only coalition-aware per-source garrison reserve that caps each
planet's spendable ships before candidate generation, so a planet the field can hit is
neither drained for a premature attack nor over-lent as a reinforcement donor.

Guards: (1) default-OFF gate; (2) 2P is byte-identical with the gate on (num_seats<4
short-circuits); (3) the agent runs legally and is NOT paralysed in 4P with the gate on.
"""
import importlib.util
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def _load_clean():
    for k in list(os.environ):
        if k.startswith("LR_"):
            os.environ.pop(k, None)
    spec = importlib.util.spec_from_file_location(
        "lr_main_r4p", str(REPO / "agents" / "least_resistance" / "main.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["lr_main_r4p"] = mod
    spec.loader.exec_module(mod)
    return mod


def _obs(seed=1393478882, n=2):
    from kaggle_environments import make
    env = make("orbit_wars", configuration={"seed": seed}, debug=False)
    env.reset(n)
    return env.state[0]["observation"]


def test_reserve_4p_default_off():
    lr = _load_clean()
    assert lr._reserve_4p() is False


def test_reserve_4p_byte_identical_in_2p():
    # num_seats < 4 must short-circuit -> 2P action identical with the gate on or off.
    lr = _load_clean()
    obs = _obs(n=2)
    os.environ.pop("LR_RESERVE_4P", None)
    base = lr.agent(obs, None)
    os.environ["LR_RESERVE_4P"] = "1"
    try:
        on = lr.agent(obs, None)
    finally:
        os.environ.pop("LR_RESERVE_4P", None)
    assert base == on, "LR_RESERVE_4P must not change 2P behaviour"


def test_reserve_4p_runs_and_not_paralysed_in_4p():
    # Over several 4P turns with the reserve on, the agent must still act (not freeze).
    lr = _load_clean()
    os.environ["LR_RESERVE_4P"] = "1"
    try:
        from kaggle_environments import make
        env = make("orbit_wars", configuration={"seed": 219030400}, debug=False)
        env.reset(4)
        acted = 0
        for _ in range(12):
            obs = env.state[0]["observation"]
            out = lr.agent(obs, None)
            assert isinstance(out, list)
            for e in out:
                assert len(e) == 3
            if out:
                acted += 1
            # advance the game one step with our move in seat 0, others idle
            actions = [out] + [[] for _ in range(3)]
            try:
                env.step([a if isinstance(a, list) else [] for a in actions])
            except Exception:
                break
        assert acted >= 1, "agent never launched in 4P with the reserve on (paralysis)"
    finally:
        os.environ.pop("LR_RESERVE_4P", None)
