"""Guard: the contagion opponent (LR_DEEP_OPP=2) — Phase 2.

`_apply_contagion(snap, me)` is the model-free dropout opponent used inside the
deep-search rollout: each step it flips NEUTRALS and MY under-defended planets to
the single STRONGEST reachable rival (max-aggregate threat, not the sum), at most
one flip per rival source per step, deterministically (no RNG). These tests are
torch-AGNOSTIC — they drive `_apply_contagion` on a tiny fake snapshot of mutable
planet rows, so they pass with or without torch installed.

Mirrors the assertions of the branch's test_native_forward_model.py (holdable
beats thin, no-threat = full survival, max-threat) but for the discrete flip.
"""
import importlib.util
import os

_MAIN = os.path.join(os.path.dirname(__file__), "..", "agents",
                     "least_resistance", "main.py")


def _load_main():
    spec = importlib.util.spec_from_file_location("lr_main_contagion_test", _MAIN)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _Obs:
    def __init__(self, planets):
        self.planets = planets


class _State:
    def __init__(self, planets):
        self.observation = _Obs(planets)


class _Snap:
    """Minimal stand-in: _apply_contagion only reads snap.state[0].observation.planets."""
    def __init__(self, planets):
        self.state = [_State(planets)]


# Planet row = [id, owner, x, y, radius, ships, production]. owner: -1 neutral.
def _p(pid, owner, x, ships, y=0.0, radius=0.0, prod=1):
    return [pid, owner, float(x), float(y), float(radius), float(ships), prod]


def test_contested_neutral_flips_to_strong_rival():
    mod = _load_main()
    os.environ.pop("LR_CONTAGION_REACH_TICKS", None)
    planets = [
        _p(0, 1, x=0.0, ships=100),   # rival (seat 1), strong, adjacent
        _p(1, -1, x=2.0, ships=5),    # contested neutral
        _p(2, 0, x=50.0, ships=200),  # me, far away
    ]
    mod._apply_contagion(_Snap(planets), me=0)
    assert planets[1][1] == 1, "neutral within reach of a strong rival must flip"
    assert planets[1][5] >= 1.0


def test_my_underdefended_flips_welldefended_holds():
    mod = _load_main()
    planets = [
        _p(0, 1, x=0.0, ships=100),    # strong rival
        _p(1, 0, x=2.0, ships=5),      # MY under-defended planet (near rival)
        _p(2, 0, x=3.0, ships=500),    # MY well-defended planet (near rival)
    ]
    mod._apply_contagion(_Snap(planets), me=0)
    assert planets[1][1] == 1, "my under-defended planet should be overrun"
    assert planets[2][1] == 0, "my well-defended planet must hold (out-masses rival)"


def test_max_threat_not_sum():
    """Two weak rivals whose SUM out-masses the target but individually do not ->
    NO flip (we model the strongest single reachable rival, not the sum)."""
    mod = _load_main()
    planets = [
        _p(0, 1, x=-2.0, ships=30),   # rival A, weak
        _p(1, 2, x=2.0, ships=30),    # rival B, weak (different seat)
        _p(2, 0, x=0.0, ships=50),    # me: 50 > 30 each, < 60 sum
    ]
    mod._apply_contagion(_Snap(planets), me=0)
    assert planets[2][1] == 0, "no single rival out-masses 50 -> must NOT flip"


def test_no_threat_full_survival():
    """A planet with no reachable rival is unchanged."""
    mod = _load_main()
    planets = [
        _p(0, 1, x=0.0, ships=100),    # strong rival
        _p(1, 0, x=10000.0, ships=5),  # me, far out of reach
    ]
    before = [row[:] for row in planets]
    mod._apply_contagion(_Snap(planets), me=0)
    assert planets[1] == before[1], "unreachable planet must not flip"


def test_one_flip_per_source_per_step():
    """A single rival source flips at most one target per contagion step."""
    mod = _load_main()
    planets = [
        _p(0, 1, x=0.0, ships=100),   # the only rival source
        _p(1, -1, x=1.0, ships=2),    # neutral A (reachable, weak)
        _p(2, -1, x=2.0, ships=2),    # neutral B (reachable, weak)
    ]
    mod._apply_contagion(_Snap(planets), me=0)
    flipped = [planets[1][1] == 1, planets[2][1] == 1]
    assert sum(flipped) == 1, "one rival source must flip at most one target per step"


def test_deterministic():
    mod = _load_main()
    def fresh():
        return [
            _p(0, 1, x=0.0, ships=100),
            _p(1, -1, x=1.0, ships=2),
            _p(2, -1, x=2.0, ships=3),
            _p(3, 0, x=2.5, ships=4),
        ]
    a = fresh(); mod._apply_contagion(_Snap(a), me=0)
    b = fresh(); mod._apply_contagion(_Snap(b), me=0)
    assert a == b, "contagion must be deterministic (no RNG)"


def test_no_rivals_is_noop():
    mod = _load_main()
    planets = [_p(0, 0, x=0.0, ships=5), _p(1, -1, x=1.0, ships=2)]
    before = [row[:] for row in planets]
    mod._apply_contagion(_Snap(planets), me=0)
    assert planets == before, "with no rivals on the board, nothing flips"
