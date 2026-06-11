"""Oracle router — match-level hybrid (PI-proposed design, 2026-06-11).

Measured weakness profiles are complementary:
  - oracle (imitation stack): 14/16 vs the v7 economist family, ~6/16 vs
    the public Producer's all-in early rush (top-meta cadence under-defends
    against off-meta tempo).
  - champion (producer_plus vetorf live bundle, ~1291 on the ladder):
    ~24/32 vs vanilla Producer per its audits, weaker vs top economists.

So: open every game with the oracle, watch the opponent's opening tempo,
and HAND THE WHOLE GAME OVER to the champion the moment the opponent
matches the rusher signature. One brain plays at a time — running both
would corrupt the champion's internal planned-launch memory (it assumes
its moves execute). The champion cold-starts fine mid-game: its movement
cache rebuilds from the observation.

Routing signal (calibrated on local probes, scripts/oracle_router_calib.py):
  enemy in-flight mass and launch count over the first ROUTE_DECIDE_T turns
  — the Producer fires sustained full-drain waves from turn ~2; economists
  hold or probe small.
"""

import os
import sys

try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _HERE = None
if _HERE is None or not os.path.isfile(os.path.join(_HERE, "main.py")):
    for _cand in ("agents/oracle_router",
                  os.path.join(os.getcwd(), "agents", "oracle_router"),
                  "/home/user/Orbit-wars-kaggle/agents/oracle_router"):
        if os.path.isfile(os.path.join(_cand, "main.py")):
            _HERE = os.path.abspath(_cand)
            break
_REPO = os.path.dirname(os.path.dirname(_HERE)) if _HERE else os.getcwd()

# the two brains (lazy modules, loaded once)
_ORACLE = None
_CHAMP = None
_IMPORT_ERROR = None


def _load_brains():
    global _ORACLE, _CHAMP, _IMPORT_ERROR
    if _ORACLE is not None or _IMPORT_ERROR is not None:
        return
    import importlib.util
    try:
        op = os.path.join(_REPO, "agents", "oracle", "main.py")
        spec = importlib.util.spec_from_file_location("oracle_brain", op)
        m = importlib.util.module_from_spec(spec)
        sys.modules["oracle_brain"] = m
        spec.loader.exec_module(m)
        _ORACLE = m
        cp = os.environ.get(
            "ROUTER_CHAMPION_PATH",
            os.path.join(_REPO, "data", "external", "live_vetorf_1291.py"))
        spec2 = importlib.util.spec_from_file_location("champ_brain", cp)
        m2 = importlib.util.module_from_spec(spec2)
        sys.modules["champ_brain"] = m2
        spec2.loader.exec_module(m2)
        _CHAMP = m2
    except Exception as e:
        _IMPORT_ERROR = e


ROUTE_DECIDE_T = int(os.environ.get("ROUTER_DECIDE_T", "12"))
# thresholds from scripts/oracle_router_calib.py measurements
RUSH_INFLIGHT_FRAC = float(os.environ.get("ROUTER_INFLIGHT_FRAC", "0.45"))
RUSH_MIN_FLEETS = int(os.environ.get("ROUTER_MIN_FLEETS", "6"))


class _State:
    def __init__(self):
        self.routed = None            # None=undecided, "oracle", "champ"
        self.enemy_fleet_peak = 0
        self.enemy_inflight_frac_peak = 0.0
        self.last_step = -1


_S = _State()


def _g(obs, key, default=None):
    if isinstance(obs, dict):
        return obs.get(key, default)
    return getattr(obs, key, default)


def agent(obs, configuration=None):
    _load_brains()
    if _IMPORT_ERROR is not None:
        raise RuntimeError(f"router import failed: {_IMPORT_ERROR}")
    me = int(_g(obs, "player", 0) or 0)
    step = int(_g(obs, "step", 0) or 0)
    if step <= _S.last_step:
        _S.__init__()                 # new episode in the same process
    _S.last_step = step

    if _S.routed is None:
        planets = _g(obs, "planets", []) or []
        fleets = _g(obs, "fleets", []) or []
        players = {p[1] for p in planets if p[1] >= 0} | \
                  {f[1] for f in fleets if f[1] >= 0}
        if len(players) > 2:
            _S.routed = "oracle"      # 4P: tempo signature unreliable; the
                                      # oracle trains on 4P expert data
        else:
            enemy_fleets = [f for f in fleets if f[1] >= 0 and f[1] != me]
            e_in = sum(f[6] for f in enemy_fleets)
            e_g = sum(p[5] for p in planets
                      if p[1] >= 0 and p[1] != me)
            tot = e_in + e_g
            frac = e_in / tot if tot > 0 else 0.0
            _S.enemy_fleet_peak = max(_S.enemy_fleet_peak,
                                      len(enemy_fleets))
            _S.enemy_inflight_frac_peak = max(
                _S.enemy_inflight_frac_peak, frac)
            if step >= ROUTE_DECIDE_T:
                rush = (_S.enemy_inflight_frac_peak >= RUSH_INFLIGHT_FRAC
                        and _S.enemy_fleet_peak >= RUSH_MIN_FLEETS)
                _S.routed = "champ" if rush else "oracle"

    brain = _CHAMP if _S.routed == "champ" else _ORACLE
    return brain.agent(obs)
