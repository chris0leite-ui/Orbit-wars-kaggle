"""Oracle router — asymmetric match-level hybrid.

Measured complementary profiles (n=16 batteries, liveness-asserted):
  champion (producer_plus vetorf, live 1291): rush-proof, ~75% vs vanilla
    Producer per its audits, but loses long economy games to the oracle
    (4/16 -> the oracle wins 12/16 of those as the aggressor side).
  oracle (imitation stack): 14/16 vs v7_0, 10/16 vs ledger, 9/16 vs
    Producer — wins EVERY style of game it survives past ~t200, dies to
    concentrated early rushes.

Cost asymmetry drives the design: misrouting an economist to the champion
costs a little edge; misrouting a rusher to the oracle is an elimination.
So the CHAMPION plays by default from turn 0, and the oracle takes over at
the decision turn only on a HIGH-PRECISION "economist" read — the opponent
never massed a fist and never booked real arrivals on our planets.

Signals are exact (oracle engine ledger, not heuristics):
  fist  = largest enemy garrison / enemy garrison total (concentration)
  threat = enemy arrivals booked on MY planets within THREAT_WINDOW ticks
           / my garrison total
Peaks tracked over t in [WATCH_FROM, DECIDE_T]; only one brain ever plays
(the champion's internal planned-launch memory stays consistent; the
oracle cold-starts fine mid-game — its caches rebuild from the obs).
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

_ORACLE = None
_CHAMP = None
_ENGINE = None
_IMPORT_ERROR = None


def _load_brains():
    global _ORACLE, _CHAMP, _ENGINE, _IMPORT_ERROR
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
        _ENGINE = sys.modules.get("oracle.engine")
        if _ENGINE is None:
            ep = os.path.join(_REPO, "agents", "oracle", "engine.py")
            spec_e = importlib.util.spec_from_file_location(
                "router_engine", ep)
            me_ = importlib.util.module_from_spec(spec_e)
            sys.modules["router_engine"] = me_
            spec_e.loader.exec_module(me_)
            _ENGINE = me_
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


WATCH_FROM = int(os.environ.get("ROUTER_WATCH_FROM", "8"))
DECIDE_T = int(os.environ.get("ROUTER_DECIDE_T", "36"))
THREAT_WINDOW = int(os.environ.get("ROUTER_THREAT_WINDOW", "14"))
# economist read requires BOTH peaks below threshold through the window
FIST_MAX = float(os.environ.get("ROUTER_FIST_MAX", "0.42"))
THREAT_MAX = float(os.environ.get("ROUTER_THREAT_MAX", "0.18"))


class _State:
    def __init__(self):
        self.routed = None            # None=watching, "oracle", "champ"
        self.fist_peak = 0.0
        self.threat_peak = 0.0
        self.last_step = -1


_S = _State()


def _g(obs, key, default=None):
    if isinstance(obs, dict):
        return obs.get(key, default)
    return getattr(obs, key, default)


def _signals(obs, me):
    """(fist, threat) from one exact ledger build at a short horizon."""
    w = _ENGINE.World(obs, horizon=max(THREAT_WINDOW + 2, 16))
    w.build_ledger()
    n = w.n_planets
    enemy_g = sorted((w.ships0[i] for i in range(n)
                      if w.owner0[i] >= 0 and w.owner0[i] != me),
                     reverse=True)
    fist = (enemy_g[0] / sum(enemy_g)) if enemy_g else 0.0
    my_g = sum(w.ships0[i] for i in range(n) if w.owner0[i] == me) or 1
    threat = 0.0
    for i in range(n):
        if w.owner0[i] != me:
            continue
        for dt, slot in w.arrivals[i].items():
            if dt > THREAT_WINDOW:
                continue
            threat += sum(s for o, s in slot.items()
                          if o >= 0 and o != me)
    return fist, threat / my_g


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
                  {int(f[1]) for f in fleets if f[1] >= 0}
        if len(players) > 2:
            # 4P: the champion's live 4P configuration is the proven one
            _S.routed = "champ"
        else:
            if WATCH_FROM <= step <= DECIDE_T:
                try:
                    fist, threat = _signals(obs, me)
                    _S.fist_peak = max(_S.fist_peak, fist)
                    _S.threat_peak = max(_S.threat_peak, threat)
                except Exception:
                    _S.fist_peak = 1.0       # fail safe: stay champion
            if step >= DECIDE_T:
                economist = (_S.fist_peak < FIST_MAX
                             and _S.threat_peak < THREAT_MAX)
                _S.routed = "oracle" if economist else "champ"

    if _S.routed == "oracle":
        return _ORACLE.agent(obs)
    return _CHAMP.agent(obs)
