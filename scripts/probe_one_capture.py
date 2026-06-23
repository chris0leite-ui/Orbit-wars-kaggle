"""Rule-38 mechanism probe: does LR_ONE_CAPTURE actually cap the emitted move to
one offensive TARGET per turn? Plays one 2P game vs V2; at each focal turn runs
agent(obs) with the cap OFF and ON on the SAME obs, and counts distinct non-our
target planets the emitted fleets head toward (heading-cone match)."""
from __future__ import annotations
import importlib.util, math, os, sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "agents" / "producer"))
from kaggle_environments import make

def _load(p, name):
    spec = importlib.util.spec_from_file_location(name, p)
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m
    spec.loader.exec_module(m); return m

lr = _load(str(REPO / "agents" / "least_resistance" / "main.py"), "lr_probe")
v2 = _load(str(REPO / "audit/external/agents/slawekbiel_the-producer-v2/main.py"), "v2_probe")

def distinct_targets(obs, action):
    """distinct non-our planets the emitted fleets aim at (cone match)."""
    me = int(obs["player"]); planets = obs["planets"]; fleets = obs.get("fleets", []) or []
    nonours = [p for p in planets if int(p[1]) != me]
    srcs = {int(p[0]): p for p in planets}; rows = {int(p[0]): p for p in planets}
    contested_t, all_t = set(), set()
    for launch in action:
        sid, ang, ships = int(launch[0]), float(launch[1]), float(launch[2])
        s = srcs.get(sid)
        if s is None: continue
        sx, sy = float(s[2]), float(s[3]); hx, hy = math.cos(ang), math.sin(ang)
        best, bp = None, 1e18
        for p in nonours:
            dx, dy = float(p[2]) - sx, float(p[3]) - sy
            proj = dx * hx + dy * hy
            if proj <= 0: continue
            d = math.hypot(dx, dy)
            if d <= 1e-6: continue
            perp = abs(dx * (-hy) + dy * hx) / d
            if perp <= 0.35 and proj < bp:
                best, bp = int(p[0]), proj
        if best is not None:
            all_t.add(best)
            tr = rows[best]
            our_eta = math.hypot(float(tr[2]) - sx, float(tr[3]) - sy) / max(1e-6, lr.fleet_speed(max(1.0, ships)))
            if lr._contested_attack(tr, our_eta, planets, fleets, me):
                contested_t.add(best)
    return len(contested_t), len(all_t)

env = make("orbit_wars", configuration={"seed": 777}, debug=False)
# capture focal (seat 0) observations by stepping with a recording wrapper
rec = []
def focal(obs, cfg=None):
    rec.append(dict(obs)); return lr.agent(obs, cfg)
env.run([focal, v2.agent])

off_c, on_c = {}, {}            # distinct CONTESTED targets/turn
off_a, on_a = {}, {}            # distinct ALL offensive targets/turn (incl neutral grabs)
for obs in rec:
    if not obs.get("planets"): continue
    os.environ["LR_ONE_CAPTURE"] = "0"; a_off = lr.agent(obs)
    os.environ["LR_ONE_CAPTURE"] = "1"; a_on = lr.agent(obs)
    (oc, oa), (nc, na) = distinct_targets(obs, a_off), distinct_targets(obs, a_on)
    off_c[oc] = off_c.get(oc, 0) + 1; on_c[nc] = on_c.get(nc, 0) + 1
    off_a[oa] = off_a.get(oa, 0) + 1; on_a[na] = on_a.get(na, 0) + 1
def fmt(h): return ", ".join("%d:%dx" % (k, h[k]) for k in sorted(h))
print("turns:", len(rec))
print("CONTESTED targets/turn  OFF:", fmt(off_c))
print("CONTESTED targets/turn  ON :", fmt(on_c), " <- must be <=1")
print("ALL offensive targets/turn OFF:", fmt(off_a))
print("ALL offensive targets/turn ON :", fmt(on_a), " <- neutral grabs may exceed 1")
print("ON turns with >=2 CONTESTED:", sum(v for k, v in on_c.items() if k >= 2))
