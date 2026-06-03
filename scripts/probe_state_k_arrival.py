"""Step-0 go/no-go probe for arrival-aware state-driven K.

Read-only. Plays a few champion-vs-champion games, and on every turn
enumerates realistic launch candidates (my planets x ORBITING targets x
proposer ship counts). For each it compares the opponent contest tick
computed at CURRENT positions (arrival_eta=0, what the shipped state-K
uses) vs at OUR ARRIVAL (arrival_eta=our_eta, the fix). Reports how often
they diverge and — the only thing that changes behaviour — how often the
*clamped* K (clamp 10..30) crosses a boundary.
"""
import os, sys
# Repo root FIRST on sys.path so the local `agents/` package wins over the
# top-level `agents.py` that kaggle_environments.lux_ai_s3 injects.
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
os.environ.setdefault("BASELINE_JOINT_AGGR", "1")
os.environ.setdefault("BASELINE_JOINT_TOP_K", "5")
os.environ.setdefault("BASELINE_JOINT_MAX_PAIRS", "60")
os.environ.setdefault("BASELINE_REINFORCE_EMIT", "1")
os.environ.setdefault("BASELINE_REINFORCE_ANTICIPATE", "1")
os.environ.setdefault("BASELINE_NEUTRAL_BONUS", "2.0")
os.environ.setdefault("BASELINE_NEUTRAL_EARLY_EXTRA", "1.5")
os.environ.setdefault("BASELINE_NEUTRAL_EARLY_HORIZON", "50")
os.environ.setdefault("BASELINE_ORBITAL_SAFETY", "1")
os.environ.setdefault("BASELINE_PV_ETA", "1")
os.environ.setdefault("BASELINE_LAUNCH_RULES", "1")
os.environ.setdefault("BASELINE_CAPTURE_HORIZON_K", "10")
os.environ.setdefault("BASELINE_STATE_DRIVEN_K", "1")
os.environ.setdefault("BASELINE_STATE_K_CEIL", "30")
os.environ.setdefault("BASELINE_KINEMATIC_TABLE", "1")
# Throttle per-turn rollout budget: we only need realistic board GEOMETRY,
# not deep play — the contest-tick divergence is a property of positions.
os.environ.setdefault("BASELINE_WALLCLOCK_MS", "40")

from kaggle_environments import make
sys.path.insert(0, _REPO)  # re-assert after kaggle_environments mutates sys.path
from agents.baseline.main import agent as champ_agent, _as_dict
from agents.baseline.proposer import aim_and_eta, enumerate_ship_counts
from lib.intent import World
from lib.world_model import WorldModel
from lib.kinematic_table import KinematicTable
from lib.orbit import is_orbiting
from kaggle_environments.envs.orbit_wars.orbit_wars import Planet

FLOOR, CEIL = 10, 30
def clampK(t):
    if t is None:
        return CEIL
    return max(FLOOR, min(CEIL, int(t)))

records = []  # list of obs_d snapshots for player 0
def recorder(obs, config=None):
    od = _as_dict(obs)
    if int(od.get("player", 0)) == 0:
        records.append({
            "planets": [list(p) for p in od.get("planets", [])],
            "fleets": [list(f) for f in od.get("fleets", [])],
            "angular_velocity": float(od.get("angular_velocity", 0.0)),
            "player": 0, "step": int(od.get("step", 0)),
        })
    return champ_agent(obs, config)

# ---- running tallies (accumulated per game so a timeout still yields data) ----
tot = diff = boundary = lower = higher = 0
delta_sum = 0
by_phase = {"open(0-50)": [0, 0], "mid(50-150)": [0, 0], "late(150+)": [0, 0]}
def phase(s):
    return "open(0-50)" if s < 50 else ("mid(50-150)" if s < 150 else "late(150+)")

def pct(a, b):
    return f"{100.0*a/b:.1f}%" if b else "n/a"

def report(tag):
    md = f"{(delta_sum/boundary):+.2f}" if boundary else "n/a"
    print(f"===== {tag}: cands={tot}  rawdiff={pct(diff,tot)}  "
          f"K-boundary-cross={boundary} ({pct(boundary,tot)})  "
          f"lowerK={pct(lower,boundary)}  meanDelta={md}", flush=True)

def analyze(recs):
    global tot, diff, boundary, lower, higher, delta_sum
    for od in recs:
        me = 0
        world = World.from_obs(od)
        world._kt = KinematicTable(); world._kt.begin_turn(world)
        model = WorldModel.from_world(world)
        omega = od["angular_velocity"]
        step = od["step"]
        planets = [Planet(*p) for p in od["planets"]]
        my_planets = [p for p in planets if int(p.owner) == me]
        targets = [p for p in planets if int(p.owner) != me and is_orbiting(list(p))]
        if not my_planets or not targets:
            continue
        for src in my_planets:
            for tgt in targets:
                if int(tgt.id) == int(src.id):
                    continue
                try:
                    ship_counts = enumerate_ship_counts(src, tgt, model, omega, me, world)
                except Exception:
                    continue
                for ships in ship_counts[:4]:
                    if ships < 1 or ships > int(src.ships):
                        continue
                    try:
                        _angle, eta = aim_and_eta(src, tgt, int(ships), omega, world=world)
                    except Exception:
                        continue
                    if eta is None or eta <= 0:
                        continue
                    stale = model.time_to_enemy_threat(int(tgt.id), me, world, arrival_eta=0)
                    arr = model.time_to_enemy_threat(int(tgt.id), me, world, lead_now=True)
                    tot += 1
                    by_phase[phase(step)][0] += 1
                    if stale != arr:
                        diff += 1
                    ks, ka = clampK(stale), clampK(arr)
                    if ks != ka:
                        boundary += 1
                        by_phase[phase(step)][1] += 1
                        delta_sum += (ka - ks)
                        if ka < ks:
                            lower += 1
                        else:
                            higher += 1

N_GAMES = int(sys.argv[1]) if len(sys.argv) > 1 else 4
for g in range(N_GAMES):
    records.clear()
    env = make("orbit_wars", configuration={"actTimeout": 1})
    env.reset(num_agents=2)
    env.run([recorder, champ_agent])
    n = len(records)
    analyze(records)
    report(f"after game {g} (+{n} turns)")

print("\n================ STATE-K ARRIVAL-AWARE DIVERGENCE PROBE ================")
print(f"games={N_GAMES}  orbiting-target candidates={tot}")
print(f"raw tick differs (stale != arrival):      {diff:6d}  ({pct(diff, tot)})")
print(f"** clamped-K boundary crossing **:        {boundary:6d}  ({pct(boundary, tot)})   <- behaviour-changing")
print(f"   of those: arrival LOWERS K (stale over-optimistic): {lower}  ({pct(lower, boundary)})")
print(f"             arrival RAISES K:                          {higher}  ({pct(higher, boundary)})")
print(f"   mean signed K delta on boundary crossings: {(delta_sum/boundary):+.2f}" if boundary else "   (no crossings)")
print("\nby phase (candidates / boundary-crossings / rate):")
for ph, (c, b) in by_phase.items():
    print(f"   {ph:14s}  {c:6d} / {b:5d}  ({pct(b, c)})")
print("=======================================================================")
