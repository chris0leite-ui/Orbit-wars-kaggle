"""Trace least_resistance over-commitment / lose-both on a given seed (4P).

Reproduces the PI's two ladder observations (seed 25260880):
  (a) launch from a large planet to a small target, lose both;
  (b) too many fleets / hyperactivity.

Runs the LIVE least_resistance (seat 0) vs 3x Producer V2 on the seed, then
post-processes env.steps to report, per turn and in aggregate:
  - launches/turn (hyperactivity);
  - each launch's source ship-count at decision time and fleet size
    (large-source launches);
  - planets we owned then LOST, and whether we launched FROM them shortly
    before losing them (lose-source), plus targets captured then reflipped
    (lose-target).
"""
from __future__ import annotations
import argparse, importlib.util, json, math, os, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
# knobs must be set BEFORE importing the focal agent
_pre = os.environ.get("_TRACE_KNOBS")
if _pre:
    for k, v in json.loads(_pre).items():
        os.environ[str(k)] = str(v)
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "agents" / "producer"))
try:
    import torch; torch.set_num_threads(1)
except Exception:
    pass


def load(path):
    spec = importlib.util.spec_from_file_location("_a_%d" % abs(hash(path)), path)
    m = importlib.util.module_from_spec(spec); sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return getattr(m, "agent")


def arity(fn):
    n = fn.__code__.co_argcount if hasattr(fn, "__code__") else 2
    return (lambda o, c=None: fn(o)) if n == 1 else (lambda o, c=None: fn(o, c))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=25260880)
    ap.add_argument("--players", type=int, default=4)
    ap.add_argument("--focal", default=str(REPO / "agents" / "least_resistance" / "main.py"))
    args = ap.parse_args()

    LR = args.focal
    V2 = str(REPO / "audit/external/agents/slawekbiel_the-producer-v2/main.py")
    from kaggle_environments import make
    focal = arity(load(LR)); v2 = load(V2)
    seats = [focal] + [arity(v2) for _ in range(args.players - 1)]
    env = make("orbit_wars", configuration={"seed": args.seed}, debug=False)
    env.run(seats)

    # ownership timeline: planet_id -> {step: (owner, ships)}
    steps = env.steps
    own = {}        # pid -> list of (step, owner, ships)
    for t, st in enumerate(steps):
        obs = st[0].observation
        planets = obs["planets"] if isinstance(obs, dict) else obs.planets
        for p in planets:
            pid = int(p[0])
            own.setdefault(pid, []).append((t, int(p[1]), float(p[5])))

    def owner_at(pid, t):
        rec = own.get(pid, [])
        last = None
        for (s, o, sh) in rec:
            if s <= t:
                last = (o, sh)
        return last

    # per-turn launches by focal (seat 0). action row = [src_id, angle, ships].
    print(f"=== seed {args.seed}  players {args.players}  steps {len(steps)} ===")
    per_turn = []
    launch_events = []   # (t, src_id, src_ships_before, fleet)
    for t, st in enumerate(steps):
        act = st[0].action or []
        obs = st[0].observation
        planets = obs["planets"] if isinstance(obs, dict) else obs.planets
        ships_by = {int(p[0]): float(p[5]) for p in planets}
        owner_by = {int(p[0]): int(p[1]) for p in planets}
        n = len(act)
        per_turn.append(n)
        for row in act:
            sid = int(row[0]); fleet = float(row[2])
            src_ships = ships_by.get(sid, float("nan"))
            launch_events.append((t, sid, src_ships, fleet))

    total_launches = sum(per_turn)
    active_turns = sum(1 for n in per_turn if n > 0)
    print(f"total focal launches: {total_launches}   active turns: {active_turns}"
          f"   mean launches/active turn: {total_launches/max(1,active_turns):.2f}")
    # hyperactivity: distribution of launches/turn
    from collections import Counter
    print("launches/turn histogram:", dict(sorted(Counter(per_turn).items())))

    # large-source launches: fleet sent while source had a big garrison
    print("\n--- launches from a high-garrison source (src_ships>=25), first 25 ---")
    big = [(t, sid, sb, fl) for (t, sid, sb, fl) in launch_events if sb >= 25]
    for (t, sid, sb, fl) in big[:25]:
        frac = fl / max(1e-9, sb)
        print(f"  step {t:3d}  src {sid:2d} had {sb:6.1f} ships -> sent {fl:6.1f}"
              f"  ({frac*100:4.0f}% of source)")
    print(f"  ...{len(big)} total launches from sources with >=25 ships")

    # lose-source: planets we OWNED that became enemy-owned, and whether we
    # launched from them in the 6 turns before losing them.
    print("\n--- planets we OWNED then LOST (lose-source / lose-both) ---")
    me = 0
    launches_by_src = {}
    for (t, sid, sb, fl) in launch_events:
        launches_by_src.setdefault(sid, []).append((t, fl, sb))
    lost = 0
    for pid, rec in own.items():
        owned_t = [s for (s, o, sh) in rec if o == me]
        if not owned_t:
            continue
        # find first step after we owned it where it is enemy-owned (>=0, !=me)
        for (s, o, sh) in rec:
            if s > owned_t[0] and o >= 0 and o != me:
                # we lost pid at step s. did we launch FROM it just before?
                pre = [(lt, fl, sb) for (lt, fl, sb) in launches_by_src.get(pid, [])
                       if s - 8 <= lt <= s]
                tag = ""
                if pre:
                    tag = "  <-- LAUNCHED FROM IT before losing: " + ", ".join(
                        f"step{lt}:sent{fl:.0f}(had{sb:.0f})" for (lt, fl, sb) in pre)
                print(f"  planet {pid:2d}: ours until ~step {max(owned_t)}, "
                      f"enemy by step {s}{tag}")
                lost += 1
                break
    print(f"  {lost} owned-then-lost planets")

    final = steps[-1][0].observation
    planets = final["planets"] if isinstance(final, dict) else final.planets
    fleets = final["fleets"] if isinstance(final, dict) else final.fleets
    sc = [0.0] * args.players
    for p in planets:
        if 0 <= int(p[1]) < args.players: sc[int(p[1])] += float(p[5])
    for f in (fleets or []):
        if 0 <= int(f[1]) < args.players: sc[int(f[1])] += float(f[6])
    print(f"\nfinal ship scores: {[round(x,1) for x in sc]}   "
          f"focal={'WIN' if sc[0]==max(sc) and max(sc)>0 else 'LOSS'}")


if __name__ == "__main__":
    main()
