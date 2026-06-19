"""Diagnose whether the position-scorer (ship advantage @ horizon 13-18) is
myopic about territory. Replays the 2-ply champion vs Producer V2 and, at
mid-game checkpoints and final, records our ship-share and planet-share, then
groups by eventual win/loss. If we are ship-AHEAD but planet-BEHIND in the games
we go on to lose, the ship-centric short-horizon objective is mis-ranking.

  python /tmp/diag_scorer.py [n_seeds]
"""
import importlib.util, os, sys
from pathlib import Path

REPO = Path("/home/user/Orbit-wars-kaggle")
os.environ["OMP_NUM_THREADS"] = "1"; os.environ["MKL_NUM_THREADS"] = "1"
for k, v in {"LR_LEADER_RELATIVE_4P": "0", "LR_VALUE_COMMIT": "0", "LR_ANYTIME": "0",
             "LR_ENEMY_BOOST": "1.0", "LR_HOLD_MARGIN": "0.5", "LR_DEFEND": "1",
             "LR_ROLLOUT_DEPTH": "0"}.items():
    os.environ[k] = v
sys.path.insert(0, str(REPO))
import torch; torch.set_num_threads(1)
from kaggle_environments import make

LR = str(REPO / "agents/least_resistance/main.py")
V2 = str(REPO / "audit/external/agents/slawekbiel_the-producer-v2/main.py")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 32
SEEDS = list(range(5000, 5000 + N))
CHECKS = [30, 50, 70]


def load(p, n):
    s = importlib.util.spec_from_file_location(n, p); m = importlib.util.module_from_spec(s)
    sys.modules[n] = m; s.loader.exec_module(m); return m


def shares(obs, me=0):
    """(our_ships, opp_ships, our_planets, opp_planets) for a 2P obs."""
    sp = [0.0, 0.0]; pl = [0, 0]
    for p in obs["planets"]:
        o = int(p[1])
        if 0 <= o < 2:
            sp[o] += float(p[5]); pl[o] += 1
    for f in (obs["fleets"] or []):
        o = int(f[1])
        if 0 <= o < 2:
            sp[o] += float(f[6])
    return sp[me], sp[1 - me], pl[me], pl[1 - me]


def main():
    lr = load(LR, "lr"); v2 = load(V2, "v2")
    nv = v2.agent.__code__.co_argcount
    v2w = (lambda o, c=None: v2.agent(o)) if nv == 1 else (lambda o, c=None: v2.agent(o, c))
    # accumulators: per checkpoint, split by eventual win/loss:
    # count games where ship-margin>0 (we look ahead) AND planet-margin<0 (truly behind)
    agg = {c: {"win": [], "loss": []} for c in CHECKS + [-1]}
    wins = 0
    for seed in SEEDS:
        env = make("orbit_wars", configuration={"seed": seed, "episodeSteps": 500}, debug=False)
        env.run([lambda o, c=None: lr.agent(o), v2w])
        steps = env.steps
        final = steps[-1]; rew = [s.reward for s in final]
        valid = [r for r in rew if r is not None]
        win = rew[0] is not None and valid and rew[0] == max(valid)
        wins += int(win)
        key = "win" if win else "loss"
        for c in CHECKS:
            if c < len(steps):
                os_, ot, op, opp = shares(steps[c][0].observation)
                agg[c][key].append((os_ - ot, op - opp))  # (ship_margin, planet_margin)
        os_, ot, op, opp = shares(final[0].observation)
        agg[-1][key].append((os_ - ot, op - opp))
    print("# 2-ply champion vs V2 | n=%d | wins=%d (%.0f%%)" % (N, wins, 100.0 * wins / N))
    print("# ship_margin = our_ships - opp_ships ; planet_margin = our_planets - opp_planets")
    print("# (the scorer optimizes ~ship_margin at horizon 13-18; planet_margin ~ true standing)")
    for c in CHECKS + [-1]:
        label = "FINAL" if c == -1 else ("step%d" % c)
        for key in ("win", "loss"):
            rows = agg[c][key]
            if not rows:
                continue
            sm = sum(r[0] for r in rows) / len(rows)
            pm = sum(r[1] for r in rows) / len(rows)
            # the tell: games where we were ship-ahead but planet-behind
            misled = sum(1 for r in rows if r[0] > 0 and r[1] < 0)
            print("%-6s %-4s n=%2d  mean ship_margin=%+7.1f  mean planet_margin=%+5.2f  "
                  "ship-ahead&planet-behind=%d/%d"
                  % (label, key, len(rows), sm, pm, misled, len(rows)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
