"""Production-race divergence diagnostic.

For each saved replay, compute seat-0's production-share over the game,
re-orient so we always view it from the EVENTUAL WINNER's side, and
average across games at fixed game-phase checkpoints. Answers: does the
winner pull ahead in the opening, the midgame, or the endgame?

Usage: PYTHONPATH=. python scripts/prodrace_divergence.py <replay_dir>
"""
import glob
import json
import sys


def planets_at(step):
    return step[0]["observation"]["planets"]


def share_seat0(planets):
    p0 = sum(p[4] for p in planets if int(p[1]) == 0)
    p1 = sum(p[4] for p in planets if int(p[1]) == 1)
    tot = p0 + p1
    return p0 / tot if tot > 0 else 0.5


def main(rdir):
    paths = sorted(glob.glob(f"{rdir}/*.json"))
    # phase checkpoints as fractions of game length
    fracs = [0.0, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.75, 0.9, 0.99]
    rows = []  # winner-oriented share per game, one per checkpoint
    winners_margin = []
    for path in paths:
        d = json.load(open(path))
        steps = d["steps"]
        n = len(steps)
        final = planets_at(steps[-1])
        f0 = sum(1 for p in final if int(p[1]) == 0)
        f1 = sum(1 for p in final if int(p[1]) == 1)
        if f0 == f1:
            continue  # skip ties
        winner0 = f0 > f1
        traj = []
        for fr in fracs:
            si = min(n - 1, int(fr * (n - 1)))
            s = share_seat0(planets_at(steps[si]))
            traj.append(s if winner0 else 1.0 - s)  # winner-oriented
        rows.append(traj)
        winners_margin.append(abs(f0 - f1))
    if not rows:
        print("no decisive games found")
        return
    ng = len(rows)
    print(f"decisive games: {ng}  (mean final planet margin {sum(winners_margin)/ng:.1f})")
    print("\nWINNER-oriented production share over game phase (0.50 = even):")
    print("  phase :  mean   min    max   frac>0.55")
    for i, fr in enumerate(fracs):
        col = [r[i] for r in rows]
        mean = sum(col) / ng
        ahead = sum(1 for c in col if c > 0.55) / ng
        print(f"  {fr:4.2f}  : {mean:5.2f}  {min(col):5.2f}  {max(col):5.2f}   {ahead:4.2f}")
    # first checkpoint where the winner is meaningfully ahead on average
    print("\nInterpretation: the phase where mean first crosses ~0.55 is where")
    print("the winner pulls ahead. If that's >0.2, the opening is even and the")
    print("midgame decides the game (lever target = midgame conversion).")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/prodrace_replays")
