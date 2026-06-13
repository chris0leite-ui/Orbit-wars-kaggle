"""Loss-phase diagnostic on REAL ladder episodes (ground truth).

Reads downloaded live-episode replays for one submission, finds our seat
(TeamNames == "ChrisLeiteScha"), and computes our planet-share and
material-share (planet ships + in-flight, ours / all players) at
checkpoint steps — split by WIN vs LOSS and by 2P vs 4P. The step where
the win/loss curves diverge is the phase we lose on the real ladder.

Far more faithful than local sim (real opponents) and fast (JSON only).

Usage:
    python scripts/diag_live_loss_phase.py 53595717 [--team ChrisLeiteScha]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LIVE = REPO / "audit" / "live-episodes"
CHECKPOINTS = [25, 50, 100, 150, 250, 400, 499]


def share_at(obs, pid):
    planets = obs.get("planets", []) or []
    fleets = obs.get("fleets", []) or []
    pl_mine = sum(1 for p in planets if int(p[1]) == pid and int(p[0]) >= 0)
    pl_tot = sum(1 for p in planets if int(p[1]) >= 0 and int(p[0]) >= 0)
    mat_mine = sum(p[5] for p in planets if int(p[1]) == pid)
    mat_mine += sum(f[6] for f in fleets if int(f[1]) == pid)
    mat_tot = sum(p[5] for p in planets if int(p[1]) >= 0)
    mat_tot += sum(f[6] for f in fleets if int(f[1]) >= 0)
    # absolute focal planet count, garrison (on planets), in-flight (fleets)
    abs_pl = pl_mine
    garrison = sum(p[5] for p in planets if int(p[1]) == pid)
    inflight = sum(f[6] for f in fleets if int(f[1]) == pid)
    return (pl_mine / pl_tot if pl_tot else 0.0,
            mat_mine / mat_tot if mat_tot else 0.0,
            abs_pl, garrison, inflight)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("sub_id")
    ap.add_argument("--team", default="ChrisLeiteScha")
    args = ap.parse_args()

    buckets = {("2P", "WIN"): [], ("2P", "LOSS"): [],
               ("4P", "WIN"): [], ("4P", "LOSS"): []}
    counts = {k: 0 for k in buckets}
    skipped = 0
    for f in sorted((LIVE / args.sub_id).glob("episode-*-replay.json")):
        try:
            r = json.loads(f.read_text())
        except Exception:
            skipped += 1
            continue
        teams = r.get("info", {}).get("TeamNames", [])
        ours = [i for i, t in enumerate(teams) if t == args.team]
        steps = r.get("steps", [])
        if len(ours) != 1 or not steps:
            skipped += 1
            continue
        pid = ours[0]
        nseats = len(steps[0])
        fmt = "2P" if nseats == 2 else "4P"
        rewards = [steps[-1][i].get("reward") for i in range(nseats)]
        rewards = [x for x in rewards if x is not None]
        my_r = steps[-1][pid].get("reward")
        if my_r is None or not rewards:
            skipped += 1
            continue
        won = my_r == max(rewards)
        key = (fmt, "WIN" if won else "LOSS")
        counts[key] += 1
        curve = {}
        for cp in CHECKPOINTS:
            if cp < len(steps):
                obs = steps[cp][pid].get("observation", {}) or {}
                curve[cp] = share_at(obs, pid)
        buckets[key].append(curve)

    print(f"sub {args.sub_id}  team={args.team}  skipped={skipped}")
    for fmt in ("2P", "4P"):
        nw, nl = counts[(fmt, "WIN")], counts[(fmt, "LOSS")]
        print(f"\n=== {fmt}: {nw} wins / {nl} losses ===")
        print(f"{'step':>5} | {'WIN  planets/garrison/inflight':>30} | {'LOSS planets/garrison/inflight':>30}")
        for cp in CHECKPOINTS:
            w = [c[cp] for c in buckets[(fmt, "WIN")] if cp in c]
            l = [c[cp] for c in buckets[(fmt, "LOSS")] if cp in c]
            def avg(xs, i): return sum(x[i] for x in xs)/len(xs) if xs else float("nan")
            print(f"{cp:>5} | {avg(w,2):>7.1f} /{avg(w,3):>7.0f} /{avg(w,4):>7.0f} | "
                  f"{avg(l,2):>7.1f} /{avg(l,3):>7.0f} /{avg(l,4):>7.0f}")
    print("\nEven share = 1/nseats (0.50 in 2P, 0.25 in 4P). The earliest "
          "step where LOSS material clearly trails WIN is the phase we lose.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
