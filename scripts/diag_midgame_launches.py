"""Isolate the mid-game (step ~40-110) expansion-conversion failure.

Over real ladder episodes (split WIN/LOSS), aggregate OUR launches in the
stall window by target owner (neutral / enemy / own=reinforce) and by ETA
bucket (near/far = reach), plus net planet change. Distinguishes:
  - low neutral-launch count        -> not trying to expand (value/defense)
  - high neutral launches, ~0 gain  -> launches fail to convert (combat/churn)
  - launches all short-ETA          -> reach problem (don't grab far neutrals)

Usage: python scripts/diag_midgame_launches.py 53595717 [--lo 40 --hi 110]
"""
from __future__ import annotations
import argparse, json, math, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LIVE = REPO / "audit" / "live-episodes"
sys.path.insert(0, str(REPO))
from scripts.label_shot_outcomes import _infer_target_pid, _fleet_speed  # noqa


def analyze(steps, pid, lo, hi):
    """Return per-window aggregates for one episode's focal seat."""
    n_neutral = n_enemy = n_own = 0
    eta_near = eta_far = 0
    sent_neutral = 0.0
    pl_lo = pl_hi = None
    for t in range(lo, min(hi, len(steps))):
        seat = steps[t][pid]
        obs = seat.get("observation", {}) or {}
        planets = obs.get("planets", []) or []
        by_id = {int(p[0]): p for p in planets}
        plcount = sum(1 for p in planets if int(p[1]) == pid and int(p[0]) >= 0)
        if t == lo: pl_lo = plcount
        pl_hi = plcount
        for a in (seat.get("action") or []):
            if not a or len(a) < 3:
                continue
            try:
                src_id = int(a[0]); ang = float(a[1]); ships = float(a[2])
            except (TypeError, ValueError):
                continue
            src = by_id.get(src_id)
            if src is None:
                continue
            tgt_id = _infer_target_pid((float(src[2]), float(src[3])), ang, planets)
            tgt = by_id.get(tgt_id) if tgt_id is not None else None
            if tgt is None:
                continue
            owner = int(tgt[1])
            if owner == pid: n_own += 1
            elif owner == -1: n_neutral += 1; sent_neutral += ships
            else: n_enemy += 1
            d = math.hypot(float(tgt[2]) - float(src[2]), float(tgt[3]) - float(src[3]))
            eta = d / max(_fleet_speed(ships), 1e-6)
            if eta <= 12: eta_near += 1
            else: eta_far += 1
    return dict(neutral=n_neutral, enemy=n_enemy, own=n_own,
               near=eta_near, far=eta_far, sent_neutral=sent_neutral,
               dplanets=(pl_hi - pl_lo) if pl_lo is not None else 0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("sub_id")
    ap.add_argument("--lo", type=int, default=40)
    ap.add_argument("--hi", type=int, default=110)
    ap.add_argument("--team", default="ChrisLeiteScha")
    ap.add_argument("--fmt", default="4P", choices=["2P", "4P"])
    args = ap.parse_args()

    agg = {"WIN": [], "LOSS": []}
    for f in sorted((LIVE / args.sub_id).glob("episode-*-replay.json")):
        try: r = json.loads(f.read_text())
        except Exception: continue
        teams = r.get("info", {}).get("TeamNames", [])
        ours = [i for i, t in enumerate(teams) if t == args.team]
        steps = r.get("steps", [])
        if len(ours) != 1 or not steps: continue
        nseats = len(steps[0])
        if (args.fmt == "2P") != (nseats == 2): continue
        pid = ours[0]
        rew = [steps[-1][i].get("reward") for i in range(nseats)]
        rew = [x for x in rew if x is not None]
        my = steps[-1][pid].get("reward")
        if my is None or not rew: continue
        agg["WIN" if my == max(rew) else "LOSS"].append(analyze(steps, pid, args.lo, args.hi))

    print(f"sub {args.sub_id}  {args.fmt}  window steps {args.lo}-{args.hi}\n")
    print(f"{'':6} | {'n':>3} | launches/game: neutral enemy own | near far | "
          f"ships->neutral | net Δplanets")
    for k in ("WIN", "LOSS"):
        rows = agg[k]
        n = len(rows)
        if not n: continue
        def m(key): return sum(x[key] for x in rows) / n
        print(f"{k:6} | {n:>3} | {m('neutral'):>14.1f} {m('enemy'):>5.1f} {m('own'):>4.1f} | "
              f"{m('near'):>4.1f} {m('far'):>4.1f} | {m('sent_neutral'):>13.0f} | {m('dplanets'):>+6.1f}")
    print("\nRead: LOSS low neutral-launches => not expanding; LOSS high "
          "neutral-launches but Δplanets~0 => not converting; LOSS far~0 "
          "while WIN far>0 => reach problem.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
