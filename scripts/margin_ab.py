"""scripts/margin_ab.py — process-isolated PAIRED MARGIN harness (fast triage).

Why this exists: the binary-outcome A/B (clean_ab.py) needs n >= 32 games for
a lift claim, but replay mining (audit/2026-06-10-top-ladder-behavior.md +
mine_decision_rules.py) shows the total-ship lead stops changing hands by
step 30-54 (p50) / <= 95 (p90) in every corpus measured — top-3 teams AND our
own live games. The material lead at step ~100 therefore predicts the final
result in >= 90% of games, so a CONTINUOUS margin read at fixed early steps
carries far more information per game than the win bit, and seat-paired
margins on deterministic seeds cancel the map draw. n = 8-16 games becomes a
meaningful triage (still not a Rule 45 submit gate — that stays n >= 32 wins).

Reported per game: focal ship-share lead (focal − opp, as share of total) at
steps 40 / 80 / 120 / 250, decision step (last lead sign flip), and outcome.
Aggregate: per-seed seat-paired mean lead + win rate.

Usage:
    python scripts/margin_ab.py focal.py opp.py --seeds 4 --workers 2
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

CHECK_STEPS = (40, 80, 120, 250)

# Runs inside the per-game subprocess: play one game, then walk env.steps and
# compute per-player ship totals (planets + fleets) at each step.
_GAME_CODE = r"""
import json, sys, time
sys.path.insert(0, {repo!r})
from kaggle_environments import make
cfg = {{'seed': {seed}}}
if {max_steps} > 0:
    cfg['episodeSteps'] = {max_steps}
env = make('orbit_wars', configuration=cfg, debug=False)
t0 = time.perf_counter()
env.run([{p0!r}, {p1!r}])
wall = time.perf_counter() - t0
totals = []
launched = [False, False]
for s in env.steps:
    obs = s[0]['observation']
    t = [0.0, 0.0]
    for p in obs.get('planets', []):
        o = int(p[1])
        if 0 <= o <= 1:
            t[o] += float(p[5])
    for f in obs.get('fleets', []):
        o = int(f[1])
        if 0 <= o <= 1:
            t[o] += float(f[6])
            launched[o] = True
    totals.append(t)
decision = 0
prev = 0
for k, t in enumerate(totals):
    lead = t[0] - t[1]
    sign = (lead > 0) - (lead < 0)
    if sign != 0 and prev != 0 and sign != prev:
        decision = k
    if sign != 0:
        prev = sign
leads = {{}}
for cs in {check_steps!r}:
    k = min(cs, len(totals) - 1)
    tot = totals[k][0] + totals[k][1]
    leads[str(cs)] = (totals[k][0] - totals[k][1]) / tot if tot > 0 else 0.0
final = env.steps[-1]
print(json.dumps({{'r0': final[0]['reward'], 'r1': final[1]['reward'],
                   'n_steps': len(env.steps), 'wall': wall,
                   'decision': decision, 'leads_p0': leads,
                   'launched': launched}}))
"""


def _worker_play(args: tuple[int, str, str, bool, int]) -> dict:
    seed, focal_path, opp_path, focal_is_p0, max_steps = args
    p0, p1 = (focal_path, opp_path) if focal_is_p0 else (opp_path, focal_path)
    code = _GAME_CODE.format(repo=str(REPO), seed=int(seed), p0=str(p0), p1=str(p1),
                             check_steps=tuple(CHECK_STEPS), max_steps=int(max_steps))
    try:
        proc = subprocess.run(
            [sys.executable, "-c", code],
            env={**os.environ}, capture_output=True, text=True, timeout=900,
        )
    except subprocess.TimeoutExpired as e:
        return {"seed": seed, "focal_is_p0": focal_is_p0, "outcome": "timeout",
                "stderr": f"timed out after {e.timeout}s"}
    out = (proc.stdout or "").strip().splitlines()
    line = next((l for l in reversed(out) if l.startswith("{")), "")
    if not line:
        return {"seed": seed, "focal_is_p0": focal_is_p0, "outcome": "error",
                "stderr": (proc.stderr or "")[:400]}
    data = json.loads(line)
    r0, r1 = data["r0"], data["r1"]
    if r0 is None or r1 is None:
        return {"seed": seed, "focal_is_p0": focal_is_p0, "outcome": "error",
                "stderr": "None reward (agent error/timeout in-game)"}
    launched = data.get("launched")
    if launched is not None and not all(launched):
        # Dead-opponent guard (ledger-branch postmortem): a player that
        # never launched a single fleet was almost certainly dead at load —
        # its games sweep like a dominated opponent and poison the stats.
        return {"seed": seed, "focal_is_p0": focal_is_p0, "outcome": "error",
                "stderr": f"DEAD PLAYER launched={launched} — game excluded"}
    sgn = 1.0 if focal_is_p0 else -1.0
    leads = {int(k): sgn * v for k, v in data["leads_p0"].items()}
    focal_r, opp_r = (r0, r1) if focal_is_p0 else (r1, r0)
    return {
        "seed": seed, "focal_is_p0": focal_is_p0,
        "outcome": "win" if focal_r > opp_r else ("loss" if focal_r < opp_r else "draw"),
        "focal_won": focal_r > opp_r,
        "n_steps": data["n_steps"], "decision": data["decision"],
        "leads": leads, "wall": data["wall"],
    }


def main() -> int:
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("focal")
    ap.add_argument("opp")
    ap.add_argument("--seeds", type=int, default=4, help="N seeds; each played twice (both seats)")
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--max-steps", type=int, default=0,
                    help="truncate games at N steps (material verdict; decision step p90<=95 in every mined corpus). 0 = full game")
    args = ap.parse_args()
    focal = str(Path(args.focal).resolve())
    opp = str(Path(args.opp).resolve())
    for p in (focal, opp):
        if not Path(p).is_file():
            print(f"not found: {p}", file=sys.stderr)
            return 2

    print(f"== margin_ab focal={Path(focal).name}  opp={Path(opp).name}  "
          f"seeds={args.seeds} (×2 seats = {2*args.seeds} games)  workers={args.workers} ==")

    tasks = []
    for s in range(args.seed_start, args.seed_start + args.seeds):
        tasks.append((s, focal, opp, True, args.max_steps))
        tasks.append((s, focal, opp, False, args.max_steps))
    t0 = time.perf_counter()
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_worker_play, t) for t in tasks]
        for fut in as_completed(futs):
            try:
                r = fut.result()
            except Exception as e:
                r = {"outcome": "error", "stderr": f"worker raised: {e}"[:400]}
            results.append(r)
            if "leads" in r:
                ls = "  ".join(f"L{k}={100*r['leads'][k]:+5.1f}%" for k in CHECK_STEPS)
                print(f"   seed={r['seed']:>3} seat={'P0' if r['focal_is_p0'] else 'P1'} "
                      f"{r['outcome']:>4} steps={r['n_steps']:>3} dec={r['decision']:>3}  {ls}")
            else:
                print(f"   seed={r.get('seed','?'):>3} {r.get('outcome','?')}: {r.get('stderr','')[:120]}")

    ok = [r for r in results if "leads" in r]
    if not ok:
        print("\n   ALL ERROR — no usable games")
        return 1
    wins = sum(1 for r in ok if r["focal_won"])
    print(f"\n   focal_wins={wins}/{len(ok)}")
    for cs in CHECK_STEPS:
        vals = [r["leads"][cs] for r in ok]
        mean = sum(vals) / len(vals)
        pos = sum(1 for v in vals if v > 0)
        # Seat-paired per-seed mean (cancels first-mover advantage on the map).
        by_seed = {}
        for r in ok:
            by_seed.setdefault(r["seed"], []).append(r["leads"][cs])
        paired = [sum(v) / len(v) for v in by_seed.values() if len(v) == 2]
        pmean = sum(paired) / len(paired) if paired else float("nan")
        ppos = sum(1 for v in paired if v > 0)
        print(f"   lead@{cs:<3}: mean={100*mean:+5.1f}%  games-ahead={pos}/{len(vals)}  "
              f"paired-mean={100*pmean:+5.1f}%  seeds-ahead={ppos}/{len(paired)}")
    decs = sorted(r["decision"] for r in ok)
    print(f"   decision step: p50={decs[len(decs)//2]}  max={decs[-1]}  "
          f"elapsed={time.perf_counter()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
