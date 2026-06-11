"""scripts/margin_ffa.py — fast 4P triage: truncated games + material reads.

Sister to margin_ab.py (2P). The full ffa_panel plays 500-step games and
reports only first place — and material ties give every seat reward 1, so
self-similar matchups inflate to 100%. This harness truncates at
``--max-steps`` (default 150 — broken 4P stacks measured to date are
eliminated by step ~160-200, and material rank forms much earlier) and
reads the focal's SHARE of total ships + material rank at fixed steps.

One game per subprocess (process isolation: no env-gate leakage between
producer_plus bundles; background MUST still be namespaced copies because
the four agents of one game share a process).

Usage:
    python scripts/margin_ffa.py focal.py --background ns_bg.py \
        --seeds 4 --max-steps 150 --workers 1
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

CHECK_STEPS = (40, 80, 120)

_GAME_CODE = r"""
import json, sys, time
sys.path.insert(0, {repo!r})
from kaggle_environments import make
cfg = {{'seed': {seed}}}
if {max_steps} > 0:
    cfg['episodeSteps'] = {max_steps}
env = make('orbit_wars', configuration=cfg, debug=False)
t0 = time.perf_counter()
env.run({agents!r})
wall = time.perf_counter() - t0
F = {focal_seat}
shares, ranks = {{}}, {{}}
totals_by_step = []
for s in env.steps:
    obs = s[0]['observation']
    t = [0.0]*4
    for p in obs.get('planets', []):
        o = int(p[1])
        if 0 <= o < 4:
            t[o] += float(p[5])
    for f in obs.get('fleets', []):
        o = int(f[1])
        if 0 <= o < 4:
            t[o] += float(f[6])
    totals_by_step.append(t)
last = len(totals_by_step) - 1
for cs in list({checks!r}) + [last]:
    k = min(cs, last)
    t = totals_by_step[k]
    tot = sum(t) or 1.0
    shares[str(cs if cs != last else 'final')] = t[F] / tot
    ranks[str(cs if cs != last else 'final')] = 1 + sum(1 for q in range(4) if q != F and t[q] > t[F])
final = env.steps[-1]
print(json.dumps({{
    'rewards': [st['reward'] for st in final],
    'steps': len(env.steps), 'wall': round(wall, 1),
    'shares': shares, 'ranks': ranks,
    'eliminated': totals_by_step[last][F] <= 0.0,
}}))
"""


def _run_game(args):
    seed, focal, background, focal_seat, max_steps = args
    agents = [background] * 4
    agents[focal_seat] = focal
    code = _GAME_CODE.format(
        repo=str(REPO), seed=seed, max_steps=max_steps,
        agents=agents, focal_seat=focal_seat, checks=CHECK_STEPS,
    )
    r = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=3600,
    )
    out = r.stdout.strip().splitlines()
    for line in reversed(out):
        if line.startswith("{"):
            return seed, focal_seat, json.loads(line)
    return seed, focal_seat, {"error": (r.stderr or "no output")[-200:]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("focal")
    ap.add_argument("--background", required=True,
                    help="namespaced bundle used for the 3 rival seats")
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=150)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--no-rotate-seats", action="store_true",
                    help="one game per map. Against a self-similar background "
                    "seats correlate ~1, so maps >> seats per unit of compute")
    args = ap.parse_args()

    seats = [0] if args.no_rotate_seats else [0, 1, 2, 3]
    jobs = [
        (seed, args.focal, args.background, seat, args.max_steps)
        for seed in range(args.seed_start, args.seed_start + args.seeds)
        for seat in seats
    ]
    print(f"== margin_ffa focal={Path(args.focal).name} bg={Path(args.background).name} "
          f"games={len(jobs)} max_steps={args.max_steps} workers={args.workers} ==",
          flush=True)

    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_run_game, j): j for j in jobs}
        for fut in as_completed(futs):
            seed, seat, res = fut.result()
            if "error" in res:
                print(f"   seed={seed} seat={seat} ERROR: {res['error']}", flush=True)
                continue
            rows.append((seed, seat, res))
            sh, rk = res["shares"], res["ranks"]
            print(f"   seed={seed} seat={seat} steps={res['steps']:3d} "
                  f"elim={'Y' if res['eliminated'] else 'n'} "
                  + "  ".join(f"@{c}: {sh[str(c)]*100:4.1f}% r{rk[str(c)]}" for c in CHECK_STEPS)
                  + f"  final: {sh['final']*100:4.1f}% r{rk['final']}",
                  flush=True)

    if not rows:
        print("no completed games"); return
    n = len(rows)
    print(f"\n   n={n}")
    for c in [str(c) for c in CHECK_STEPS] + ["final"]:
        mean_share = sum(r[2]["shares"][c] for r in rows) / n
        mean_rank = sum(r[2]["ranks"][c] for r in rows) / n
        r1 = sum(1 for r in rows if r[2]["ranks"][c] == 1)
        print(f"   @{c:>5}: share={mean_share*100:5.1f}% (even=25.0%)  "
              f"mean-rank={mean_rank:.2f}  rank1={r1}/{n}")
    elim = sum(1 for r in rows if r[2]["eliminated"])
    print(f"   eliminated by truncation: {elim}/{n}")


if __name__ == "__main__":
    main()
