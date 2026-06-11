"""Build the oracle POLICY dataset: per-(state, candidate-pair) expert labels.

For every top-rated seat in the downloaded replays, every k-th turn:
enumerate the candidate (source -> target) pairs with the SAME shortlist the
runtime planner uses, featurize each with the shared policy features, and
label a pair 1 when the expert actually launched source->target that turn
(attributed by exact flight simulation of the recorded action angles).
Positives also record the launched size as a fraction of the source's
launchable spare.

Coverage stats (what fraction of expert launches the shortlist contains)
print at the end — if low, widen the shortlist before trusting training.

Usage:
  python scripts/oracle_policy_dataset.py [--min-seat-score 1480]
      [--stride 2] [--workers 6] [--limit N]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from agents.oracle.engine import World, PLAN_HORIZON         # noqa: E402
from agents.oracle.planner import source_states, shortlist_pairs  # noqa: E402
from agents.oracle.policy_features import (                  # noqa: E402
    PolicyContext, N_POLICY_FEATURES)

REPLAY_DIR = Path(os.environ.get("ORACLE_REPLAY_DIR", str(REPO / "data" / "external" / "replays")))
EPISODES_PATH = REPO / "data" / "external" / "episodes.jsonl"
EXTRACT_HORIZON = PLAN_HORIZON

_SEAT_SCORES = None


def seat_scores():
    global _SEAT_SCORES
    if _SEAT_SCORES is None:
        m = {}
        with open(EPISODES_PATH) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                m[r["id"]] = {a.get("index", k): (a.get("score") or 0.0)
                              for k, a in enumerate(r["agents"])}
        _SEAT_SCORES = m
    return _SEAT_SCORES


def attribute_moves(world, moves, me):
    """Recorded action list -> {(src_idx, hit_idx): ships}; misses counted."""
    out = {}
    misses = 0
    for mv in moves or []:
        if not mv or len(mv) < 3:
            continue
        pid, angle, ships = int(mv[0]), float(mv[1]), int(mv[2])
        if pid not in world.idx_of or ships < 1:
            continue
        src = world.idx_of[pid]
        sx, sy = world.px[src], world.py[src]
        lx = sx + math.cos(angle) * (world.pr[src] + 0.1)
        ly = sy + math.sin(angle) * (world.pr[src] + 0.1)
        hit, dt = world.fly(lx, ly, angle, ships, 1)
        if hit is None:
            misses += 1
            continue
        out[(src, hit)] = out.get((src, hit), 0) + ships
    return out, misses


def process(job):
    path, min_score, stride = job
    try:
        with open(path) as f:
            d = json.load(f)
        steps = d.get("steps", [])
        if len(steps) < 60:
            return None
        ep_id = int(d.get("info", {}).get("EpisodeId", 0) or 0)
        scores = seat_scores().get(ep_id, {})
        seats = [p for p in range(len(steps[0]))
                 if scores.get(p, 0.0) >= min_score]
        if not seats:
            return None

        X, y, frac, meta = [], [], [], []
        cov_hit = cov_tot = sun_miss = 0
        state_id = 0
        for t in range(1, len(steps) - 1, stride):
            base_obs = steps[t][0].get("observation")
            if not base_obs or not base_obs.get("planets"):
                continue
            for seat in seats:
                moves = steps[t][seat].get("action") or []
                obs = dict(base_obs)
                obs["player"] = seat
                obs["step"] = t
                w = World(obs, horizon=EXTRACT_HORIZON)
                if not any(o == seat for o in w.owner0):
                    continue
                w.build_ledger()
                sp = source_states(w, seat)
                pairs = shortlist_pairs(w, sp)
                if not pairs:
                    continue
                expert, misses = attribute_moves(w, moves, seat)
                sun_miss += misses
                pair_set = {(s, tt) for (_k, s, tt) in pairs}
                for (s, tt), ships in expert.items():
                    cov_tot += 1
                    if (s, tt) in pair_set:
                        cov_hit += 1
                pctx = PolicyContext(w, sp)
                state_id += 1
                rng = (ep_id * 1000003 + t * 131 + seat)
                for kn, (kind, s, tt) in enumerate(pairs):
                    g, safe, doomed = sp.get(s, (0, 0, False))
                    fired = (s, tt) in expert
                    # negative subsampling (deterministic): keep all
                    # positives + ~30% of negatives with weight 1/0.3
                    if not fired and ((rng + kn * 7919) % 10) >= 3:
                        continue
                    X.append(pctx.pair(s, tt, g, safe, doomed))
                    y.append(1.0 if fired else 0.0)
                    frac.append(min(1.0, expert[(s, tt)] / max(1.0, g))
                                if fired else -1.0)
                    meta.append((ep_id, seat, t, state_id))
        if not X:
            return None
        return (np.asarray(X, np.float32), np.asarray(y, np.float32),
                np.asarray(frac, np.float32), np.asarray(meta, np.int64),
                cov_hit, cov_tot, sun_miss)
    except Exception as e:
        return ("ERR", os.path.basename(str(path)), str(e)[:200])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(
        REPO / "data" / "external" / "oracle_policy_ds.npz"))
    ap.add_argument("--min-seat-score", type=float, default=1480.0)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    seat_scores()    # build once pre-fork (workers inherit)
    paths = sorted(REPLAY_DIR.glob("episode-*-replay.json"))
    if args.limit:
        paths = paths[: args.limit]
    jobs = [(str(p), args.min_seat_score, args.stride) for p in paths]
    print(f"{len(jobs)} replays, {args.workers} workers, "
          f"{N_POLICY_FEATURES} policy features")

    Xs, ys, fr, metas = [], [], [], []
    cov_hit = cov_tot = sun_miss = errs = 0
    with Pool(args.workers) as pool:
        for k, res in enumerate(pool.imap_unordered(process, jobs)):
            if res is None:
                continue
            if isinstance(res[0], str):
                errs += 1
                if errs <= 5:
                    print("  err:", res[1], res[2])
                continue
            X, y, f_, m, ch, ct, sm = res
            Xs.append(X); ys.append(y); fr.append(f_); metas.append(m)
            cov_hit += ch; cov_tot += ct; sun_miss += sm
            if (k + 1) % 100 == 0:
                print(f"  {k+1}/{len(jobs)} ({sum(x.shape[0] for x in Xs)} rows)")
    X = np.concatenate(Xs)
    y = np.concatenate(ys)
    f_ = np.concatenate(fr)
    meta = np.concatenate(metas)
    np.savez_compressed(args.out, X=X, y=y, frac=f_, meta=meta)
    print(f"saved {X.shape} -> {args.out} ({errs} errors)")
    print(f"positives: {int(y.sum())} ({100*y.mean():.2f}%)")
    print(f"expert-launch shortlist coverage: {cov_hit}/{cov_tot} "
          f"({100*cov_hit/max(1,cov_tot):.1f}%), "
          f"expert sun/oob misses: {sun_miss}")


if __name__ == "__main__":
    main()
