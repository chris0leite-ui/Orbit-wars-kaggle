"""Build the oracle value-model dataset from downloaded ladder replays.

For every replay in data/external/replays/, every seat, every k-th step:
run the oracle engine's exact ledger on that frame and extract the shared
feature vector (agents/oracle/features.py — the SAME code the agent runs at
inference), labelled with that seat's final outcome.

Labels:
  y_win   1.0 win / 0.5 tied-for-best / 0.0 loss  (from final rewards)
  y_share final (garrison + in-flight) share of the focal seat

Usage:
  python scripts/oracle_dataset.py --out data/external/oracle_ds.npz \
      --stride 2 --workers 8 [--limit N]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from agents.oracle.engine import World            # noqa: E402
from agents.oracle import features as F           # noqa: E402

REPLAY_DIR = REPO / "data" / "external" / "replays"
EXTRACT_HORIZON = 36   # > max(PROBES); keeps per-frame cost low


def final_share(last_obs, seat):
    per = {}
    for p in last_obs.get("planets", []):
        if p[1] >= 0:
            per[p[1]] = per.get(p[1], 0) + p[5]
    for f in last_obs.get("fleets", []):
        if f[1] >= 0:
            per[f[1]] = per.get(f[1], 0) + f[6]
    tot = sum(per.values())
    return per.get(seat, 0) / tot if tot > 0 else 0.0


def process_replay(path: str):
    try:
        with open(path) as f:
            d = json.load(f)
        steps = d.get("steps", [])
        if len(steps) < 60:
            return None
        n_seats = len(steps[0])
        rewards = [steps[-1][p].get("reward") for p in range(n_seats)]
        rewards = [(-1.0 if r is None else float(r)) for r in rewards]
        rmax = max(rewards)
        n_best = sum(1 for r in rewards if r == rmax)
        wins = [(1.0 if n_best == 1 else 0.5) if r == rmax else 0.0
                for r in rewards]
        last_obs = steps[-1][0]["observation"]
        shares = [final_share(last_obs, p) for p in range(n_seats)]

        X, yw, ys, meta = [], [], [], []
        ep_id = int(d.get("info", {}).get("EpisodeId", 0) or 0)
        stride = 2
        for t in range(1, len(steps) - 1, stride):
            base_obs = steps[t][0].get("observation")
            if not base_obs or not base_obs.get("planets"):
                continue
            for seat in range(n_seats):
                obs = dict(base_obs)
                obs["player"] = seat
                obs["step"] = t
                w = World(obs, horizon=EXTRACT_HORIZON)
                w.build_ledger()
                X.append(F.extract(w))
                yw.append(wins[seat])
                ys.append(shares[seat])
                meta.append((ep_id, seat, t, n_seats))
        if not X:
            return None
        return (np.asarray(X, dtype=np.float32),
                np.asarray(yw, dtype=np.float32),
                np.asarray(ys, dtype=np.float32),
                np.asarray(meta, dtype=np.int64))
    except Exception as e:
        return ("ERR", os.path.basename(path), str(e)[:200])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "data" / "external" / "oracle_ds.npz"))
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    paths = sorted(str(p) for p in REPLAY_DIR.glob("episode-*-replay.json"))
    if args.limit:
        paths = paths[: args.limit]
    print(f"{len(paths)} replays, {args.workers} workers, "
          f"{F.N_FEATURES} features")

    Xs, yws, yss, metas = [], [], [], []
    errs = 0
    with Pool(args.workers) as pool:
        for k, res in enumerate(pool.imap_unordered(process_replay, paths)):
            if res is None:
                continue
            if isinstance(res[0], str):  # ("ERR", file, msg) sentinel
                errs += 1
                if errs <= 5:
                    print("  err:", res[1], res[2])
                continue
            X, yw, ys, meta = res
            Xs.append(X); yws.append(yw); yss.append(ys); metas.append(meta)
            if (k + 1) % 100 == 0:
                print(f"  {k+1}/{len(paths)} replays "
                      f"({sum(x.shape[0] for x in Xs)} rows, {errs} errs)")
    X = np.concatenate(Xs)
    yw = np.concatenate(yws)
    ys = np.concatenate(yss)
    meta = np.concatenate(metas)
    np.savez_compressed(args.out, X=X, y_win=yw, y_share=ys, meta=meta,
                        feature_names=np.array(F.FEATURE_NAMES))
    print(f"saved {X.shape} -> {args.out} ({errs} replay errors)")
    print(f"win-rate balance: {yw.mean():.3f}, share mean {ys.mean():.3f}")


if __name__ == "__main__":
    main()
