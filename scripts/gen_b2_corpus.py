"""Reframe B.2 corpus generator — pv_eta self-play with feature emission.

Two-stage pipeline:

  Stage 1 — self-play (`probe_pveta_selfplay._run_one_game` reused
            verbatim). pv_eta plays itself; `BASELINE_VH_TRACE_FEATURES=1`
            makes `trace_accepted` emit the 14-d feature vector per
            solo accepted candidate (joints skipped — head is solo-only).
            Each game writes:
              <out>/game-<seed>/accepted.jsonl  — per-candidate features
              <out>/game-<seed>/replay.jsonl    — per-tick ship totals

  Stage 2 — label pairing (`probe_pveta_leaf_residual.pair_records`).
            For each seat-{0,1} accepted-solo record at step T, compute
            the focal-seat ship-delta over the next K turns. Both seats
            are labelled (each is a focal-seat decision in self-play).
            Output: <out>/corpus.jsonl with one row per (game, candidate)
            carrying:
              {game_id, seat, step, src_id, tgt_id, ships, eta,
               delta_pred, features, label}
            where features is the 15-d full vector (14 base + leaf_delta
            as feats[14]) and label is the actual K=10 ship-delta.

CLI:
    python scripts/gen_b2_corpus.py \\
        --games 100 --seed 1000 \\
        --out data/value_head/corpus_runs/2026-05-29-100games \\
        --wallclock-ms 100 --workers 8

Compute envelope: 100 games × 2 seats ≈ 18k labelled candidates.
Wallclock ≈ 30 min on 8 workers at WALLCLOCK_MS=100.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _ensure_feature_env():
    """Set BASELINE_VH_TRACE_FEATURES=1 in the PARENT process so the
    workers (spawn context) inherit it before they import the chooser."""
    os.environ["BASELINE_VH_TRACE_FEATURES"] = "1"


def _run_selfplay(out_dir: Path, games: int, seed: int,
                  wallclock_ms: int, workers: int,
                  agent_path: str) -> None:
    """Delegate to the B.1 runner; it's the same selfplay shape."""
    from scripts.probe_pveta_selfplay import main as _probe_main
    argv = [
        "--games", str(games),
        "--seed", str(seed),
        "--out-dir", str(out_dir),
        "--wallclock-ms", str(wallclock_ms),
        "--workers", str(workers),
        "--agent", agent_path,
    ]
    rc = _probe_main(argv)
    if rc != 0:
        raise RuntimeError(f"selfplay runner returned {rc}")


def _build_labels(out_dir: Path, K: int) -> dict:
    """Pair accepted-solo candidates with focal-seat ship-delta over
    K turns. Writes <out_dir>/corpus.jsonl. Returns a small summary
    dict for logging."""
    # Reuse the B.1 pair_records logic; it already attaches owner_at_launch
    # but for B.2 we ALSO need the per-candidate `features` field which
    # B.1 didn't carry. pair_records reads `accepted` rows verbatim, so
    # any extra fields just pass through silently — but pair_records is
    # hard-coded to focal=seat-0. Re-implement minimally here, handling
    # both seats.
    import numpy as np  # noqa: F401  (kept for typed nd arrays in callers)

    game_dirs = sorted(
        d for d in out_dir.iterdir()
        if d.is_dir() and d.name.startswith("game-")
    )
    if not game_dirs:
        raise RuntimeError(f"no game-* subdirs in {out_dir}")

    corpus_path = out_dir / "corpus.jsonl"
    n_rows = 0
    n_solo_skipped = 0
    n_no_features = 0
    n_truncated = 0
    seat_counts = {0: 0, 1: 0}

    with corpus_path.open("w") as out_fh:
        for game_dir in game_dirs:
            accepted_path = game_dir / "accepted.jsonl"
            replay_path = game_dir / "replay.jsonl"
            if not accepted_path.exists() or not replay_path.exists():
                continue

            replay_by_step: dict[int, dict] = {}
            with replay_path.open() as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    replay_by_step[int(rec["step"])] = rec
            if not replay_by_step:
                continue
            last_step = max(replay_by_step.keys())
            game_id = game_dir.name

            with accepted_path.open() as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    # Solo-only training (joints don't get a head correction).
                    if str(rec.get("kind", "solo")) != "solo":
                        n_solo_skipped += 1
                        continue
                    feats = rec.get("features")
                    if feats is None or len(feats) != 14:
                        n_no_features += 1
                        continue
                    seat = int(rec.get("me", 0))
                    t = int(rec["step"])
                    t_future = t + K
                    if t_future > last_step or t_future not in replay_by_step:
                        n_truncated += 1
                        continue
                    if t not in replay_by_step:
                        n_truncated += 1
                        continue
                    ships_t = int(replay_by_step[t][
                        f"focal{seat}_ships_total"])
                    ships_future = int(replay_by_step[t_future][
                        f"focal{seat}_ships_total"])
                    label = ships_future - ships_t
                    delta_pred = float(rec["delta_pred"])
                    # Full 15-d vector: 14 base + leaf_delta as feats[14].
                    full = list(feats) + [delta_pred]
                    out_row = {
                        "game_id": game_id,
                        "seat": seat,
                        "step": t,
                        "src_id": int(rec["src_id"]),
                        "tgt_id": int(rec["tgt_id"]),
                        "ships": int(rec["ships"]),
                        "eta": int(rec["eta"]),
                        "delta_pred": delta_pred,
                        "features": full,
                        "label": float(label),
                    }
                    out_fh.write(json.dumps(out_row) + "\n")
                    n_rows += 1
                    seat_counts[seat] = seat_counts.get(seat, 0) + 1

    return {
        "n_rows": n_rows,
        "n_solo_skipped": n_solo_skipped,
        "n_no_features": n_no_features,
        "n_truncated": n_truncated,
        "seat_counts": seat_counts,
        "corpus_path": str(corpus_path),
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--games", type=int, required=True)
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--out", type=Path, required=True,
                   help="Output directory; per-game subdirs + corpus.jsonl")
    p.add_argument("--wallclock-ms", type=int, default=100)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--K", type=int, default=10,
                   help="Label horizon (default 10)")
    p.add_argument(
        "--agent",
        default=str(REPO / "agents" / "baseline_pv_eta_probe" / "main.py"),
        help="Agent path (default baseline_pv_eta_probe: pv_eta @ λ_ml=0)",
    )
    p.add_argument("--skip-selfplay", action="store_true",
                   help="Skip Stage 1; only re-run label pairing on "
                        "existing game-*/ subdirs in --out.")
    args = p.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_selfplay:
        _ensure_feature_env()
        t0 = time.perf_counter()
        _run_selfplay(out_dir, args.games, args.seed,
                      args.wallclock_ms, args.workers, args.agent)
        print(f"=== stage 1 (self-play): "
              f"{time.perf_counter()-t0:.0f}s ===", flush=True)

    t1 = time.perf_counter()
    summary = _build_labels(out_dir, K=args.K)
    print(f"=== stage 2 (label pairing): "
          f"{time.perf_counter()-t1:.0f}s ===", flush=True)
    print(f"  corpus rows: {summary['n_rows']}")
    print(f"  seat counts: {summary['seat_counts']}")
    print(f"  skipped solo (kind!=solo): {summary['n_solo_skipped']}")
    print(f"  skipped no-features: {summary['n_no_features']}")
    print(f"  skipped truncated (T+K past end): {summary['n_truncated']}")
    print(f"  output: {summary['corpus_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
