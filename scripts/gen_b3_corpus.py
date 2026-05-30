"""Reframe B.3 corpus generator — pv_eta self-play with CRN-paired advantage labels.

Two-stage pipeline:

  Stage 1 — self-play. Reuses `probe_pveta_selfplay._run_one_game` with
            two extra env vars set:
              PRERANK_TRACE_ON=1                  (enables prerank trace)
              BASELINE_VH_TRACE_FEATURES=1        (writes 14-d features)
            Each game writes:
              <out>/game-<seed>/prerank.jsonl  — per-scored-candidate
              <out>/game-<seed>/replay.jsonl   — extended (full obs +
                                                 per-tick actions)

  Stage 2 — CRN advantage labelling via `compute_crn_advantage`. For
            each top-N prerank candidate per (state, seat), runs two
            K-tick rollouts (idle + action) with pv_eta as the opponent
            throughout. Writes `<out>/corpus.jsonl` rows with
            `label = margin_action - margin_idle`.

CLI:
    python scripts/gen_b3_corpus.py \\
        --games 50 --seed 1000 \\
        --out data/value_head/b3-run \\
        --top-n 5 --K 5 --wallclock-ms 100 --workers 1

For smoke (~30 min):
    python scripts/gen_b3_corpus.py --games 4 --top-n 5 --K 5 \\
        --out data/value_head/b3-smoke

Plan: /root/.claude/plans/go-step-0-transient-newt.md (deprecated by
this run; the B.3 design questions are now answered).
Verdict / cost model: audit/2026-05-30-b3-step0-bench.md
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _stage1(out_dir: Path, games: int, seed: int,
            wallclock_ms: int, workers: int, agent_path: str) -> None:
    """Selfplay with prerank + replay + accepted traces."""
    os.environ["PRERANK_TRACE_ON"] = "1"
    os.environ["BASELINE_VH_TRACE_FEATURES"] = "1"
    from scripts.probe_pveta_selfplay import main as _selfplay_main
    argv = [
        "--games", str(games),
        "--seed", str(seed),
        "--out-dir", str(out_dir),
        "--wallclock-ms", str(wallclock_ms),
        "--workers", str(workers),
        "--agent", agent_path,
    ]
    rc = _selfplay_main(argv)
    if rc != 0:
        raise RuntimeError(f"stage 1 (selfplay) returned {rc}")


def _stage2(out_dir: Path, top_n: int, K: int,
            wallclock_ms: int, workers: int,
            max_games: int | None) -> None:
    from scripts.compute_crn_advantage import main as _label_main
    argv = [
        "--in", str(out_dir),
        "--top-n", str(top_n),
        "--K", str(K),
        "--wallclock-ms", str(wallclock_ms),
        "--workers", str(workers),
    ]
    if max_games is not None:
        argv += ["--max-games", str(max_games)]
    rc = _label_main(argv)
    if rc != 0:
        raise RuntimeError(f"stage 2 (labelling) returned {rc}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--games", type=int, required=True)
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--top-n", type=int, default=5)
    p.add_argument("--K", type=int, default=5)
    p.add_argument("--wallclock-ms", type=int, default=100)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument(
        "--agent",
        default=str(REPO / "agents" / "baseline_pv_eta_probe" / "main.py"),
        help="Agent path (default baseline_pv_eta_probe: pv_eta @ λ_ml=0)",
    )
    p.add_argument("--skip-stage1", action="store_true",
                   help="Reuse existing game-*/ subdirs in --out.")
    p.add_argument("--skip-stage2", action="store_true",
                   help="Stop after stage 1 (e.g. for inspection).")
    args = p.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_stage1:
        t0 = time.perf_counter()
        _stage1(out_dir, args.games, args.seed,
                args.wallclock_ms, args.workers, args.agent)
        print(f"=== stage 1 selfplay: "
              f"{time.perf_counter()-t0:.0f}s ===", flush=True)

    if not args.skip_stage2:
        t1 = time.perf_counter()
        _stage2(out_dir, args.top_n, args.K,
                args.wallclock_ms, args.workers, args.games)
        print(f"=== stage 2 labelling: "
              f"{time.perf_counter()-t1:.0f}s ===", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
