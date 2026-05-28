"""Generate self-play training data for the learned value head.

Runs games between pairs of agents via `kaggle_environments.make + env.run`
(same path as `scripts/tournament.py:_run_one`), then for each game:
  - samples K positions uniformly across the game timeline,
  - extracts a 40-dim feature vector at each sampled position from each
    seat's perspective via `lib.value_features.extract_features`,
  - labels each example with the final ship-margin from that seat's
    perspective.

Output: a .npz with `X (N, 40) float32` and `y (N,) float32`. Default
pairings reproduce the MVP plan's mixed-opponent pool (4 opponents x
2500 games = 10k games total -> 10 sample points x 2 seats x 10k =
200k examples).

Per-chunk checkpointing: each pairing writes its own .npz to the output
directory; a final `--merge` step concatenates them. Crash-safe.

Usage:
  # Smoke (8 games, validates pipeline)
  python scripts/gen_value_training_data.py \\
      --pairing self_play:agents/baseline_full/main.py:8 \\
      --out data/value_head/smoke

  # MVP target (10k games across 4 opponents)
  python scripts/gen_value_training_data.py \\
      --preset mvp --out data/value_head --workers 8

  # Merge per-pairing chunks into a single training.npz
  python scripts/gen_value_training_data.py --merge data/value_head
"""

from __future__ import annotations

import argparse
import importlib.util
import multiprocessing as mp
import random
import sys
import time
from pathlib import Path

import numpy as np

# Add project root to path so workers can import lib.*
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kaggle_environments import make  # noqa: E402

from lib.value_features import FEATURE_DIM, extract_features  # noqa: E402

SAMPLES_PER_GAME = 10
MIN_SAMPLE_TURN = 10  # skip opening fluff
SEATS = 2  # MVP: 2P only (matches current chooser)


# ---------------------------------------------------------------------------
# Agent specs
# ---------------------------------------------------------------------------


def _load_callable_from_path(path: str):
    """Load `agent` callable from a Python file (mirrors tournament.py)."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"agent file not found: {path}")
    spec = importlib.util.spec_from_file_location(
        f"_agent_{p.stem}_{abs(hash(path))}", p
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"could not import agent at {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if not hasattr(module, "agent"):
        raise AttributeError(f"{path} has no `agent` callable")
    return module.agent


def _resolve_agent_spec(spec: str):
    """Resolve a spec string to a callable.

    Specs:
      - 'lite_greedy'      -> lib.opp_model.lite_greedy_policy
      - '<path>.py'        -> module.agent
    """
    if spec == "lite_greedy":
        from lib.opp_model import lite_greedy_policy
        return lite_greedy_policy
    return _load_callable_from_path(spec)


# ---------------------------------------------------------------------------
# Per-game worker
# ---------------------------------------------------------------------------


def _final_margin_per_seat(state) -> dict[int, float]:
    """Final (ships + in-flight) per seat at game end."""
    obs0 = state[0].observation
    by_owner: dict[int, float] = {}
    for p in obs0.get("planets", []):
        owner, ships = int(p[1]), float(p[5])
        if owner >= 0:
            by_owner[owner] = by_owner.get(owner, 0.0) + ships
    for f in obs0.get("fleets", []):
        owner, ships = int(f[1]), float(f[6])
        if owner >= 0:
            by_owner[owner] = by_owner.get(owner, 0.0) + ships
    return by_owner


def _run_one_capture_samples(
    task: tuple,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Worker: run one game, return (X, y, meta).

    `task` is `(p0_spec, p1_spec, seed, samples_per_game, rng_seed)`.
    """
    p0_spec, p1_spec, seed, samples_per_game, rng_seed = task
    p0 = _resolve_agent_spec(p0_spec)
    p1 = _resolve_agent_spec(p1_spec)

    env = make("orbit_wars", configuration={"seed": int(seed)}, debug=False)
    env.run([p0, p1])
    final = env.steps[-1]
    n_steps = len(env.steps)

    # Final-margin label (from each seat's perspective).
    final_ships = _final_margin_per_seat(final)
    margin_p0 = float(final_ships.get(0, 0.0) - final_ships.get(1, 0.0))
    margin_per_seat = {0: margin_p0, 1: -margin_p0}

    # Sample positions: random uniform over [MIN_SAMPLE_TURN, n_steps-1].
    # Use a per-game RNG so workers don't share state.
    rng = random.Random(rng_seed)
    if n_steps - MIN_SAMPLE_TURN <= samples_per_game:
        sample_turns = list(range(MIN_SAMPLE_TURN, n_steps))
    else:
        sample_turns = sorted(
            rng.sample(range(MIN_SAMPLE_TURN, n_steps), samples_per_game)
        )

    X_rows: list[np.ndarray] = []
    y_rows: list[float] = []
    for t in sample_turns:
        # env.steps[t] is a list of per-seat states; observation is the
        # shared world view (full info per comp spec). Read seat 0's obs.
        obs_t = env.steps[t][0].observation
        for me in range(SEATS):
            X_rows.append(extract_features(obs_t, me=me, num_seats=SEATS))
            y_rows.append(margin_per_seat[me])

    X = np.stack(X_rows, axis=0).astype(np.float32)
    y = np.asarray(y_rows, dtype=np.float32)
    meta = {"seed": int(seed), "n_steps": int(n_steps), "margin_p0": margin_p0}
    return X, y, meta


# ---------------------------------------------------------------------------
# Pairing orchestration
# ---------------------------------------------------------------------------


def _parse_pairing(spec: str) -> tuple[str, str, int]:
    """`name:p0:p1:count` OR `self_play:agent:count` -> (p0, p1, count)."""
    parts = spec.split(":")
    if parts[0] == "self_play":
        if len(parts) != 3:
            raise ValueError(f"bad self_play spec: {spec!r}")
        agent, count = parts[1], int(parts[2])
        return agent, agent, count
    if len(parts) != 3:
        raise ValueError(
            f"pairing must be 'self_play:<agent>:<count>' or"
            f" '<p0>:<p1>:<count>': {spec!r}"
        )
    return parts[0], parts[1], int(parts[2])


def _run_pairing(
    p0: str, p1: str, count: int, seed_offset: int,
    workers: int, samples_per_game: int, out_path: Path,
) -> dict:
    """Run `count` games in parallel; write X, y to `out_path`."""
    tasks = [
        (p0, p1, seed_offset + i, samples_per_game, seed_offset + i + 999983)
        for i in range(count)
    ]

    t0 = time.perf_counter()
    X_chunks: list[np.ndarray] = []
    y_chunks: list[np.ndarray] = []
    n_done = 0
    margin_sum = 0.0
    n_steps_sum = 0

    with mp.Pool(processes=workers) as pool:
        for X, y, meta in pool.imap_unordered(
            _run_one_capture_samples, tasks, chunksize=1
        ):
            X_chunks.append(X)
            y_chunks.append(y)
            n_done += 1
            margin_sum += meta["margin_p0"]
            n_steps_sum += meta["n_steps"]
            if n_done % max(1, count // 20) == 0 or n_done == count:
                elapsed = time.perf_counter() - t0
                rate = n_done / elapsed if elapsed > 0 else 0.0
                eta = (count - n_done) / rate if rate > 0 else 0.0
                print(
                    f"  [{p0[:30]} vs {p1[:30]}] "
                    f"{n_done}/{count} games, "
                    f"{rate:.2f} games/s, ETA {eta:.0f}s",
                    flush=True,
                )

    X = np.concatenate(X_chunks, axis=0)
    y = np.concatenate(y_chunks, axis=0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, X=X, y=y)
    elapsed = time.perf_counter() - t0
    summary = {
        "p0": p0,
        "p1": p1,
        "n_games": count,
        "n_examples": int(X.shape[0]),
        "mean_margin_p0": margin_sum / count if count else 0.0,
        "mean_n_steps": n_steps_sum / count if count else 0.0,
        "elapsed_s": elapsed,
        "out_path": str(out_path),
    }
    print(
        f"  -> wrote {X.shape[0]} examples to {out_path} in {elapsed:.1f}s",
        flush=True,
    )
    return summary


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------

MVP_PAIRINGS = [
    # name, spec
    ("self_play_baseline_full",
     "self_play:agents/baseline_full/main.py:2500"),
    ("baseline_full_vs_orbitfix",
     "agents/baseline_full/main.py:"
     "submissions/baseline_joint_aggr_consolidated_orbitfix.py:2500"),
    ("baseline_full_vs_v351",
     "agents/baseline_full/main.py:agents/v3.5.1/main.py:2500"),
    ("baseline_full_vs_litegreedy",
     "agents/baseline_full/main.py:lite_greedy:2500"),
]

VALIDATION_PAIRING = (
    "validation_vs_trajectory",
    "agents/baseline_full/main.py:agents/baseline/main.py:1000",
)


# ---------------------------------------------------------------------------
# Merge helper
# ---------------------------------------------------------------------------


def _merge_chunks(chunk_dir: Path, out_name: str = "training.npz") -> None:
    """Concatenate all *.npz under chunk_dir into chunk_dir/<out_name>."""
    chunks = sorted(p for p in chunk_dir.glob("*.npz") if p.name != out_name)
    if not chunks:
        raise FileNotFoundError(f"no .npz chunks found in {chunk_dir}")
    X_all: list[np.ndarray] = []
    y_all: list[np.ndarray] = []
    for p in chunks:
        d = np.load(p)
        X_all.append(d["X"])
        y_all.append(d["y"])
        print(f"  {p.name}: X={d['X'].shape} y={d['y'].shape}")
    X = np.concatenate(X_all, axis=0)
    y = np.concatenate(y_all, axis=0)
    out_path = chunk_dir / out_name
    np.savez(out_path, X=X, y=y)
    print(
        f"merged {len(chunks)} chunks -> {out_path}, X={X.shape} y={y.shape}"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--pairing", action="append", default=[],
        help="pairing spec: 'self_play:<agent>:<count>' or '<p0>:<p1>:<count>'",
    )
    p.add_argument("--preset", choices=["mvp"], default=None,
                   help="use a preset pairing list")
    p.add_argument("--out", type=Path, required=False,
                   help="output directory (chunks land here)")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--samples-per-game", type=int, default=SAMPLES_PER_GAME)
    p.add_argument("--seed-offset", type=int, default=0)
    p.add_argument("--merge", type=Path, default=None,
                   help="merge chunks under this dir into training.npz")
    p.add_argument("--include-validation", action="store_true",
                   help="also run the held-out validation pairing (mvp preset)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    if args.merge is not None:
        _merge_chunks(args.merge)
        return 0

    if args.preset == "mvp":
        pairing_specs = [(name, spec) for name, spec in MVP_PAIRINGS]
        if args.include_validation:
            pairing_specs.append(VALIDATION_PAIRING)
    else:
        if not args.pairing:
            print("error: provide --pairing or --preset", file=sys.stderr)
            return 2
        pairing_specs = [
            (f"pairing_{i:02d}", spec) for i, spec in enumerate(args.pairing)
        ]

    if args.out is None:
        print("error: --out required", file=sys.stderr)
        return 2

    summaries: list[dict] = []
    seed_offset = args.seed_offset
    for name, spec in pairing_specs:
        p0, p1, count = _parse_pairing(spec)
        print(f"\n=== {name}: {p0} vs {p1}, n={count} ===", flush=True)
        out_path = args.out / f"{name}.npz"
        summary = _run_pairing(
            p0, p1, count, seed_offset,
            args.workers, args.samples_per_game, out_path,
        )
        summaries.append(summary)
        seed_offset += count

    print("\n=== summary ===")
    for s in summaries:
        print(
            f"  {Path(s['out_path']).stem}: "
            f"{s['n_examples']} examples, "
            f"margin_p0={s['mean_margin_p0']:+.1f}, "
            f"steps={s['mean_n_steps']:.0f}, "
            f"{s['elapsed_s']:.1f}s"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
