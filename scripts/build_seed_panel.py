"""Build the 128-seed geometry-diverse evaluation panel.

Pipeline:
  1. Enumerate seeds [0, POOL_SIZE), extract geometry features for each.
  2. Persist raw features to audit/seed-panel/features.json.
  3. Stratify into 4 (total_production) x 4 (rotating_share) x 2 (size_split) = 32 cells.

  Note: total_production replaces n_planets as the primary "density / pace" axis
  because n_planets is structurally correlated with rotating_share via the
  simulator's MIN_STATIC_GROUPS=3 constraint (a 20-planet board has at most 2
  rotating groups, so "very-sparse + mostly-rotating" is unreachable).
  total_production has a wider range and captures the PI's "fast games + high
  production" flavour directly.
  4. Within each cell, pick 4 seeds by farthest-point sampling on secondary
     features so the in-cell choices also spread.
  5. Write data/seed_panel_128.json and audit/seed-panel/build-report.md.

Bin edges are EMPIRICAL percentiles, not hard constants, so the pipeline
adapts if the underlying RNG behaviour shifts.

Run:
  python scripts/build_seed_panel.py
  python scripts/build_seed_panel.py --pool-size 50000   # expand if a cell is short
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from lib.geometry_features import extract_geometry

POOL_SIZE_DEFAULT = 10_000
PROD_BINS = 4
ROT_SHARE_BINS = 4
SIZE_SPLIT_BINS = 2
SEEDS_PER_CELL = 4
TOTAL_PANEL = PROD_BINS * ROT_SHARE_BINS * SIZE_SPLIT_BINS * SEEDS_PER_CELL


# Names for archetype cells — index tuples (prod_bin, rot_bin, split_bin) map
# to human-readable labels used downstream in regression reports.
PROD_NAMES = ["low_prod", "med_low_prod", "med_high_prod", "high_prod"]
ROT_SHARE_NAMES = ["mostly_static", "mixed_static", "mixed_rotating", "mostly_rotating"]
SIZE_SPLIT_NAMES = ["big_static", "big_rotating"]


def archetype_name(prod_bin: int, rot_bin: int, split_bin: int) -> str:
    return f"{PROD_NAMES[prod_bin]}__{ROT_SHARE_NAMES[rot_bin]}__{SIZE_SPLIT_NAMES[split_bin]}"


def percentile_edges(values: list[float], n_bins: int) -> list[float]:
    """Return n_bins-1 interior edges that split `values` into equal-count bins."""
    sorted_v = sorted(values)
    n = len(sorted_v)
    edges = []
    for i in range(1, n_bins):
        idx = int(round(n * i / n_bins))
        idx = max(1, min(n - 1, idx))
        edges.append(sorted_v[idx])
    return edges


def bin_index(value: float, edges: list[float]) -> int:
    for i, e in enumerate(edges):
        if value < e:
            return i
    return len(edges)


def standardize(records: list[dict], keys: list[str]) -> dict:
    """Return per-key (mean, std) used for farthest-point distances."""
    stats = {}
    for k in keys:
        vs = [r[k] for r in records]
        mu = statistics.fmean(vs)
        sd = statistics.pstdev(vs) or 1.0
        stats[k] = (mu, sd)
    return stats


def farthest_point_sample(
    candidates: list[dict],
    n_pick: int,
    feature_keys: list[str],
    stats: dict,
) -> list[dict]:
    """Greedy farthest-point sampling in standardized feature space."""
    if len(candidates) <= n_pick:
        return list(candidates)

    def zfeat(r: dict) -> tuple:
        return tuple((r[k] - stats[k][0]) / stats[k][1] for k in feature_keys)

    seed_vecs = [zfeat(r) for r in candidates]

    # Seed with the centroid-farthest record so the result is deterministic.
    centroid = tuple(sum(v[i] for v in seed_vecs) / len(seed_vecs) for i in range(len(feature_keys)))

    def dist(a, b):
        return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(len(a))))

    picked_idx = [max(range(len(seed_vecs)), key=lambda i: dist(seed_vecs[i], centroid))]
    while len(picked_idx) < n_pick:
        best_i, best_d = -1, -1.0
        for i, v in enumerate(seed_vecs):
            if i in picked_idx:
                continue
            d = min(dist(v, seed_vecs[j]) for j in picked_idx)
            if d > best_d:
                best_d, best_i = d, i
        picked_idx.append(best_i)
    return [candidates[i] for i in sorted(picked_idx)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool-size", type=int, default=POOL_SIZE_DEFAULT)
    parser.add_argument(
        "--features-cache",
        type=Path,
        default=REPO / "audit" / "seed-panel" / "features.json",
        help="Reuse cached features if file exists.",
    )
    args = parser.parse_args()

    audit_dir = REPO / "audit" / "seed-panel"
    audit_dir.mkdir(parents=True, exist_ok=True)

    features: list[dict] = []
    if args.features_cache.exists():
        cached = json.loads(args.features_cache.read_text())
        if len(cached) >= args.pool_size:
            features = cached[: args.pool_size]
            print(f"loaded {len(features)} cached features from {args.features_cache}")

    if not features:
        print(f"extracting features for seeds [0, {args.pool_size}) ...")
        t0 = time.perf_counter()
        for s in range(args.pool_size):
            features.append(extract_geometry(s))
            if (s + 1) % 500 == 0:
                rate = (s + 1) / (time.perf_counter() - t0)
                eta = (args.pool_size - s - 1) / rate
                print(f"  seed {s+1}/{args.pool_size}  {rate:.1f}/s  eta {eta:.0f}s")
        dt = time.perf_counter() - t0
        print(f"extraction done in {dt:.1f}s ({args.pool_size / dt:.1f} seeds/s)")
        args.features_cache.write_text(json.dumps(features))
        print(f"wrote {args.features_cache}")

    # Distribution summary
    print("\n=== feature distributions over pool ===")
    for k in ("n_planets", "rotating_share", "total_production", "size_split",
              "angular_velocity", "home_orbital_radius"):
        vs = [r[k] for r in features]
        if isinstance(vs[0], bool):
            print(f"  {k}: True={sum(vs)}/{len(vs)}")
        else:
            qs = statistics.quantiles(vs, n=4)
            print(f"  {k}: min={min(vs):.3f} q1={qs[0]:.3f} med={qs[1]:.3f} q3={qs[2]:.3f} max={max(vs):.3f}")

    # Bin edges (empirical percentiles)
    prod_edges = percentile_edges([r["total_production"] for r in features], PROD_BINS)
    rot_edges = percentile_edges([r["rotating_share"] for r in features], ROT_SHARE_BINS)
    split_edges = percentile_edges([r["size_split"] for r in features], SIZE_SPLIT_BINS)

    print(f"\ntotal_production edges: {prod_edges}")
    print(f"rotating_share edges: {[f'{e:.3f}' for e in rot_edges]}")
    print(f"size_split edges: {[f'{e:.3f}' for e in split_edges]}")

    # Stratify
    cells: dict[tuple[int, int, int], list[dict]] = {}
    for r in features:
        key = (
            bin_index(r["total_production"], prod_edges),
            bin_index(r["rotating_share"], rot_edges),
            bin_index(r["size_split"], split_edges),
        )
        cells.setdefault(key, []).append(r)

    # Stats for farthest-point sampling
    secondary_keys = [
        "total_production",
        "angular_velocity",
        "home_orbital_radius",
        "nearest_neighbor_mean",
    ]
    stats = standardize(features, secondary_keys)

    # Pick seeds per cell
    panel: list[dict] = []
    underfilled: list[str] = []
    for prod_bin in range(PROD_BINS):
        for rot_bin in range(ROT_SHARE_BINS):
            for split_bin in range(SIZE_SPLIT_BINS):
                key = (prod_bin, rot_bin, split_bin)
                pool = cells.get(key, [])
                name = archetype_name(prod_bin, rot_bin, split_bin)
                if len(pool) < SEEDS_PER_CELL:
                    underfilled.append(f"{name}: {len(pool)}/{SEEDS_PER_CELL}")
                picks = farthest_point_sample(pool, SEEDS_PER_CELL, secondary_keys, stats)
                for r in picks:
                    panel.append({
                        "seed": r["seed"],
                        "archetype": name,
                        "archetype_bins": list(key),
                        "features": r,
                    })

    print(f"\npanel size: {len(panel)} (target {TOTAL_PANEL})")
    if underfilled:
        print("UNDER-FILLED CELLS:")
        for u in underfilled:
            print(f"  {u}")
        print("  -> consider --pool-size 50000")

    # Persist
    out_json = REPO / "data" / "seed_panel_128.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps({
        "pool_size": args.pool_size,
        "bin_edges": {
            "total_production": prod_edges,
            "rotating_share": rot_edges,
            "size_split": split_edges,
        },
        "archetype_names": {
            "total_production": PROD_NAMES,
            "rotating_share": ROT_SHARE_NAMES,
            "size_split": SIZE_SPLIT_NAMES,
        },
        "panel": panel,
    }, indent=2))
    print(f"wrote {out_json}")

    # Build report
    report = REPO / "audit" / "seed-panel" / "build-report.md"
    lines = [
        "# Seed-panel build report",
        "",
        f"- pool size: {args.pool_size}",
        f"- panel size: {len(panel)} / {TOTAL_PANEL}",
        f"- under-filled cells: {len(underfilled)}",
        "",
        "## Bin edges (empirical percentiles)",
        f"- total_production: {prod_edges}",
        f"- rotating_share: {[f'{e:.3f}' for e in rot_edges]}",
        f"- size_split: {[f'{e:.3f}' for e in split_edges]}",
        "",
        "## Per-archetype seed counts",
    ]
    cell_counts: dict[str, int] = {}
    for entry in panel:
        cell_counts[entry["archetype"]] = cell_counts.get(entry["archetype"], 0) + 1
    for name, n in sorted(cell_counts.items()):
        lines.append(f"- {name}: {n}")
    if underfilled:
        lines += ["", "## Under-filled cells", ""] + [f"- {u}" for u in underfilled]
    report.write_text("\n".join(lines) + "\n")
    print(f"wrote {report}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
