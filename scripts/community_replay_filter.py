"""Stream-filter the daily orbit-wars episode zip from
kaggle/orbit-wars-episodes-2026-05-XX. Keeps only replays involving
top ladder players + a random sample for archetype variety. Avoids
extracting the full 20 GB.

CLI: python3 scripts/community_replay_filter.py \
    --zip /tmp/community-zip/orbit-wars-episodes-2026-05-20.zip \
    --out audit/community-replays/2026-05-20 \
    --random-sample 100
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import zipfile
from pathlib import Path

# Current top-15 by μ snapshot (Rule 43 re-pull 2026-05-21 evening).
# Snapshots drift; treat as a working list, not a fixed roster.
TOP_PLAYERS = {
    "3Comets",         # μ 1668
    "bowwowforeach",   # μ 1645
    "Vadasz",          # μ 1615
    "Jake Will",       # μ 1576
    "typeIIIfairy",    # μ 1576
    "TonyK",           # μ 1548
    "Audun Ljone Henriksen",  # μ 1500
    "flg",             # μ 1461
    "Ebi",             # μ 1443
    "Erfan Eshratifar", # μ 1437
    "saharan",         # μ 1434
    "Shun_PI",         # μ 1431
    "Vincent Schuler", # μ 1430
    "kovi",            # μ 1427
    "213tubo",         # μ 1427
    "ShunkiKyoya",     # μ 1411
    "🛰️ Low-Orbit Losers",  # μ 1411
    "ChrisLeiteScha",  # us — keep all our games for cross-reference
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True, help="path to downloaded daily zip")
    ap.add_argument("--out", required=True, help="output dir for kept JSONs")
    ap.add_argument("--random-sample", type=int, default=100,
                    help="random non-top-player sample size for archetype diversity")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    zip_path = Path(args.zip)
    if not zip_path.is_file():
        print(f"ERROR: missing zip {zip_path}", file=sys.stderr)
        return 1

    random.seed(args.seed)

    n_total = 0
    n_kept_top = 0
    n_kept_sample = 0
    n_skipped = 0
    bytes_kept = 0
    top_hit_counter: dict[str, int] = {}

    # Two-pass: pass 1 enumerates files + identifies top-player matches +
    # picks random sample from non-matches; pass 2 extracts.
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = [m for m in zf.namelist() if m.endswith(".json")]
        print(f"zip contains {len(members)} json files; scanning…",
              flush=True)

        keep_set: set[str] = set()
        sample_candidates: list[str] = []

        for i, member in enumerate(members):
            if i % 500 == 0:
                print(f"  scan {i}/{len(members)}…", flush=True)
            try:
                with zf.open(member) as f:
                    # Read just enough to find TeamNames. Quick-and-dirty:
                    # parse the first ~8 KB if possible. If TeamNames is
                    # late, fall back to full parse.
                    raw = f.read(16 * 1024)
                    # Try cheap substring check first
                    keep_for_top = False
                    for player in TOP_PLAYERS:
                        if f'"{player}"'.encode() in raw:
                            keep_for_top = True
                            top_hit_counter[player] = \
                                top_hit_counter.get(player, 0) + 1
                            break
                    if keep_for_top:
                        keep_set.add(member)
                    else:
                        sample_candidates.append(member)
            except Exception as e:
                print(f"  WARN scan {member}: {e}", file=sys.stderr)

            n_total += 1

        # Random sample for archetype variety
        if args.random_sample and len(sample_candidates) > args.random_sample:
            sample_chosen = random.sample(sample_candidates, args.random_sample)
        else:
            sample_chosen = sample_candidates

        print(f"\nscan complete:")
        print(f"  total: {n_total}")
        print(f"  top-player matches: {len(keep_set)}")
        print(f"  random sample (non-top): {len(sample_chosen)}")
        print(f"\ntop-player hit counts (substring match — may include "
              f"false matches from other JSON fields):")
        for player, count in sorted(top_hit_counter.items(),
                                    key=lambda kv: -kv[1]):
            print(f"  {player:32s} {count:>4d}")
        print(f"\nextracting kept replays to {out_dir}…", flush=True)

        all_keep = sorted(keep_set) + sorted(sample_chosen)
        for member in all_keep:
            try:
                data = zf.read(member)
                # Sanity-check that it actually contains a top player
                # (substring match in 16 KB could be a false positive)
                if member in keep_set:
                    parsed = json.loads(data)
                    teams = parsed.get("info", {}).get("TeamNames", [])
                    if not any(t in TOP_PLAYERS for t in teams):
                        n_skipped += 1
                        continue
                out_path = out_dir / Path(member).name
                out_path.write_bytes(data)
                bytes_kept += len(data)
                if member in keep_set:
                    n_kept_top += 1
                else:
                    n_kept_sample += 1
            except Exception as e:
                print(f"  WARN extract {member}: {e}", file=sys.stderr)

    print(f"\ndone:")
    print(f"  kept (top-player verified): {n_kept_top}")
    print(f"  kept (random sample): {n_kept_sample}")
    print(f"  skipped (false positives): {n_skipped}")
    print(f"  bytes written: {bytes_kept // (1024*1024)} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
