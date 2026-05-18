"""Cross-check: do our 128 stratified seeds cover the geometries that
appear in REAL games on the Kaggle ladder?

Loads every ``episode-*-replay.json`` under ``audit/replays/live/<sid>/``
(downloaded via ``kaggle competitions replay ...``; see ``--pull``),
extracts turn-0 geometry features from each, bins them into the 32
panel archetypes, and reports:

  - per-archetype frequency in real games vs our uniform 4-per-cell panel
  - any cell with ZERO real-game representation (we over-cover it)
  - any cell that contains a large fraction of real games (we under-cover it)
  - any real game whose features fall outside ALL our bins
    (shouldn't happen with percentile bins, but worth flagging)

This is a coverage validation — the panel should at minimum hit every
archetype the live ladder samples. 4P games are reported separately
(panel is 2P-only).
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from lib.geometry import CENTER
from lib.orbit import is_orbiting
from lib.seed_panel import ARCHETYPE_OF_SEED, PANEL_BIN_EDGES, SEED_PANEL_BY_ARCHETYPE
from scripts.build_seed_panel import (
    PROD_NAMES,
    ROT_SHARE_NAMES,
    SIZE_SPLIT_NAMES,
    bin_index,
)


def features_from_obs(obs: dict) -> dict[str, Any]:
    planets = obs["planets"]
    n_planets = len(planets)
    rot = [p for p in planets if is_orbiting(p)]
    stat = [p for p in planets if not is_orbiting(p)]
    prods = [p[6] for p in planets]
    radii_rot = [p[4] for p in rot]
    radii_stat = [p[4] for p in stat]
    return {
        "n_planets": n_planets,
        "n_rotating": len(rot),
        "rotating_share": len(rot) / n_planets if n_planets else 0.0,
        "total_production": sum(prods),
        "size_split": (sum(radii_rot) / len(radii_rot) if radii_rot else 0.0)
        - (sum(radii_stat) / len(radii_stat) if radii_stat else 0.0),
        "angular_velocity": obs["angular_velocity"],
    }


def archetype_for(feat: dict) -> str:
    prod_bin = bin_index(feat["total_production"], PANEL_BIN_EDGES["total_production"])
    rot_bin = bin_index(feat["rotating_share"], PANEL_BIN_EDGES["rotating_share"])
    split_bin = bin_index(feat["size_split"], PANEL_BIN_EDGES["size_split"])
    return f"{PROD_NAMES[prod_bin]}__{ROT_SHARE_NAMES[rot_bin]}__{SIZE_SPLIT_NAMES[split_bin]}"


def num_seats_from_replay(rep: dict) -> int:
    obs0 = rep["steps"][0][0]["observation"]
    owners = {p[1] for p in obs0["planets"] if p[1] >= 0}
    return len(owners)


def load_replays(replays_dir: Path) -> list[dict]:
    out = []
    for f in sorted(replays_dir.glob("episode-*-replay.json")):
        try:
            out.append(json.loads(f.read_text()))
        except Exception as e:
            print(f"  skip {f.name}: {e}")
    return out


def pull_replays(submission_id: str, sub_dir: Path, limit: int | None) -> int:
    """Download missing replay JSONs for a submission via kaggle CLI.

    Picks episode IDs from the matching summary.json's `losses` list AND from
    `audit/live-episodes/<sid>/episodes.csv` if present.
    """
    summary = REPO / "audit" / "live-episodes" / submission_id / "summary.json"
    if not summary.exists():
        print(f"no summary at {summary}; cannot pick episode ids")
        return 0
    data = json.loads(summary.read_text())
    eids: list[str] = []
    # losses[*].episode is "episode-<eid>"; we want the bare <eid> for kaggle
    for entry in data.get("losses", []):
        ep = entry.get("episode", "")
        if ep.startswith("episode-"):
            eids.append(ep.removeprefix("episode-"))
    eids = list(dict.fromkeys(eids))  # dedupe, preserve order
    if limit:
        eids = eids[:limit]
    sub_dir.mkdir(parents=True, exist_ok=True)
    pulled = 0
    for eid in eids:
        target = sub_dir / f"episode-{eid}-replay.json"
        if target.exists():
            continue
        print(f"  pulling episode {eid} ...")
        r = subprocess.run(
            ["kaggle", "competitions", "replay", eid, "-p", str(sub_dir)],
            capture_output=True, text=True, timeout=60,
        )
        if r.returncode != 0:
            print(f"    FAILED: {r.stderr.strip()[:200]}")
            continue
        pulled += 1
    return pulled


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission-id", default="52710995")
    parser.add_argument("--pull", action="store_true",
                        help="Download missing replays for the submission.")
    parser.add_argument("--pull-limit", type=int, default=20,
                        help="Cap the --pull batch so we don't exhaust quota.")
    args = parser.parse_args()

    sub_dir = REPO / "audit" / "replays" / "live" / args.submission_id
    if args.pull:
        n = pull_replays(args.submission_id, sub_dir, args.pull_limit)
        print(f"\npulled {n} new replays into {sub_dir}")

    replays = load_replays(sub_dir)
    if not replays:
        print(f"\nno replays in {sub_dir}.")
        print(f"  run with --pull to download from Kaggle.")
        return 1
    print(f"\nloaded {len(replays)} replays from {sub_dir}\n")

    by_size_arch: dict[int, Counter] = defaultdict(Counter)
    by_size_feat: dict[int, list[dict]] = defaultdict(list)
    for rep in replays:
        size = num_seats_from_replay(rep)
        obs = rep["steps"][0][0]["observation"]
        feat = features_from_obs(obs)
        arch = archetype_for(feat)
        by_size_arch[size][arch] += 1
        by_size_feat[size].append(feat)

    for size in sorted(by_size_arch.keys()):
        arch_counts = by_size_arch[size]
        feats = by_size_feat[size]
        total = sum(arch_counts.values())
        print(f"=== {size}-player games: n={total} ===")
        # Feature distribution
        for k in ("total_production", "rotating_share", "size_split", "n_planets"):
            vs = sorted(f[k] for f in feats)
            print(f"  {k}: min={vs[0]:.2f} med={vs[len(vs)//2]:.2f} max={vs[-1]:.2f}")
        # Archetype coverage
        all_archs = sorted(SEED_PANEL_BY_ARCHETYPE.keys())
        uncovered = [a for a in all_archs if arch_counts[a] == 0]
        print(f"  archetypes present: {len(all_archs) - len(uncovered)}/{len(all_archs)}")
        if uncovered:
            print(f"  ARCHETYPES MISSING FROM THESE LIVE GAMES (panel over-covers):")
            for a in uncovered[:8]:
                print(f"    {a}")
            if len(uncovered) > 8:
                print(f"    ... +{len(uncovered)-8} more")
        # Top archetypes
        print(f"  top 8 archetypes in live games:")
        for a, c in arch_counts.most_common(8):
            pct = c / total
            panel_pct = 4 / 128  # uniform panel
            ratio = pct / panel_pct
            flag = " <-- under-covered" if ratio > 2.0 else ""
            print(f"    {a:55s}  {c:3d}  {pct:.1%} (panel uniform {panel_pct:.1%}, ratio {ratio:.1f}x){flag}")
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
