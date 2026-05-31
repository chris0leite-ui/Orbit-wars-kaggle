"""Download top-rated 2P episode replays from a Kaggle daily-episodes dataset.

The daily dataset `kaggle/orbit-wars-episodes-YYYY-MM-DD` is itself a pre-filtered
"top average rated, up to 20 GB" subset (see Kaggle discussion topic 701894).
So we just need to:

  1. Enumerate all episode JSONs in the dataset.
  2. Sort by file size ascending (smaller files correlate with 2P games —
     2P has half the state-per-step of 4P).
  3. Download in order; after each download, peek the JSON and keep iff 2P.
  4. Stop when we have `--target` 2P episodes.

The script is idempotent: already-downloaded files are skipped, and the
dataset listing is cached to a JSON so reruns don't re-paginate.

Output: `/tmp/ow_replays/<episode_id>.json` for each kept episode +
`/tmp/ow_replays/_manifest.json` with the kept set + sizes + seed teams.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def _run(cmd: list[str], timeout: int = 300) -> str:
    """Run a CLI command, return stdout (str). Raises on non-zero exit."""
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if res.returncode != 0:
        raise RuntimeError(
            f"command failed (exit={res.returncode}): {' '.join(cmd)}\n"
            f"stderr: {res.stderr[-2000:]}"
        )
    return res.stdout


def list_dataset_files(slug: str, page_size: int = 200) -> list[tuple[str, int]]:
    """Paginate through every file in a Kaggle dataset.

    Returns list of (filename, size_bytes). Filters to *.json only (the daily
    dataset only contains episode JSONs, but the filter protects against
    future schema changes that drop a manifest in there).
    """
    out: list[tuple[str, int]] = []
    token: str | None = None
    page = 0
    while True:
        page += 1
        cmd = ["kaggle", "datasets", "files", slug, "-v",
               "--page-size", str(page_size)]
        if token:
            cmd += ["--page-token", token]
        raw = _run(cmd, timeout=120)
        # Output format with -v (CSV):
        #   "Next Page Token = <token>\n" (optional)
        #   "name,size,creationDate\n"
        #   "<file>,<bytes>,<iso>\n" × N
        next_token: str | None = None
        body_lines: list[str] = []
        for line in raw.splitlines():
            if line.startswith("Next Page Token = "):
                next_token = line.split("=", 1)[1].strip()
            else:
                body_lines.append(line)
        rdr = csv.DictReader(io.StringIO("\n".join(body_lines)))
        page_count = 0
        for row in rdr:
            name = row.get("name", "")
            if not name.endswith(".json"):
                continue
            try:
                size = int(row.get("size", "0"))
            except ValueError:
                continue
            out.append((name, size))
            page_count += 1
        print(f"  page {page}: +{page_count} files (total {len(out)}; "
              f"next={'yes' if next_token else 'no'})",
              file=sys.stderr, flush=True)
        if not next_token or page_count == 0:
            break
        token = next_token
        # Be a polite client to the listing endpoint.
        time.sleep(0.1)
    return out


def download_one(slug: str, name: str, dest_dir: Path,
                 timeout: int = 120) -> Path:
    """Download a single file from a dataset to dest_dir. Returns the path."""
    cmd = ["kaggle", "datasets", "download", slug, "-f", name,
           "-p", str(dest_dir), "--quiet"]
    _run(cmd, timeout=timeout)
    # Kaggle CLI writes either <name> or <name>.zip; episode JSONs come
    # through unzipped at the file level, but be defensive.
    direct = dest_dir / name
    if direct.exists():
        return direct
    zipped = dest_dir / (name + ".zip")
    if zipped.exists():
        # Extract single file inline.
        import zipfile
        with zipfile.ZipFile(zipped) as zf:
            zf.extractall(dest_dir)
        zipped.unlink()
        if direct.exists():
            return direct
    raise FileNotFoundError(f"expected {direct} after download, not found")


def peek_episode(path: Path) -> dict:
    """Open an episode JSON and extract minimal metadata for filtering.

    Returns {'num_seats': int, 'team_names': list[str], 'steps': int,
             'rewards': list[int]}.
    """
    with path.open() as f:
        ep = json.load(f)
    steps = ep.get("steps") or []
    if not steps:
        return {"num_seats": 0, "team_names": [], "steps": 0, "rewards": []}
    info = ep.get("info") or {}
    team_names = []
    for a in info.get("Agents") or []:
        n = a.get("Name") if isinstance(a, dict) else None
        team_names.append(n or "?")
    return {
        "num_seats": len(steps[0]),
        "team_names": team_names,
        "steps": len(steps),
        "rewards": ep.get("rewards") or [],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--date", default="2026-05-30",
        help="UTC date for the daily dataset (default: 2026-05-30)",
    )
    ap.add_argument(
        "--target", type=int, default=940,
        help="Number of 2P episodes to keep (default: 940 ≈ top-20%% of one day)",
    )
    ap.add_argument(
        "--out", default="/tmp/ow_replays",
        help="Destination directory for kept episodes (default: /tmp/ow_replays)",
    )
    ap.add_argument(
        "--max-mb-per-file", type=float, default=6.0,
        help="Skip listing entries above this size (MB); 2P games are typically "
             "1-5 MB, 4P games 5-18 MB (default: 6.0)",
    )
    ap.add_argument(
        "--max-tries", type=int, default=2000,
        help="Hard cap on download attempts (safety net)",
    )
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = f"kaggle/orbit-wars-episodes-{args.date}"

    listing_cache = out_dir / f"_listing_{args.date}.json"
    if listing_cache.exists():
        print(f"reusing cached listing: {listing_cache}", file=sys.stderr)
        listing = json.loads(listing_cache.read_text())
    else:
        print(f"enumerating {slug} ...", file=sys.stderr)
        listing = list_dataset_files(slug)
        listing_cache.write_text(json.dumps(listing))
        print(f"  wrote {len(listing)} entries to {listing_cache}",
              file=sys.stderr)

    # Sort by size ascending (smaller → more likely 2P).
    listing.sort(key=lambda t: t[1])
    size_cap = int(args.max_mb_per_file * 1024 * 1024)
    listing = [(n, s) for (n, s) in listing if s <= size_cap]
    print(f"considering {len(listing)} files ≤ {args.max_mb_per_file:.1f} MB",
          file=sys.stderr)

    manifest_path = out_dir / "_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    else:
        manifest = {"date": args.date, "kept": [], "skipped_4p": 0,
                    "failed_download": 0}

    kept_set = {row["name"] for row in manifest["kept"]}
    tries = 0
    for name, size in listing:
        if len(manifest["kept"]) >= args.target:
            break
        if tries >= args.max_tries:
            print("hit --max-tries cap; stopping", file=sys.stderr)
            break
        tries += 1
        dest = out_dir / name
        if name in kept_set and dest.exists():
            continue
        if dest.exists():
            # Already downloaded but not in manifest (resume case) — peek.
            pass
        else:
            try:
                download_one(slug, name, out_dir, timeout=120)
            except Exception as exc:
                print(f"  download failed for {name}: {exc}",
                      file=sys.stderr)
                manifest["failed_download"] += 1
                continue
        try:
            meta = peek_episode(dest)
        except Exception as exc:
            print(f"  peek failed for {name}: {exc}", file=sys.stderr)
            dest.unlink(missing_ok=True)
            continue
        if meta["num_seats"] != 2:
            manifest["skipped_4p"] += 1
            dest.unlink(missing_ok=True)
            continue
        manifest["kept"].append({
            "name": name,
            "size": size,
            "steps": meta["steps"],
            "team_names": meta["team_names"],
            "rewards": meta["rewards"],
        })
        kept_set.add(name)
        if len(manifest["kept"]) % 25 == 0:
            manifest_path.write_text(json.dumps(manifest, indent=1))
            print(
                f"  kept {len(manifest['kept'])}/{args.target} 2P "
                f"(skipped {manifest['skipped_4p']} 4P, "
                f"{manifest['failed_download']} failed dl) — last: {name}",
                file=sys.stderr, flush=True,
            )

    manifest_path.write_text(json.dumps(manifest, indent=1))
    total_bytes = sum(row["size"] for row in manifest["kept"])
    print(
        f"DONE: kept {len(manifest['kept'])} 2P episodes, "
        f"{total_bytes / 1024**2:.1f} MB; "
        f"skipped {manifest['skipped_4p']} 4P; "
        f"{manifest['failed_download']} dl-failures",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
