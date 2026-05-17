"""Replay-mine — observe→bake-in loop, step 1.

Walks `audit/live-episodes/<submission_id>/episode-*-replay.json` files
that `scripts/live_episode_summary.py --pull` already fetched from
Kaggle, classifies every fleet WE launched into PI-facing buckets,
and writes a frequency-weighted error catalog.

Buckets (PI-facing):
  - `win`             — captured (flipped target to us)
  - `defense`         — reinforced our own planet
  - `waste_attack`    — bounced (target too defended, neutral or enemy)
  - `waste_trajectory`— sun-death, oob, vanished-in-space (never hit anything)
  - `inflight`        — still moving when episode ended
  - `unknown`         — hit_planet_unknown_flip / other

Output:
  audit/replays/replay-mine-<DATE>.json   machine-readable per-sub roll-up
  audit/replays/replay-mine-<DATE>.md     PI-readable summary

CLI:
    python scripts/replay_mine.py <submission_id> [<submission_id> ...]
    python scripts/replay_mine.py --recent N        last N submissions
    python scripts/replay_mine.py --pull <sub_id>   pull then mine

Reuses `scripts.episode_postmortem.attribute_fleets` for classification.
No agent re-execution; this is pure replay-walking.

Origin: 2026-05-17 audit-workflow-performance session — PI's "observe
the games, replay them, bake insights into the architecture" mandate.
Gates pivot #2 (composite_capture_value wire-up) and pivot #5 (sun-fix).
"""

from __future__ import annotations

import argparse
import collections
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.episode_postmortem import attribute_fleets  # noqa: E402
from scripts.live_episode_summary import detect_team_name  # noqa: E402


# Raw `attribute_fleets` outcome -> PI-facing bucket.
BUCKET_OF: dict[str, str] = {
    "captured": "win",
    "reinforced_self": "defense",
    "bounced_neutral": "waste_attack",
    "bounced_enemy": "waste_attack",
    "arrived_but_lost": "waste_attack",
    "sun": "waste_trajectory",
    "oob": "waste_trajectory",
    "vanished_in_space": "waste_trajectory",
    "alive_at_end": "inflight",
    "hit_planet_unknown_flip": "unknown",
    "unknown": "unknown",
}

BUCKETS = ("win", "defense", "waste_attack", "waste_trajectory",
           "inflight", "unknown")


def mine_one_submission(sub_id: str, team_name: str | None = None,
                        ) -> dict:
    """Classify every fleet from every replay for one submission.

    Returns a dict with raw counts, PI buckets, and per-outcome detail.
    """
    sub_dir = REPO / "audit" / "live-episodes" / str(sub_id)
    if not sub_dir.is_dir():
        return {"submission_id": sub_id, "error": f"{sub_dir} missing",
                "hint": "run scripts/live_episode_summary.py --pull first"}

    replays = sorted(sub_dir.glob("episode-*-replay.json"))
    if not replays:
        return {"submission_id": sub_id,
                "error": f"no episode-*-replay.json in {sub_dir}",
                "hint": "run scripts/live_episode_summary.py --pull first"}

    if team_name is None:
        team_name = detect_team_name(replays, None)

    raw_outcomes: collections.Counter = collections.Counter()
    by_bucket: collections.Counter = collections.Counter()
    ships_by_bucket: collections.Counter = collections.Counter()
    per_episode_summary: list[dict] = []
    n_episodes_ok = 0

    for path in replays:
        try:
            replay = json.load(open(path))
        except Exception as e:
            per_episode_summary.append({"eid": path.stem,
                                        "error": f"{type(e).__name__}: {e}"})
            continue
        teams = replay.get("info", {}).get("TeamNames", [])
        our_seats = [i for i, t in enumerate(teams) if t == team_name]
        if not our_seats:
            per_episode_summary.append({"eid": path.stem,
                                        "error": "team not in seats"})
            continue
        our_seat = our_seats[0]
        our_player_id = replay["steps"][0][our_seat]["observation"].get(
            "player", our_seat,
        )

        try:
            fleets = attribute_fleets(replay, our_seat, our_player_id)
        except Exception as e:
            per_episode_summary.append({"eid": path.stem,
                                        "error": f"attribute_fleets: {type(e).__name__}: {e}"})
            continue

        ep_outcomes: collections.Counter = collections.Counter()
        for f in fleets:
            outcome = f.get("outcome", "unknown")
            ships = int(f.get("ships", 0) or 0)
            bucket = BUCKET_OF.get(outcome, "unknown")
            raw_outcomes[outcome] += 1
            by_bucket[bucket] += 1
            ships_by_bucket[bucket] += ships
            ep_outcomes[bucket] += 1

        n_episodes_ok += 1
        per_episode_summary.append({
            "eid": path.stem.replace("-replay", ""),
            "n_fleets": len(fleets),
            "by_bucket": dict(ep_outcomes),
        })

    total_fleets = sum(by_bucket.values())
    total_ships = sum(ships_by_bucket.values())
    return {
        "submission_id": sub_id,
        "team_name": team_name,
        "n_episodes": n_episodes_ok,
        "n_fleets": total_fleets,
        "n_ships_launched": total_ships,
        "raw_outcomes": dict(raw_outcomes),
        "by_bucket": dict(by_bucket),
        "ships_by_bucket": dict(ships_by_bucket),
        "pct_by_bucket": {
            b: round(100 * by_bucket.get(b, 0) / total_fleets, 1)
            if total_fleets else 0.0
            for b in BUCKETS
        },
        "pct_ships_by_bucket": {
            b: round(100 * ships_by_bucket.get(b, 0) / total_ships, 1)
            if total_ships else 0.0
            for b in BUCKETS
        },
        "per_episode": per_episode_summary,
    }


def recent_submission_ids(n: int) -> list[str]:
    """Pull the N most recent COMPLETE submission IDs via Kaggle CLI."""
    try:
        out = subprocess.check_output(
            ["kaggle", "competitions", "submissions", "orbit-wars", "--csv"],
            text=True, stderr=subprocess.STDOUT,
        )
    except subprocess.CalledProcessError as e:
        print(f"ERROR: kaggle CLI failed: {e.output}", file=sys.stderr)
        return []
    ids = []
    import csv as _csv
    import io
    reader = _csv.DictReader(io.StringIO(out))
    for row in reader:
        status = (row.get("status") or "").lower()
        # Kaggle reports status like "SubmissionStatus.COMPLETE" or "complete"
        if "complete" not in status:
            continue
        ref = row.get("ref") or row.get("submission_id") or row.get("id")
        if ref:
            ids.append(str(ref))
        if len(ids) >= n:
            break
    return ids


def pull_replays(sub_id: str) -> bool:
    """Invoke live_episode_summary --pull to populate replay JSONs."""
    cmd = [sys.executable, "-m", "scripts.live_episode_summary",
           str(sub_id), "--pull"]
    print(f"--- pulling replays for {sub_id} ---")
    rc = subprocess.call(cmd, cwd=str(REPO))
    return rc == 0


def render_markdown(rollup: dict) -> str:
    """Compact PI-readable summary across all mined submissions."""
    lines = []
    lines.append(f"# replay-mine — {rollup['date']}")
    lines.append("")
    lines.append("PI buckets: `win`=captured, `defense`=reinforced own, "
                 "`waste_attack`=bounced, `waste_trajectory`=sun/oob/vanished, "
                 "`inflight`=alive at end, `unknown`=other.")
    lines.append("")
    lines.append("## per-submission roll-up")
    lines.append("")
    headers = ["sub_id", "ep", "fleets", "win%", "def%",
               "waste_atk%", "waste_traj%", "inflight%", "unknown%"]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for sub in rollup["submissions"]:
        if "error" in sub:
            lines.append(f"| {sub['submission_id']} | ERR | "
                         + " | ".join(["-"] * (len(headers) - 2))
                         + " | " + sub.get("error", "?") + " |")
            continue
        p = sub["pct_by_bucket"]
        row = [
            sub["submission_id"],
            str(sub["n_episodes"]),
            str(sub["n_fleets"]),
            f"{p.get('win', 0):.1f}",
            f"{p.get('defense', 0):.1f}",
            f"{p.get('waste_attack', 0):.1f}",
            f"{p.get('waste_trajectory', 0):.1f}",
            f"{p.get('inflight', 0):.1f}",
            f"{p.get('unknown', 0):.1f}",
        ]
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    lines.append("## cross-submission totals")
    agg = rollup["aggregate"]
    if agg.get("n_fleets"):
        lines.append("")
        lines.append(f"- fleets launched: {agg['n_fleets']} "
                     f"across {agg['n_episodes']} episodes "
                     f"in {len(rollup['submissions'])} submissions")
        lines.append(f"- ships launched: {agg['n_ships_launched']}")
        lines.append("")
        lines.append("By bucket (count and percentage):")
        for b in BUCKETS:
            n = agg["by_bucket"].get(b, 0)
            pct = agg["pct_by_bucket"].get(b, 0.0)
            sn = agg["ships_by_bucket"].get(b, 0)
            sp = agg["pct_ships_by_bucket"].get(b, 0.0)
            lines.append(f"- `{b:<17}` {n:>5} fleets ({pct:>4.1f}%) — "
                         f"{sn:>6} ships ({sp:>4.1f}%)")
        lines.append("")
        lines.append("Raw outcomes (debug):")
        for outcome, n in sorted(agg["raw_outcomes"].items(),
                                 key=lambda kv: -kv[1]):
            lines.append(f"- `{outcome}` {n}")
    return "\n".join(lines) + "\n"


def aggregate_across(submissions: list[dict]) -> dict:
    """Sum buckets across all successful submissions."""
    raw_outcomes: collections.Counter = collections.Counter()
    by_bucket: collections.Counter = collections.Counter()
    ships_by_bucket: collections.Counter = collections.Counter()
    n_episodes = 0
    n_fleets = 0
    n_ships = 0
    for sub in submissions:
        if "error" in sub:
            continue
        n_episodes += sub.get("n_episodes", 0)
        n_fleets += sub.get("n_fleets", 0)
        n_ships += sub.get("n_ships_launched", 0)
        for k, v in sub.get("raw_outcomes", {}).items():
            raw_outcomes[k] += v
        for k, v in sub.get("by_bucket", {}).items():
            by_bucket[k] += v
        for k, v in sub.get("ships_by_bucket", {}).items():
            ships_by_bucket[k] += v
    return {
        "n_episodes": n_episodes,
        "n_fleets": n_fleets,
        "n_ships_launched": n_ships,
        "raw_outcomes": dict(raw_outcomes),
        "by_bucket": dict(by_bucket),
        "ships_by_bucket": dict(ships_by_bucket),
        "pct_by_bucket": {
            b: round(100 * by_bucket.get(b, 0) / n_fleets, 1)
            if n_fleets else 0.0
            for b in BUCKETS
        },
        "pct_ships_by_bucket": {
            b: round(100 * ships_by_bucket.get(b, 0) / n_ships, 1)
            if n_ships else 0.0
            for b in BUCKETS
        },
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="replay_mine")
    ap.add_argument("submission_ids", nargs="*",
                    help="Submission IDs to mine (e.g. 52721807 52710995).")
    ap.add_argument("--recent", type=int, default=None,
                    help="Mine the N most recent COMPLETE submissions instead.")
    ap.add_argument("--pull", action="store_true",
                    help="Run live_episode_summary --pull before mining "
                         "(downloads any missing replays).")
    ap.add_argument("--team", default=None,
                    help="Our Kaggle team name (auto-detect if omitted).")
    ap.add_argument("--out-dir", default=None,
                    help="Output dir (default: audit/replays/).")
    args = ap.parse_args(argv)

    sub_ids = list(args.submission_ids)
    if args.recent:
        sub_ids = recent_submission_ids(args.recent)
        if not sub_ids:
            print("ERROR: no recent submissions found.", file=sys.stderr)
            return 1
        print(f"--- recent submissions: {sub_ids}")

    if not sub_ids:
        ap.print_help()
        return 1

    if args.pull:
        for sid in sub_ids:
            pull_replays(sid)

    submissions = [mine_one_submission(sid, args.team) for sid in sub_ids]
    aggregate = aggregate_across(submissions)

    out_dir = Path(args.out_dir) if args.out_dir else REPO / "audit" / "replays"
    out_dir.mkdir(parents=True, exist_ok=True)
    date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    rollup = {
        "date": date,
        "submission_ids": sub_ids,
        "submissions": submissions,
        "aggregate": aggregate,
    }
    json_path = out_dir / f"replay-mine-{date}.json"
    md_path = out_dir / f"replay-mine-{date}.md"
    json_path.write_text(json.dumps(rollup, indent=2) + "\n")
    md_path.write_text(render_markdown(rollup))

    print()
    print(f"=== replay-mine {date} ===")
    if not aggregate["n_fleets"]:
        print("WARNING: zero fleets classified. Most likely cause: no "
              "episode-*-replay.json files exist for the requested "
              "submissions. Re-run with --pull or invoke "
              "scripts/live_episode_summary.py --pull <sub_id> first.")
    else:
        p = aggregate["pct_by_bucket"]
        print(f"  fleets={aggregate['n_fleets']}  "
              f"ships={aggregate['n_ships_launched']}  "
              f"episodes={aggregate['n_episodes']}")
        print(f"  win={p.get('win', 0):.1f}%  "
              f"defense={p.get('defense', 0):.1f}%  "
              f"waste_attack={p.get('waste_attack', 0):.1f}%  "
              f"waste_trajectory={p.get('waste_trajectory', 0):.1f}%  "
              f"inflight={p.get('inflight', 0):.1f}%  "
              f"unknown={p.get('unknown', 0):.1f}%")
    print(f"  -> {json_path}")
    print(f"  -> {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
