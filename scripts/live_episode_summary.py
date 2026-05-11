"""Aggregate the live-ladder episodes pulled via the Kaggle CLI into a
per-submission summary (winrate, 2P vs 4P split, opponent map, episode-length
stats, seat distribution).

Workflow:

1. `kaggle competitions episodes <submission_id>` populates
   `audit/live-episodes/<submission_id>/episodes.csv`.
2. (Optional) `--pull` downloads any missing replay JSONs via
   `kaggle competitions replay <episode_id> -p <dir>`.
3. Aggregate every `episode-*-replay.json` on disk for that submission;
   write `audit/live-episodes/<submission_id>/summary.json` and print a
   table.

CLI:
    python -m scripts.live_episode_summary 52532938 [--pull] [--team NAME]

`--team`'s auto-detect picks the `TeamNames` entry that appears in ≥80% of
non-self-only episodes. Falls back to `$KAGGLE_USERNAME` (case-normalised
against any team name observed) when no clear majority exists.

The replay JSON schema (live-ladder games) is documented in
`audit/2026-05-11-block-bootstrap-live-episodes.md`; the relevant slices are
`info.TeamNames` (list of player names, length 2 or 4) and `rewards`
(parallel list of -1/+1 outcomes).
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

REPO = Path(__file__).resolve().parents[1]
LIVE_DIR = REPO / "audit" / "live-episodes"


# ---------------------------------------------------------------------------
# Disk I/O
# ---------------------------------------------------------------------------


def submission_dir(submission_id: str) -> Path:
    return LIVE_DIR / str(submission_id)


def refresh_episodes_csv(submission_id: str, sub_dir: Path) -> Path:
    """Run `kaggle competitions episodes <id> -v` and persist to csv."""
    sub_dir.mkdir(parents=True, exist_ok=True)
    csv_path = sub_dir / "episodes.csv"
    proc = subprocess.run(
        ["kaggle", "competitions", "episodes", str(submission_id), "-v"],
        capture_output=True, text=True, check=False,
    )
    if proc.returncode != 0:
        print(
            f"WARN: `kaggle competitions episodes {submission_id}` exited "
            f"{proc.returncode}: {proc.stderr.strip()[:200]}",
            file=sys.stderr,
        )
        return csv_path
    csv_path.write_text(proc.stdout)
    return csv_path


def read_episodes_csv(csv_path: Path) -> list[dict]:
    if not csv_path.is_file():
        return []
    with csv_path.open() as fh:
        return list(csv.DictReader(fh))


def pull_missing_replays(submission_id: str, sub_dir: Path,
                         episode_rows: Iterable[dict]) -> int:
    """Download any episode-<id>-replay.json missing from sub_dir.

    Returns count of new replays pulled. Skips non-COMPLETED episodes.
    """
    pulled = 0
    for row in episode_rows:
        if "COMPLETED" not in row.get("state", ""):
            continue
        eid = row["id"]
        out = sub_dir / f"episode-{eid}-replay.json"
        if out.is_file():
            continue
        proc = subprocess.run(
            ["kaggle", "competitions", "replay", eid,
             "-p", str(sub_dir), "-q"],
            capture_output=True, text=True, check=False,
        )
        if proc.returncode != 0:
            print(f"  WARN: replay pull failed for {eid}: "
                  f"{proc.stderr.strip()[:120]}", file=sys.stderr)
            continue
        pulled += 1
    return pulled


def replay_files(sub_dir: Path) -> list[Path]:
    return sorted(sub_dir.glob("episode-*-replay.json"))


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def detect_team_name(replays: list[Path], hint: str | None = None) -> str:
    """Pick the TeamNames entry that appears in ≥80% of episodes.

    `hint` (typically $KAGGLE_USERNAME) is used as a tie-break / fallback,
    matched case-insensitively against observed names.
    """
    counts: collections.Counter[str] = collections.Counter()
    for f in replays:
        teams = json.load(open(f))["info"]["TeamNames"]
        for name in set(teams):
            counts[name] += 1
    if not counts:
        raise RuntimeError(f"no replays in {replays}")
    total = len(replays)
    for name, n in counts.most_common():
        if n / total >= 0.8:
            return name
    if hint:
        hint_low = hint.lower()
        for name in counts:
            if name.lower() == hint_low:
                return name
    raise RuntimeError(
        f"could not auto-detect team name; pass --team explicitly. "
        f"Observed: {dict(counts.most_common(5))}"
    )


def _first_place_for_team(rewards: list[float], teams: list[str],
                          team_name: str) -> bool:
    """True iff ANY seat held by `team_name` finished first.

    The live env's reward semantics: a seat scores +1 iff its final ship
    count tied the max ship count; otherwise -1. In 2P that's a clean
    win/loss; in 4P FFA the same logic gives co-winners on a tie. We treat
    "any of our seats took first" as a win for the submission.
    """
    return any(
        rewards[i] == max(rewards)
        for i, t in enumerate(teams)
        if t == team_name and rewards[i] is not None
    )


def aggregate(replays: list[Path], team_name: str) -> dict:
    """Per-submission summary from already-on-disk replays."""
    n_total = len(replays)
    by_size = collections.Counter()
    wins_by_size = collections.Counter()
    opp_seen: collections.Counter[str] = collections.Counter()
    opp_wins: collections.Counter[str] = collections.Counter()  # times opp took 1st
    seat_dist: collections.Counter[int] = collections.Counter()
    self_match_count = 0
    ep_steps: list[int] = []
    losses: list[dict] = []  # episode_id, opponents, winner, steps

    for f in replays:
        d = json.load(open(f))
        teams = d["info"]["TeamNames"]
        rewards = d["rewards"]
        if any(r is None for r in rewards):
            # crashed / timed-out games — skip but log
            continue
        size = len(teams)
        by_size[size] += 1
        our_seats = [i for i, t in enumerate(teams) if t == team_name]
        if not our_seats:
            continue
        if len(our_seats) > 1:
            self_match_count += 1
        for s in our_seats:
            seat_dist[s] += 1
        we_won = _first_place_for_team(rewards, teams, team_name)
        if we_won:
            wins_by_size[size] += 1
        # opponent stats — exclude every seat of ours
        for j, name in enumerate(teams):
            if name == team_name:
                continue
            opp_seen[name] += 1
            if rewards[j] == max(rewards):
                opp_wins[name] += 1
        ep_steps.append(len(d.get("steps", [])))
        if not we_won:
            winner_idx = rewards.index(max(rewards))
            losses.append({
                "episode": f.stem.replace("-replay", ""),
                "size": size,
                "opponents": [t for t in teams if t != team_name],
                "winner": teams[winner_idx],
                "steps": len(d.get("steps", [])),
            })

    def _q(xs, pct):
        if not xs: return None
        xs = sorted(xs); return xs[min(len(xs)-1, int(pct * (len(xs)-1) + 0.5))]

    n_played = sum(by_size.values())
    summary = {
        "submission_id": replays[0].parent.name if replays else None,
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "team_name": team_name,
        "n_episodes": n_total,
        "n_played": n_played,
        "wins_total": sum(wins_by_size.values()),
        "winrate_total": (sum(wins_by_size.values()) / n_played) if n_played else 0.0,
        "by_size": {
            str(sz): {
                "n": by_size[sz],
                "wins": wins_by_size[sz],
                "winrate": (wins_by_size[sz] / by_size[sz]) if by_size[sz] else 0.0,
            }
            for sz in sorted(by_size)
        },
        "self_match_episodes": self_match_count,
        "seat_distribution": dict(sorted(seat_dist.items())),
        "episode_length_steps": {
            "n": len(ep_steps),
            "min": min(ep_steps) if ep_steps else None,
            "q1": _q(ep_steps, 0.25),
            "median": _q(ep_steps, 0.5),
            "q3": _q(ep_steps, 0.75),
            "max": max(ep_steps) if ep_steps else None,
            "mean": (statistics.mean(ep_steps) if ep_steps else None),
        },
        "opponents": [
            {"name": name, "seen": n, "beat_us": opp_wins[name]}
            for name, n in opp_seen.most_common()
        ],
        "losses": losses,
    }
    return summary


# ---------------------------------------------------------------------------
# Console rendering
# ---------------------------------------------------------------------------


def render_table(summary: dict) -> str:
    lines = [
        f"=== submission {summary['submission_id']} as `{summary['team_name']}` ===",
        f"played: {summary['n_played']} / {summary['n_episodes']}  "
        f"wins: {summary['wins_total']}  "
        f"winrate: {summary['winrate_total']:.1%}",
        "",
        "by player-count:",
    ]
    for sz, row in summary["by_size"].items():
        lines.append(
            f"  {sz}-player  n={row['n']:3d}  wins={row['wins']:3d}  "
            f"winrate={row['winrate']:.1%}"
        )
    el = summary["episode_length_steps"]
    if el["n"]:
        lines.append("")
        lines.append(
            f"episode steps: min={el['min']} q1={el['q1']} median={el['median']} "
            f"q3={el['q3']} max={el['max']} mean={el['mean']:.0f}"
        )
    if summary["self_match_episodes"]:
        lines.append("")
        lines.append(
            f"self-match episodes (multiple of our submissions): "
            f"{summary['self_match_episodes']}"
        )
    seats = summary["seat_distribution"]
    if seats:
        lines.append("")
        lines.append(f"seat distribution: " + "  ".join(
            f"P{i}={seats[i]}" for i in sorted(seats)))
    losers = [o for o in summary["opponents"] if o["beat_us"]]
    if losers:
        lines.append("")
        lines.append("opponents who beat us (1st-place wins vs us):")
        for o in losers:
            lines.append(f"  {o['name']:30s}  seen={o['seen']:2d}  beat_us={o['beat_us']}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Aggregate live-ladder episodes into a per-submission summary."
    )
    parser.add_argument("submission_id", help="Kaggle submission ID (e.g. 52532938).")
    parser.add_argument("--pull", action="store_true",
                        help="Download any missing replay JSONs before aggregating.")
    parser.add_argument("--no-refresh", action="store_true",
                        help="Skip `kaggle competitions episodes` refresh.")
    parser.add_argument("--team", default=None,
                        help="Our team name on Kaggle (auto-detect by default).")
    args = parser.parse_args(argv)

    sub_dir = submission_dir(args.submission_id)
    sub_dir.mkdir(parents=True, exist_ok=True)

    if not args.no_refresh:
        refresh_episodes_csv(args.submission_id, sub_dir)
    rows = read_episodes_csv(sub_dir / "episodes.csv")
    if args.pull and rows:
        pulled = pull_missing_replays(args.submission_id, sub_dir, rows)
        print(f"pulled {pulled} new replays into {sub_dir}", file=sys.stderr)

    replays = replay_files(sub_dir)
    if not replays:
        print(f"ERROR: no replay JSONs in {sub_dir}. Run with --pull.",
              file=sys.stderr)
        return 1

    team = args.team or detect_team_name(replays, os.environ.get("KAGGLE_USERNAME"))
    summary = aggregate(replays, team)

    out_path = sub_dir / "summary.json"
    out_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(render_table(summary))
    print(f"\nJSON: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
