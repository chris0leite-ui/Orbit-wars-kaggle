"""Measure hold_fraction on live-ladder replays.

For each focal-capture segment in each replay, compute
`hold_fraction = hold_time / (T_end - t_capture)` and the
production-weighted analogue `p̃ · hold_time`. Aggregate per game and
report medians split by won/lost, by game-size, and by sub_id.

This is the empirical gate for the reach-frontier doctrine
(`knowledge-base/concepts/reach-frontier-doctrine.md` §9).

Usage:
    python -m scripts.measure_hold_times --sub-ids 52744856 52894340 \\
        --team ChrisLeiteScha --pull --out audit/2026-05-27-hold-time
    python -m scripts.measure_hold_times --sub-ids 52532938 --dry-run

`--dry-run` skips the kaggle pull and only aggregates replays already
on disk under `audit/live-episodes/<sub_id>/`.

Planet schema (per `lib/archetype_binning.py` + `lib/geometry_features.py`):
    p = [id, owner, x, y, radius, ships, production, ...]

Reward semantics (per `scripts/live_episode_summary._first_place_for_team`):
    seat reward == max(rewards)  =>  finished 1st (a "win").
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

REPO = Path(__file__).resolve().parents[1]
LIVE_DIR = REPO / "audit" / "live-episodes"


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def sub_dir(sub_id: str) -> Path:
    return LIVE_DIR / str(sub_id)


def refresh_episodes_csv(sub_id: str) -> Path:
    out = sub_dir(sub_id)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "episodes.csv"
    proc = subprocess.run(
        ["kaggle", "competitions", "episodes", str(sub_id), "-v"],
        capture_output=True, text=True, check=False,
    )
    if proc.returncode != 0:
        print(f"WARN: episodes refresh failed for {sub_id}: "
              f"{proc.stderr.strip()[:200]}", file=sys.stderr)
        return csv_path
    csv_path.write_text(proc.stdout)
    return csv_path


def read_episodes_csv(csv_path: Path) -> list[dict]:
    import csv as _csv
    if not csv_path.is_file():
        return []
    with csv_path.open() as fh:
        return list(_csv.DictReader(fh))


def pull_replays(sub_id: str, max_pulls: int) -> int:
    """Pull up to `max_pulls` missing replays for this submission.

    Returns count of new replays downloaded.
    """
    out = sub_dir(sub_id)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "episodes.csv"
    if not csv_path.is_file():
        refresh_episodes_csv(sub_id)
    rows = read_episodes_csv(csv_path)
    pulled = 0
    for row in rows:
        if pulled >= max_pulls:
            break
        state = row.get("state") or ""
        if "COMPLETED" not in state:
            continue
        eid = row["id"]
        target = out / f"episode-{eid}-replay.json"
        if target.is_file():
            continue
        proc = subprocess.run(
            ["kaggle", "competitions", "replay", eid,
             "-p", str(out), "-q"],
            capture_output=True, text=True, check=False,
        )
        if proc.returncode != 0:
            print(f"  WARN: pull failed for {eid}: "
                  f"{proc.stderr.strip()[:120]}", file=sys.stderr)
            continue
        pulled += 1
    return pulled


def replay_files(sub_id: str) -> list[Path]:
    return sorted(sub_dir(sub_id).glob("episode-*-replay.json"))


# ---------------------------------------------------------------------------
# Per-replay hold-time extraction
# ---------------------------------------------------------------------------


def _global_obs(step_entry: list) -> dict | None:
    """Return one seat's observation for this step (any ACTIVE seat).

    Per `scripts/episode_postmortem._global_obs`: any ACTIVE seat's
    observation is the canonical post-combat state for that step.
    """
    for seat in step_entry:
        if seat.get("status") == "ACTIVE":
            return seat.get("observation")
    if step_entry:
        return step_entry[0].get("observation")
    return None


def _planet_records(obs: dict) -> dict:
    """Map planet_id -> [id, owner, x, y, radius, ships, production, ...]."""
    return {int(p[0]): p for p in obs.get("planets", [])}


def extract_capture_segments(replay: dict, focal_seats: list[int]
                             ) -> list[dict]:
    """For one replay, emit per-(planet, focal-capture) segments.

    A *focal-capture segment* is a contiguous interval `[t_capture, t_end]`
    during which one of `focal_seats` held the planet. The segment ends
    either when ownership flips away or when the game ends.

    Returns list of dicts with keys:
        planet_id, t_capture, t_end, end_reason, T_end, production,
        hold_time, hold_fraction, weighted_hold.

    end_reason ∈ {"flipped_to_opp", "game_end", "planet_vanished"}.
    Planet vanishing applies to comets that expire mid-game.
    """
    steps = replay["steps"]
    T_end = len(steps)
    focal = set(focal_seats)

    # Track per-planet: current segment owner-is-focal flag + t_capture.
    open_segments: dict[int, dict] = {}
    last_seen_planets: set[int] = set()
    segments: list[dict] = []

    # We also need to know per-planet production. It's constant per planet;
    # read at first sighting.
    production_of: dict[int, float] = {}

    for t in range(T_end):
        obs = _global_obs(steps[t])
        if obs is None:
            continue
        rec = _planet_records(obs)
        seen_now = set(rec.keys())

        # Update production cache from first sighting.
        for pid, p in rec.items():
            if pid not in production_of and len(p) >= 7:
                production_of[pid] = float(p[6])

        # Planets that vanished (comet expired): close any open segment.
        for pid in last_seen_planets - seen_now:
            if pid in open_segments:
                seg = open_segments.pop(pid)
                segments.append(_finish_segment(seg, t, "planet_vanished",
                                                T_end, production_of))

        # For each currently-visible planet, check ownership.
        for pid, p in rec.items():
            owner = int(p[1])
            owner_is_focal = (owner in focal)

            if pid in open_segments:
                if not owner_is_focal:
                    # Lost the planet at this step.
                    seg = open_segments.pop(pid)
                    segments.append(_finish_segment(seg, t, "flipped_to_opp",
                                                    T_end, production_of))
            else:
                if owner_is_focal:
                    # New focal-owned segment opens at this step.
                    open_segments[pid] = {
                        "planet_id": pid,
                        "t_capture": t,
                    }

        last_seen_planets = seen_now

    # Any segments still open at game end close at T_end.
    for pid, seg in open_segments.items():
        segments.append(_finish_segment(seg, T_end, "game_end", T_end,
                                        production_of))

    return segments


def _finish_segment(seg: dict, t_end: int, end_reason: str, T_end: int,
                    production_of: dict[int, float]) -> dict:
    """Close an open segment with derived metrics."""
    pid = seg["planet_id"]
    t_cap = seg["t_capture"]
    hold_time = t_end - t_cap
    denom = T_end - t_cap
    hold_fraction = (hold_time / denom) if denom > 0 else 0.0
    prod = production_of.get(pid, 0.0)
    return {
        "planet_id": pid,
        "t_capture": t_cap,
        "t_end": t_end,
        "end_reason": end_reason,
        "T_end": T_end,
        "production": prod,
        "hold_time": hold_time,
        "hold_fraction": hold_fraction,
        "weighted_hold": prod * hold_time,
    }


# ---------------------------------------------------------------------------
# Per-game aggregation
# ---------------------------------------------------------------------------


def game_record(replay_path: Path, team_name: str) -> dict | None:
    """Walk one replay, emit per-game aggregates + raw segment list."""
    try:
        replay = json.load(open(replay_path))
    except Exception as e:
        print(f"  WARN: read failed {replay_path.name}: {e}", file=sys.stderr)
        return None

    teams = replay.get("info", {}).get("TeamNames") or []
    rewards = replay.get("rewards") or []
    if not teams or not rewards or any(r is None for r in rewards):
        return None  # crashed / mismatched

    focal_seats = [i for i, t in enumerate(teams) if t == team_name]
    if not focal_seats:
        return None

    # Won = any focal seat held max(rewards).
    rmax = max(rewards)
    won = any(rewards[i] == rmax for i in focal_seats)

    # Also compute opp-side hold (for confirmatory metric).
    opp_seats = [i for i in range(len(teams)) if i not in focal_seats]

    focal_segments = extract_capture_segments(replay, focal_seats)
    opp_segments = extract_capture_segments(replay, opp_seats)

    # Per-game aggregates.
    fold_hf = [s["hold_fraction"] for s in focal_segments]
    fold_wh = sum(s["weighted_hold"] for s in focal_segments)
    oold_wh = sum(s["weighted_hold"] for s in opp_segments)

    return {
        "episode": replay_path.stem.replace("-replay", ""),
        "size": len(teams),
        "team_name": team_name,
        "focal_seats": focal_seats,
        "won": won,
        "T_end": len(replay.get("steps", [])),
        "n_focal_segments": len(focal_segments),
        "n_opp_segments": len(opp_segments),
        "focal_hold_fraction_median": (statistics.median(fold_hf)
                                        if fold_hf else None),
        "focal_weighted_hold_total": fold_wh,
        "opp_weighted_hold_total": oold_wh,
        "share_focal": (fold_wh / (fold_wh + oold_wh)
                        if (fold_wh + oold_wh) > 0 else 0.0),
        "segments": focal_segments,  # raw rows for JSON dump
    }


# ---------------------------------------------------------------------------
# Cross-game aggregation + gate evaluation
# ---------------------------------------------------------------------------


def _median_safe(xs: list[float]) -> float | None:
    return statistics.median(xs) if xs else None


def aggregate_games(games: list[dict]) -> dict:
    """Cross-game medians split by won/lost, size, sub_id."""
    out: dict = {
        "n_games": len(games),
        "n_wins": sum(1 for g in games if g["won"]),
        "n_losses": sum(1 for g in games if not g["won"]),
    }

    # Per-capture (not per-game) hold_fraction split.
    wins_hf: list[float] = []
    loss_hf: list[float] = []
    wins_share: list[float] = []
    loss_share: list[float] = []
    by_size: dict = defaultdict(lambda: {"wins_hf": [], "loss_hf": []})
    by_sub: dict = defaultdict(lambda: {"wins_hf": [], "loss_hf": []})

    for g in games:
        sub = g.get("sub_id", "?")
        size = g["size"]
        bucket_size = by_size[size]
        bucket_sub = by_sub[sub]
        for s in g["segments"]:
            if g["won"]:
                wins_hf.append(s["hold_fraction"])
                bucket_size["wins_hf"].append(s["hold_fraction"])
                bucket_sub["wins_hf"].append(s["hold_fraction"])
            else:
                loss_hf.append(s["hold_fraction"])
                bucket_size["loss_hf"].append(s["hold_fraction"])
                bucket_sub["loss_hf"].append(s["hold_fraction"])
        (wins_share if g["won"] else loss_share).append(g["share_focal"])

    out["wins_median_hold_fraction"] = _median_safe(wins_hf)
    out["loss_median_hold_fraction"] = _median_safe(loss_hf)
    out["n_captures_win_games"] = len(wins_hf)
    out["n_captures_loss_games"] = len(loss_hf)
    out["wins_median_share_focal"] = _median_safe(wins_share)
    out["loss_median_share_focal"] = _median_safe(loss_share)
    out["share_separation"] = (
        (out["wins_median_share_focal"] or 0.0)
        - (out["loss_median_share_focal"] or 0.0)
    )

    out["by_size"] = {
        str(sz): {
            "n_wins_captures": len(b["wins_hf"]),
            "n_loss_captures": len(b["loss_hf"]),
            "wins_median_hold_fraction": _median_safe(b["wins_hf"]),
            "loss_median_hold_fraction": _median_safe(b["loss_hf"]),
        }
        for sz, b in sorted(by_size.items())
    }
    out["by_sub"] = {
        str(sub): {
            "n_wins_captures": len(b["wins_hf"]),
            "n_loss_captures": len(b["loss_hf"]),
            "wins_median_hold_fraction": _median_safe(b["wins_hf"]),
            "loss_median_hold_fraction": _median_safe(b["loss_hf"]),
        }
        for sub, b in sorted(by_sub.items())
    }

    out["gate_verdict"] = evaluate_gate(
        out["wins_median_hold_fraction"],
        out["loss_median_hold_fraction"],
        out["share_separation"],
    )
    return out


def evaluate_gate(wins_med: float | None, loss_med: float | None,
                  share_sep: float) -> dict:
    """Apply the pre-registered §9 gates."""
    if wins_med is None or loss_med is None:
        return {"verdict": "no_data", "rationale": "insufficient captures."}

    # Falsified: wins < 0.60 OR losses >= wins.
    if wins_med < 0.60 or loss_med >= wins_med:
        return {
            "verdict": "falsified",
            "rationale": (
                f"wins_median={wins_med:.3f}, loss_median={loss_med:.3f}: "
                "fails either the 0.60-floor or ordering-correctness check."
            ),
            "share_separation": share_sep,
        }

    # Strong: wins >= 0.70 AND losses <= 0.45.
    if wins_med >= 0.70 and loss_med <= 0.45:
        return {
            "verdict": "strong",
            "rationale": (
                f"wins_median={wins_med:.3f} >= 0.70 AND "
                f"loss_median={loss_med:.3f} <= 0.45. "
                "Doctrine confirmed; proceed to chooser build."
            ),
            "share_separation": share_sep,
        }

    # Weak-positive: wins >= 0.60 AND losses <= 0.50.
    if wins_med >= 0.60 and loss_med <= 0.50:
        return {
            "verdict": "weak_positive",
            "rationale": (
                f"wins_median={wins_med:.3f} >= 0.60 AND "
                f"loss_median={loss_med:.3f} <= 0.50. "
                "Directionally correct; complementary signal needed."
            ),
            "share_separation": share_sep,
        }

    # Anything else — conservative tie-break to falsified per Part C.
    return {
        "verdict": "falsified",
        "rationale": (
            f"wins_median={wins_med:.3f}, loss_median={loss_med:.3f}: "
            "between Weak-positive and Falsified; tie-break to Falsified "
            "per pre-registered no-goalpost-shift clause."
        ),
        "share_separation": share_sep,
    }


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------


def render_report(agg: dict, sub_ids: list[str], team: str) -> str:
    lines = []
    lines.append("# Empirical verification — reach-frontier doctrine")
    lines.append("")
    lines.append(f"Date: 2026-05-27. Team: `{team}`. Sub IDs: "
                 f"{', '.join(sub_ids)}.")
    lines.append("")
    lines.append(f"Games analysed: {agg['n_games']} "
                 f"({agg['n_wins']} wins, {agg['n_losses']} losses).")
    lines.append("")
    lines.append("## Primary metric — hold_fraction medians (per capture)")
    lines.append("")
    lines.append("| Outcome | Captures | Median hold_fraction |")
    lines.append("|---|---:|---:|")
    wm = agg["wins_median_hold_fraction"]
    lm = agg["loss_median_hold_fraction"]
    lines.append(f"| Wins   | {agg['n_captures_win_games']} | "
                 f"{wm if wm is None else f'{wm:.3f}'} |")
    lines.append(f"| Losses | {agg['n_captures_loss_games']} | "
                 f"{lm if lm is None else f'{lm:.3f}'} |")
    lines.append("")
    lines.append("## Confirmatory — production-share medians (per game)")
    lines.append("")
    lines.append("| Outcome | Games | Median share_focal | "
                 "Σp̃·τ_me / (Σp̃·τ_me + Σp̃·τ_opp) |")
    lines.append("|---|---:|---:|---:|")
    wsh = agg["wins_median_share_focal"]
    lsh = agg["loss_median_share_focal"]
    lines.append(f"| Wins   | {agg['n_wins']} | "
                 f"{wsh if wsh is None else f'{wsh:.3f}'} | — |")
    lines.append(f"| Losses | {agg['n_losses']} | "
                 f"{lsh if lsh is None else f'{lsh:.3f}'} | — |")
    lines.append(f"| Separation | — | {agg['share_separation']:.3f} | "
                 f"(gate: > 0.15 = confirmatory) |")
    lines.append("")
    lines.append("## By game size")
    lines.append("")
    lines.append("| Size | Wins captures | Wins-median | "
                 "Loss captures | Loss-median |")
    lines.append("|---|---:|---:|---:|---:|")
    for sz, b in agg["by_size"].items():
        wm = b["wins_median_hold_fraction"]
        lm = b["loss_median_hold_fraction"]
        lines.append(
            f"| {sz} | {b['n_wins_captures']} | "
            f"{wm if wm is None else f'{wm:.3f}'} | "
            f"{b['n_loss_captures']} | "
            f"{lm if lm is None else f'{lm:.3f}'} |"
        )
    lines.append("")
    lines.append("## By submission id")
    lines.append("")
    lines.append("| Sub | Wins captures | Wins-median | "
                 "Loss captures | Loss-median |")
    lines.append("|---|---:|---:|---:|---:|")
    for sub, b in agg["by_sub"].items():
        wm = b["wins_median_hold_fraction"]
        lm = b["loss_median_hold_fraction"]
        lines.append(
            f"| {sub} | {b['n_wins_captures']} | "
            f"{wm if wm is None else f'{wm:.3f}'} | "
            f"{b['n_loss_captures']} | "
            f"{lm if lm is None else f'{lm:.3f}'} |"
        )
    lines.append("")
    lines.append("## Gate verdict")
    lines.append("")
    lines.append(f"**{agg['gate_verdict']['verdict'].upper()}** — "
                 f"{agg['gate_verdict']['rationale']}")
    lines.append("")
    lines.append("Pre-registered thresholds (§9 of "
                 "`knowledge-base/concepts/reach-frontier-doctrine.md`):")
    lines.append("")
    lines.append("- Strong: wins ≥ 0.70 AND losses ≤ 0.45")
    lines.append("- Weak-positive: wins ≥ 0.60 AND losses ≤ 0.50")
    lines.append("- Falsified: wins < 0.60 OR losses ≥ wins")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Measure focal hold_fraction on Kaggle live replays.")
    ap.add_argument("--sub-ids", nargs="+", required=True,
                    help="submission ids to aggregate")
    ap.add_argument("--team", default="ChrisLeiteScha",
                    help="focal team name as it appears in TeamNames")
    ap.add_argument("--pull", action="store_true",
                    help="invoke `kaggle competitions replay` for missing "
                         "episodes (otherwise dry-run mode)")
    ap.add_argument("--dry-run", action="store_true",
                    help="alias for `not --pull`; explicit no-pull mode")
    ap.add_argument("--max-pulls-per-sub", type=int, default=30,
                    help="cap on episodes to pull per sub_id "
                         "(rate-limit safety)")
    ap.add_argument("--out", default="audit/2026-05-27-hold-time-empirical",
                    help="output basename; .md and .json suffixes added")
    args = ap.parse_args(argv)

    if args.pull and args.dry_run:
        print("ERROR: cannot specify both --pull and --dry-run.",
              file=sys.stderr)
        return 2

    games: list[dict] = []
    for sub in args.sub_ids:
        if args.pull:
            n_pulled = pull_replays(sub, args.max_pulls_per_sub)
            print(f"[{sub}] pulled {n_pulled} new replays")
        files = replay_files(sub)
        print(f"[{sub}] {len(files)} replays on disk")
        for f in files:
            rec = game_record(f, args.team)
            if rec is None:
                continue
            rec["sub_id"] = sub
            games.append(rec)

    if not games:
        print("ERROR: no valid games found.", file=sys.stderr)
        return 3

    agg = aggregate_games(games)
    report = render_report(agg, args.sub_ids, args.team)

    md_path = REPO / f"{args.out}.md"
    json_path = REPO / f"{args.out}.json"
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(report)
    json_path.write_text(json.dumps({
        "sub_ids": args.sub_ids,
        "team": args.team,
        "aggregate": agg,
        "games": games,
    }, indent=2, default=str))

    print()
    print(report)
    print(f"Wrote {md_path}")
    print(f"Wrote {json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
