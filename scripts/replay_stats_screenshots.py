"""Per-side aggregate statistics for the three PI-flagged screenshot
games. Surfaces behavioral differences (tempo, ship sizing, exploitation
of new captures, target diversity) that can become testable gates.

Reads the per-game action JSONs produced by replay_compare_screenshots.py.

CLI: python3 scripts/replay_stats_screenshots.py
"""
from __future__ import annotations

import json
import sys
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ACTIONS_DIR = REPO / "audit" / "replays"
LIVE_DIR = REPO / "audit" / "live-episodes" / "52894340"

LABELS = ["S1-mille-step14", "S2-koshin-step12", "S3-aidan-step44"]


def stats_for_side(actions_per_step: list[dict], side: str,
                   window_end: int, replay: dict, our_pid: int,
                   opp_pid: int) -> dict:
    """Compute aggregate behavioral stats for one side (us or opp)
    in the first `window_end` steps."""
    side_pid = our_pid if side == "our" else opp_pid

    n_launches = 0
    ships_per_launch: list[int] = []
    sources_used: dict[int, int] = {}
    launches_per_step: list[int] = []
    same_source_streaks: list[int] = []
    cur_streak_src: int | None = None
    cur_streak_len = 0

    # Track per-planet capture step (when this side first owned it)
    planet_captured_at: dict[int, int] = {}
    obs0 = replay["steps"][0][0]["observation"]
    for p in obs0.get("planets", []):
        if p[1] == side_pid:
            planet_captured_at[int(p[0])] = 0

    # Track first-launch-from-new-capture delays
    first_launch_from_new_capture_delay: list[int] = []
    launched_from: set[int] = set()  # planets we've launched FROM at least once

    # Track target distribution (approximated by angle — coarse proxy)
    # Actual target inference requires aim_orbiting; skipping for first cut.

    for step_idx in range(window_end):
        actions = actions_per_step[step_idx][side]
        launches_per_step.append(len(actions))
        n_launches += len(actions)

        # Update planet ownership (which planets we own now)
        if step_idx + 1 < len(replay["steps"]):
            obs_next = replay["steps"][step_idx + 1][0]["observation"]
            for p in obs_next.get("planets", []):
                if p[1] == side_pid and int(p[0]) not in planet_captured_at:
                    planet_captured_at[int(p[0])] = step_idx + 1

        for act in actions:
            if not isinstance(act, list) or len(act) != 3:
                continue
            src_id = int(act[0])
            ships = int(act[2])
            ships_per_launch.append(ships)
            sources_used[src_id] = sources_used.get(src_id, 0) + 1

            # Same-source streak tracking
            if src_id == cur_streak_src:
                cur_streak_len += 1
            else:
                if cur_streak_len > 0:
                    same_source_streaks.append(cur_streak_len)
                cur_streak_src = src_id
                cur_streak_len = 1

            # First-launch-from-new-capture delay (per source)
            if src_id not in launched_from:
                launched_from.add(src_id)
                cap_step = planet_captured_at.get(src_id)
                if cap_step is not None and cap_step > 0:
                    # cap_step > 0 means it's not an initial planet
                    delay = step_idx - cap_step
                    first_launch_from_new_capture_delay.append(max(0, delay))

    if cur_streak_len > 0:
        same_source_streaks.append(cur_streak_len)

    return {
        "n_launches": n_launches,
        "n_silent_steps": sum(1 for l in launches_per_step if l == 0),
        "n_active_steps": sum(1 for l in launches_per_step if l > 0),
        "tempo_pct": round(100 * sum(1 for l in launches_per_step if l > 0)
                           / max(1, len(launches_per_step)), 1),
        "ships_per_launch_mean": round(statistics.mean(ships_per_launch), 1)
        if ships_per_launch else 0,
        "ships_per_launch_median": int(statistics.median(ships_per_launch))
        if ships_per_launch else 0,
        "ships_per_launch_max": max(ships_per_launch) if ships_per_launch else 0,
        "distinct_sources": len(sources_used),
        "top_source_pct": round(100 * max(sources_used.values())
                                / max(1, n_launches), 1) if sources_used else 0,
        "max_same_source_streak": max(same_source_streaks) if same_source_streaks else 0,
        "delay_to_first_launch_from_new_planet_median":
        int(statistics.median(first_launch_from_new_capture_delay))
        if first_launch_from_new_capture_delay else None,
        "delay_to_first_launch_from_new_planet_max":
        max(first_launch_from_new_capture_delay)
        if first_launch_from_new_capture_delay else None,
    }


def detect_mass_bursts(actions_per_step: list[dict], side: str) -> list[dict]:
    """Find turns where one side fires ≥3 launches with same src AND
    nearly-identical angle (within 0.1 rad). Spec-min-cap stacking."""
    bursts = []
    for step_idx, step in enumerate(actions_per_step):
        actions = step[side]
        if len(actions) < 3:
            continue
        from collections import defaultdict
        by_src_angle: dict[tuple, int] = defaultdict(int)
        by_src_angle_ships: dict[tuple, list[int]] = defaultdict(list)
        for act in actions:
            if not isinstance(act, list) or len(act) != 3:
                continue
            src = int(act[0])
            angle = round(act[1] * 10) / 10  # round to 0.1 rad
            by_src_angle[(src, angle)] += 1
            by_src_angle_ships[(src, angle)].append(int(act[2]))
        for (src, angle), count in by_src_angle.items():
            if count >= 3:
                bursts.append({
                    "step": step_idx,
                    "src": src,
                    "angle": angle,
                    "count": count,
                    "ships_total": sum(by_src_angle_ships[(src, angle)]),
                })
    return bursts


def main() -> int:
    rows = []
    for label in LABELS:
        action_path = ACTIONS_DIR / f"{label}-actions.json"
        actions_data = json.load(open(action_path))
        ep_id = actions_data["episode_id"]
        replay_path = LIVE_DIR / f"episode-{ep_id}-replay.json"
        replay = json.load(open(replay_path))

        our_seat = actions_data["our_seat"]
        opp_seat = actions_data["opp_seat"]
        our_pid = replay["steps"][0][our_seat]["observation"].get(
            "player", our_seat)
        opp_pid = replay["steps"][0][opp_seat]["observation"].get(
            "player", opp_seat)
        teams = actions_data["teams"]
        n_steps = len(replay["steps"])

        for window in [30, 60, 100, n_steps]:
            window = min(window, n_steps)
            our_stats = stats_for_side(
                actions_data["actions_per_step"], "our", window,
                replay, our_pid, opp_pid)
            opp_stats = stats_for_side(
                actions_data["actions_per_step"], "opp", window,
                replay, our_pid, opp_pid)
            rows.append({
                "label": label,
                "window_end": window,
                "our_team": teams[our_seat],
                "opp_team": teams[opp_seat],
                "our": our_stats,
                "opp": opp_stats,
            })

        # Mass burst detection
        our_bursts = detect_mass_bursts(actions_data["actions_per_step"], "our")
        opp_bursts = detect_mass_bursts(actions_data["actions_per_step"], "opp")
        rows.append({
            "label": label + "-bursts",
            "our_bursts": our_bursts[:10],
            "opp_bursts": opp_bursts[:10],
            "our_burst_count": len(our_bursts),
            "opp_burst_count": len(opp_bursts),
        })

    out_path = ACTIONS_DIR / "screenshot-stats.json"
    out_path.write_text(json.dumps(rows, indent=2))
    print(f"wrote {out_path}")

    # Compact markdown summary
    md = ["# Screenshot games — per-side statistics", ""]
    for label in LABELS:
        windowed_rows = [r for r in rows
                         if r.get("label") == label]
        burst_row = next((r for r in rows
                          if r.get("label") == label + "-bursts"), None)
        if not windowed_rows:
            continue
        first = windowed_rows[0]
        md.append(f"## {label}")
        md.append(f"- Us: **{first['our_team']}**, Opp: **{first['opp_team']}**")
        md.append("")
        md.append("| window | side | launches | tempo% | mean ships | "
                  "distinct src | max streak | delay→1st_launch_median |")
        md.append("|---:|:---:|---:|---:|---:|---:|---:|---:|")
        for r in windowed_rows:
            for side, side_lbl in [("our", "OUR"), ("opp", "OPP")]:
                s = r[side]
                md.append(
                    f"| {r['window_end']} | {side_lbl} | "
                    f"{s['n_launches']} | {s['tempo_pct']}% | "
                    f"{s['ships_per_launch_mean']} | "
                    f"{s['distinct_sources']} | "
                    f"{s['max_same_source_streak']} | "
                    f"{s.get('delay_to_first_launch_from_new_planet_median')} |")
        md.append("")
        if burst_row:
            md.append(f"**Mass-burst count** (≥3 launches same src+angle): "
                      f"OURS={burst_row['our_burst_count']}, "
                      f"OPP={burst_row['opp_burst_count']}")
            if burst_row["our_bursts"]:
                md.append("")
                md.append("OUR bursts:")
                for b in burst_row["our_bursts"][:5]:
                    md.append(f"- step {b['step']}: src={b['src']} angle={b['angle']:+.2f} "
                              f"×{b['count']} = {b['ships_total']} ships")
            if burst_row["opp_bursts"]:
                md.append("")
                md.append("OPP bursts:")
                for b in burst_row["opp_bursts"][:5]:
                    md.append(f"- step {b['step']}: src={b['src']} angle={b['angle']:+.2f} "
                              f"×{b['count']} = {b['ships_total']} ships")
            md.append("")
        md.append("---")
        md.append("")

    md_path = ACTIONS_DIR / "screenshot-stats.md"
    md_path.write_text("\n".join(md))
    print(f"wrote {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
