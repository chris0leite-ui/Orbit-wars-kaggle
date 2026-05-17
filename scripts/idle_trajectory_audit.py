"""Fleet idling + trajectory length analysis.

Walks `audit/live-episodes/<submission_id>/episode-*-replay.json` and
computes:
  - Idle-ship-turn density: ship-turns spent on planets > IDLE_RADIUS
    from any non-our planet, broken down by distance bucket.
  - Launch ETA distribution: how many of OUR launches fall into
    short/medium/long/very_long ETA buckets.
  - Staging-opportunity rate: % of long launches that had an own-or-
    neutral planet in the ±15deg corridor at <60% target distance.

Output: audit/replays/idle-trajectory-<DATE>.md

CLI:
    python scripts/idle_trajectory_audit.py <submission_id> [<submission_id> ...]
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.live_episode_summary import detect_team_name  # noqa: E402

# Buckets for min-distance to nearest non-our planet
DIST_BUCKETS = [
    ("frontier", 0, 20),     # adjacent / near contested
    ("mid", 20, 35),         # mid-range
    ("rear", 35, 50),        # back of own territory
    ("isolated", 50, 999),   # very far from action
]

# Buckets for launch ETA
ETA_BUCKETS = [
    ("short", 0, 10),
    ("medium", 11, 20),
    ("long", 21, 30),
    ("very_long", 31, 999),
]

LONG_ETA_THRESHOLD = 20  # for staging-opportunity check
STAGING_ANGLE_TOL_DEG = 15.0
STAGING_DIST_FRAC = 0.6  # candidate must be < 60% of target distance


def planet_dict(planet_row) -> dict:
    return {
        "id": int(planet_row[0]),
        "owner": int(planet_row[1]),
        "x": float(planet_row[2]),
        "y": float(planet_row[3]),
        "ships": int(planet_row[5]),
        "production": int(planet_row[6]),
    }


def dist(a, b) -> float:
    return math.hypot(a["x"] - b["x"], a["y"] - b["y"])


def bucket_of(value: float, buckets) -> str:
    for name, lo, hi in buckets:
        if lo <= value < hi:
            return name
    return "unknown"


def analyze_idle_ship_turns(replay: dict, my_seat: int) -> dict:
    """For each step + each of my planets, attribute ships to a
    distance-bucket based on min-distance to nearest non-our planet."""
    steps = replay["steps"]
    ship_turns_by_bucket: dict[str, int] = collections.Counter()
    total_ship_turns = 0
    # Track per-planet idle ship-turns (for top-offender breakdown)
    per_planet_idle: dict[int, int] = collections.Counter()
    n_steps_analyzed = 0

    for step_data in steps:
        if my_seat >= len(step_data):
            continue
        obs = step_data[my_seat].get("observation") or {}
        planets_raw = obs.get("planets") or []
        if not planets_raw:
            continue
        my_id = int(obs.get("player", my_seat))
        planets = [planet_dict(p) for p in planets_raw]
        my_planets = [p for p in planets if p["owner"] == my_id]
        non_our = [p for p in planets if p["owner"] != my_id]
        if not non_our:
            # All planets ours — degenerate
            continue
        n_steps_analyzed += 1
        for p in my_planets:
            d_min = min(dist(p, q) for q in non_our)
            bucket = bucket_of(d_min, DIST_BUCKETS)
            ship_turns_by_bucket[bucket] += p["ships"]
            total_ship_turns += p["ships"]
            if bucket in ("rear", "isolated"):
                per_planet_idle[p["id"]] += p["ships"]

    pct_by_bucket = {
        name: round(100 * ship_turns_by_bucket.get(name, 0) / total_ship_turns, 1)
        if total_ship_turns else 0.0
        for name, _lo, _hi in DIST_BUCKETS
    }
    # Top-5 idle offenders
    top_idle = sorted(per_planet_idle.items(), key=lambda kv: -kv[1])[:5]

    return {
        "n_steps_analyzed": n_steps_analyzed,
        "total_ship_turns": total_ship_turns,
        "ship_turns_by_bucket": dict(ship_turns_by_bucket),
        "pct_by_bucket": pct_by_bucket,
        "top_idle_planets": [
            {"planet_id": pid, "idle_ship_turns": st}
            for pid, st in top_idle
        ],
    }


def analyze_launches(replay: dict, my_seat: int) -> dict:
    """For each launch in this replay, extract:
       - eta (ceil(distance / fleet_speed))
       - whether a staging alternative existed (own/neutral planet in
         narrow corridor at <60% target distance)
    """
    from lib.fleet import speed as fleet_speed

    steps = replay["steps"]
    eta_by_bucket: dict[str, int] = collections.Counter()
    ships_by_bucket: dict[str, int] = collections.Counter()
    total_launches = 0
    total_ships = 0
    long_launches_with_staging: int = 0
    long_launches_total: int = 0

    for step_data in steps:
        if my_seat >= len(step_data):
            continue
        seat_data = step_data[my_seat]
        action = seat_data.get("action") or []
        obs = seat_data.get("observation") or {}
        planets_raw = obs.get("planets") or []
        if not action or not planets_raw:
            continue
        my_id = int(obs.get("player", my_seat))
        planets = [planet_dict(p) for p in planets_raw]
        by_id = {p["id"]: p for p in planets}
        for a in action:
            try:
                src_id, angle, ships = int(a[0]), float(a[1]), int(a[2])
            except (ValueError, TypeError, IndexError):
                continue
            src = by_id.get(src_id)
            if not src or src["owner"] != my_id:
                continue
            # Predict target by ray-casting along angle (or just use the
            # nearest non-self planet in that direction)
            cos_a, sin_a = math.cos(angle), math.sin(angle)
            best_tgt = None
            best_t = float("inf")
            for tgt in planets:
                if tgt["id"] == src_id:
                    continue
                dx, dy = tgt["x"] - src["x"], tgt["y"] - src["y"]
                # Project onto angle direction
                t = dx * cos_a + dy * sin_a
                if t <= 0:
                    continue
                # Perpendicular distance from line
                perp = abs(dx * (-sin_a) + dy * cos_a)
                # Hit if perp <= ~2 (planet radius ~1)
                if perp < 3.0 and t < best_t:
                    best_t = t
                    best_tgt = tgt
            if best_tgt is None:
                # No clear target — likely sun-bound or vanished
                continue
            flight_dist = math.hypot(
                best_tgt["x"] - src["x"], best_tgt["y"] - src["y"]
            )
            spd = fleet_speed(ships)
            if spd <= 0:
                continue
            eta = int(math.ceil(flight_dist / spd))
            bucket = bucket_of(eta, ETA_BUCKETS)
            eta_by_bucket[bucket] += 1
            ships_by_bucket[bucket] += ships
            total_launches += 1
            total_ships += ships

            # Staging-opportunity check for long+very_long launches
            if eta > LONG_ETA_THRESHOLD:
                long_launches_total += 1
                tol_rad = math.radians(STAGING_ANGLE_TOL_DEG)
                max_stage_dist = STAGING_DIST_FRAC * flight_dist
                has_stage = False
                for B in planets:
                    if B["id"] in (src_id, best_tgt["id"]):
                        continue
                    if B["owner"] != my_id and B["owner"] != -1:
                        continue  # need own or neutral
                    bx, by = B["x"] - src["x"], B["y"] - src["y"]
                    bdist = math.hypot(bx, by)
                    if bdist >= max_stage_dist or bdist <= 0:
                        continue
                    b_angle = math.atan2(by, bx)
                    diff = abs(((b_angle - angle + math.pi) % (2 * math.pi))
                               - math.pi)
                    if diff <= tol_rad:
                        has_stage = True
                        break
                if has_stage:
                    long_launches_with_staging += 1

    pct_by_bucket = {
        name: round(100 * eta_by_bucket.get(name, 0) / total_launches, 1)
        if total_launches else 0.0
        for name, _lo, _hi in ETA_BUCKETS
    }
    pct_ships_by_bucket = {
        name: round(100 * ships_by_bucket.get(name, 0) / total_ships, 1)
        if total_ships else 0.0
        for name, _lo, _hi in ETA_BUCKETS
    }
    staging_rate = (
        round(100 * long_launches_with_staging / long_launches_total, 1)
        if long_launches_total else 0.0
    )

    return {
        "total_launches": total_launches,
        "total_ships_launched": total_ships,
        "eta_by_bucket": dict(eta_by_bucket),
        "ships_by_bucket": dict(ships_by_bucket),
        "pct_by_bucket": pct_by_bucket,
        "pct_ships_by_bucket": pct_ships_by_bucket,
        "long_launches_total": long_launches_total,
        "long_launches_with_staging": long_launches_with_staging,
        "staging_opportunity_rate": staging_rate,
    }


def mine_one_submission(sub_id: str, team_name: str | None = None,
                        ) -> dict:
    sub_dir = REPO / "audit" / "live-episodes" / str(sub_id)
    replays = sorted(sub_dir.glob("episode-*-replay.json"))
    if not replays:
        return {"submission_id": sub_id, "error": f"no replays in {sub_dir}"}

    if team_name is None:
        team_name = detect_team_name(replays, None)

    idle_total = collections.Counter()
    idle_steps = 0
    launch_total = collections.Counter()
    launch_ships = collections.Counter()
    total_launches = 0
    total_ships = 0
    long_total = 0
    long_staged = 0
    per_planet_idle = collections.Counter()
    n_episodes = 0

    for path in replays:
        try:
            replay = json.load(open(path))
        except Exception:
            continue
        teams = replay.get("info", {}).get("TeamNames", []) or []
        our_seats = [i for i, t in enumerate(teams) if t == team_name]
        if not our_seats:
            continue
        seat = our_seats[0]

        idle = analyze_idle_ship_turns(replay, seat)
        launches = analyze_launches(replay, seat)

        for b, n in idle["ship_turns_by_bucket"].items():
            idle_total[b] += n
        idle_steps += idle["n_steps_analyzed"]
        for d in idle["top_idle_planets"]:
            per_planet_idle[d["planet_id"]] += d["idle_ship_turns"]

        for b, n in launches["eta_by_bucket"].items():
            launch_total[b] += n
        for b, n in launches["ships_by_bucket"].items():
            launch_ships[b] += n
        total_launches += launches["total_launches"]
        total_ships += launches["total_ships_launched"]
        long_total += launches["long_launches_total"]
        long_staged += launches["long_launches_with_staging"]

        n_episodes += 1

    total_ship_turns = sum(idle_total.values())
    pct_idle_by_bucket = {
        name: round(100 * idle_total.get(name, 0) / total_ship_turns, 1)
        if total_ship_turns else 0.0
        for name, _lo, _hi in DIST_BUCKETS
    }
    pct_launch_by_bucket = {
        name: round(100 * launch_total.get(name, 0) / total_launches, 1)
        if total_launches else 0.0
        for name, _lo, _hi in ETA_BUCKETS
    }
    pct_launch_ships = {
        name: round(100 * launch_ships.get(name, 0) / total_ships, 1)
        if total_ships else 0.0
        for name, _lo, _hi in ETA_BUCKETS
    }
    staging_rate = (
        round(100 * long_staged / long_total, 1) if long_total else 0.0
    )

    return {
        "submission_id": sub_id,
        "team_name": team_name,
        "n_episodes": n_episodes,
        "idle": {
            "n_steps_analyzed": idle_steps,
            "total_ship_turns": total_ship_turns,
            "ship_turns_by_bucket": dict(idle_total),
            "pct_by_bucket": pct_idle_by_bucket,
        },
        "launches": {
            "total_launches": total_launches,
            "total_ships_launched": total_ships,
            "eta_by_bucket": dict(launch_total),
            "ships_by_bucket": dict(launch_ships),
            "pct_by_bucket": pct_launch_by_bucket,
            "pct_ships_by_bucket": pct_launch_ships,
            "long_launches_total": long_total,
            "long_launches_with_staging": long_staged,
            "staging_opportunity_rate": staging_rate,
        },
    }


def render_markdown(rollup: dict) -> str:
    lines = []
    lines.append(f"# idle-trajectory audit — {rollup['date']}")
    lines.append("")
    lines.append("Idle ship-turns: my-planet ships attributed by "
                 "min-distance to nearest non-our planet.")
    lines.append("Launch ETA: ceil(launch flight distance / fleet speed).")
    lines.append("Staging opportunity: long launch had own/neutral planet "
                 "in ±15° corridor at <60% target distance.")
    lines.append("")
    for sub in rollup["submissions"]:
        if "error" in sub:
            lines.append(f"## {sub['submission_id']} — ERROR: {sub['error']}")
            continue
        lines.append(f"## {sub['submission_id']} ({sub['n_episodes']} eps)")
        idle = sub["idle"]
        lines.append("")
        lines.append("**Idle ship-turns by distance bucket:**")
        lines.append("")
        lines.append("| bucket | range | ship-turns | % |")
        lines.append("|---|---|---:|---:|")
        for name, lo, hi in DIST_BUCKETS:
            n = idle["ship_turns_by_bucket"].get(name, 0)
            pct = idle["pct_by_bucket"].get(name, 0.0)
            lines.append(f"| {name} | {lo}-{hi} | {n} | {pct:.1f}% |")
        lines.append(f"| **TOTAL** | | **{idle['total_ship_turns']}** | "
                     f"{idle['n_steps_analyzed']} steps |")
        lines.append("")
        lc = sub["launches"]
        lines.append(f"**Launch ETA distribution** "
                     f"({lc['total_launches']} launches, "
                     f"{lc['total_ships_launched']} ships):")
        lines.append("")
        lines.append("| bucket | range | launches | % | ships% |")
        lines.append("|---|---|---:|---:|---:|")
        for name, lo, hi in ETA_BUCKETS:
            n = lc["eta_by_bucket"].get(name, 0)
            pct = lc["pct_by_bucket"].get(name, 0.0)
            sp = lc["pct_ships_by_bucket"].get(name, 0.0)
            lines.append(f"| {name} | {lo}-{hi} | {n} | {pct:.1f}% | "
                         f"{sp:.1f}% |")
        lines.append("")
        lines.append(f"**Staging opportunity (ETA > {LONG_ETA_THRESHOLD})**: "
                     f"{lc['long_launches_with_staging']} / "
                     f"{lc['long_launches_total']} = "
                     f"{lc['staging_opportunity_rate']:.1f}%")
        lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("**Decision gates:**")
    lines.append("- isolated+rear > 30% of ship-turns → spatial leaf fix high-leverage")
    lines.append("- long+very_long > 25% of launches AND staging-rate > 50% → staging proposer high-leverage")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="idle_trajectory_audit")
    ap.add_argument("submission_ids", nargs="+")
    ap.add_argument("--team", default=None)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args(argv)

    submissions = [mine_one_submission(sid, args.team) for sid in args.submission_ids]
    out_dir = (Path(args.out_dir) if args.out_dir
               else REPO / "audit" / "replays")
    out_dir.mkdir(parents=True, exist_ok=True)
    date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    rollup = {"date": date, "submissions": submissions}
    json_path = out_dir / f"idle-trajectory-{date}.json"
    md_path = out_dir / f"idle-trajectory-{date}.md"
    json_path.write_text(json.dumps(rollup, indent=2) + "\n")
    md_path.write_text(render_markdown(rollup))

    print(f"=== idle-trajectory audit {date} ===")
    for sub in submissions:
        if "error" in sub:
            print(f"  {sub['submission_id']}: {sub['error']}")
            continue
        idle_isolated = sub["idle"]["pct_by_bucket"].get("isolated", 0.0)
        idle_rear = sub["idle"]["pct_by_bucket"].get("rear", 0.0)
        lc = sub["launches"]
        long_pct = (lc["pct_by_bucket"].get("long", 0.0)
                    + lc["pct_by_bucket"].get("very_long", 0.0))
        print(f"  {sub['submission_id']} ({sub['n_episodes']} eps): "
              f"isolated={idle_isolated:.1f}% rear={idle_rear:.1f}% "
              f"long-launch={long_pct:.1f}% "
              f"staging-rate={lc['staging_opportunity_rate']:.1f}%")
    print(f"-> {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
