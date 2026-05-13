"""Loss-mode classifier for v7_0_drop_one's live ladder games.

Reads replays in `audit/live-episodes/<submission_id>/`, filters to
non-self-play games where our team lost, and assigns each one to a
single bucket from the five-class taxonomy:

  opening_lost       — we launched too late OR fell behind in planet
                       count by step 30 by ≥ 10 ships.
  mid_economy_lost   — we lost ≥ 30 ships of relative position between
                       step 30 and step 200 (the production-share window).
  gang_up_failure    — (4P only) we were not the last to die AND a
                       different player ran away with the lead.
  tactical_missizing — we launched ≥ 50 % more ships than the eventual
                       winner over the game, suggesting bounced /
                       wasted fleets.
  endgame_drift      — none of the above AND ship_delta(step 400) > 0
                       AND ship_delta(final) < 0.

Cases matching none → `unclassified` (we accept up to 20 %).

CLI:
    python -m scripts.classify_losses <submission_id> [--team NAME]
    [--out audit/2026-05-13-v7-0-loss-modes.csv]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

LIVE_DIR = REPO / "audit" / "live-episodes"


def detect_team_name(replays: list[Path], hint: str | None) -> str:
    """Pick the team name that appears in ≥ 80 % of non-self-only episodes."""
    if hint:
        return hint
    names: Counter = Counter()
    for rp in replays:
        try:
            with rp.open() as f:
                rep = json.load(f)
        except Exception:
            continue
        teams = rep.get("info", {}).get("TeamNames") or []
        unique = set(teams)
        if len(unique) <= 1:
            continue
        for t in teams:
            names[t] += 1
    if not names:
        return os.environ.get("KaggleUserName") or "ChrisLeiteScha"
    top, _ = names.most_common(1)[0]
    return top


def _our_seats(rep: dict, team: str) -> list[int]:
    """Return seat indices belonging to `team`. Multiple seats means
    self-vs-self (skip the game)."""
    teams = rep.get("info", {}).get("TeamNames") or []
    return [i for i, n in enumerate(teams) if n == team]


def _seat_totals_at_step(step_state: list, num_seats: int) -> tuple[list[float], list[int], list[int]]:
    """Compute (ship_totals, planet_counts, fleet_counts) per seat at one step.

    `step_state` is the per-agent list from `replay["steps"][i]`. The seat-0
    observation carries the full planet + fleet list for everyone.
    """
    obs = step_state[0]["observation"]
    ship_totals = [0.0] * num_seats
    planet_counts = [0] * num_seats
    fleet_counts = [0] * num_seats
    for p in obs.get("planets", []) or []:
        owner = int(p[1])
        if 0 <= owner < num_seats:
            ship_totals[owner] += float(p[5])
            planet_counts[owner] += 1
    for f in obs.get("fleets", []) or []:
        owner = int(f[1])
        if 0 <= owner < num_seats:
            ship_totals[owner] += float(f[6])
            fleet_counts[owner] += 1
    return ship_totals, planet_counts, fleet_counts


def _first_launch_step(rep: dict, our_seat: int) -> int | None:
    """First step at which our seat emitted a non-empty action."""
    for idx, step in enumerate(rep.get("steps", [])):
        action = step[our_seat].get("action")
        if action:  # non-empty list
            return idx
    return None


def _total_ships_launched(rep: dict, our_seat: int) -> int:
    """Sum of ships emitted by our seat across the whole game."""
    total = 0
    for step in rep.get("steps", []):
        action = step[our_seat].get("action") or []
        for mv in action:
            if isinstance(mv, list) and len(mv) == 3:
                try:
                    total += int(mv[2])
                except (TypeError, ValueError):
                    pass
    return total


def classify_game(rep: dict, team: str) -> dict | None:
    """Bucket a single replay. Returns None for self-vs-self (skip).

    Output dict: seed, episode_id, num_seats, our_reward, bucket,
    plus the raw signals used.
    """
    our_seats = _our_seats(rep, team)
    teams = rep.get("info", {}).get("TeamNames") or []
    num_seats = len(teams)
    if not our_seats:
        return None
    if len(our_seats) == num_seats:
        return None  # self-vs-self
    if num_seats not in (2, 4):
        return None

    our_seat = our_seats[0]
    rewards = rep.get("rewards") or []
    our_reward = rewards[our_seat] if our_seat < len(rewards) else None
    if our_reward != -1:
        # Only losses are classified here.
        return None

    steps = rep.get("steps", [])
    if not steps:
        return None
    n_steps = len(steps)

    # Per-step totals at the diagnostic windows.
    def _at(target_step: int):
        idx = min(target_step, n_steps - 1)
        return _seat_totals_at_step(steps[idx], num_seats)

    ships_30, planets_30, _ = _at(30)
    ships_200, planets_200, _ = _at(200)
    ships_400, planets_400, _ = _at(400)
    ships_final, planets_final, _ = _at(n_steps - 1)

    # Opp-aggregate (sum across non-us seats).
    def _opp_sum(arr):
        return sum(v for i, v in enumerate(arr) if i != our_seat)

    def _opp_max(arr):
        return max((v for i, v in enumerate(arr) if i != our_seat), default=0.0)

    our_ships30 = ships_30[our_seat]
    opp_ships30 = _opp_sum(ships_30)
    our_planets30 = planets_30[our_seat]
    opp_planets30_max = max(
        (v for i, v in enumerate(planets_30) if i != our_seat), default=0,
    )

    our_ships200 = ships_200[our_seat]
    opp_ships200 = _opp_sum(ships_200)

    our_ships400 = ships_400[our_seat]
    opp_ships400 = _opp_sum(ships_400)
    our_shipsF = ships_final[our_seat]
    opp_shipsF = _opp_sum(ships_final)

    delta30 = our_ships30 - opp_ships30
    delta200 = our_ships200 - opp_ships200
    delta400 = our_ships400 - opp_ships400
    deltaF = our_shipsF - opp_shipsF

    first_launch = _first_launch_step(rep, our_seat)
    our_total_launched = _total_ships_launched(rep, our_seat)
    # Best estimate of the winner's launches (use their position-final
    # ship total as a coarse proxy when total_launches is hard to get).
    # We compare our total launches to the seat with highest final ships.
    winner_seat = max(range(num_seats), key=lambda i: ships_final[i])
    winner_launched = _total_ships_launched(rep, winner_seat) or 1

    # ---- Bucket assignment ----

    # 1. opening_lost
    opening_lost = (
        (first_launch is None or first_launch > 6)
        or (our_planets30 < opp_planets30_max and delta30 < -10)
    )

    # 2. mid_economy_lost
    mid_economy_lost = False
    if not opening_lost:
        # Relative position dropped ≥ 30 ships between step 30 and 200.
        if delta200 - delta30 < -30:
            mid_economy_lost = True

    # 3. gang_up_failure (4P only)
    gang_up_failure = False
    if not opening_lost and not mid_economy_lost and num_seats == 4:
        # Were we eliminated before someone else? Check whether at any
        # late step we had zero ships+planets while another non-winner
        # still had presence.
        # Heuristic: at step 300 we're behind the winner by ≥50 ships AND
        # behind the 2nd-place by ≥ 20 ships → we got squeezed mid-game.
        ships_300, _, _ = _at(300)
        winner_300 = max(ships_300)
        sorted_300 = sorted(ships_300, reverse=True)
        second_300 = sorted_300[1] if len(sorted_300) > 1 else 0
        gap_winner = winner_300 - ships_300[our_seat]
        gap_second = second_300 - ships_300[our_seat]
        if gap_winner >= 50 and gap_second >= 20 and ships_300[our_seat] <= 30:
            gang_up_failure = True

    # 4. tactical_missizing — we launched a lot more than the winner
    #    but still lost (suggests bounced / wasted fleets).
    tactical_missizing = False
    if not (opening_lost or mid_economy_lost or gang_up_failure):
        if our_total_launched >= int(winner_launched * 1.5):
            tactical_missizing = True

    # 5. endgame_drift — we were ahead at step 400, lost at final.
    endgame_drift = False
    if not (opening_lost or mid_economy_lost or gang_up_failure or tactical_missizing):
        if delta400 > 0 and deltaF < 0:
            endgame_drift = True

    bucket = (
        "opening_lost" if opening_lost else
        "mid_economy_lost" if mid_economy_lost else
        "gang_up_failure" if gang_up_failure else
        "tactical_missizing" if tactical_missizing else
        "endgame_drift" if endgame_drift else
        "unclassified"
    )

    return {
        "episode_id": rep.get("info", {}).get("EpisodeId"),
        "seed": rep.get("info", {}).get("seed"),
        "num_seats": num_seats,
        "our_seat": our_seat,
        "our_reward": our_reward,
        "n_steps": n_steps,
        "first_launch": first_launch if first_launch is not None else -1,
        "our_total_launched": our_total_launched,
        "winner_launched": winner_launched,
        "delta30": delta30,
        "delta200": delta200,
        "delta400": delta400,
        "deltaF": deltaF,
        "our_planets30": our_planets30,
        "opp_planets30_max": opp_planets30_max,
        "bucket": bucket,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("submission_id", help="Kaggle submission id (e.g. 52588156)")
    parser.add_argument("--team", default=None,
                        help="our team name (auto-detected if omitted)")
    parser.add_argument("--out", default=None,
                        help="output CSV path (default audit/...loss-modes.csv)")
    parser.add_argument("--limit", type=int, default=None,
                        help="cap on number of replays processed")
    args = parser.parse_args()

    sub_dir = LIVE_DIR / args.submission_id
    replays = sorted(sub_dir.glob("episode-*-replay.json"))
    if not replays:
        print(f"no replays in {sub_dir}", file=sys.stderr)
        sys.exit(1)
    if args.limit:
        replays = replays[: args.limit]
    team = detect_team_name(replays, args.team)
    print(f"team: {team}; replays: {len(replays)}")

    rows: list[dict] = []
    for rp in replays:
        try:
            with rp.open() as f:
                rep = json.load(f)
        except Exception as exc:
            print(f"  WARN: failed to parse {rp.name}: {exc}", file=sys.stderr)
            continue
        out = classify_game(rep, team)
        if out is None:
            continue
        rows.append(out)

    if not rows:
        print("no classified games (only self-vs-self or wins?).")
        return

    # Aggregate.
    bucket_counts: Counter = Counter(r["bucket"] for r in rows)
    n = len(rows)
    print(f"\n=== loss-mode breakdown ({n} lost games) ===")
    for bucket, count in sorted(
        bucket_counts.items(), key=lambda kv: -kv[1],
    ):
        pct = 100.0 * count / n
        print(f"  {bucket:25s} {count:4d}  ({pct:5.1f} %)")
    unclassified_pct = 100.0 * bucket_counts.get("unclassified", 0) / n
    if unclassified_pct > 20.0:
        print(f"  ⚠ unclassified rate {unclassified_pct:.1f} % > 20 %; "
              f"thresholds may need retuning.")

    # Sample game IDs per bucket.
    print("\n=== sample episode IDs per bucket ===")
    by_bucket: dict = {}
    for r in rows:
        by_bucket.setdefault(r["bucket"], []).append(r["episode_id"])
    for bucket, ids in by_bucket.items():
        sample = ids[:3]
        print(f"  {bucket}: {sample}")

    out_path = REPO / (args.out or f"audit/2026-05-13-loss-modes-{args.submission_id}.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
