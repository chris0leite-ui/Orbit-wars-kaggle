"""Mine 3 — Opening atlas (first 30 turns).

For each replay, extract the WINNER's first 30 turns of launches:
turn, source planet, target planet (inferred from launch angle +
arriving fleet), ships sent, target's t0 production / starting ships /
distance / visibility. Emits per-game sequence + per-game summary
(first_launch_step, n_distinct_targets, median_ships_per_launch,
median_target_production_first3).

CLI:
    python -m scripts.eda.opening_atlas --replay-dirs audit/external/replays --out audit/<date>-opening-atlas.json
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from lib.geometry import path_clears_sun  # noqa: E402

OPENING_TURNS = 30


def _winner_seat(replay):
    rewards = replay.get("rewards") or []
    if rewards.count(1) == 1:
        return rewards.index(1)
    return None


def _angle_diff(a, b):
    d = (a - b) % (2 * math.pi)
    if d > math.pi:
        d -= 2 * math.pi
    return abs(d)


def _infer_target(action, planets_at_launch, src_xy):
    """Heuristic: pick the planet whose direction-from-src best matches `angle`."""
    from_pid, angle, ships = action[0], action[1], action[2]
    best = None
    best_score = float("inf")
    for p in planets_at_launch:
        if p[0] == from_pid:
            continue
        dx = p[2] - src_xy[0]
        dy = p[3] - src_xy[1]
        if dx == 0 and dy == 0:
            continue
        target_angle = math.atan2(dy, dx)
        ad = _angle_diff(angle, target_angle)
        d = math.hypot(dx, dy)
        # Score: angle deviation weighted by inverse distance (closer
        # planets with small angle deviation are most likely)
        score = ad * d
        if score < best_score:
            best_score = score
            best = p
    return best


def extract(replay_path: Path) -> dict:
    replay = json.load(open(replay_path))
    winner = _winner_seat(replay)
    if winner is None:
        return None
    steps = replay["steps"]
    first_launch = None
    launches = []
    for step_idx in range(min(OPENING_TURNS, len(steps))):
        step = steps[step_idx]
        if winner >= len(step):
            continue
        obs = step[winner]["observation"]
        planets = obs["planets"]
        planets_by_id = {p[0]: p for p in planets}
        action = step[winner].get("action") or []
        if not action:
            continue
        for a in action:
            if not a or len(a) < 3:
                continue
            from_pid = a[0]
            src = planets_by_id.get(from_pid)
            if src is None:
                continue
            tgt = _infer_target(a, planets, (src[2], src[3]))
            if tgt is None:
                continue
            if first_launch is None:
                first_launch = step_idx
            tgt_pid, tgt_owner, tx, ty, tr, tships, tprod = tgt
            d = math.hypot(tx - src[2], ty - src[3])
            launches.append({
                "step": step_idx,
                "from_pid": from_pid,
                "target_pid": tgt_pid,
                "ships_sent": a[2],
                "target_owner_t0": tgt_owner,
                "target_production": tprod,
                "target_ships": tships,
                "target_distance": d,
                "target_sun_clear": path_clears_sun((src[2], src[3]), (tx, ty)),
            })
    # Per-game summary
    if not launches:
        return {
            "replay": replay_path.name,
            "winner_seat": winner,
            "first_launch_step": None,
            "n_launches_30": 0,
            "n_distinct_targets": 0,
            "neutral_target_frac": None,
            "enemy_target_frac": None,
            "median_target_production": None,
            "median_ships_sent": None,
            "median_target_distance": None,
            "launches": [],
        }
    targets = {l["target_pid"] for l in launches}
    return {
        "replay": replay_path.name,
        "winner_seat": winner,
        "first_launch_step": first_launch,
        "n_launches_30": len(launches),
        "n_distinct_targets": len(targets),
        "neutral_target_frac": sum(1 for l in launches if l["target_owner_t0"] == -1) / len(launches),
        "enemy_target_frac": sum(1 for l in launches if l["target_owner_t0"] >= 0 and l["target_owner_t0"] != winner) / len(launches),
        "median_target_production": statistics.median(l["target_production"] for l in launches),
        "median_ships_sent": statistics.median(l["ships_sent"] for l in launches),
        "median_target_distance": statistics.median(l["target_distance"] for l in launches),
        "launches": launches,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay-dirs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--glob", default="*.json")
    args = ap.parse_args(argv)

    games = []
    for d in args.replay_dirs:
        for fp in sorted(Path(d).glob(args.glob)):
            try:
                g = extract(fp)
                if g:
                    games.append(g)
            except Exception as e:
                print(f"  SKIP {fp.name}: {e}", file=sys.stderr)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"n_games": len(games), "games": games}))
    print(f"wrote {len(games)} games -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
