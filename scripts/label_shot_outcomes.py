"""Walk replays under audit/external/replays/ and emit labeled training
examples for the konbu17-style shot validator MLP.

For each replay × focal_seat × launch_step:
  - Encode the 24-dim shot context (src, target, shot, in-flight, meta).
  - Label = 1 iff target.owner == focal_seat at
    min(launch_step + eta + 10, end_of_game).

Output: data/shot_validator/labels.jsonl (one labeled example per line).
Schema in `data/shot_validator/schema.json` (versioned).

Usage:
    python -m scripts.label_shot_outcomes [--replay-dir DIR]
                                          [--out PATH]
                                          [--limit N]

The 24-dim feature spec matches konbu17/orbit-wars-rule-base-ml-shot-
validator-hybrid (Kaggle notebook) so a future MLP training session can
load this dataset directly. See `data/shot_validator/README.md`.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_REPLAY_DIR = REPO / "audit" / "external" / "replays"
DEFAULT_OUT = REPO / "data" / "shot_validator" / "labels.jsonl"
SCHEMA_PATH = REPO / "data" / "shot_validator" / "schema.json"

# Normalisation constants — from schema.json
_NORM = {
    "max_ships": 2000.0,
    "max_production": 5.0,
    "max_radius": 3.0,
    "max_fleet_speed": 6.0,
    "max_eta": 200.0,
    "board_diagonal": 141.42,
    "max_planets": 40.0,
    "episode_steps": 500.0,
}

LABEL_BUFFER = 10   # steps after eta to check ownership


def _fleet_speed(ships):
    """Lifted from lib/fleet.py to keep this script env-free."""
    if ships <= 0:
        return 0.0
    return 1.0 + (6.0 - 1.0) * (math.log(ships) / math.log(1000.0)) ** 1.5


def _infer_target_pid(src_xy, angle, planets):
    """Project ray from src; return planet id with smallest perpendicular
    distance among forward candidates (mirror of `scripts/extended_features._infer_target_pid`)."""
    sx, sy = src_xy
    dx, dy = math.cos(angle), math.sin(angle)
    best_id = None
    best_score = float("inf")
    for p in planets:
        pid, _, px, py, prad, _, _ = p[0], p[1], p[2], p[3], p[4], p[5], p[6]
        if abs(px - sx) < 1e-6 and abs(py - sy) < 1e-6:
            continue
        rx, ry = px - sx, py - sy
        fwd = rx * dx + ry * dy
        if fwd <= 0:
            continue
        perp = math.hypot(rx - fwd * dx, ry - fwd * dy)
        score = perp + 0.001 * fwd
        if score < best_score:
            best_score = score
            best_id = int(pid)
    return best_id


def _encode_features(
    src_planet, target_planet, ships_sent, distance, eta, fleet_speed,
    all_planets, all_fleets, focal_seat, step,
) -> list[float]:
    """Build the 24-dim feature vector. All values normalised to [0, 1]."""
    sps_ships = src_planet[5] / _NORM["max_ships"]
    sps_prod = src_planet[6] / _NORM["max_production"]
    sps_rad = src_planet[4] / _NORM["max_radius"]

    tgt_ships = target_planet[5] / _NORM["max_ships"]
    tgt_prod = target_planet[6] / _NORM["max_production"]
    tgt_rad = target_planet[4] / _NORM["max_radius"]

    tgt_owner = int(target_planet[1])
    owner_mine = 1.0 if tgt_owner == focal_seat else 0.0
    owner_neutral = 1.0 if tgt_owner == -1 else 0.0
    owner_enemy = 1.0 if (tgt_owner != -1 and tgt_owner != focal_seat) else 0.0

    src_garrison = max(1, src_planet[5])
    shot_ships = min(1.0, ships_sent / _NORM["max_ships"])
    # Logically a launch can't exceed the source garrison; cap at 1.0
    # (records sometimes have stale src.ships from pre-launch state).
    shot_frac = min(1.0, ships_sent / src_garrison)
    shot_dist = min(1.0, distance / _NORM["board_diagonal"])
    shot_eta = min(1.0, eta / _NORM["max_eta"])
    shot_fs = min(1.0, fleet_speed / _NORM["max_fleet_speed"])

    n_allied = 0
    ship_allied = 0.0
    n_enemy = 0
    ship_enemy = 0.0
    for f in all_fleets:
        owner = int(f[1])
        ships = float(f[6])
        if owner == focal_seat:
            n_allied += 1
            ship_allied += ships
        elif owner != -1:
            n_enemy += 1
            ship_enemy += ships
    in_flight_n_allied = min(1.0, n_allied / _NORM["max_planets"])
    in_flight_n_enemy = min(1.0, n_enemy / _NORM["max_planets"])
    in_flight_ship_allied = min(1.0, ship_allied / _NORM["max_ships"])
    in_flight_ship_enemy = min(1.0, ship_enemy / _NORM["max_ships"])

    my_total_ships = sum(p[5] for p in all_planets if int(p[1]) == focal_seat) + ship_allied
    enemy_total_ships = sum(p[5] for p in all_planets
                              if int(p[1]) not in (-1, focal_seat)) + ship_enemy
    # ship_diff is signed in [-1, 1] (clipped). Top-10 games can produce
    # thousands of total ships; norm chosen to keep the typical
    # distribution centred without saturating extreme blowouts.
    ship_diff = max(-1.0, min(1.0,
        (my_total_ships - enemy_total_ships) / _NORM["max_ships"]))
    my_total_ships_n = min(1.0, my_total_ships / _NORM["max_ships"])
    enemy_total_ships_n = min(1.0, enemy_total_ships / _NORM["max_ships"])
    meta_turn = step / _NORM["episode_steps"]
    my_planet_count = sum(1 for p in all_planets if int(p[1]) == focal_seat)
    enemy_planet_count = sum(1 for p in all_planets if int(p[1]) not in (-1, focal_seat))
    my_pc_n = my_planet_count / _NORM["max_planets"]
    enemy_pc_n = enemy_planet_count / _NORM["max_planets"]

    return [
        sps_ships, sps_prod, sps_rad,
        tgt_ships, tgt_prod, tgt_rad,
        owner_mine, owner_neutral, owner_enemy,
        shot_ships, shot_frac, shot_dist, shot_eta, shot_fs,
        in_flight_n_allied, in_flight_ship_allied,
        in_flight_n_enemy, in_flight_ship_enemy,
        meta_turn, my_total_ships_n, enemy_total_ships_n,
        ship_diff, my_pc_n, enemy_pc_n,
    ]


def _process_replay(path: Path) -> list[dict]:
    """Returns a list of {"features": [...], "label": int, "meta": {...}}."""
    try:
        replay = json.loads(path.read_text())
    except Exception:
        return []
    steps = replay.get("steps", [])
    if not steps:
        return []
    info = replay.get("info", {})
    team_names = info.get("TeamNames", [])
    # We label the focal seat's launches. For top-10 replays the focal is
    # the player we sampled the replay around (encoded in filename).
    # For midpack replays, focal is "ChrisLeiteScha" (our team).
    if path.name.startswith("midpack-"):
        focal_team = "ChrisLeiteScha"
    else:
        # r{rank:02d}-{team}-{2P|4P}-{W|L}-{eid}.json
        # team is dash-separated; rebuild it by stripping the rank/format/outcome/eid parts
        parts = path.stem.split("-")
        if len(parts) < 5:
            return []
        # parts: ['r01', 'bowwowforeach', '2P', 'W', '76308932']
        # but team may contain spaces converted to underscores; tolerate _
        focal_team = parts[1].replace("_", " ")
    focal_seat = None
    for i, t in enumerate(team_names):
        if t == focal_team or t.replace(" ", "_") == focal_team:
            focal_seat = i
            break
    if focal_seat is None:
        # fuzzy fallback
        for i, t in enumerate(team_names):
            if focal_team.lower().replace("_", "") in t.lower().replace(" ", ""):
                focal_seat = i
                break
    if focal_seat is None:
        return []

    examples = []
    n_steps = len(steps)
    for step_idx, step in enumerate(steps):
        if focal_seat >= len(step):
            continue
        obs = step[focal_seat].get("observation", {}) or {}
        planets = obs.get("planets", []) or []
        fleets = obs.get("fleets", []) or []
        action = step[focal_seat].get("action") or []
        by_id = {int(p[0]): p for p in planets}
        for a in action:
            if not a or len(a) < 3:
                continue
            try:
                src_pid = int(a[0])
                angle = float(a[1])
                ships = float(a[2])
            except (TypeError, ValueError):
                continue
            src = by_id.get(src_pid)
            if src is None:
                continue
            target_pid = _infer_target_pid(
                (float(src[2]), float(src[3])), angle, planets,
            )
            if target_pid is None:
                continue
            target = by_id.get(target_pid)
            if target is None:
                continue
            d = math.hypot(float(target[2]) - float(src[2]),
                            float(target[3]) - float(src[3]))
            v = _fleet_speed(ships)
            eta = int(math.ceil(d / max(v, 1e-6))) if v > 0 else 0
            # Label: ownership at step_idx + eta + LABEL_BUFFER (capped).
            check_step = min(step_idx + eta + LABEL_BUFFER, n_steps - 1)
            if check_step >= len(steps):
                continue
            check_obs = steps[check_step][focal_seat].get("observation", {}) or {}
            check_planets = check_obs.get("planets", []) or []
            check_by_id = {int(p[0]): p for p in check_planets}
            target_check = check_by_id.get(target_pid)
            if target_check is None:
                continue
            label = 1 if int(target_check[1]) == focal_seat else 0

            features = _encode_features(
                src, target, ships, d, eta, v,
                planets, fleets, focal_seat, step_idx,
            )
            examples.append({
                "features": features,
                "label": label,
                "meta": {
                    "src_path": path.name,
                    "step": step_idx,
                    "src_pid": src_pid,
                    "target_pid": target_pid,
                    "focal_seat": focal_seat,
                    "focal_team": focal_team,
                },
            })
    return examples


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--replay-dir", default=str(DEFAULT_REPLAY_DIR))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--limit", type=int, default=None,
                        help="Process at most this many replay files (debug).")
    args = parser.parse_args(argv)

    replay_dir = Path(args.replay_dir)
    if not replay_dir.is_dir():
        print(f"ERROR: replay-dir not found: {replay_dir}", file=sys.stderr)
        return 1
    files = sorted(replay_dir.glob("*.json"))
    if args.limit:
        files = files[:args.limit]
    if not files:
        print(f"ERROR: no replay JSONs in {replay_dir}", file=sys.stderr)
        return 1

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    positives = 0
    with out_path.open("w") as fh:
        for f in files:
            try:
                rows = _process_replay(f)
            except Exception as e:
                print(f"  WARN {f.name}: {e}", file=sys.stderr)
                continue
            for r in rows:
                fh.write(json.dumps(r) + "\n")
                total += 1
                positives += r["label"]
    print(f"Wrote {total} examples ({positives} positives, "
          f"{total-positives} negatives) to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
