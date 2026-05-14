"""Mine 2 — Per-planet importance features.

For each planet on each replay, compute geometry-driven value features
and a label "captured by eventual winner by step 100." Output is suitable
for a logistic-regression fit (sklearn) to identify which features
predict importance.

CLI:
    python -m scripts.eda.planet_features --replay-dirs audit/external/replays --out audit/<date>-planet-importance-raw.json
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

from lib.geometry import CENTER, path_clears_sun  # noqa: E402
from lib.orbit import is_orbiting, predict_relative  # noqa: E402
from lib.fleet import travel_time  # noqa: E402

LABEL_STEP = 100   # "captured by winner by step LABEL_STEP"
ORBIT_SAMPLES = 24


def _planet_xy(p):
    return (p[2], p[3])


def _step0_obs(replay):
    return replay["steps"][0][0]["observation"]


def _winner_seat(replay) -> int | None:
    rewards = replay.get("rewards") or []
    if not rewards:
        return None
    if rewards.count(1) == 1:
        return rewards.index(1)
    # Tie or no winner
    return None


def planet_capture_step(replay, planet_id: int, winner_seat: int) -> int | None:
    """First step at which `planet_id` is owned by `winner_seat`."""
    for step_idx, step in enumerate(replay["steps"]):
        obs = step[0]["observation"]
        for p in obs["planets"]:
            if p[0] == planet_id:
                if p[1] == winner_seat:
                    return step_idx
                break
    return None


def per_planet_features(replay, planet_idx: int) -> dict:
    obs0 = _step0_obs(replay)
    planets = obs0["planets"]
    p = planets[planet_idx]
    pid, owner, x, y, radius, ships, prod = p
    omega = float(obs0.get("angular_velocity", 0.0))
    homes = [pl for pl in planets if pl[1] >= 0]
    seat_homes = {h[1]: h for h in homes}

    n = len(planets)

    # Time-distance centrality with 100-ship reference fleet
    centrality = 0.0
    for other in planets:
        if other[0] == pid:
            continue
        tt = travel_time(_planet_xy(p), _planet_xy(other), ships=100)
        centrality += 1.0 / (1.0 + tt)

    # Sun-shadow visibility
    visible_n, total_n = 0, 0
    for other in planets:
        if other[0] == pid:
            continue
        if path_clears_sun(_planet_xy(p), _planet_xy(other)):
            visible_n += 1
        total_n += 1
    visibility = visible_n / total_n if total_n else 0.0

    # Distance to each home (winner home = home of the eventual winner; both for diff)
    home_dists = {seat: math.hypot(x - h[2], y - h[3]) for seat, h in seat_homes.items()}
    min_home_dist = min(home_dists.values()) if home_dists else None
    max_home_dist = max(home_dists.values()) if home_dists else None

    # Voronoi front position (2P): signed distance to perp-bisector
    perp_signed = None
    if len(homes) == 2:
        (x0, y0) = (homes[0][2], homes[0][3])
        (x1, y1) = (homes[1][2], homes[1][3])
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        nx, ny = x1 - x0, y1 - y0
        nlen = math.hypot(nx, ny)
        if nlen > 0:
            perp_signed = ((x - mx) * nx + (y - my) * ny) / nlen

    # Production density
    prod_density = prod / (1.0 + radius)

    # Neutral denial (rahul's term): production of OTHER neutrals within 25
    # units of THIS planet but farther from every home than they are from
    # this planet. Captures "if I get this, I deny that neutral too."
    denial_score = 0.0
    for other in planets:
        if other[0] == pid or other[1] >= 0:
            continue
        d_self = math.hypot(other[2] - x, other[3] - y)
        if d_self > 25:
            continue
        if home_dists and d_self < min(math.hypot(other[2] - h[2], other[3] - h[3]) for h in homes):
            denial_score += other[6]

    # Orbital-phase reachability: for orbital planets only, sample positions
    # around the orbital period and report fraction where this planet is
    # closer to home[0] than to home[1] (2P only).
    orbital_reach = None
    if is_orbiting(p) and len(homes) == 2 and omega > 0:
        period = 2 * math.pi / abs(omega)
        closer_to_0 = 0
        for k in range(ORBIT_SAMPLES):
            lead = (k / ORBIT_SAMPLES) * period
            px, py = predict_relative(p, omega, lead)
            d0 = math.hypot(px - homes[0][2], py - homes[0][3])
            d1 = math.hypot(px - homes[1][2], py - homes[1][3])
            if d0 < d1:
                closer_to_0 += 1
        orbital_reach = closer_to_0 / ORBIT_SAMPLES

    return {
        "planet_id": pid,
        "is_home": owner >= 0,
        "owner_t0": owner,
        "x": x,
        "y": y,
        "production": prod,
        "radius": radius,
        "starting_ships": ships,
        "is_orbiting": bool(is_orbiting(p)),
        "is_comet": False,
        "centrality": centrality,
        "visibility": visibility,
        "min_home_dist": min_home_dist,
        "max_home_dist": max_home_dist,
        "perp_signed_dist": perp_signed,
        "prod_density": prod_density,
        "denial_score": denial_score,
        "orbital_reach_seat0": orbital_reach,
    }


def extract_replay(replay_path: Path) -> list[dict]:
    replay = json.load(open(replay_path))
    obs0 = _step0_obs(replay)
    planets = obs0["planets"]
    winner = _winner_seat(replay)
    rows = []
    for i, p in enumerate(planets):
        feats = per_planet_features(replay, i)
        # Label: captured by winner by step 100. None if no winner.
        cap_step = None
        captured = None
        if winner is not None:
            cap_step = planet_capture_step(replay, p[0], winner)
            captured = bool(cap_step is not None and cap_step <= LABEL_STEP)
        feats.update({
            "replay": replay_path.name,
            "winner_seat": winner,
            "capture_step_by_winner": cap_step,
            "captured_by_winner_by_step100": captured,
        })
        rows.append(feats)
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay-dirs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--glob", default="*.json")
    args = ap.parse_args(argv)

    all_rows = []
    for d in args.replay_dirs:
        for fp in sorted(Path(d).glob(args.glob)):
            try:
                all_rows.extend(extract_replay(fp))
            except Exception as e:
                print(f"  SKIP {fp.name}: {e}", file=sys.stderr)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"n_rows": len(all_rows), "rows": all_rows}))
    print(f"wrote {len(all_rows)} rows -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
