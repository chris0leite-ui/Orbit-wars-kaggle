"""Mine 1 — Board-geometry feature extractor.

Reads one replay JSON (live-ladder format or self-play format from
`scripts/generate_selfplay_replays.py`) and emits a per-board feature
row. Reuses `lib/geometry`, `lib/orbit`. No agent execution; features
are computed purely from step-0 observation.

CLI:
    python -m scripts.eda.board_features --replay-dir <dir> --out audit/<date>-board-taxonomy.json
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Iterable

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from lib.geometry import CENTER, SUN_RADIUS, path_clears_sun  # noqa: E402
from lib.orbit import is_orbiting  # noqa: E402


def _step0_obs(replay: dict) -> dict:
    """Return the step-0 observation. Robust to live-ladder and self-play schemas."""
    return replay["steps"][0][0]["observation"]


def _home_planets(obs: dict) -> list:
    return [p for p in obs["planets"] if p[1] >= 0]


def _planet_xy(p) -> tuple:
    return (p[2], p[3])


def board_features(replay_path: Path) -> dict:
    replay = json.load(open(replay_path))
    obs = _step0_obs(replay)
    planets = obs["planets"]
    n = len(planets)
    homes = _home_planets(obs)
    n_seats = len(homes)
    omega = float(obs.get("angular_velocity", 0.0))

    # Geometry features
    orb_count = sum(1 for p in planets if is_orbiting(p))
    prods = [p[6] for p in planets]
    ships = [p[5] for p in planets]
    radii = [p[4] for p in planets]

    # Sun-shadow pair fraction
    blocked, total = 0, 0
    for i in range(n):
        a = _planet_xy(planets[i])
        for j in range(i + 1, n):
            b = _planet_xy(planets[j])
            if not path_clears_sun(a, b, safety=0.0):
                blocked += 1
            total += 1
    sun_shadow_frac = blocked / total if total else 0.0

    # Home-pair distance
    home_pair_min = None
    if n_seats >= 2:
        ds = []
        for i in range(n_seats):
            for j in range(i + 1, n_seats):
                ds.append(math.hypot(homes[i][2] - homes[j][2], homes[i][3] - homes[j][3]))
        home_pair_min = min(ds)

    # Production-weighted centroid distance from sun
    tot_prod = sum(prods)
    cx = sum(p[2] * p[6] for p in planets) / tot_prod
    cy = sum(p[3] * p[6] for p in planets) / tot_prod
    prod_centroid_dist_sun = math.hypot(cx - CENTER, cy - CENTER)

    # For 2P only: distance from each non-home planet to the perpendicular
    # bisector of the home-pair line (Voronoi front).
    perp_med = None
    perp_mean = None
    if n_seats == 2:
        (x0, y0) = (homes[0][2], homes[0][3])
        (x1, y1) = (homes[1][2], homes[1][3])
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        # Line through midpoint, perpendicular to home-pair line.
        # Normal vector to perp-bisector = (x1-x0, y1-y0); distance from
        # arbitrary point P to bisector = |(P-M) . n_hat|.
        nx, ny = x1 - x0, y1 - y0
        nlen = math.hypot(nx, ny)
        if nlen > 0:
            distances = []
            for p in planets:
                if p[1] >= 0:
                    continue
                dx, dy = p[2] - mx, p[3] - my
                d = abs(dx * nx + dy * ny) / nlen
                distances.append(d)
            if distances:
                perp_med = statistics.median(distances)
                perp_mean = statistics.fmean(distances)

    # Mean home-distance-to-sun (asymmetric maps have homes pushed inward)
    home_dist_sun = (
        statistics.fmean(math.hypot(h[2] - CENTER, h[3] - CENTER) for h in homes) if homes else None
    )

    return {
        "file": replay_path.name,
        "n_planets": n,
        "n_seats": n_seats,
        "orbital_frac": orb_count / n,
        "mean_prod": statistics.fmean(prods),
        "std_prod": statistics.stdev(prods) if n > 1 else 0.0,
        "mean_init_ships": statistics.fmean(ships),
        "std_init_ships": statistics.stdev(ships) if n > 1 else 0.0,
        "mean_radius": statistics.fmean(radii),
        "sun_shadow_frac": sun_shadow_frac,
        "home_pair_min_dist": home_pair_min,
        "home_dist_sun": home_dist_sun,
        "prod_centroid_dist_sun": prod_centroid_dist_sun,
        "perp_bisector_median_dist": perp_med,
        "perp_bisector_mean_dist": perp_mean,
        "angular_velocity": omega,
        "n_steps": len(replay.get("steps", [])),
        "rewards": replay.get("rewards"),
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--replay-dirs", nargs="+", required=True, help="One or more directories of replays")
    p.add_argument("--out", required=True, help="Output JSON path")
    p.add_argument("--glob", default="*.json")
    args = p.parse_args(argv)

    rows = []
    for d in args.replay_dirs:
        for fp in sorted(Path(d).glob(args.glob)):
            try:
                rows.append(board_features(fp))
            except Exception as e:
                print(f"  SKIP {fp.name}: {e}", file=sys.stderr)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"n_rows": len(rows), "rows": rows}, indent=2))
    print(f"wrote {len(rows)} rows -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
