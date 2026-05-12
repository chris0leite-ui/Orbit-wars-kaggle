"""Battlefield geometry report — empirical pass over 100 game seeds.

Samples kaggle_environments orbit_wars over seeds 0-99 and computes the
distributions listed in audit/2026-05-12-battlefield-geometry-report.md
(planet counts, production, orbital radii, omega, 4-fold symmetry, sun
no-fly zones, nearest-neighbour distances, home-to-enemy distance over
time, comet trajectories, frontier-line).

Writes:
- audit/2026-05-12-battlefield-geometry-data.json  (raw per-seed stats)
- prints a markdown-friendly summary to stdout, which we then paste into
  audit/2026-05-12-battlefield-geometry-report.md.

Run from repo root: python3 scripts/geometry_report.py
"""

from __future__ import annotations

import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from kaggle_environments import make  # noqa: E402

from lib.geometry import (  # noqa: E402
    CENTER,
    ROTATION_RADIUS_LIMIT,
    SUN_RADIUS,
    point_to_segment_distance,
)

N_SEEDS = 100
NUM_AGENTS = 2
COMET_SPAWN_STEPS = (50, 150, 250, 350, 450)


def _is_orbiting(p) -> bool:
    px, py, pr = p[2], p[3], p[4]
    orb_r = math.hypot(px - CENTER, py - CENTER)
    return (orb_r + pr) < ROTATION_RADIUS_LIMIT


def _quadrant(x: float, y: float) -> int:
    """Returns 1..4 in standard math convention (Q1: x>50,y>50)."""
    if x >= CENTER and y >= CENTER:
        return 1
    if x < CENTER and y >= CENTER:
        return 2
    if x < CENTER and y < CENTER:
        return 3
    return 4


def _polar(p) -> tuple[float, float]:
    """Return (orbital_radius, angle_rad in [-pi, pi])."""
    dx, dy = p[2] - CENTER, p[3] - CENTER
    return math.hypot(dx, dy), math.atan2(dy, dx)


def _detect_quartets(planets):
    """Cluster planets by orbital radius (tolerance 0.01) and check each
    cluster has size that's a multiple of 4 AND can be partitioned into
    4-fold-rotationally-symmetric subsets.

    Different quartets can share an orbital radius; in that case the
    cluster has 8 planets, but should still partition into two groups of
    4 each at 90 degree offsets.

    Returns: (n_clusters, n_quartet_planets, n_total_planets).
    """
    radii = sorted({round(_polar(p)[0], 2) for p in planets})
    # Greedy clustering by orbital radius tolerance.
    by_radius: dict[float, list] = {}
    for p in planets:
        r = _polar(p)[0]
        match = None
        for r0 in by_radius:
            if abs(r - r0) < 0.05:
                match = r0
                break
        if match is None:
            by_radius[r] = [p]
        else:
            by_radius[match].append(p)
    n_quartet_planets = 0
    n_clusters = len(by_radius)
    for r0, ps in by_radius.items():
        if len(ps) % 4 != 0:
            continue
        # Sort by angle; partition into len(ps)//4 quartets of 4 at 90 deg.
        ps_sorted = sorted(ps, key=lambda p: _polar(p)[1])
        # If all consecutive angular gaps are equal multiples of pi/(2k)
        # where len(ps)=4k, the configuration is a regular 4k-gon — which
        # partitions into k quartets, each with 90 deg spacing.
        n = len(ps_sorted)
        angles = [_polar(p)[1] for p in ps_sorted]
        gaps = [
            (angles[(i + 1) % n] - angles[i]) % (2 * math.pi)
            for i in range(n)
        ]
        expected_gap = 2 * math.pi / n
        if all(abs(g - expected_gap) < 1e-2 for g in gaps):
            n_quartet_planets += n
    return n_clusters, n_quartet_planets, len(planets)


def _nearest_neighbour_distances(planets) -> list[float]:
    dists = []
    for i, p in enumerate(planets):
        best = float("inf")
        for j, q in enumerate(planets):
            if i == j:
                continue
            d = math.hypot(p[2] - q[2], p[3] - q[3])
            if d < best:
                best = d
        dists.append(best)
    return dists


def _sun_blocked_pair_fraction(planets) -> float:
    """% of unordered planet pairs whose straight chord clips the sun."""
    if len(planets) < 2:
        return 0.0
    blocked = 0
    total = 0
    for i, p in enumerate(planets):
        for q in planets[i + 1:]:
            total += 1
            d = point_to_segment_distance(
                (CENTER, CENTER), (p[2], p[3]), (q[2], q[3]),
            )
            if d <= SUN_RADIUS:
                blocked += 1
    return blocked / total if total else 0.0


def _home_planets(planets) -> list:
    """Home group = planets with owner != -1 at step 0 (10 ships each)."""
    return [p for p in planets if p[1] != -1]


def _distance_over_time(home_a, home_b, omega: float, n_turns: int = 500) -> list[float]:
    """Compute centre-of-planet distance for each turn 0..n_turns-1, taking
    orbit into account when applicable."""
    rA, thetaA = _polar(home_a)
    rB, thetaB = _polar(home_b)
    a_orbit = _is_orbiting(home_a)
    b_orbit = _is_orbiting(home_b)
    out = []
    for t in range(n_turns):
        if a_orbit:
            ax = CENTER + rA * math.cos(thetaA + omega * t)
            ay = CENTER + rA * math.sin(thetaA + omega * t)
        else:
            ax, ay = home_a[2], home_a[3]
        if b_orbit:
            bx = CENTER + rB * math.cos(thetaB + omega * t)
            by = CENTER + rB * math.sin(thetaB + omega * t)
        else:
            bx, by = home_b[2], home_b[3]
        out.append(math.hypot(ax - bx, ay - by))
    return out


def _orbit_sweep_stats(planets, my_home, omega: float, n_turns: int = 500):
    """For each ORBITING non-home planet, compute the percentage gap
    between t=0 distance from my_home and the minimum distance over the
    next n_turns turns. Returns aggregated stats per seed.
    """
    my_home_static = not _is_orbiting(my_home)
    gaps = []  # (t0 - min) / t0  in [0, 1]
    inits = []
    mins = []
    for p in planets:
        if p[0] == my_home[0]:
            continue
        if not _is_orbiting(p):
            continue
        d_series = _distance_over_time(my_home, p, omega, n_turns)
        d0 = d_series[0]
        dmin = min(d_series)
        if d0 > 1e-6:
            gaps.append((d0 - dmin) / d0)
            inits.append(d0)
            mins.append(dmin)
    return {
        "n_orbiting_targets": len(gaps),
        "mean_pct_gap": statistics.mean(gaps) if gaps else None,
        "max_pct_gap": max(gaps) if gaps else None,
        "median_pct_gap": statistics.median(gaps) if gaps else None,
        "mean_init_dist": statistics.mean(inits) if inits else None,
        "mean_min_dist": statistics.mean(mins) if mins else None,
    }


def _frontier_planets(planets, home_a, home_b, band: float = 10.0):
    """Static planets whose perpendicular distance to the home-home
    bisector is within `band` units. Bisector passes through midpoint
    perpendicular to home_a -> home_b."""
    mx = (home_a[2] + home_b[2]) / 2.0
    my = (home_a[3] + home_b[3]) / 2.0
    dx = home_b[2] - home_a[2]
    dy = home_b[3] - home_a[3]
    norm = math.hypot(dx, dy)
    if norm < 1e-9:
        return []
    # Normal vector of the home-home line is (dx, dy)/norm; bisector is
    # the line through (mx,my) perpendicular to that vector. A point P's
    # signed distance to the bisector along the direction (dx,dy) is
    # ((P-M) . (dx,dy)) / norm.
    out = []
    for p in planets:
        proj = ((p[2] - mx) * dx + (p[3] - my) * dy) / norm
        if abs(proj) <= band:
            out.append((p[0], abs(proj)))
    return sorted(out, key=lambda x: x[1])


def _comet_atlas(env, seed: int):
    """Step the env forward to step ~455 capturing comet quartets at each
    spawn step. Each `comets[i]` is a single spawn event describing 4
    comets (planet_ids list of 4 + paths list of 4 polylines).

    Returns: list of {spawn_step, n_comets_in_event, path_lengths,
    starts, ends}.
    """
    env.reset(num_agents=NUM_AGENTS)
    out = []
    next_capture_idx = 0
    while env.state[0]["status"] == "ACTIVE" and next_capture_idx < len(COMET_SPAWN_STEPS):
        env.step([[], []])
        step = env.state[0]["observation"]["step"]
        if step == COMET_SPAWN_STEPS[next_capture_idx]:
            comets = env.state[0]["observation"].get("comets", []) or []
            if comets:
                event = comets[-1]  # latest spawn event
                paths = event.get("paths") or []
                planet_ids = event.get("planet_ids") or []
                out.append({
                    "spawn_step": step,
                    "n_comets_in_event": len(planet_ids),
                    "path_lengths": [len(p) for p in paths],
                    "starts": [list(p[0]) if p else None for p in paths],
                    "ends": [list(p[-1]) if p else None for p in paths],
                })
            next_capture_idx += 1
        if step >= 500:
            break
    return out


def collect_seed(seed: int, do_comet_atlas: bool = False) -> dict:
    env = make("orbit_wars", configuration={"seed": seed})
    env.reset(num_agents=NUM_AGENTS)
    obs0 = env.state[0]["observation"]
    planets = obs0["planets"]
    omega = obs0.get("angular_velocity", 0.0)
    homes = _home_planets(planets)
    # Production distribution
    productions = [p[6] for p in planets]
    radii = [_polar(p)[0] for p in planets]
    orbiting = [_is_orbiting(p) for p in planets]
    # Quartet check
    n_clusters, n_quartet_planets, n_total = _detect_quartets(planets)
    # Nearest-neighbour
    nn = _nearest_neighbour_distances(planets)
    # Sun-blocked pair fraction
    sun_frac = _sun_blocked_pair_fraction(planets)
    # Home quadrants
    home_quads = [_quadrant(h[2], h[3]) for h in homes]
    # Distance-over-time for the home pair (need both)
    dot = None
    frontier = None
    sweep_stats = None
    if len(homes) == 2:
        dot = _distance_over_time(homes[0], homes[1], omega, n_turns=500)
        frontier = _frontier_planets(planets, homes[0], homes[1], band=10.0)
        # Pick MY home (owner == 0); compute sweep stats vs every orbiting planet.
        my_home = next((h for h in homes if h[1] == 0), homes[0])
        sweep_stats = _orbit_sweep_stats(planets, my_home, omega, n_turns=500)
    # Comet atlas only for some seeds (expensive)
    comet_atlas = _comet_atlas(env, seed) if do_comet_atlas else None
    return {
        "seed": seed,
        "n_planets": len(planets),
        "productions": productions,
        "orbital_radii": radii,
        "orbiting_flags": orbiting,
        "n_orbiting": sum(orbiting),
        "n_static": sum(1 for o in orbiting if not o),
        "omega": omega,
        "n_radius_clusters": n_clusters,
        "n_quartet_planets": n_quartet_planets,
        "n_total_planets_for_quartet_check": n_total,
        "pct_in_4fold_arrangement": n_quartet_planets / n_total if n_total else 0.0,
        "nearest_neighbour_distances": nn,
        "nn_mean": statistics.mean(nn) if nn else None,
        "nn_min": min(nn) if nn else None,
        "sun_blocked_pair_fraction": sun_frac,
        "home_quadrants": home_quads,
        "home_positions": [[h[2], h[3]] for h in homes],
        "home_distance_over_time": dot,
        "home_distance_min": min(dot) if dot else None,
        "home_distance_max": max(dot) if dot else None,
        "home_distance_t0": dot[0] if dot else None,
        "frontier_planets_band10": frontier,
        "orbit_sweep_stats": sweep_stats,
        "comet_atlas": comet_atlas,
    }


def histogram(values: list[float], n_bins: int, low: float | None = None, high: float | None = None):
    if not values:
        return []
    low = low if low is not None else min(values)
    high = high if high is not None else max(values)
    if high <= low:
        return [(low, high, len(values))]
    step = (high - low) / n_bins
    bins = [0] * n_bins
    for v in values:
        idx = min(int((v - low) / step), n_bins - 1)
        if idx >= 0:
            bins[idx] += 1
    return [(low + i * step, low + (i + 1) * step, bins[i]) for i in range(n_bins)]


def _bar(count: int, max_count: int, width: int = 40) -> str:
    if max_count <= 0:
        return ""
    return "#" * max(1, int(count / max_count * width)) if count > 0 else ""


def render_text_histogram(label: str, hist, fmt: str = "{:6.2f}-{:<6.2f}"):
    lines = [f"### {label}"]
    if not hist:
        lines.append("(empty)")
        return "\n".join(lines)
    max_count = max(h[2] for h in hist) or 1
    for lo, hi, c in hist:
        lines.append(f"{fmt.format(lo, hi)}  {c:5d}  {_bar(c, max_count)}")
    return "\n".join(lines)


def main():
    print(f"Sampling {N_SEEDS} seeds...", file=sys.stderr)
    rows = []
    comet_seeds = list(range(0, N_SEEDS, 10))  # 10 seeds with comet atlas
    for seed in range(N_SEEDS):
        try:
            row = collect_seed(seed, do_comet_atlas=(seed in comet_seeds))
        except Exception as e:
            print(f"seed {seed}: {e}", file=sys.stderr)
            continue
        rows.append(row)
        if seed % 20 == 0:
            print(f"  ...seed {seed}", file=sys.stderr)
    # Save raw data
    out_path = REPO / "audit" / "2026-05-12-battlefield-geometry-data.json"
    out_path.write_text(json.dumps(rows, default=float))
    print(f"wrote {out_path}", file=sys.stderr)

    # Aggregate stats for the markdown summary.
    all_n_planets = [r["n_planets"] for r in rows]
    all_omega = [r["omega"] for r in rows]
    all_productions = [v for r in rows for v in r["productions"]]
    all_radii = [v for r in rows for v in r["orbital_radii"]]
    n_orbiting = [r["n_orbiting"] for r in rows]
    n_static = [r["n_static"] for r in rows]
    pct_orbiting_per_seed = [
        r["n_orbiting"] / r["n_planets"] for r in rows if r["n_planets"]
    ]
    all_nn = [v for r in rows for v in r["nearest_neighbour_distances"]]
    well_formed_pct = [r["pct_in_4fold_arrangement"] for r in rows]
    sun_fracs = [r["sun_blocked_pair_fraction"] for r in rows]
    quadrant_pair_counter = Counter()
    for r in rows:
        hq = tuple(sorted(r["home_quadrants"]))
        quadrant_pair_counter[hq] += 1
    diag_home_count = sum(
        c for k, c in quadrant_pair_counter.items() if k in ((1, 3), (2, 4))
    )

    # Frontier planet counts per seed (only for 2P)
    frontier_counts = [
        len(r["frontier_planets_band10"]) for r in rows if r["frontier_planets_band10"] is not None
    ]
    home_dist_mins = [r["home_distance_min"] for r in rows if r["home_distance_min"] is not None]
    home_dist_maxs = [r["home_distance_max"] for r in rows if r["home_distance_max"] is not None]
    home_dist_t0s = [r["home_distance_t0"] for r in rows if r["home_distance_t0"] is not None]

    # Compute the sweep amplitude (max-min)/t0 per seed.
    sweep_pct = [
        (r["home_distance_max"] - r["home_distance_min"]) / r["home_distance_t0"]
        for r in rows
        if r["home_distance_t0"] and r["home_distance_t0"] > 0
    ]

    # ---- Print markdown summary ----
    def s(values):
        if not values:
            return "(empty)"
        return (
            f"n={len(values)} mean={statistics.mean(values):.3f} "
            f"min={min(values):.3f} max={max(values):.3f} "
            f"median={statistics.median(values):.3f}"
        )

    print("\n=== SUMMARY ===\n")
    print(f"## 1. Planet count per seed\n{s(all_n_planets)}")
    print(render_text_histogram(
        "histogram", histogram(all_n_planets, 10, 16, 44),
        fmt="{:5.1f}-{:<5.1f}",
    ))
    print()
    print(f"## 2. Production distribution (per-planet)\nn={len(all_productions)}")
    pc = Counter(all_productions)
    for k in sorted(pc):
        print(f"  production={k}: {pc[k]:5d}  {_bar(pc[k], max(pc.values()))}")
    print()
    print(f"## 3. Orbital radii\n{s(all_radii)}")
    print(f"orbiting/static counts per seed: orb={s(n_orbiting)} static={s(n_static)}")
    print(f"pct orbiting per seed: {s(pct_orbiting_per_seed)}")
    print(render_text_histogram(
        "orbital radius histogram",
        histogram(all_radii, 12, 0, 60),
        fmt="{:5.1f}-{:<5.1f}",
    ))
    print()
    print(f"## 4. Per-game omega\n{s(all_omega)}")
    print(render_text_histogram(
        "omega histogram", histogram(all_omega, 10, 0.02, 0.055),
        fmt="{:.4f}-{:.4f}",
    ))
    print()
    print(f"## 5. 4-fold rotational symmetry check")
    print(f"  fraction of planets in 4-fold-symmetric radius clusters: {s(well_formed_pct)}")
    bad_seeds = [r["seed"] for r in rows if r["pct_in_4fold_arrangement"] < 0.999]
    print(f"  seeds with ANY non-4-fold planet cluster: {len(bad_seeds)} (e.g., {bad_seeds[:5]})")
    print()
    print(f"## 6. Sun-blocked pair fraction per seed\n{s(sun_fracs)}")
    print(render_text_histogram(
        "sun-blocked %", histogram([100 * v for v in sun_fracs], 10, 0, 50),
        fmt="{:5.1f}-{:<5.1f}",
    ))
    print()
    print(f"## 7. Nearest-neighbour distance (all planets)\n{s(all_nn)}")
    print(render_text_histogram(
        "nn distance", histogram(all_nn, 10, 0, 40),
        fmt="{:5.1f}-{:<5.1f}",
    ))
    print()
    print(f"## 8. Home-to-enemy distance over 500 turns (homes are STATIC)")
    print(f"  min: {s(home_dist_mins)}")
    print(f"  max: {s(home_dist_maxs)}")
    print(f"  t=0: {s(home_dist_t0s)}")
    print(f"  sweep amplitude (max-min)/t0: {s(sweep_pct)}  <- expected 0; homes don't move")
    print()
    # ----- 8b: distance from MY home to each ORBITING planet -----
    sweep_means = [r["orbit_sweep_stats"]["mean_pct_gap"] for r in rows if r["orbit_sweep_stats"] and r["orbit_sweep_stats"]["mean_pct_gap"] is not None]
    sweep_max_per_seed = [r["orbit_sweep_stats"]["max_pct_gap"] for r in rows if r["orbit_sweep_stats"] and r["orbit_sweep_stats"]["max_pct_gap"] is not None]
    n_orbit_targets = [r["orbit_sweep_stats"]["n_orbiting_targets"] for r in rows if r["orbit_sweep_stats"]]
    mean_init = [r["orbit_sweep_stats"]["mean_init_dist"] for r in rows if r["orbit_sweep_stats"] and r["orbit_sweep_stats"]["mean_init_dist"] is not None]
    mean_min = [r["orbit_sweep_stats"]["mean_min_dist"] for r in rows if r["orbit_sweep_stats"] and r["orbit_sweep_stats"]["mean_min_dist"] is not None]
    print(f"## 8b. My home to every ORBITING enemy/neutral planet (closest-approach payoff)")
    print(f"  orbiting targets per seed: {s(n_orbit_targets)}")
    print(f"  mean initial distance (t=0):   {s(mean_init)}")
    print(f"  mean closest-approach distance: {s(mean_min)}")
    print(f"  per-target percent gap (t=0 - min)/t=0, averaged within seed: {s(sweep_means)}")
    print(f"  per-target percent gap, MAX within seed: {s(sweep_max_per_seed)}")
    print(render_text_histogram(
        "mean pct-gap per seed",
        histogram([100 * v for v in sweep_means], 10, 0, 80),
        fmt="{:5.1f}-{:<5.1f}",
    ))
    print()
    print(f"## 9. Home-quadrant pairs (2P)")
    for k in sorted(quadrant_pair_counter, key=quadrant_pair_counter.get, reverse=True):
        print(f"  quads {k}: {quadrant_pair_counter[k]}")
    print(f"  diagonal (Q1+Q3 or Q2+Q4): {diag_home_count}/{len(rows)} = {100*diag_home_count/len(rows):.1f}%")
    print()
    print(f"## 10. Comet atlas (10 seeds with stepped env)")
    for r in rows:
        if r["comet_atlas"]:
            n_per_window = [w["n_comets_in_event"] for w in r["comet_atlas"]]
            path_lens = [w["path_lengths"] for w in r["comet_atlas"]]
            print(f"  seed {r['seed']}: comets/event = {n_per_window}, path lengths = {path_lens}")
    print()
    print(f"## 11. Frontier planets in ±10-unit band of bisector")
    print(s(frontier_counts))
    print(render_text_histogram(
        "frontier count", histogram(frontier_counts, 8, 0, 16),
        fmt="{:5.1f}-{:<5.1f}",
    ))
    print()


if __name__ == "__main__":
    main()
