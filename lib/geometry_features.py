"""Geometry feature extraction for the seed-panel pipeline.

Single entry point: ``extract_geometry(seed)`` runs ``env.reset()`` on the
official ``orbit_wars`` interpreter and returns a flat feature dict
covering the axes the PI named (sparse/dense, rotating/static split,
size split, production, angular velocity, home-planet exposure).

Consumed by ``scripts/build_seed_panel.py``.
"""

from __future__ import annotations

import math
from typing import Any

from kaggle_environments import make

from .geometry import CENTER
from .orbit import is_orbiting


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _nearest_neighbor_mean(planets: list) -> float:
    if len(planets) < 2:
        return 0.0
    coords = [(p[2], p[3]) for p in planets]
    nn = []
    for i, (xi, yi) in enumerate(coords):
        best = math.inf
        for j, (xj, yj) in enumerate(coords):
            if i == j:
                continue
            d = math.hypot(xi - xj, yi - yj)
            if d < best:
                best = d
        nn.append(best)
    return _mean(nn)


def extract_geometry(seed: int) -> dict[str, Any]:
    """Initialise a fresh orbit_wars game on ``seed`` and read turn-0 state.

    Returns a flat dict of geometry features (no nested structures, all
    numeric / bool / scalar) so it can be loaded into a pandas DataFrame
    or a small JSON record without extra parsing.
    """
    env = make("orbit_wars", configuration={"seed": seed}, debug=False)
    env.reset()
    obs = env.state[0].observation

    planets = list(obs.planets)
    n_planets = len(planets)

    rotating = [p for p in planets if is_orbiting(p)]
    static = [p for p in planets if not is_orbiting(p)]
    n_rotating = len(rotating)
    n_static = len(static)

    prods = [p[6] for p in planets]
    radii = [p[4] for p in planets]
    prods_rot = [p[6] for p in rotating]
    prods_stat = [p[6] for p in static]
    radii_rot = [p[4] for p in rotating]
    radii_stat = [p[4] for p in static]

    total_production = sum(prods)

    # Home planet for P0 — owner index 0 (the env always assigns one to P0).
    home = next((p for p in planets if p[1] == 0), None)
    if home is None:
        home_orbital_radius = 0.0
        home_is_rotating = False
    else:
        home_orbital_radius = math.hypot(home[2] - CENTER, home[3] - CENTER)
        home_is_rotating = is_orbiting(home)

    return {
        "seed": seed,
        "n_planets": n_planets,
        "n_groups": n_planets // 4,
        "n_rotating": n_rotating,
        "n_static": n_static,
        "rotating_share": n_rotating / n_planets if n_planets else 0.0,
        "total_production": total_production,
        "production_rotating_share": sum(prods_rot) / total_production
        if total_production
        else 0.0,
        "production_static_share": sum(prods_stat) / total_production
        if total_production
        else 0.0,
        "radius_mean": _mean(radii),
        "radius_max": max(radii) if radii else 0.0,
        "radius_rotating_mean": _mean(radii_rot),
        "radius_static_mean": _mean(radii_stat),
        "size_split": _mean(radii_rot) - _mean(radii_stat),
        "angular_velocity": float(obs.angular_velocity),
        "home_orbital_radius": home_orbital_radius,
        "home_is_rotating": home_is_rotating,
        "nearest_neighbor_mean": _nearest_neighbor_mean(planets),
    }
