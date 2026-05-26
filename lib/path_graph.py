"""path_graph — static feasibility graph for SA admissibility.

Precomputes, for every (src_planet, tgt_planet, t_dep_bucket), the
closed-form aim/eta/arrival from src to tgt assuming departure at t_dep.
Ship cost is NOT stored (it depends on dynamic garrison state at arrival).

This is geometry-only: depends only on planet positions, orbits, comet
paths, sun. Does NOT depend on ownership, fleets, or game-state at the
moment of query.

PI 2026-05-28: cascade-aware admissibility needs cheap (src, tgt, t_dep)
lookups. Precomputing the geometry once at agent startup amortises the
~50µs aim_orbiting / aim_comet call across every SA refine. Lazy-built
on first SA invocation (deferred past Kaggle's `actTimeout=1s` turn 0).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Optional

from lib.aim import aim_comet, aim_orbiting
from lib.orbit import predict_relative
from lib.world_model import _comet_paths_by_id


@dataclass(frozen=True)
class PathEdge:
    """A geometry-feasible fleet path from src to tgt with departure at t_dep.

    `t_arr` is `t_dep + eta` (turn 0-relative). `angle` is the aim required
    at departure to intercept the orbiting / comet target on arrival.

    Ship cost is *not* part of the edge — it depends on garrison at t_arr,
    which is dynamic. The edge only certifies that fleet motion from src
    at t_dep can reach tgt at t_arr under the closed-form aim.
    """
    src_id: int
    tgt_id: int
    t_dep: int
    t_arr: int
    eta: int
    angle: float
    is_comet_target: bool


class PathGraph:
    """Lookup table of geometry-feasible edges keyed by (src_id, tgt_id).

    Edges within each (src, tgt) entry are sorted by `t_dep` ascending.
    `lookup(src, tgt, t_dep)` returns the edge whose `t_dep` is the
    largest bucket boundary `<=` the requested turn — caller can use
    that edge's angle/eta as a close approximation. With orbiting_bucket=4
    the geometric error is small (orbital phase advances slowly).
    """

    def __init__(self) -> None:
        self.edges_by_src_tgt: dict[tuple[int, int], list[PathEdge]] = {}
        self.orbiting_bucket: int = 4
        self.comet_bucket: int = 1
        self.t_max: int = 500

    def edges_from(self, src_id: int) -> Iterable[PathEdge]:
        for (s, _t), edges in self.edges_by_src_tgt.items():
            if s == int(src_id):
                for e in edges:
                    yield e

    def edges_to(self, tgt_id: int) -> Iterable[PathEdge]:
        for (_s, t), edges in self.edges_by_src_tgt.items():
            if t == int(tgt_id):
                for e in edges:
                    yield e

    def lookup(self, src_id: int, tgt_id: int,
               t_dep: int) -> Optional[PathEdge]:
        """Return the edge whose `t_dep` is the largest value <= the
        requested `t_dep`. Returns None if no such edge exists (no
        feasible aim from src to tgt at or before the requested turn)."""
        edges = self.edges_by_src_tgt.get((int(src_id), int(tgt_id)))
        if not edges:
            return None
        best: Optional[PathEdge] = None
        for e in edges:
            if e.t_dep > int(t_dep):
                break
            best = e
        return best

    def __len__(self) -> int:
        return sum(len(v) for v in self.edges_by_src_tgt.values())


def _planet_tuple(p) -> list:
    return [int(p.id), int(p.owner), float(p.x), float(p.y),
            float(p.radius), float(p.production)]


def _provisional_ships(tgt) -> int:
    """Aim functions only use `ships` to compute fleet speed (log-curve
    in lib.fleet). Within a wide ships range the eta changes by <1 turn,
    so a static provisional value gives a faithful precompute."""
    return max(1, int(getattr(tgt, "ships", 1)) + 1)


def build_path_graph(world, *, t_max: int = 500,
                     orbiting_bucket: int = 4,
                     comet_bucket: int = 1) -> PathGraph:
    """Enumerate geometry-feasible (src, tgt, t_dep) → edges from `world`.

    For every distinct pair of planets, sample `t_dep` at bucket cadence
    from `[0, t_max)`. For each sample, run closed-form aim and store
    the resulting (angle, eta, t_arr) if a valid lead exists. Comet
    targets use `comet_bucket` (finer); orbiting targets use
    `orbiting_bucket` (orbital phase advances slowly).

    Build cost on a typical 18-planet world with defaults:
        ~306 pairs × ~125 buckets ≈ 38k aim calls × ~50µs ≈ 1.9s.
    Most pairs return None quickly (sun-blocked / too far), so real
    cost is lower. Bucket up to halve cost; cost is one-time per game.
    """
    pg = PathGraph()
    pg.orbiting_bucket = int(orbiting_bucket)
    pg.comet_bucket = int(comet_bucket)
    pg.t_max = int(t_max)

    if world is None:
        return pg

    omega = float(getattr(world, "omega", 0.0))
    comet_ids = frozenset(int(c)
                           for c in (getattr(world, "comet_ids", None) or ()))
    comet_paths = _comet_paths_by_id(world) if comet_ids else {}
    planets_by_id = dict(getattr(world, "planets_by_id", {}))

    for src_id, src in planets_by_id.items():
        src_tuple = _planet_tuple(src)
        for tgt_id, tgt in planets_by_id.items():
            if int(src_id) == int(tgt_id):
                continue
            tgt_tuple = _planet_tuple(tgt)
            is_comet = int(tgt_id) in comet_ids
            bucket = comet_bucket if is_comet else orbiting_bucket
            prov_ships = _provisional_ships(tgt)
            edges: list[PathEdge] = []
            for t_dep in range(0, int(t_max), int(bucket)):
                # Source position at the departure turn
                if t_dep == 0 or omega == 0.0:
                    src_xy = (float(src.x), float(src.y))
                else:
                    src_xy = predict_relative(src_tuple, omega, int(t_dep))
                # Target position at the departure turn (input to aim)
                if is_comet and int(tgt_id) in comet_paths:
                    cpath, base_idx = comet_paths[int(tgt_id)]
                    tgt_path_idx = int(base_idx) + int(t_dep)
                    if tgt_path_idx < 0 or tgt_path_idx >= len(cpath):
                        continue  # comet exited before this t_dep
                    pt = cpath[tgt_path_idx]
                    tgt_xy = (float(pt[0]), float(pt[1]))
                    tgt_tuple_at = [tgt_tuple[0], tgt_tuple[1],
                                     tgt_xy[0], tgt_xy[1],
                                     tgt_tuple[4], tgt_tuple[5]]
                    aim_result = aim_comet(
                        src_xy, float(src.radius),
                        tgt_tuple_at, float(tgt.radius),
                        prov_ships, cpath, float(tgt_path_idx),
                    )
                else:
                    if t_dep == 0 or omega == 0.0:
                        tgt_xy = (float(tgt.x), float(tgt.y))
                    else:
                        tgt_xy = predict_relative(
                            tgt_tuple, omega, int(t_dep))
                    tgt_tuple_at = [tgt_tuple[0], tgt_tuple[1],
                                     tgt_xy[0], tgt_xy[1],
                                     tgt_tuple[4], tgt_tuple[5]]
                    aim_result = aim_orbiting(
                        src_xy, float(src.radius),
                        tgt_tuple_at, float(tgt.radius),
                        prov_ships, omega,
                    )
                if aim_result is None:
                    continue
                angle, _arrival_xy, eta_float = aim_result
                eta = max(1, int(math.ceil(float(eta_float))))
                t_arr = int(t_dep) + eta
                if t_arr >= int(t_max):
                    continue
                edges.append(PathEdge(
                    src_id=int(src_id),
                    tgt_id=int(tgt_id),
                    t_dep=int(t_dep),
                    t_arr=t_arr,
                    eta=eta,
                    angle=float(angle),
                    is_comet_target=bool(is_comet),
                ))
            if edges:
                pg.edges_by_src_tgt[(int(src_id), int(tgt_id))] = edges
    return pg
