"""Unit tests for lib/path_graph.py — the static feasibility graph.

PI 2026-05-28: cascade-aware admissibility needs the precomputed
graph; these tests gate Step 1 of the architecture redesign.
"""
from __future__ import annotations

import math
import random
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

from lib.path_graph import PathEdge, PathGraph, build_path_graph  # noqa: E402


def _world_from_seed(seed: int = 0):
    """Build a World object from a fresh env at the given seed."""
    from kaggle_environments import make
    from lib.intent import World

    env = make("orbit_wars",
               configuration={"seed": seed, "episodeSteps": 50},
               debug=False)
    env.reset(num_agents=2)
    obs0 = (env.steps[0][0]["observation"]
            if isinstance(env.steps[0][0], dict)
            else env.steps[0][0].observation)
    od: dict = {}
    for k in ("player", "step", "planets", "fleets", "comets",
              "comet_planet_ids", "angular_velocity"):
        v = obs0.get(k) if isinstance(obs0, dict) else getattr(obs0, k, None)
        if v is not None:
            od[k] = list(v) if isinstance(v, list) else v
    return World.from_obs(od)


def test_build_path_graph_returns_edges():
    """A non-degenerate world should produce at least one feasible edge."""
    world = _world_from_seed(seed=7542)
    pg = build_path_graph(world, t_max=100, orbiting_bucket=4, comet_bucket=1)
    assert len(pg) > 0, (
        f"path graph empty for a real world — geometry impossible? "
        f"planets={len(world.planets_by_id)}")
    # Every stored edge must be a PathEdge instance.
    for (src_id, tgt_id), edges in pg.edges_by_src_tgt.items():
        assert src_id != tgt_id, "self-loop in graph"
        for e in edges:
            assert isinstance(e, PathEdge)
            assert e.src_id == src_id
            assert e.tgt_id == tgt_id
            assert e.eta >= 1
            assert e.t_arr == e.t_dep + e.eta


def test_path_graph_eta_matches_aim_orbiting():
    """For every stored orbiting edge, calling aim_orbiting directly with
    the same (src_xy, src_radius, tgt_tuple_at_turn, tgt_radius, ships,
    omega) must yield an eta within 1 turn of the stored value.

    This certifies the graph faithfully records aim_orbiting's output —
    not a stale or quantised approximation.
    """
    from lib.aim import aim_orbiting
    from lib.orbit import predict_relative
    from lib.path_graph import _planet_tuple, _provisional_ships

    world = _world_from_seed(seed=7542)
    pg = build_path_graph(world, t_max=80, orbiting_bucket=4, comet_bucket=1)
    omega = float(world.omega)
    comet_ids = frozenset(int(c) for c in (world.comet_ids or ()))

    rng = random.Random(0)
    all_orbiting_edges = []
    for (src_id, tgt_id), edges in pg.edges_by_src_tgt.items():
        if int(tgt_id) in comet_ids:
            continue  # this test exercises orbiting only; comets tested via build
        for e in edges:
            all_orbiting_edges.append(e)
    if not all_orbiting_edges:
        pytest.skip("no orbiting edges produced for this seed; skip parity check")

    sample = rng.sample(all_orbiting_edges, k=min(50, len(all_orbiting_edges)))
    n_mismatch = 0
    for edge in sample:
        src = world.planets_by_id[edge.src_id]
        tgt = world.planets_by_id[edge.tgt_id]
        src_tuple = _planet_tuple(src)
        tgt_tuple = _planet_tuple(tgt)
        if edge.t_dep == 0 or omega == 0.0:
            src_xy = (float(src.x), float(src.y))
            tgt_xy = (float(tgt.x), float(tgt.y))
        else:
            src_xy = predict_relative(src_tuple, omega, edge.t_dep)
            tgt_xy = predict_relative(tgt_tuple, omega, edge.t_dep)
        tgt_tuple_at = [tgt_tuple[0], tgt_tuple[1],
                         tgt_xy[0], tgt_xy[1],
                         tgt_tuple[4], tgt_tuple[5]]
        result = aim_orbiting(src_xy, float(src.radius),
                               tgt_tuple_at, float(tgt.radius),
                               _provisional_ships(tgt), omega)
        assert result is not None, (
            f"aim_orbiting returned None for an edge that was stored: "
            f"src={edge.src_id} tgt={edge.tgt_id} t_dep={edge.t_dep}")
        _angle, _arr, eta_direct = result
        eta_direct_int = max(1, int(math.ceil(float(eta_direct))))
        if abs(eta_direct_int - edge.eta) > 1:
            n_mismatch += 1
    assert n_mismatch == 0, (
        f"{n_mismatch} of {len(sample)} edges had eta drift > 1 turn vs "
        f"direct aim_orbiting — graph stale or off-by-one")


def test_path_graph_lookup_returns_largest_bucket_le_t_dep():
    """lookup(src, tgt, t_dep) must return the edge whose t_dep is the
    largest bucket boundary <= the requested turn. This is the contract
    cascade-aware admissibility relies on.
    """
    world = _world_from_seed(seed=7542)
    pg = build_path_graph(world, t_max=60, orbiting_bucket=4, comet_bucket=1)
    if not pg.edges_by_src_tgt:
        pytest.skip("no edges produced for this seed")
    # Pick any (src, tgt) with at least 2 edges
    pair_with_edges = None
    for pair, edges in pg.edges_by_src_tgt.items():
        if len(edges) >= 2:
            pair_with_edges = (pair, edges)
            break
    if pair_with_edges is None:
        pytest.skip("no (src, tgt) pair has 2+ edges; cannot test bucket selection")
    (src_id, tgt_id), edges = pair_with_edges
    edges_sorted = sorted(edges, key=lambda e: e.t_dep)
    # Query halfway between two adjacent buckets — should return the earlier one
    e_lo, e_hi = edges_sorted[0], edges_sorted[1]
    query_turn = e_lo.t_dep + (e_hi.t_dep - e_lo.t_dep) // 2
    if query_turn == e_lo.t_dep:
        query_turn = e_lo.t_dep + 1  # ensure strictly between
    if query_turn >= e_hi.t_dep:
        pytest.skip("buckets too tight to test bucket selection")
    hit = pg.lookup(src_id, tgt_id, query_turn)
    assert hit is not None
    assert hit.t_dep == e_lo.t_dep, (
        f"expected bucket t_dep={e_lo.t_dep} for query={query_turn}, "
        f"got t_dep={hit.t_dep}")
    # Query before any bucket — should return None
    if e_lo.t_dep > 0:
        miss = pg.lookup(src_id, tgt_id, e_lo.t_dep - 1)
        assert miss is None
    # Query at a bucket boundary — should return that bucket
    exact = pg.lookup(src_id, tgt_id, e_lo.t_dep)
    assert exact is not None and exact.t_dep == e_lo.t_dep
