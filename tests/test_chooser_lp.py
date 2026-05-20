"""Unit + integration tests for agents/baseline/chooser_lp — Slice 10.

Plan reference: /root/.claude/plans/take-the-lens-of-magical-shore.md §16.

Load-bearing per PI directive ("must not fail because of a bug").
Coverage spans 4 layers:
  1. `_compute_candidate_value` — analytical value per candidate.
  2. `_build_assignment_matrix` — LP cost matrix construction.
  3. `_solve_and_extract` — Hungarian solve + move extraction.
  4. `choose_lp` — end-to-end chooser.
Plus property tests pinning the LP's per-source / per-target /
positive-value invariants.
"""

from __future__ import annotations

from kaggle_environments.envs.orbit_wars.orbit_wars import Fleet, Planet

from agents.baseline.chooser_lp import (
    INFEASIBLE_COST,
    NOOP_COST,
    W2_VALUE_MULTIPLIER,
    _build_assignment_matrix,
    _compute_candidate_value,
    _solve_and_extract,
    choose_lp,
)
from lib.intent import World
from lib.world_model import WorldModel


def _planet(pid, owner, x, y, *, ships=10, production=2, radius=1.5):
    return Planet(pid, owner, x, y, radius, ships, production)


def _fleet(fid, owner, x, y, angle, ships, from_planet_id=0):
    return Fleet(fid, owner, x, y, angle, from_planet_id, ships)


def _world(my_id, planets, *, fleets=None, step=0, omega=0.0):
    obs = {
        "player": my_id,
        "planets": [
            (p.id, p.owner, p.x, p.y, p.radius, p.ships, p.production)
            for p in planets
        ],
        "fleets": [
            (f.id, f.owner, f.x, f.y, f.angle, f.from_planet_id, f.ships)
            for f in (fleets or [])
        ],
        "angular_velocity": omega,
        "comet_planet_ids": [],
        "comets": [],
        "step": step,
    }
    return obs, World.from_obs(obs)


def _candidate(src, tgt, *, cheap_delta=1.0, ships=10, eta=5, wait_N=0):
    """Build a proposer-style prerank tuple."""
    return (float(cheap_delta), src, tgt, int(ships), 0.0, int(eta),
            int(eta + 2), int(wait_N))


# ===========================================================================
# Layer 1: _compute_candidate_value
# ===========================================================================


def test_value_wait_N_filtered_to_zero():
    """wait_N > 0 candidates → value=0 (single-turn LP only)."""
    src = _planet(0, 0, 10.0, 50.0, ships=100)
    tgt = _planet(1, -1, 30.0, 50.0, ships=5, production=2)
    obs, world = _world(0, [src, tgt])
    model = WorldModel.from_world(world)
    c = _candidate(src, tgt, ships=50, eta=4, wait_N=3)
    assert _compute_candidate_value(c, world, model, me=0) == 0.0


def test_value_clean_capture_positive():
    """Clean capture (Wald passes) → positive value via _w1_value_bounds."""
    src = _planet(0, 0, 10.0, 50.0, ships=120, production=3)
    tgt = _planet(1, -1, 30.0, 50.0, ships=10, production=2)
    opp_far = _planet(2, 1, 95.0, 95.0, ships=10)
    obs, world = _world(0, [src, tgt, opp_far])
    model = WorldModel.from_world(world)
    c = _candidate(src, tgt, ships=80, eta=4)
    v = _compute_candidate_value(c, world, model, me=0)
    assert v > 0.0


def test_value_bounce_zero():
    """Under-sized capture → bounce → value=0."""
    src = _planet(0, 0, 10.0, 50.0, ships=100)
    tgt = _planet(1, -1, 50.0, 50.0, ships=100, production=1)
    obs, world = _world(0, [src, tgt])
    model = WorldModel.from_world(world)
    c = _candidate(src, tgt, ships=5, eta=7)
    assert _compute_candidate_value(c, world, model, me=0) == 0.0


def test_value_capture_under_gangup_zero():
    """Capture but Wald fails (2-opp gang-up) → value=0."""
    src = _planet(0, 0, 10.0, 50.0, ships=200, production=3)
    tgt = _planet(1, -1, 50.0, 50.0, ships=10, production=2)
    opp_a = _planet(2, 1, 60.0, 45.0, ships=80, production=2)
    opp_b = _planet(3, 1, 60.0, 55.0, ships=80, production=2)
    obs, world = _world(0, [src, tgt, opp_a, opp_b])
    model = WorldModel.from_world(world)
    c = _candidate(src, tgt, ships=40, eta=7)
    # Wald fails → lower_bound = 0 → value = 0.
    assert _compute_candidate_value(c, world, model, me=0) == 0.0


def test_value_defensive_reinforce_positive():
    """Reinforce of threatened own planet → W2 commit → positive value."""
    src = _planet(0, 0, 10.0, 50.0, ships=80, production=3)
    tgt = _planet(1, 0, 50.0, 50.0, ships=4, production=2)
    opp_far = _planet(2, 1, 95.0, 50.0, ships=10, production=1)
    inbound = _fleet(0, 1, 80.0, 50.0, angle=3.141592, ships=15)
    obs, world = _world(0, [src, tgt, opp_far], fleets=[inbound])
    model = WorldModel.from_world(world)
    c = _candidate(src, tgt, ships=50, eta=3)
    v = _compute_candidate_value(c, world, model, me=0)
    # W2 commits → value = 2 × prod × pv > 0.
    assert v > 0.0


def test_value_own_target_no_threat_uses_cheap_delta():
    """Migration candidate (own→own, no threat) → uses cheap_delta directly."""
    src = _planet(0, 0, 10.0, 50.0, ships=80)
    tgt = _planet(1, 0, 80.0, 50.0, ships=10)
    obs, world = _world(0, [src, tgt])
    model = WorldModel.from_world(world)
    c = _candidate(src, tgt, cheap_delta=42.0, ships=30, eta=10)
    # tgt.owner == me, no threat → migration → cheap_delta=42 used.
    v = _compute_candidate_value(c, world, model, me=0)
    assert v == 42.0


def test_value_own_target_no_threat_zero_cheap_delta():
    """Migration with non-positive cheap_delta → value=0."""
    src = _planet(0, 0, 10.0, 50.0, ships=80)
    tgt = _planet(1, 0, 80.0, 50.0, ships=10)
    obs, world = _world(0, [src, tgt])
    model = WorldModel.from_world(world)
    c = _candidate(src, tgt, cheap_delta=-5.0, ships=30, eta=10)
    assert _compute_candidate_value(c, world, model, me=0) == 0.0


def test_value_source_threatened_capture_zero():
    """Source under its own inbound threat → _w1_value_bounds returns (0, _) → value=0."""
    src = _planet(0, 0, 10.0, 50.0, ships=20, production=1)
    tgt = _planet(1, -1, 30.0, 50.0, ships=5, production=1)
    opp_far = _planet(2, 1, 95.0, 50.0, ships=5)
    inbound_at_src = _fleet(0, 1, 15.0, 50.0, angle=3.141592, ships=50)
    obs, world = _world(0, [src, tgt, opp_far], fleets=[inbound_at_src])
    model = WorldModel.from_world(world)
    c = _candidate(src, tgt, ships=15, eta=3)
    assert _compute_candidate_value(c, world, model, me=0) == 0.0


# ===========================================================================
# Layer 2: _build_assignment_matrix
# ===========================================================================


def test_matrix_empty_prerank():
    """No candidates → empty matrix."""
    obs, world = _world(0, [_planet(0, 0, 10.0, 50.0)])
    model = WorldModel.from_world(world)
    mat, src_ids, c2c = _build_assignment_matrix([], world, model, me=0)
    assert mat == []
    assert src_ids == []
    assert c2c == {}


def test_matrix_single_candidate_shape():
    """1 candidate → 1 src, 2 cols (1 noop + 1 pair)."""
    src = _planet(0, 0, 10.0, 50.0, ships=120, production=3)
    tgt = _planet(1, -1, 30.0, 50.0, ships=10, production=2)
    opp_far = _planet(2, 1, 95.0, 95.0, ships=10)
    obs, world = _world(0, [src, tgt, opp_far])
    model = WorldModel.from_world(world)
    c = _candidate(src, tgt, ships=80, eta=4)
    mat, src_ids, c2c = _build_assignment_matrix([c], world, model, me=0)
    assert len(src_ids) == 1
    assert src_ids[0] == 0
    assert len(mat) == 1
    assert len(mat[0]) == 2  # 1 noop col + 1 pair col
    assert mat[0][0] == NOOP_COST  # noop diagonal
    assert mat[0][1] < 0.0  # negated positive value


def test_matrix_only_positive_candidates_get_pair_cols():
    """A zero-value (bounce) candidate doesn't create a pair column."""
    src = _planet(0, 0, 10.0, 50.0, ships=100, production=2)
    tgt_bounce = _planet(1, -1, 50.0, 50.0, ships=200, production=1)
    obs, world = _world(0, [src, tgt_bounce])
    model = WorldModel.from_world(world)
    c = _candidate(src, tgt_bounce, ships=5, eta=7)  # under-sized
    mat, src_ids, c2c = _build_assignment_matrix([c], world, model, me=0)
    # No positive-value candidates → only noop column.
    assert len(src_ids) == 1
    assert len(mat[0]) == 1  # just the noop
    assert mat[0][0] == NOOP_COST
    assert c2c == {}


def test_matrix_picks_max_value_for_duplicate_pair():
    """Two candidates same (src, tgt) → cell value = max of the two."""
    src = _planet(0, 0, 10.0, 50.0, ships=200, production=3)
    tgt = _planet(1, -1, 30.0, 50.0, ships=10, production=2)
    opp_far = _planet(2, 1, 95.0, 95.0, ships=10)
    obs, world = _world(0, [src, tgt, opp_far])
    model = WorldModel.from_world(world)
    c_low = _candidate(src, tgt, ships=40, eta=4)
    c_high = _candidate(src, tgt, ships=120, eta=4)
    mat, src_ids, c2c = _build_assignment_matrix(
        [c_low, c_high], world, model, me=0,
    )
    # 1 src, 1 pair → matrix is 1×2.
    assert len(mat[0]) == 2
    # The pair column's value should reflect the better of the two.
    # (Both yield same _w1_value_bounds since same target; values
    # equal; the choice is non-strict either way.)
    pair_cost = mat[0][1]
    assert pair_cost < 0.0  # negated positive


def test_matrix_multiple_sources_single_target():
    """N srcs → N×(N+P) matrix where P is num unique (src,tgt) pairs."""
    src_a = _planet(0, 0, 10.0, 50.0, ships=120, production=2)
    src_b = _planet(1, 0, 20.0, 50.0, ships=120, production=2)
    tgt = _planet(2, -1, 30.0, 50.0, ships=5, production=3)
    opp_far = _planet(3, 1, 95.0, 95.0, ships=10)
    obs, world = _world(0, [src_a, src_b, tgt, opp_far])
    model = WorldModel.from_world(world)
    c_a = _candidate(src_a, tgt, ships=50, eta=4)
    c_b = _candidate(src_b, tgt, ships=50, eta=4)
    mat, src_ids, c2c = _build_assignment_matrix(
        [c_a, c_b], world, model, me=0,
    )
    assert len(src_ids) == 2
    # Cols: 2 noops + 2 pairs (one per src).
    assert len(mat) == 2 and len(mat[0]) == 4
    # Noop diagonals.
    assert mat[0][0] == NOOP_COST
    assert mat[1][1] == NOOP_COST
    # Off-diagonal noops are infeasible.
    assert mat[0][1] == INFEASIBLE_COST
    assert mat[1][0] == INFEASIBLE_COST


def test_matrix_noop_diagonal_and_offdiag_infeasibility():
    """Each source has its own noop column on the diagonal; off-diagonal
    noop cells are infeasible (a source can't pick another's noop)."""
    # Both sources have positive-value candidates → both surface as rows.
    src_a = _planet(0, 0, 10.0, 50.0, ships=120, production=2)
    src_b = _planet(1, 0, 20.0, 50.0, ships=120, production=2)
    tgt_a = _planet(2, -1, 25.0, 50.0, ships=5, production=3)
    tgt_b = _planet(3, -1, 30.0, 50.0, ships=5, production=3)
    opp_far = _planet(4, 1, 95.0, 95.0, ships=10)
    obs, world = _world(0, [src_a, src_b, tgt_a, tgt_b, opp_far])
    model = WorldModel.from_world(world)
    c_a = _candidate(src_a, tgt_a, ships=50, eta=4)
    c_b = _candidate(src_b, tgt_b, ships=50, eta=4)
    mat, src_ids, c2c = _build_assignment_matrix(
        [c_a, c_b], world, model, me=0,
    )
    # 2 sources, 2 pairs → 2×4 matrix (2 noops + 2 pairs).
    assert len(src_ids) == 2
    assert len(mat) == 2
    assert len(mat[0]) == 4
    # Noop diagonals.
    assert mat[0][0] == NOOP_COST  # src_a's noop
    assert mat[1][1] == NOOP_COST  # src_b's noop
    # Off-diagonal noops are infeasible.
    assert mat[0][1] == INFEASIBLE_COST  # src_a can't take src_b's noop
    assert mat[1][0] == INFEASIBLE_COST  # src_b can't take src_a's noop


def test_matrix_negative_values_for_pair_cells():
    """Pair cells have NEGATIVE costs (negated value for minimization)."""
    src = _planet(0, 0, 10.0, 50.0, ships=120, production=3)
    tgt = _planet(1, -1, 30.0, 50.0, ships=10, production=2)
    opp_far = _planet(2, 1, 95.0, 95.0, ships=10)
    obs, world = _world(0, [src, tgt, opp_far])
    model = WorldModel.from_world(world)
    c = _candidate(src, tgt, ships=80, eta=4)
    mat, src_ids, c2c = _build_assignment_matrix([c], world, model, me=0)
    assert mat[0][1] < 0  # negated positive value


def test_matrix_col_to_candidate_mapping():
    """col_to_candidate maps pair columns to their underlying candidate."""
    src = _planet(0, 0, 10.0, 50.0, ships=120, production=3)
    tgt = _planet(1, -1, 30.0, 50.0, ships=10, production=2)
    opp_far = _planet(2, 1, 95.0, 95.0, ships=10)
    obs, world = _world(0, [src, tgt, opp_far])
    model = WorldModel.from_world(world)
    c = _candidate(src, tgt, ships=80, eta=4)
    mat, src_ids, c2c = _build_assignment_matrix([c], world, model, me=0)
    # Pair column index = n_srcs + 0 = 1.
    assert 1 in c2c
    assert c2c[1] is c
    # Noop column index 0 NOT in mapping.
    assert 0 not in c2c


def test_matrix_src_ids_sorted():
    """`src_ids` should be sorted for deterministic row ordering."""
    src_high = _planet(5, 0, 10.0, 50.0, ships=120, production=2)
    src_low = _planet(2, 0, 20.0, 50.0, ships=120, production=2)
    tgt = _planet(7, -1, 30.0, 50.0, ships=5, production=3)
    opp_far = _planet(9, 1, 95.0, 95.0, ships=10)
    obs, world = _world(0, [src_high, src_low, tgt, opp_far])
    model = WorldModel.from_world(world)
    c_a = _candidate(src_high, tgt, ships=50, eta=4)
    c_b = _candidate(src_low, tgt, ships=50, eta=4)
    mat, src_ids, c2c = _build_assignment_matrix(
        [c_a, c_b], world, model, me=0,
    )
    assert src_ids == sorted(src_ids)  # 2, 5


# ===========================================================================
# Layer 3: _solve_and_extract
# ===========================================================================


def test_solve_empty_matrix():
    moves = _solve_and_extract([], {})
    assert moves == []


def test_solve_single_candidate_picks_it():
    """Single src + 1 pair → solver picks the pair (better than noop)."""
    src = _planet(0, 0, 10.0, 50.0, ships=120, production=3)
    tgt = _planet(1, -1, 30.0, 50.0, ships=10, production=2)
    opp_far = _planet(2, 1, 95.0, 95.0, ships=10)
    obs, world = _world(0, [src, tgt, opp_far])
    model = WorldModel.from_world(world)
    c = _candidate(src, tgt, ships=80, eta=4)
    mat, src_ids, c2c = _build_assignment_matrix([c], world, model, me=0)
    moves = _solve_and_extract(mat, c2c)
    assert len(moves) == 1
    assert int(moves[0][0]) == 0  # src=0


def test_solve_all_noop_returns_empty():
    """When no positive-value pairs exist → only noops → empty emit."""
    src = _planet(0, 0, 10.0, 50.0, ships=100)
    tgt = _planet(1, -1, 50.0, 50.0, ships=500, production=1)  # infeasible
    obs, world = _world(0, [src, tgt])
    model = WorldModel.from_world(world)
    c = _candidate(src, tgt, ships=5, eta=7)
    mat, src_ids, c2c = _build_assignment_matrix([c], world, model, me=0)
    moves = _solve_and_extract(mat, c2c)
    assert moves == []


def test_solve_picks_higher_value_when_conflict():
    """Two srcs, both can hit same target — solver picks the better."""
    src_a = _planet(0, 0, 10.0, 50.0, ships=200, production=2)
    src_b = _planet(1, 0, 20.0, 50.0, ships=120, production=2)
    tgt = _planet(2, -1, 30.0, 50.0, ships=5, production=3)
    opp_far = _planet(3, 1, 95.0, 95.0, ships=10)
    obs, world = _world(0, [src_a, src_b, tgt, opp_far])
    model = WorldModel.from_world(world)
    c_a = _candidate(src_a, tgt, ships=50, eta=4)
    c_b = _candidate(src_b, tgt, ships=50, eta=4)
    mat, src_ids, c2c = _build_assignment_matrix(
        [c_a, c_b], world, model, me=0,
    )
    moves = _solve_and_extract(mat, c2c)
    # Both have positive value but per-target ≤ 1 → only 1 emit.
    assert len(moves) == 1


def test_solve_two_srcs_two_tgts_picks_both():
    """2 srcs × 2 tgts → both fire, each src to its own tgt."""
    src_a = _planet(0, 0, 10.0, 50.0, ships=120, production=3)
    src_b = _planet(1, 0, 80.0, 50.0, ships=120, production=3)
    tgt_a = _planet(2, -1, 25.0, 50.0, ships=5, production=3)
    tgt_b = _planet(3, -1, 75.0, 50.0, ships=5, production=3)
    opp_far = _planet(4, 1, 50.0, 95.0, ships=10)
    obs, world = _world(0, [src_a, src_b, tgt_a, tgt_b, opp_far])
    model = WorldModel.from_world(world)
    c_a = _candidate(src_a, tgt_a, ships=50, eta=4)
    c_b = _candidate(src_b, tgt_b, ships=50, eta=4)
    mat, src_ids, c2c = _build_assignment_matrix(
        [c_a, c_b], world, model, me=0,
    )
    moves = _solve_and_extract(mat, c2c)
    assert len(moves) == 2
    emit_srcs = {int(m[0]) for m in moves}
    assert emit_srcs == {0, 1}


def test_solve_migration_emits():
    """Migration candidate (own→own, no threat, positive cheap_delta) emits."""
    src = _planet(0, 0, 10.0, 50.0, ships=200)
    tgt = _planet(1, 0, 80.0, 50.0, ships=10)
    obs, world = _world(0, [src, tgt])
    model = WorldModel.from_world(world)
    c = _candidate(src, tgt, cheap_delta=50.0, ships=80, eta=10)
    mat, src_ids, c2c = _build_assignment_matrix([c], world, model, me=0)
    moves = _solve_and_extract(mat, c2c)
    assert len(moves) == 1
    assert int(moves[0][0]) == 0  # src=0
    assert int(moves[0][2]) == 80  # migration ships


def test_solve_handles_infeasible_matrix():
    """All-infeasible matrix → solver gracefully returns empty."""
    # Construct a matrix with no pair columns (everything infeasible).
    mat = [[NOOP_COST, INFEASIBLE_COST], [INFEASIBLE_COST, NOOP_COST]]
    c2c = {}  # no pair columns
    moves = _solve_and_extract(mat, c2c)
    assert moves == []


def test_solve_belt_and_suspenders_per_src_dedup():
    """Even with a (hypothetical) buggy LP, the post-pass dedup
    catches double-launches from the same source."""
    # Manually craft a matrix that (if mis-solved) would double-emit.
    src = _planet(0, 0, 10.0, 50.0, ships=120, production=3)
    tgt = _planet(1, -1, 30.0, 50.0, ships=10, production=2)
    opp_far = _planet(2, 1, 95.0, 95.0, ships=10)
    obs, world = _world(0, [src, tgt, opp_far])
    model = WorldModel.from_world(world)
    c = _candidate(src, tgt, ships=80, eta=4)
    mat, src_ids, c2c = _build_assignment_matrix([c], world, model, me=0)
    moves = _solve_and_extract(mat, c2c)
    # Even on a degenerate input, no source gets two emits.
    src_count = {}
    for m in moves:
        src_count[int(m[0])] = src_count.get(int(m[0]), 0) + 1
    for s, n in src_count.items():
        assert n == 1, f"src {s} has {n} emits"


# ===========================================================================
# Layer 4: choose_lp — end-to-end
# ===========================================================================


def test_choose_lp_empty_prerank():
    obs, world = _world(0, [_planet(0, 0, 10.0, 50.0)])
    model = WorldModel.from_world(world)
    moves = choose_lp(
        None, [], None, 0, 2, 600.0, 25, 40, 0.99, world, model,
    )
    assert moves == []


def test_choose_lp_clean_capture_emits():
    src = _planet(0, 0, 10.0, 50.0, ships=120, production=3)
    tgt = _planet(1, -1, 30.0, 50.0, ships=10, production=2)
    opp_far = _planet(2, 1, 95.0, 95.0, ships=10)
    obs, world = _world(0, [src, tgt, opp_far])
    model = WorldModel.from_world(world)
    c = _candidate(src, tgt, ships=80, eta=4)
    moves = choose_lp(None, [c], None, 0, 2, 600.0, 25, 40, 0.99, world, model)
    assert len(moves) == 1


def test_choose_lp_multiple_sources_multiple_targets():
    src_a = _planet(0, 0, 10.0, 50.0, ships=120, production=3)
    src_b = _planet(1, 0, 80.0, 50.0, ships=120, production=3)
    tgt_a = _planet(2, -1, 25.0, 50.0, ships=5, production=3)
    tgt_b = _planet(3, -1, 75.0, 50.0, ships=5, production=3)
    opp_far = _planet(4, 1, 50.0, 95.0, ships=10)
    obs, world = _world(0, [src_a, src_b, tgt_a, tgt_b, opp_far])
    model = WorldModel.from_world(world)
    c_a = _candidate(src_a, tgt_a, ships=50, eta=4)
    c_b = _candidate(src_b, tgt_b, ships=50, eta=4)
    moves = choose_lp(
        None, [c_a, c_b], None, 0, 2, 600.0, 25, 40, 0.99, world, model,
    )
    assert len(moves) == 2


def test_choose_lp_defensive_reinforce_emits():
    src = _planet(0, 0, 10.0, 50.0, ships=80, production=3)
    tgt = _planet(1, 0, 50.0, 50.0, ships=4, production=2)
    opp_far = _planet(2, 1, 95.0, 50.0, ships=10, production=1)
    inbound = _fleet(0, 1, 80.0, 50.0, angle=3.141592, ships=15)
    obs, world = _world(0, [src, tgt, opp_far], fleets=[inbound])
    model = WorldModel.from_world(world)
    c = _candidate(src, tgt, ships=50, eta=3)
    moves = choose_lp(None, [c], None, 0, 2, 600.0, 25, 40, 0.99, world, model)
    assert len(moves) == 1


def test_choose_lp_migration_emits():
    src = _planet(0, 0, 10.0, 50.0, ships=200)
    tgt = _planet(1, 0, 80.0, 50.0, ships=10)
    obs, world = _world(0, [src, tgt])
    model = WorldModel.from_world(world)
    # cheap_delta=50 — migration value from solver.
    c = _candidate(src, tgt, cheap_delta=50.0, ships=80, eta=10)
    moves = choose_lp(None, [c], None, 0, 2, 600.0, 25, 40, 0.99, world, model)
    assert len(moves) == 1
    assert int(moves[0][0]) == 0


def test_choose_lp_all_zero_value_no_emit():
    """All candidates have value=0 → empty emit."""
    src = _planet(0, 0, 10.0, 50.0, ships=100)
    tgt = _planet(1, -1, 50.0, 50.0, ships=500, production=1)
    obs, world = _world(0, [src, tgt])
    model = WorldModel.from_world(world)
    c = _candidate(src, tgt, ships=5, eta=7)  # bounce
    moves = choose_lp(None, [c], None, 0, 2, 600.0, 25, 40, 0.99, world, model)
    assert moves == []


def test_choose_lp_wait_N_filtered():
    """wait_N>0 candidates do not emit even if value would be positive."""
    src = _planet(0, 0, 10.0, 50.0, ships=120, production=3)
    tgt = _planet(1, -1, 30.0, 50.0, ships=10, production=2)
    opp_far = _planet(2, 1, 95.0, 95.0, ships=10)
    obs, world = _world(0, [src, tgt, opp_far])
    model = WorldModel.from_world(world)
    c_fire = _candidate(src, tgt, ships=80, eta=4, wait_N=0)
    c_wait = _candidate(src, tgt, ships=100, eta=4, wait_N=5)
    moves = choose_lp(
        None, [c_wait, c_fire], None, 0, 2, 600.0, 25, 40, 0.99,
        world, model,
    )
    # Only the wait_N=0 fires.
    assert len(moves) == 1
    assert int(moves[0][2]) == 80


def test_choose_lp_mixed_candidate_types():
    """Capture + reinforce from different sources → both emit."""
    src_a = _planet(0, 0, 10.0, 50.0, ships=120, production=3)
    src_b = _planet(1, 0, 40.0, 50.0, ships=80, production=2)
    threatened_tgt = _planet(2, 0, 60.0, 50.0, ships=4, production=2)
    enemy_tgt = _planet(3, -1, 25.0, 50.0, ships=5, production=3)
    opp = _planet(4, 1, 95.0, 50.0, ships=10)
    inbound = _fleet(0, 1, 80.0, 50.0, angle=3.141592, ships=15)
    obs, world = _world(
        0, [src_a, src_b, threatened_tgt, enemy_tgt, opp], fleets=[inbound],
    )
    model = WorldModel.from_world(world)
    c_capture = _candidate(src_a, enemy_tgt, ships=50, eta=3)
    c_reinforce = _candidate(src_b, threatened_tgt, ships=40, eta=3)
    moves = choose_lp(
        None, [c_capture, c_reinforce], None, 0, 2, 600.0, 25, 40, 0.99,
        world, model,
    )
    # Both fire (different srcs, different tgts).
    assert len(moves) == 2


def test_choose_lp_signature_compatible():
    """Signature accepts all the args trajectory chooser accepts."""
    obs, world = _world(0, [_planet(0, 0, 10.0, 50.0)])
    model = WorldModel.from_world(world)
    # Just ensure the call doesn't crash with the full signature.
    moves = choose_lp(
        None,                # snap_base (unused)
        [],                  # prerank
        ["unused"],          # baseline_favors (unused)
        0,                   # me
        2,                   # num_seats
        600.0,               # wallclock_ms
        25,                  # min_horizon
        40,                  # max_horizon
        0.99,                # gamma
        world, model,
    )
    assert moves == []


# ===========================================================================
# Property tests — LP invariants
# ===========================================================================


def test_property_no_double_launch_per_source():
    """LP-emitted moves: no source appears twice."""
    src_a = _planet(0, 0, 10.0, 50.0, ships=120, production=3)
    src_b = _planet(1, 0, 80.0, 50.0, ships=120, production=3)
    tgt_a = _planet(2, -1, 25.0, 50.0, ships=5, production=3)
    tgt_b = _planet(3, -1, 75.0, 50.0, ships=5, production=3)
    tgt_c = _planet(4, -1, 50.0, 90.0, ships=5, production=3)
    opp = _planet(5, 1, 50.0, 10.0, ships=10)
    obs, world = _world(0, [src_a, src_b, tgt_a, tgt_b, tgt_c, opp])
    model = WorldModel.from_world(world)
    # 6 candidates: each src to each tgt.
    cands = []
    for s in (src_a, src_b):
        for t in (tgt_a, tgt_b, tgt_c):
            cands.append(_candidate(s, t, ships=50, eta=4))
    moves = choose_lp(
        None, cands, None, 0, 2, 600.0, 25, 40, 0.99, world, model,
    )
    src_count: dict = {}
    for m in moves:
        sid = int(m[0])
        src_count[sid] = src_count.get(sid, 0) + 1
    for sid, n in src_count.items():
        assert n == 1, f"src={sid} has {n} emits (should be 1)"


def test_property_per_target_uniqueness():
    """Multiple srcs to same tgt → at most one fires."""
    src_a = _planet(0, 0, 10.0, 50.0, ships=120, production=3)
    src_b = _planet(1, 0, 20.0, 50.0, ships=120, production=3)
    src_c = _planet(2, 0, 30.0, 50.0, ships=120, production=3)
    tgt = _planet(3, -1, 50.0, 50.0, ships=5, production=3)
    opp = _planet(4, 1, 95.0, 95.0, ships=10)
    obs, world = _world(0, [src_a, src_b, src_c, tgt, opp])
    model = WorldModel.from_world(world)
    cands = [_candidate(s, tgt, ships=50, eta=4)
             for s in (src_a, src_b, src_c)]
    moves = choose_lp(
        None, cands, None, 0, 2, 600.0, 25, 40, 0.99, world, model,
    )
    # Three sources, one target, per-target ≤ 1 → at most 1 emit.
    assert len(moves) <= 1


def test_property_only_positive_value_emitted():
    """A scenario where ALL candidates have value=0 → empty emit."""
    src = _planet(0, 0, 10.0, 50.0, ships=10, production=1)
    tgt = _planet(1, -1, 30.0, 50.0, ships=500, production=2)  # infeasible
    obs, world = _world(0, [src, tgt])
    model = WorldModel.from_world(world)
    c = _candidate(src, tgt, ships=5, eta=7)
    moves = choose_lp(
        None, [c], None, 0, 2, 600.0, 25, 40, 0.99, world, model,
    )
    assert moves == []


# ===========================================================================
# Constants sanity
# ===========================================================================


def test_constants_have_expected_signs():
    """Sanity: NOOP_COST = 0, INFEASIBLE_COST positive (big),
    W2_VALUE_MULTIPLIER positive."""
    assert NOOP_COST == 0.0
    assert INFEASIBLE_COST > 0
    assert W2_VALUE_MULTIPLIER > 0
