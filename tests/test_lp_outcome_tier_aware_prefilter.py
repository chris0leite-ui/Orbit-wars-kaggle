"""Tier-aware LP prefilter (2026-05-21 night).

The pre-prefilter dedup at `lib/joint_solver/lp_outcome.py:_build_per_planet_arrivals`
now groups optional columns by `(src_id, tier_class)` and keeps the best
per group BEFORE applying the per-target cap at `MAX_CONTESTERS_PER_PLANET`.

Motivation: `agents/baseline/proposer.py::enumerate_ship_counts_with_tier`
emits up to 4 fire-now ship sizes per (src, tgt): `spec_min`, `buffered`,
`double`, `budget`. Pre-fix, the per-target cap sorted by `(value, ships,
-wait_N, -column_id)` descending — for `prerank_passthrough`'s uniform
value=1.0, the tiebreak collapsed to "highest ships wins", so `budget`
always survived and the confidence buffer (`buffered`) was never visible
to the LP.

Post-fix, the pre-pass keeps one column per (src_id, tier_class) where
tier_class is: 0=spec_min, 1=buffered, 2=other-overkill (double / budget
/ wait / unknown). This lets a single (src, tgt) pair contribute up to
3 distinct columns competing for the per-target cap budget instead of
N redundant overkill variants from the same source.
"""

from __future__ import annotations

from lib.intent import Planet, World
from lib.joint_solver.columns import Column
from lib.world_model import WorldModel


def _make_col(*, column_id, ships, tier="unknown", value=1.0, src_id=0,
              tgt_id=5, wait_N=0, eta=5):
    return Column(
        column_id=column_id, src_id=src_id, tgt_id=tgt_id,
        ships=ships, wait_N=wait_N, angle=0.0, eta=eta,
        owner=0, value=value, tier=tier,
    )


def _make_world_model():
    planets = [
        Planet(id=0, owner=0, x=20.0, y=20.0, radius=5.0, ships=2000,
               production=2),
        Planet(id=5, owner=1, x=80.0, y=20.0, radius=5.0, ships=10,
               production=2),
    ]
    world = World(my_id=0, planets_by_id={p.id: p for p in planets},
                  omega=0.0, comet_ids=frozenset(), step=0, obs_raw={})
    model = WorldModel(ledger={5: []}, timelines={}, horizon=200)
    return planets, world, model


def test_prefilter_keeps_both_spec_min_and_overkill_per_pair():
    """A single (src=0, tgt=5) pair emits spec_min + budget. Other src
    planets fill the remaining cap slots with overkill columns. Pre-fix:
    only budget from src=0 survives (highest ships wins ties). Post-fix:
    both spec_min and budget from src=0 reach the LP — the cap budget is
    spent across tier-distinct columns, not duplicate overkill from one
    source.
    """
    from lib.joint_solver.lp_outcome import (
        _build_per_planet_arrivals, MAX_CONTESTERS_PER_PLANET,
    )
    planets, world, model = _make_world_model()

    cols = []
    # src=0 emits both spec_min and budget toward tgt=5. Same value=1.0.
    cols.append(_make_col(column_id=0, ships=11, tier="spec_min", src_id=0))
    cols.append(_make_col(column_id=1, ships=120, tier="budget", src_id=0))

    # 7 other sources emit budget toward the same target — competes for
    # the remaining cap slots. cap is MAX_CONTESTERS_PER_PLANET (6) so
    # we MUST have > cap candidates for the prefilter to fire.
    for i in range(7):
        cols.append(_make_col(
            column_id=10 + i, ships=100 - i, tier="budget", src_id=i + 1,
        ))
    assert len(cols) > MAX_CONTESTERS_PER_PLANET

    out = _build_per_planet_arrivals(cols, world, model, my_id=0, step_now=0)
    survivors_by_id = {a.column_id for a in out[5][1]}  # candidate arrivals

    # Pre-fix: only column_id=1 (budget from src=0) survives. spec_min
    # from src=0 (column_id=0) is dropped because budget out-cheaps it
    # by ships.
    # Post-fix: BOTH column_id=0 (spec_min, src=0) AND column_id=1
    # (budget, src=0) survive — they're in different tier_class buckets.
    assert 0 in survivors_by_id, (
        "spec_min column from src=0 must survive the prefilter even when "
        "overkill candidates exist. tier_class=0 (spec_min) and tier_class=1+ "
        "(overkill) should be deduped INDEPENDENTLY. Got survivors: "
        f"{sorted(survivors_by_id)}"
    )
    assert 1 in survivors_by_id, (
        "budget column from src=0 must also survive — overkill bucket "
        "for src=0 picks the best by sort key. Got survivors: "
        f"{sorted(survivors_by_id)}"
    )


def test_prefilter_keeps_buffered_separate_from_budget():
    """Three-tier split: a single source emits spec_min, buffered, AND
    budget. The buffered variant must survive even though budget has
    higher ships and (by virtue of shorter eta → higher cheap) would
    out-rank it in the legacy single-band sort.
    """
    from lib.joint_solver.lp_outcome import _build_per_planet_arrivals
    planets, world, model = _make_world_model()

    cols = []
    # src=0 emits all three tiers — same value=1.0, eta proxy via ships.
    cols.append(_make_col(column_id=0, ships=11, tier="spec_min", src_id=0))
    cols.append(_make_col(column_id=1, ships=17, tier="buffered", src_id=0))
    cols.append(_make_col(column_id=2, ships=120, tier="budget", src_id=0))

    # 4 more sources emit budget → 7 total > cap=6.
    for i in range(4):
        cols.append(_make_col(
            column_id=10 + i, ships=100 - i, tier="budget", src_id=i + 1,
        ))

    out = _build_per_planet_arrivals(cols, world, model, my_id=0, step_now=0)
    survivors_by_id = {a.column_id for a in out[5][1]}

    # All three of src=0's tiers must survive: spec_min (col 0),
    # buffered (col 1), and budget (col 2). The 3 remaining cap slots
    # go to the best of the 4 other-source budgets.
    assert 0 in survivors_by_id, (
        f"spec_min from src=0 should survive. Got: {sorted(survivors_by_id)}"
    )
    assert 1 in survivors_by_id, (
        "buffered from src=0 should survive — its own tier_class bucket "
        f"is distinct from budget. Got: {sorted(survivors_by_id)}"
    )
    assert 2 in survivors_by_id, (
        f"budget from src=0 should survive. Got: {sorted(survivors_by_id)}"
    )


def test_prefilter_handles_unknown_tier_gracefully():
    """Columns with tier='unknown' (the dataclass default for direct
    construction paths) must not crash the prefilter and should land in
    tier_class=2 (other-overkill). Determinism preserved.
    """
    from lib.joint_solver.lp_outcome import (
        _build_per_planet_arrivals, MAX_CONTESTERS_PER_PLANET,
    )
    planets, world, model = _make_world_model()

    # 8 columns, all tier='unknown', different sources/ships.
    cols = []
    for i in range(8):
        cols.append(_make_col(
            column_id=i, ships=100 - i, tier="unknown", src_id=i,
        ))
    assert len(cols) > MAX_CONTESTERS_PER_PLANET

    out = _build_per_planet_arrivals(cols, world, model, my_id=0, step_now=0)
    survivors = list(out[5][1])

    # All distinct (src_id, tier_class=2) keys → 8 keys → 8 survivors
    # pre-cap, then cap drops to 6.
    assert len(survivors) == MAX_CONTESTERS_PER_PLANET, (
        f"With 8 distinct-src unknown-tier columns and cap={MAX_CONTESTERS_PER_PLANET}, "
        f"expected {MAX_CONTESTERS_PER_PLANET} survivors. Got {len(survivors)}."
    )


def test_prefilter_preserves_parent_keepset_under_tier_dedup():
    """A compound column referencing a parent must survive the
    prefilter even when the parent's tier-class group would otherwise
    pick a different column. Mirrors the existing parent_keepset
    guarantee from tests/test_lp_outcome_parent_keepset.py.
    """
    from lib.joint_solver.lp_outcome import _build_per_planet_arrivals
    planets, world, model = _make_world_model()

    # src=0 emits spec_min (col 0) and budget (col 1). budget has higher
    # ships → wins sort key.
    cols = [
        _make_col(column_id=0, ships=11, tier="spec_min", src_id=0, value=0.5),
        _make_col(column_id=1, ships=120, tier="budget", src_id=0, value=1.0),
    ]
    # Compound that depends on col 0 (the spec_min) — parent must be
    # force-kept regardless of how the dedup ranks it.
    cols.append(Column(
        column_id=99, src_id=0, tgt_id=5, ships=50, wait_N=0,
        angle=0.0, eta=5, owner=0, value=2.0, tier="budget",
        parent_column_id=0,
    ))

    # Pad with extra src budgets to force the prefilter to fire.
    for i in range(5):
        cols.append(_make_col(
            column_id=20 + i, ships=80 - i, tier="budget", src_id=i + 1,
            value=0.5,
        ))

    out = _build_per_planet_arrivals(cols, world, model, my_id=0, step_now=0)
    survivors_by_id = {a.column_id for a in out[5][1]}

    assert 0 in survivors_by_id, (
        "Parent column (col 0, spec_min) must survive — referenced by "
        f"compound col 99. Got: {sorted(survivors_by_id)}"
    )
    assert 99 in survivors_by_id, (
        f"Compound col 99 must survive. Got: {sorted(survivors_by_id)}"
    )
