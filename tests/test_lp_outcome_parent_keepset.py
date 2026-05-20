"""Bug #6 — pre-filter drops parent columns silently.

`lib/joint_solver/lp_outcome.py:154-157` caps each planet's candidate
column count at MAX_CONTESTERS_PER_PLANET (64) by sorting on
`Column.value` and dropping the lowest. The drop is silent.

When a dropped column is referenced as `parent_column_id` by another
column (Phase F2a compound), the linkage constraint added later
(lines 504-518) force-zeros the orphaned compound. The candidate
action space silently shrinks; no diagnostic surfaces it.

Pin (Rule 38) — pre-fix this test fails. Build 65 columns targeting
planet 5, with a low-value parent at column_id=0; add a compound on
planet 7 with `parent_column_id=0`. The parent should be force-kept
because something references it as a parent — but pre-fix it's
dropped by the value-rank pre-filter and the compound is orphaned.
"""

from __future__ import annotations

from lib.intent import Planet, World
from lib.joint_solver.columns import Column
from lib.world_model import WorldModel


def _make_col(*, column_id, src_id, tgt_id, value, ships=10, wait_N=0, eta=5,
              parent_column_id=None, owner=0):
    return Column(
        column_id=column_id, src_id=src_id, tgt_id=tgt_id,
        ships=ships, wait_N=wait_N, angle=0.0, eta=eta,
        owner=owner, value=value, parent_column_id=parent_column_id,
    )


def _make_world_model():
    planets = [
        Planet(id=0, owner=0, x=20.0, y=20.0, radius=5.0, ships=200, production=2),
        Planet(id=5, owner=1, x=80.0, y=20.0, radius=5.0, ships=10, production=2),
        Planet(id=7, owner=1, x=80.0, y=80.0, radius=5.0, ships=5, production=1),
    ]
    world = World(my_id=0, planets_by_id={p.id: p for p in planets}, omega=0.0,
                  comet_ids=frozenset(), step=0, obs_raw={})
    model = WorldModel(ledger={5: [], 7: []}, timelines={}, horizon=200)
    return planets, world, model


def test_prefilter_force_keeps_parent_columns():
    """Build 65 columns targeting planet 5, all sourced from planet 0,
    with the lowest-value column being the parent of a compound on
    planet 7. Verify the parent survives the per-planet pre-filter.
    """
    from lib.joint_solver.lp_outcome import (
        _build_per_planet_arrivals, MAX_CONTESTERS_PER_PLANET,
    )
    planets, world, model = _make_world_model()

    # 65 columns on planet 5; column_id 0 is the low-value parent.
    n_columns = MAX_CONTESTERS_PER_PLANET + 1
    cols = []
    for i in range(n_columns):
        # column_id 0 = lowest value (so the pre-filter would drop it
        # first if we didn't force-keep parents).
        value = 0.01 if i == 0 else float(i)
        cols.append(_make_col(
            column_id=i, src_id=0, tgt_id=5, value=value, ships=5,
            wait_N=0, eta=5,
        ))

    # Compound on planet 7 referencing column_id=0 as parent.
    compound = _make_col(
        column_id=999, src_id=5, tgt_id=7, value=10.0,
        ships=3, wait_N=5, eta=3, parent_column_id=0,
    )

    active = cols + [compound]
    per_planet = _build_per_planet_arrivals(
        active, world, model, my_id=0, step_now=0,
    )

    # Planet 5's candidate arrival set MUST include the parent (column_id 0).
    fixed_p5, cands_p5 = per_planet[5]
    cand_ids_p5 = {int(c.column_id) for c in cands_p5}
    assert 0 in cand_ids_p5, (
        f"parent column (id=0, low value) was dropped by the per-planet "
        f"pre-filter. Surviving column ids on planet 5: "
        f"{sorted(cand_ids_p5)[:5]}...{sorted(cand_ids_p5)[-5:]} "
        f"(total {len(cand_ids_p5)}). Compound column 999's "
        f"parent_column_id=0 is now orphaned, force-zero'd by the LP "
        f"linkage constraint. The keep-set must protect parent columns."
    )


def test_prefilter_keepset_does_not_starve_high_value_winners():
    """Sanity: force-keeping parents must not starve out the highest-value
    columns; the budget is `MAX - len(forced)`.
    """
    from lib.joint_solver.lp_outcome import (
        _build_per_planet_arrivals, MAX_CONTESTERS_PER_PLANET,
    )
    planets, world, model = _make_world_model()

    n_columns = MAX_CONTESTERS_PER_PLANET + 5
    cols = []
    for i in range(n_columns):
        # column_id 0..2 are low-value parents; rest are high-value.
        value = 0.01 if i < 3 else 100.0 + i
        cols.append(_make_col(
            column_id=i, src_id=0, tgt_id=5, value=value, ships=5,
            wait_N=0, eta=5,
        ))
    # Three compounds reference columns 0, 1, 2 as parents.
    compounds = [
        _make_col(column_id=1000 + j, src_id=5, tgt_id=7, value=10.0,
                  ships=3, wait_N=5, eta=3, parent_column_id=j)
        for j in range(3)
    ]
    active = cols + compounds

    per_planet = _build_per_planet_arrivals(
        active, world, model, my_id=0, step_now=0,
    )
    fixed_p5, cands_p5 = per_planet[5]
    cand_ids_p5 = {int(c.column_id) for c in cands_p5}

    # All three parents must survive.
    assert {0, 1, 2}.issubset(cand_ids_p5), (
        f"not all parents survived. ids: {sorted(cand_ids_p5)[:10]}"
    )
    # Total surviving columns must respect the cap.
    assert len(cand_ids_p5) <= MAX_CONTESTERS_PER_PLANET, (
        f"per-planet cap exceeded: {len(cand_ids_p5)} > "
        f"{MAX_CONTESTERS_PER_PLANET}"
    )
    # Some high-value columns must survive too (not 100% parents).
    high_value_survived = {
        cid for cid in cand_ids_p5 if cid >= 3
    }
    expected_high_value_count = MAX_CONTESTERS_PER_PLANET - 3  # budget after forced
    assert len(high_value_survived) == expected_high_value_count, (
        f"expected {expected_high_value_count} high-value columns to "
        f"survive after force-keeping 3 parents, got "
        f"{len(high_value_survived)}"
    )
