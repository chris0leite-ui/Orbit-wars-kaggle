"""Bug #7 — pre-filter tie-break determinism for prerank_passthrough.

`lib/pipeline/prerank_passthrough.py` rewrites every Column's `value`
to 1.0 to neutralize lp_outcome's value-based drop. But the per-planet
pre-filter at `lib/joint_solver/lp_outcome.py:154` sorts solely by
`Column.value`: with all values equal, the sort is unstable across
runs (Python sort is stable on equal keys, but the INPUT order depends
on dict iteration order from upstream stages, which can vary).

Result: when a planet has > MAX_CONTESTERS_PER_PLANET candidates, the
specific subset that survives is non-deterministic.

Fix: extend the pre-filter sort with a total-order tie-break — e.g.
`(value, ships, -wait_N, -column_id)` reversed. The exact tie-break
keys don't matter as long as they're deterministic given the column
set.

Pin (Rule 38): two runs of the pre-filter with the SAME columns but
different input orderings must produce the IDENTICAL surviving set.
Pre-fix this fails (Python's sort is stable; different input orders
yield different surviving subsets when keys tie). Post-fix the
secondary key makes the survivors deterministic by column identity.
"""

from __future__ import annotations

from lib.intent import Planet, World
from lib.joint_solver.columns import Column
from lib.world_model import WorldModel


def _make_col(*, column_id, ships, wait_N=0, eta=5, value=1.0):
    return Column(
        column_id=column_id, src_id=0, tgt_id=5,
        ships=ships, wait_N=wait_N, angle=0.0, eta=eta,
        owner=0, value=value,
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


def test_prefilter_is_order_independent_with_tied_values():
    """Two runs with the same column set but reversed input order must
    keep the SAME surviving column ids. Pre-fix: input-order dependent.
    """
    from lib.joint_solver.lp_outcome import (
        _build_per_planet_arrivals, MAX_CONTESTERS_PER_PLANET,
    )
    planets, world, model = _make_world_model()

    # 70 columns, all value=1.0, varying ships + column_id.
    n = MAX_CONTESTERS_PER_PLANET + 6
    cols_a = [
        _make_col(column_id=i, ships=(i % 17) + 1, wait_N=i % 4,
                  eta=5 + (i % 3), value=1.0)
        for i in range(n)
    ]
    cols_b = list(reversed(cols_a))

    per_planet_a = _build_per_planet_arrivals(
        cols_a, world, model, my_id=0, step_now=0,
    )
    per_planet_b = _build_per_planet_arrivals(
        cols_b, world, model, my_id=0, step_now=0,
    )

    surviving_a = sorted({int(c.column_id) for c in per_planet_a[5][1]})
    surviving_b = sorted({int(c.column_id) for c in per_planet_b[5][1]})

    assert surviving_a == surviving_b, (
        f"pre-filter is order-dependent with tied values. "
        f"surviving_a={surviving_a[:10]}..., surviving_b={surviving_b[:10]}... "
        f"sym_diff = {set(surviving_a) ^ set(surviving_b)}"
    )
