"""Bugs #3, #4 — Phase F2a compound candidate ship-count + ownership.

`lib/pipeline/candidates_production_feedback.py:96` computes
`ships_avail = int(tgt.production) * int(delay)` for compound (Phase F2a)
candidates. This is a heuristic that:

  Bug #3: ignores the capture residual (ships left over from the base
          fleet capturing the target). If I send 18 ships at a 10-ship
          opp planet, I capture with 7 residual; the heuristic says 0.
  Bug #4: never checks whether I still own the planet at compound_fire
          time. If opp can re-capture between my base arrival and the
          compound fire, the compound is impossible — but the column is
          still emitted.

`simulate_planet_timeline` (lib/world_model.py:171) gives the exact
`owner_at[t]` and `ships_at[t]` for any planet and arrival schedule —
the fix uses this primitive instead of the heuristic.

Pin tests (Rule 38) — pre-fix both fail:

  Scenario A (ship-count): a base capture with surplus produces a
  larger ship_avail than `production * delay`. Pre-fix: the emitted
  column's `ships` field equals the heuristic; post-fix it equals the
  exact garrison.

  Scenario B (ownership): inject an opp re-capture arrival between
  base arrival and compound fire. Pre-fix: a compound column is still
  emitted; post-fix: no compound column is emitted (planet is opp's at
  compound_fire_rel).
"""

from __future__ import annotations

from lib.intent import Planet, World
from lib.joint_solver.columns import Column
from lib.pipeline.types import TurnContext
from lib.world_model import WorldModel


def _make_planet(*, id, owner, x, y, ships, production, radius=5.0):
    return Planet(id=id, owner=owner, x=x, y=y, radius=radius, ships=ships,
                  production=production)


def _make_ctx(planets, ledger, step_now, me=0, num_seats=2, horizon=100):
    """Build a synthetic TurnContext."""
    world = World(
        my_id=me,
        planets_by_id={p.id: p for p in planets},
        omega=0.0,
        comet_ids=frozenset(),
        step=step_now,
        obs_raw={},
    )
    # build empty timelines (the fix will populate per-planet on demand)
    timelines = {}
    model = WorldModel(ledger=ledger, timelines=timelines, horizon=horizon)
    return TurnContext(
        obs_d={"player": me},
        configuration=None,
        me=me,
        num_seats=num_seats,
        step_now=step_now,
        omega=0.0,
        planets=planets,
        fleets=[],
        my_planets=[p for p in planets if p.owner == me],
        other_planets=[p for p in planets if p.owner != me],
        world=world,
        model=model,
    )


def _make_base_column(*, src_id, tgt_id, ships, wait_N, eta, column_id=0,
                     angle=0.0):
    return Column(
        column_id=column_id, src_id=src_id, tgt_id=tgt_id,
        ships=ships, wait_N=wait_N, angle=angle, eta=eta,
        owner=0, value=1.0,
    )


def test_compound_ship_count_uses_exact_garrison_not_heuristic():
    """Bug #3: post-capture garrison must come from the exact
    `simulate_planet_timeline`, not the heuristic `production * delay`.

    Setup:
      planet 0 (mine): far from action.
      planet 5 (opp): garrison 10, production 2.
      base column: 30 ships, arrival_rel=5 → capture with surplus.
      compound delay∈{1,2,3} → compound_fire_rel∈{6,7,8}.

    Pre-capture growth at t=5: opp garrison = 10 + 2*5 = 20. My 30
    ships - 20 = 10 surplus. Post-capture: garrison grows by 2/turn.
    Heuristic value at compound_fire_rel=8 (delay=3): 2*3 = 6.
    Exact garrison at t=8: 10 (capture surplus) + 2*3 (production) = 16.
    After reserving 1 ship: ships_avail = 15.

    Post-fix expectation: c.ships > 6 (heuristic ceiling) for every
    emitted compound — the exact garrison must include capture surplus.
    """
    from lib.pipeline.candidates_production_feedback import (
        generate_compound_candidates,
    )

    # Board is 100x100, sun at (50,50). Place planets inside.
    planets = [
        _make_planet(id=0, owner=0, x=20.0, y=50.0, ships=20, production=2),
        _make_planet(id=5, owner=1, x=50.0, y=20.0, ships=10, production=2),
        _make_planet(id=7, owner=1, x=80.0, y=50.0, ships=5, production=1),
    ]
    ctx = _make_ctx(planets, ledger={5: [], 7: []}, step_now=0)
    base = _make_base_column(
        src_id=0, tgt_id=5, ships=30, wait_N=0, eta=5, column_id=42,
    )
    compounds = generate_compound_candidates([base], ctx, next_col_id_start=100)

    # At least one compound should survive trajectory + ownership checks.
    assert len(compounds) > 0, (
        "no compound candidates emitted; the scenario should produce "
        "at least one (planet 7 is a valid opp target near captured "
        "planet 5 at compound_fire_rel ∈ {6,7,8})"
    )
    for c in compounds:
        assert int(c.src_id) == 5
        compound_delay = int(c.wait_N) - 5  # wait_N == compound_fire_rel; base arrival_rel = 5
        heuristic_value = 2 * compound_delay  # pre-fix would have used production * delay
        assert int(c.ships) > heuristic_value, (
            f"compound column ships={c.ships} ≤ heuristic ceiling "
            f"{heuristic_value} (production*delay). The exact "
            f"simulate_planet_timeline-based garrison should INCLUDE the "
            f"capture surplus from the base fleet."
        )


def test_compound_skipped_when_opp_recaptures_before_fire():
    """Bug #4: if opp re-captures the planet between my base arrival
    and compound_fire_rel, the compound column must NOT be emitted.

    Setup:
      planet 5 (opp): I capture at t=5 with sufficient ships.
      Inject opp re-capture arrival at t=6 (large enough to flip).
      compound delay=3 → compound_fire_rel=8.

    At t=8, planet 5 is opp's (re-captured at t=6). The compound column
    is invalid (I can't fire from a planet I don't own).

    Pre-fix: code doesn't check owner_at[compound_fire_rel] → still
    emits the compound. Post-fix: ownership check skips it.
    """
    from lib.pipeline.candidates_production_feedback import (
        generate_compound_candidates,
    )

    # Board is 100x100, sun at (50,50). Place planets inside.
    planets = [
        _make_planet(id=0, owner=0, x=20.0, y=50.0, ships=20, production=2),
        _make_planet(id=5, owner=1, x=50.0, y=20.0, ships=10, production=2),
        _make_planet(id=7, owner=1, x=80.0, y=50.0, ships=5, production=1),
    ]
    # Existing ledger: planet 5 has an opp re-capture arrival at t=6
    # with 100 ships (enough to flip from my freshly-captured state).
    ledger = {5: [(6, 1, 100)], 7: []}
    ctx = _make_ctx(planets, ledger=ledger, step_now=0)

    # Base: I capture planet 5 at t=5 with substantial surplus (so the
    # pre-fix heuristic wouldn't immediately discard the compound for
    # other reasons; we want ownership to be the ONLY rejection cause).
    base = _make_base_column(
        src_id=0, tgt_id=5, ships=30, wait_N=0, eta=5, column_id=42,
    )

    compounds = generate_compound_candidates([base], ctx, next_col_id_start=100)

    # Filter to compounds parented on planet 5.
    p5_compounds = [c for c in compounds if c.parent_column_id == 42]
    assert len(p5_compounds) == 0, (
        f"compound candidates emitted from planet 5 even though opp "
        f"re-captures it at t=6, before any compound_fire_rel ≥ 6. "
        f"Got {len(p5_compounds)} compound(s): "
        f"{[(c.column_id, c.src_id, c.tgt_id, c.wait_N) for c in p5_compounds]}"
    )
