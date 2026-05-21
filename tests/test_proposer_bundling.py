"""Pin tests for proposer.py — bundle blind spot + strategic stockpile.

Two fixes diagnosed against seed 2121761784 (4P FFA, step 46/168, PI
observation: "we capture small planets and expose our big planets
rather than bundling forces to protect ours and capture the big ones"):

- Fix 1 (bundle blind spot): `enumerate_ship_counts` refuses to emit
  a partial-budget candidate when `budget < cap` (source can't solo-
  capture). The LP's outcome_table.enumerate_outcomes already correctly
  sums multi-source arrivals via subset enumeration — but it can only
  bundle the columns the proposer emits. Post-fix: always emit `budget`
  as a candidate when `budget >= MIN_FLEET_SIZE`, enabling both
  offensive bundles (big-target multi-source captures) and defensive
  bundles (multi-source reinforce when no single source can cover the
  shortfall).

- Fix 2 (strategic stockpile): `capture_size` for own targets returns
  0 when `shortfall <= 0` (garrison already covers current threat).
  This blinds the LP to strategic defense of high-prod own planets
  before opp builds up. Post-fix: floor the reinforce target to
  STRATEGIC_STOCKPILE_TICKS × tgt.production when production >=
  STRATEGIC_DEFENSE_PROD, creating a preemptive buffer.
"""

from __future__ import annotations

from lib.intent import World
from lib.world_model import WorldModel, simulate_planet_timeline


def _build_world_and_model(planets, *, step=0, omega=0.0, ledger=None):
    """Synthetic World + WorldModel from [pid, owner, x, y, r, ships,
    prod] rows. `ledger` is `{pid: [(eta, owner, ships), ...]}`.

    Timelines are populated via `simulate_planet_timeline` so callers
    that rely on `model.ships_at` / `model.owner_at` get realistic
    values (neutral planets stay at their initial ships; owned planets
    accrue production over the horizon)."""
    obs = {
        "player": 0,
        "planets": planets,
        "fleets": [],
        "angular_velocity": omega,
        "initial_planets": [],
        "comet_planet_ids": [],
        "comets": [],
        "step": step,
    }
    world = World.from_obs(obs)
    base_ledger = {int(p[0]): [] for p in planets}
    if ledger:
        for pid, arrivals in ledger.items():
            base_ledger[int(pid)] = arrivals
    horizon = 50
    timelines = {
        p.id: simulate_planet_timeline(p, base_ledger.get(p.id, []), horizon)
        for p in world.planets_by_id.values()
    }
    model = WorldModel(ledger=base_ledger, timelines=timelines, horizon=horizon)
    return world, model


# ---------------------------------------------------------------------------
# Fix 1 — partial-budget candidates (bundle blind spot)
# ---------------------------------------------------------------------------


def test_partial_budget_candidates_enable_bundle():
    """Pin: 3 sources clustered near a big neutral target. Each source
    has fewer ships than the target's capture-size, so none can solo-
    capture. Pre-fix: `enumerate_ship_counts` returns [] for every
    (src, tgt) pair → no candidate targets the big planet. Post-fix:
    each source emits a partial-budget candidate so the LP can bundle.

    Mirrors the seed-2121761784 mid-game scenario where multiple
    medium-ship sources can't solo a high-defense big planet.
    """
    from agents.baseline.proposer import (
        MIN_FLEET_SIZE,
        capture_size,
        enumerate_ship_counts,
        propose,
    )

    # Layout: 3 my sources clustered top-right around (80, 80), neutral
    # big target T at (60, 60), one distant opp planet for threat-eta
    # plumbing. Coordinates avoid the sun at (50, 50, r=10).
    planets = [
        [0, 0, 80.0, 80.0, 1.69, 15, 2],   # me: S1 (15 ships)
        [1, 0, 85.0, 75.0, 1.69, 12, 2],   # me: S2 (12 ships)
        [2, 0, 75.0, 85.0, 1.69, 10, 2],   # me: S3 (10 ships)
        [10, -1, 60.0, 60.0, 2.61, 29, 4], # neutral: big target T
        [99, 1, 5.0, 5.0, 1.69, 10, 2],    # distant opp
    ]
    world, model = _build_world_and_model(planets, step=0)

    me = 0
    my_planets = [p for p in world.planets_by_id.values() if int(p.owner) == me]
    big_t = world.planets_by_id[10]

    # Sanity: capture_size from each source exceeds that source's budget.
    for src in my_planets:
        cap = capture_size(src, big_t, model, 0.0, me, world)
        assert cap > int(src.ships), (
            f"test prerequisite: source S{src.id} ({src.ships} ships) must "
            f"have budget < cap ({cap}) for the bundle scenario. Got "
            f"cap={cap}, budget={src.ships}."
        )

    # Direct check on enumerate_ship_counts — the unit under test.
    for src in my_planets:
        sizes = enumerate_ship_counts(src, big_t, model, 0.0, me, world)
        assert sizes, (
            f"enumerate_ship_counts returned no sizes for source S{src.id} "
            f"({src.ships} ships) toward big target T (cap > budget). "
            f"Post-fix: at least the partial-budget candidate ({src.ships} "
            f"ships) should be emitted so the LP can consider bundling. "
            f"Got: {sizes}"
        )

    # End-to-end via propose(): at least one candidate per source targets T.
    target_pool = [big_t]
    prerank = propose(my_planets, target_pool, world, model, me, 0.0,
                      baseline_len=50)
    by_src = {int(p.id): 0 for p in my_planets}
    for entry in prerank:
        _cheap, src, tgt, _ships, _angle, _eta, _h, _w = entry
        if int(tgt.id) == 10:
            by_src[int(src.id)] = by_src.get(int(src.id), 0) + 1
    missing = [sid for sid, n in by_src.items() if n == 0]
    assert not missing, (
        f"propose() emitted no candidate targeting big planet T from "
        f"sources {missing} (post-fix should emit at least one each). "
        f"Counts: {by_src}"
    )


# ---------------------------------------------------------------------------
# Fix 2 — strategic stockpile for high-prod own planets
# ---------------------------------------------------------------------------


def test_strategic_stockpile_for_high_prod():
    """Pin: a high-prod own planet (prod=5) with garrison already
    covering immediate shortfall. Pre-fix: `capture_size` returns 0 →
    `enumerate_ship_counts` returns [] → no reinforce candidate
    emitted. Post-fix: capture_size floors to
    STRATEGIC_STOCKPILE_TICKS × production = 25, so a preemptive
    reinforce candidate is generated.

    Captures the symptom from seed 2121761784 where our prod-5 home
    sits unreinforced while opp builds up.
    """
    from agents.baseline.proposer import (
        STRATEGIC_DEFENSE_PROD,
        STRATEGIC_STOCKPILE_TICKS,
        capture_size,
        enumerate_ship_counts,
    )

    # Layout: own prod-5 planet T at (75, 75) with 200 ships (well above
    # any immediate shortfall). Distant opp planet so time_to_enemy_threat
    # returns a value (the "no opp anywhere" short-circuit is preserved
    # by Fix 2 — only strategically defend when opp actually exists).
    # Nearby own source S to satisfy the propose() loop later.
    planets = [
        [10, 0, 75.0, 75.0, 2.61, 200, 5],  # me: high-prod T
        [11, 0, 78.0, 72.0, 1.69, 30, 2],   # me: nearby source S
        [99, 1, 5.0, 5.0, 1.69, 10, 2],     # distant opp
    ]
    world, model = _build_world_and_model(planets, step=0)
    me = 0
    tgt = world.planets_by_id[10]
    src = world.planets_by_id[11]

    # Sanity: prod meets strategic threshold.
    assert int(tgt.production) >= STRATEGIC_DEFENSE_PROD, (
        f"test prerequisite: tgt.production ({tgt.production}) must be "
        f">= STRATEGIC_DEFENSE_PROD ({STRATEGIC_DEFENSE_PROD})."
    )

    # capture_size for the high-prod own target must return at least
    # the strategic stockpile.
    cap = capture_size(src, tgt, model, 0.0, me, world)
    expected_floor = STRATEGIC_STOCKPILE_TICKS * int(tgt.production)
    assert cap >= expected_floor, (
        f"capture_size for high-prod own planet (prod={tgt.production}, "
        f"garrison={tgt.ships}) should return at least "
        f"STRATEGIC_STOCKPILE_TICKS × production = {expected_floor}. "
        f"Got {cap}. Pre-fix returns 0 when shortfall <= 0; post-fix "
        f"applies the strategic floor."
    )

    # enumerate_ship_counts should emit at least one positive size.
    sizes = enumerate_ship_counts(src, tgt, model, 0.0, me, world)
    assert sizes, (
        f"enumerate_ship_counts emitted no reinforce candidate for "
        f"high-prod own target (cap={cap}, budget={src.ships}). "
        f"Post-fix: at least the cap-sized candidate should be present. "
        f"Got: {sizes}"
    )
