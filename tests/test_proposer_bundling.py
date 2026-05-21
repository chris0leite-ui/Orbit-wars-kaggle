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


# ---------------------------------------------------------------------------
# Fix 3 — partial-budget candidates gated when no bundle partner exists
# ---------------------------------------------------------------------------


def test_partial_budget_solo_filtered_when_alone():
    """Pin: a single small source with no peers nearby should NOT emit a
    sub-cap partial-budget candidate (it would fire solo and bounce).

    Captures the 2026-05-21 introspect finding (seeds 384458460/42/7):
    6 confirmed Type-A regressions from the prior partial-budget fix,
    where 6-12 ship partial candidates fired solo when no peer
    contributed. Post-fix: `peer_sources_in_reach == 0` gates the
    sub-cap candidate; the bundle case is preserved when peers exist.
    """
    from agents.baseline.proposer import (
        capture_size,
        enumerate_ship_counts,
        propose,
    )

    # Layout: ONE my source S (6 ships) at top-right corner, target T
    # at (60, 60), distant opp for threat-eta plumbing. No peer
    # sources exist within reach of T.
    planets = [
        [0, 0, 90.0, 90.0, 1.69, 6, 2],     # me: S (6 ships, alone)
        [10, -1, 60.0, 60.0, 2.61, 16, 2],  # neutral big T (cap > 6)
        [99, 1, 5.0, 5.0, 1.69, 10, 2],     # distant opp
    ]
    world, model = _build_world_and_model(planets, step=0)
    me = 0
    src = world.planets_by_id[0]
    tgt = world.planets_by_id[10]
    target_pool = [tgt]
    my_planets = [p for p in world.planets_by_id.values() if int(p.owner) == me]

    cap = capture_size(src, tgt, model, 0.0, me, world)
    assert cap > int(src.ships), (
        f"test prerequisite: src budget ({src.ships}) must be < cap ({cap})."
    )

    # Direct call with explicit peer_sources_in_reach=0: gate engages,
    # partial-budget candidate (6 ships) is NOT emitted.
    sizes = enumerate_ship_counts(
        src, tgt, model, 0.0, me, world, peer_sources_in_reach=0
    )
    assert int(src.ships) not in sizes, (
        f"Sub-cap partial-budget candidate ({src.ships} ships, cap={cap}) "
        f"was emitted despite no peer in reach. Post-fix should gate this. "
        f"Got: {sizes}"
    )

    # End-to-end via propose: no FIRE-NOW partial-budget candidate
    # (ships == src.ships, wait_N == 0) should be emitted. Wait-grid
    # candidates that accumulate ships before firing are unrelated to
    # this gate and may still appear.
    prerank = propose(my_planets, target_pool, world, model, me, 0.0,
                      baseline_len=50)
    fire_now_partials = [
        c for c in prerank
        if int(c[2].id) == int(tgt.id)
        and int(c[3]) == int(src.ships)
        and int(c[7]) == 0
    ]
    assert not fire_now_partials, (
        f"propose() emitted fire-now partial-budget candidate(s) "
        f"(ships={src.ships}, wait_N=0) targeting T from a lone source "
        f"with no bundle partner: {[(int(c[1].id), int(c[3]), int(c[7])) for c in fire_now_partials]}"
    )

    # Companion: adding a peer source restores the partial-budget emission.
    planets_with_peer = planets + [[1, 0, 87.0, 87.0, 1.69, 8, 2]]
    world2, model2 = _build_world_and_model(planets_with_peer, step=0)
    src2 = world2.planets_by_id[0]
    tgt2 = world2.planets_by_id[10]
    sizes_with_peer = enumerate_ship_counts(
        src2, tgt2, model2, 0.0, me, world2, peer_sources_in_reach=1
    )
    assert int(src2.ships) in sizes_with_peer, (
        f"Partial-budget candidate ({src2.ships} ships) should be "
        f"emitted when a peer is in reach. Got: {sizes_with_peer}"
    )


# ---------------------------------------------------------------------------
# Fix 4 — confidence-aware capture buffer
# ---------------------------------------------------------------------------


def test_confidence_buffer_emitted_as_extra_variant():
    """Pin: `enumerate_ship_counts` emits the buffered variant alongside
    spec-min cap so the LP can choose between ship-efficient (cap) and
    robust (cap + ε) sizing per outcome value.

    For eta≈10, prod=3, 2P: expected ε ≈ 6; buffered ≈ 17. Spec-min
    cap stays at 11. Both must appear in the emitted sizes list.
    """
    from agents.baseline.proposer import (
        capture_size,
        confidence_buffered_size,
        enumerate_ship_counts,
    )

    planets = [
        [10, 0, 70.0, 80.0, 1.69, 50, 2],   # me: S (50 ships, plenty)
        [11, -1, 70.0, 60.0, 2.61, 10, 3],  # neutral T (10 ships, prod=3)
        [99, 1, 5.0, 5.0, 1.69, 10, 2],     # opp (makes it 2P)
    ]
    world, model = _build_world_and_model(planets, step=0)
    me = 0
    src = world.planets_by_id[10]
    tgt = world.planets_by_id[11]

    cap = capture_size(src, tgt, model, 0.0, me, world)
    buffered = confidence_buffered_size(src, tgt, model, 0.0, me, world)
    sizes = enumerate_ship_counts(src, tgt, model, 0.0, me, world)

    assert cap == 11, (
        f"capture_size should be spec-min (pred + 1 = 11). Got {cap}."
    )
    assert buffered >= 15, (
        f"confidence_buffered_size expected ≥ 15 for prod=3, eta≈10, 2P. "
        f"Got {buffered}. ε = base(1) + eta_scale(0.5) × eta × prod/3."
    )
    assert cap in sizes, (
        f"enumerate_ship_counts must keep spec-min cap ({cap}). Got: {sizes}"
    )
    assert buffered in sizes, (
        f"enumerate_ship_counts must emit the buffered variant ({buffered}). "
        f"Got: {sizes}"
    )


def test_confidence_buffer_discounted_in_4p():
    """Pin: 4P discount shrinks the buffered variant so it's smaller than
    the 2P case but still above spec-min. Both spec-min and buffered are
    emitted as candidates.
    """
    from agents.baseline.proposer import (
        capture_size,
        confidence_buffered_size,
        enumerate_ship_counts,
    )

    planets = [
        [10, 0, 70.0, 80.0, 1.69, 50, 2],   # me: S
        [11, -1, 70.0, 60.0, 2.61, 10, 3],  # neutral T (prod=3)
        [99, 1, 5.0, 5.0, 1.69, 10, 2],     # opp 1
        [98, 2, 5.0, 95.0, 1.69, 10, 2],    # opp 2
        [97, 3, 95.0, 5.0, 1.69, 10, 2],    # opp 3 (makes it 4P)
    ]
    world, model = _build_world_and_model(planets, step=0)
    me = 0
    src = world.planets_by_id[10]
    tgt = world.planets_by_id[11]

    cap = capture_size(src, tgt, model, 0.0, me, world)
    buffered_4p = confidence_buffered_size(src, tgt, model, 0.0, me, world)
    sizes = enumerate_ship_counts(src, tgt, model, 0.0, me, world)

    assert cap == 11, f"Spec-min cap unchanged in 4P. Got {cap}."
    # 4P: ε ≈ 6 × 0.4 = 2.4 → ceil(10 + 2.4) + 1 = 14.
    assert 12 <= buffered_4p <= 15, (
        f"4P confidence buffer expected to be discounted: 12 ≤ buffered ≤ 15. "
        f"Got {buffered_4p}. The 4P discount (×0.4) shrinks ε from ~6 in 2P "
        f"to ~2.4 in 4P; spec-min cap stays at 11."
    )
    assert cap in sizes and buffered_4p in sizes, (
        f"4P enumerate_ship_counts must emit both spec-min ({cap}) and "
        f"buffered ({buffered_4p}). Got: {sizes}"
    )
