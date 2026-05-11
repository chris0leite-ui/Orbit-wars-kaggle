"""Tests for lib/missions/snipe.propose_snipe_missions."""

from __future__ import annotations

from types import SimpleNamespace

from lib.intent import World
from lib.missions.snipe import propose_snipe_missions
from lib.world_model import WorldModel


def _planet(pid, owner, x, y, *, ships=10, production=2, radius=1.5):
    return SimpleNamespace(
        id=pid, owner=owner, x=x, y=y, radius=radius,
        ships=ships, production=production,
    )


def _world(my_id, planets, *, step=0):
    obs = {
        "player": my_id,
        "planets": [
            (p.id, p.owner, p.x, p.y, p.radius, p.ships, p.production)
            for p in planets
        ],
        "fleets": [],
        "angular_velocity": 0.0,
        "comet_planet_ids": [],
        "step": step,
    }
    return World.from_obs(obs)


def test_no_missions_when_we_own_nothing():
    world = _world(my_id=0, planets=[
        _planet(0, 1, 10.0, 50.0),
        _planet(1, 2, 90.0, 50.0),
    ])
    model = WorldModel.from_world(world)
    assert propose_snipe_missions(world, model) == []


def test_no_missions_when_no_non_our_targets():
    world = _world(my_id=0, planets=[
        _planet(0, 0, 10.0, 50.0),
        _planet(1, 0, 90.0, 50.0),
    ])
    model = WorldModel.from_world(world)
    assert propose_snipe_missions(world, model) == []


def test_cross_product_count_2_sources_3_targets():
    """2 owned, 3 enemy planets → 6 candidate snipe missions."""
    world = _world(my_id=0, planets=[
        _planet(0, 0, 10.0, 50.0, ships=20),     # us
        _planet(1, 0, 90.0, 50.0, ships=20),     # us
        _planet(2, 1, 50.0, 10.0, ships=5),      # enemy
        _planet(3, 2, 50.0, 90.0, ships=5),      # enemy
        _planet(4, -1, 50.0, 50.0, ships=10),    # neutral
    ])
    model = WorldModel.from_world(world)
    ms = propose_snipe_missions(world, model)
    assert len(ms) == 6
    assert all(m.mission_class == "snipe" for m in ms)
    assert all(m.src_id in {0, 1} for m in ms)
    assert all(m.target_id in {2, 3, 4} for m in ms)
    assert all(m.ships >= 1 for m in ms)
    assert all(m.score > 0.0 for m in ms)


def test_score_scales_with_production_inverse_distance():
    """Closer / higher-production target ranks above farther / lower."""
    world = _world(my_id=0, planets=[
        _planet(0, 0, 0.0, 0.0, ships=20),
        _planet(1, 1, 10.0, 0.0, ships=1, production=5),   # near, high-prod
        _planet(2, 1, 100.0, 0.0, ships=1, production=5),  # far, same prod
    ])
    model = WorldModel.from_world(world)
    ms = propose_snipe_missions(world, model)
    # Two missions: src=0 -> {1, 2}.
    near = next(m for m in ms if m.target_id == 1)
    far = next(m for m in ms if m.target_id == 2)
    assert near.score > far.score


# ---------------------------------------------------------------------------
# Airtime penalty (v3.5)
# ---------------------------------------------------------------------------


def test_airtime_penalty_lowers_long_flight_scores():
    """Same target attributes at two different distances → the closer
    (lower-eta) target scores higher. Holds at any AIRTIME_PENALTY_WEIGHT
    > 0 because the airtime term is in the denominator."""
    world = _world(my_id=0, planets=[
        _planet(0, 0, 0.0, 0.0, ships=50),
        _planet(1, 1, 15.0, 0.0, ships=1, production=2),    # close
        _planet(2, 1, 90.0, 0.0, ships=1, production=2),    # far
    ])
    model = WorldModel.from_world(world)
    ms = propose_snipe_missions(world, model)
    close = next(m for m in ms if m.target_id == 1)
    far = next(m for m in ms if m.target_id == 2)
    assert close.eta < far.eta
    # The score ratio (close/far) must beat what a pure distance-only
    # penalty would give, because eta also penalises the far target.
    distance_only_ratio = (max(1, 1 + 1) + 90.0 + 1.0) / (
        max(1, 1 + 1) + 15.0 + 1.0
    )
    actual_ratio = close.score / far.score
    assert actual_ratio > distance_only_ratio, (
        f"airtime penalty should amplify near/far gap; "
        f"distance-only={distance_only_ratio:.3f}, actual={actual_ratio:.3f}"
    )


def test_airtime_penalty_disabled_at_zero_weight_matches_legacy():
    """Setting AIRTIME_PENALTY_WEIGHT=0 should recover the pre-v3.5 score
    formula exactly. Guards against future refactors smuggling extra terms."""
    import lib.missions.snipe as snipe_mod
    original = snipe_mod.AIRTIME_PENALTY_WEIGHT
    try:
        snipe_mod.AIRTIME_PENALTY_WEIGHT = 0.0
        world = _world(my_id=0, planets=[
            _planet(0, 0, 0.0, 0.0, ships=100),
            _planet(1, 1, 50.0, 0.0, ships=10, production=2),
        ])
        model = WorldModel.from_world(world)
        ms = propose_snipe_missions(world, model)
        m = ms[0]
        expected = 1.0 * (2 * max(1, 500 - 0 - m.eta)) / (
            max(1, 10 + 1) + 50.0 + 1.0
        )
        assert abs(m.score - expected) < 1e-9
    finally:
        snipe_mod.AIRTIME_PENALTY_WEIGHT = original


# ---------------------------------------------------------------------------
# Endgame neutral boost (v3.5, Exp 1)
# ---------------------------------------------------------------------------


def test_endgame_neutral_outscores_pre_endgame_neutral():
    """At step >= ENDGAME_STEP, an identical neutral target gets the
    endgame burn bonus on top of NEUTRAL_BONUS. Compare priority-only
    by extracting eta+geometry from the returned Mission."""
    import lib.missions.snipe as snipe_mod
    src = _planet(0, 0, 0.0, 0.0, ships=100)
    neutral = _planet(1, -1, 30.0, 0.0, ships=2, production=2)
    # Pre-endgame.
    pre = _world(my_id=0, planets=[src, neutral], step=100)
    pre_m = next(
        m for m in propose_snipe_missions(pre, WorldModel.from_world(pre))
        if m.target_id == 1
    )
    # Endgame.
    post = _world(my_id=0, planets=[src, neutral], step=snipe_mod.ENDGAME_STEP)
    post_m = next(
        m for m in propose_snipe_missions(post, WorldModel.from_world(post))
        if m.target_id == 1
    )
    # Geometry is identical; both Missions emit the same eta because
    # base_ships and distance are unchanged. So denom is the same.
    # value differs by time_to_hold (500-step-eta). Recover the effective
    # priority by dividing score by (value / denom).
    assert pre_m.eta == post_m.eta
    denom = max(1, 2 + 1) + 30.0 + snipe_mod.AIRTIME_PENALTY_WEIGHT * pre_m.eta + 1.0
    pre_value = 2 * max(1, 500 - 100 - pre_m.eta)
    post_value = 2 * max(1, 500 - snipe_mod.ENDGAME_STEP - post_m.eta)
    pre_priority = pre_m.score / (pre_value / denom)
    post_priority = post_m.score / (post_value / denom)
    assert abs(pre_priority - snipe_mod.NEUTRAL_BONUS) < 1e-6
    assert abs(
        post_priority - snipe_mod.NEUTRAL_BONUS * snipe_mod.ENDGAME_NEUTRAL_BONUS
    ) < 1e-6


def test_endgame_neutral_outscores_endgame_enemy_at_equal_distance():
    """At endgame, identical-distance neutral target beats identical-
    distance enemy target because endgame neutrals get the burn bonus
    and enemies don't."""
    import lib.missions.snipe as snipe_mod
    src = _planet(0, 0, 0.0, 0.0, ships=100)
    neutral = _planet(1, -1, 30.0, 0.0, ships=2, production=2)
    enemy = _planet(2, 1, 30.0, 5.0, ships=2, production=2)  # ~same distance
    world = _world(
        my_id=0,
        planets=[src, neutral, enemy],
        step=snipe_mod.ENDGAME_STEP,
    )
    model = WorldModel.from_world(world)
    ms = propose_snipe_missions(world, model)
    by_t = {m.target_id: m for m in ms}
    assert by_t[1].score > by_t[2].score
    # Ratio should be ~ENDGAME_NEUTRAL_BONUS (modulo the slight distance
    # difference; floor to 1.3 keeps the test robust).
    ratio = by_t[1].score / by_t[2].score
    assert ratio >= 1.3, f"expected endgame neutral preference; got ratio={ratio:.3f}"


def test_affordability_filter_off_keeps_all_proposals():
    """With PROPOSER_AFFORDABILITY_FILTER=False (default), the proposer
    emits Missions even when src.ships < base_ships. Validates default
    behaviour is unchanged."""
    import lib.missions.snipe as snipe_mod
    assert snipe_mod.PROPOSER_AFFORDABILITY_FILTER is False
    world = _world(my_id=0, planets=[
        _planet(0, 0, 0.0, 0.0, ships=3),                # small source
        _planet(1, 1, 50.0, 0.0, ships=20, production=2),  # large target
    ])
    model = WorldModel.from_world(world)
    ms = propose_snipe_missions(world, model)
    assert len(ms) == 1
    assert ms[0].ships == 21   # base_ships = target.ships + 1


def test_affordability_filter_on_skips_unaffordable_pair():
    """With PROPOSER_AFFORDABILITY_FILTER=True, the proposer skips
    pairs where base_ships > src.ships."""
    import lib.missions.snipe as snipe_mod
    original = snipe_mod.PROPOSER_AFFORDABILITY_FILTER
    try:
        snipe_mod.PROPOSER_AFFORDABILITY_FILTER = True
        world = _world(my_id=0, planets=[
            _planet(0, 0, 0.0, 0.0, ships=3),                   # small
            _planet(1, 1, 50.0, 0.0, ships=20, production=2),   # unaffordable
            _planet(2, 1, 30.0, 0.0, ships=2, production=1),    # affordable
        ])
        model = WorldModel.from_world(world)
        ms = propose_snipe_missions(world, model)
        # Source 0 (3 ships) can't fund target 1 (needs 21) but can fund target 2 (needs 3).
        targets = {m.target_id for m in ms}
        assert 2 in targets
        assert 1 not in targets
    finally:
        snipe_mod.PROPOSER_AFFORDABILITY_FILTER = original


def test_endgame_boost_does_not_fire_pre_endgame():
    """Below ENDGAME_STEP, no endgame boost — neutral score uses
    NEUTRAL_BONUS only."""
    import lib.missions.snipe as snipe_mod
    src = _planet(0, 0, 0.0, 0.0, ships=100)
    neutral = _planet(1, -1, 30.0, 0.0, ships=2, production=2)
    world = _world(
        my_id=0,
        planets=[src, neutral],
        step=snipe_mod.ENDGAME_STEP - 1,
    )
    model = WorldModel.from_world(world)
    ms = propose_snipe_missions(world, model)
    score = ms[0].score
    # Reconstruct the un-boosted score (NEUTRAL_BONUS only).
    eta = ms[0].eta
    value = 2 * max(1, 500 - (snipe_mod.ENDGAME_STEP - 1) - eta)
    expected = snipe_mod.NEUTRAL_BONUS * value / (
        max(1, 2 + 1) + 30.0 + snipe_mod.AIRTIME_PENALTY_WEIGHT * eta + 1.0
    )
    assert abs(score - expected) < 1e-9


def test_skips_target_already_ours_at_arrival():
    """If WorldModel predicts target is ours with surplus at our arrival,
    no mission is produced for that pair."""
    # Setup: source at (0,0) owned. Target at (5,0) currently enemy with
    # 1 ship and production=1. Our own in-flight fleet arrives in 1 turn
    # with 50 ships -> target becomes ours.
    target = _planet(1, 1, 5.0, 0.0, ships=1, production=1, radius=0.5)
    src = _planet(0, 0, 0.0, 0.0, ships=100, production=1)
    obs = {
        "player": 0,
        "planets": [
            (src.id, src.owner, src.x, src.y, src.radius, src.ships, src.production),
            (target.id, target.owner, target.x, target.y, target.radius,
             target.ships, target.production),
        ],
        # Pre-existing fleet from us heading at the target.
        "fleets": [(900, 0, 4.0, 0.0, 0.0, src.id, 50)],
        "angular_velocity": 0.0,
        "comet_planet_ids": [],
        "step": 0,
    }
    world = World.from_obs(obs)
    model = WorldModel.from_world(world)
    # Sanity: at eta=1 the target should already be ours with surplus.
    pred_o = model.owner_at(target.id, 1)
    pred_s = model.ships_at(target.id, 1) or 0.0
    # If the live WorldModel does see our 50-ship fleet hitting, the
    # mission is suppressed:
    if pred_o == 0 and pred_s >= 2:
        ms = propose_snipe_missions(world, model)
        assert all(m.target_id != target.id for m in ms), (
            f"snipe to target {target.id} should be suppressed; got "
            f"{[(m.src_id, m.target_id, m.ships) for m in ms]}"
        )
