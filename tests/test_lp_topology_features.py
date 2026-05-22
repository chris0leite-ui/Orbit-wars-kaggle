"""Level 1 — per-planet topology features in the LP leaf value.

PI directive 2026-05-21: "we need joint optimization that considers
topology." The LP at `lib/joint_solver/lp_outcome.py::_value_for_outcome`
was scoring only `me_prod − α · opp_prod + endgame_bonus` — purely
per-planet additive, no awareness of:
  - reachability: owning p enables future captures of neutrals near p
  - mutual defense: p in/near my cluster is cheaper to reinforce
  - recapture risk: p adjacent to opp is hard to hold

Level 1 adds three closed-form per-planet bonuses (computed from
`lib.geo.sense.sense_state` once per turn, cached, looked up per
(planet, subset) and credited iff `row.owner_T == my_id`).

Pin tests (Rule 38). Tests 1-3 exercise each topology component in
isolation against the helper. Test 4 locks the OWNERSHIP gate
(no bonus for planets opp ends up owning). Test 5 locks the no-op
contract when the feature is disabled.

The Rule 38 cycle: each helper-level test FAILS pre-fix (with the
relevant feature flag set to False or the underlying lambda set to
0); each PASSES post-fix.
"""
from __future__ import annotations

import pytest

from kaggle_environments.envs.orbit_wars.orbit_wars import Planet

from lib.intent import World
from lib.joint_solver.outcome_table import OutcomeRow
from lib.joint_solver import lp_outcome as lo
from lib.world_model import WorldModel


# ---------------------------------------------------------------------------
# Test fixtures (mirror tests/test_lp_endgame_predicate.py idioms)
# ---------------------------------------------------------------------------


def _planet(pid, owner, *, ships=10, production=2, x=0.0, y=0.0, radius=1.5):
    return Planet(pid, owner, x, y, radius, ships, production)


def _world_from_planets(my_id, planets, *, step=0, fleets=None):
    obs = {
        "player": my_id,
        "planets": [
            (p.id, p.owner, p.x, p.y, p.radius, p.ships, p.production)
            for p in planets
        ],
        "fleets": fleets or [],
        "angular_velocity": 0.0,
        "comet_planet_ids": [],
        "step": step,
    }
    return World.from_obs(obs)


def _model(world):
    """WorldModel for a clean world (no in-flight fleets)."""
    return WorldModel.from_world(world)


def _row(*, subset, owner_T, me_prod=0, opp_prod=0):
    stream = {}
    if me_prod:
        stream[0] = int(me_prod)
    if opp_prod:
        stream[1] = int(opp_prod)
    return OutcomeRow(
        subset=tuple(subset),
        owner_T=int(owner_T),
        ships_T=0.0,
        prod_stream=stream,
        prod_stream_discounted={},
    )


@pytest.fixture
def all_features_on(monkeypatch):
    """Enable all three topology features for the duration of the test."""
    monkeypatch.setenv("LP_TOPOLOGY_FEATURES", "1")
    monkeypatch.setenv("LP_REACH_BONUS", "1")
    monkeypatch.setenv("LP_DEFENSE_BONUS", "1")
    monkeypatch.setenv("LP_FRONT_PENALTY", "1")


# ---------------------------------------------------------------------------
# Test 1 — reachability_bonus rewards planets near unclaimed neutrals.
# ---------------------------------------------------------------------------

def test_reachability_bonus_rewards_planets_near_neutrals(monkeypatch):
    """Planet p1 sits near 3 unclaimed neutrals (in our voronoi);
    planet p2 sits alone. p1's reachability score must exceed p2's.

    Pre-fix (LP_REACH_BONUS=0): both planets score 0 → no preference.
    Post-fix (LP_REACH_BONUS=1, LAMBDA_REACH=50): p1 scores positive.
    """
    me = [_planet(0, 0, production=2, ships=10, x=0.0, y=0.0)]
    # p1 surrounded by 3 nearby neutrals → high reachability
    p1 = _planet(10, -1, production=2, x=12.0, y=0.0)
    neutrals_near_p1 = [
        _planet(20, -1, production=2, x=14.0, y=0.0),
        _planet(21, -1, production=2, x=16.0, y=2.0),
        _planet(22, -1, production=2, x=15.0, y=-2.0),
    ]
    # p2 is isolated far from any neutral
    p2 = _planet(30, -1, production=2, x=-30.0, y=-30.0)

    world = _world_from_planets(
        my_id=0, planets=me + [p1] + neutrals_near_p1 + [p2])
    model = _model(world)
    from lib.geo.sense import sense_state
    sense = sense_state(world, model)

    # Pre-fix: feature disabled → both scores are 0.
    monkeypatch.setenv("LP_TOPOLOGY_FEATURES", "0")
    monkeypatch.setenv("LP_REACH_BONUS", "0")
    pre_p1 = lo._per_planet_topology_score(10, world, model, sense, my_id=0)
    pre_p2 = lo._per_planet_topology_score(30, world, model, sense, my_id=0)
    assert pre_p1 == 0.0 and pre_p2 == 0.0, "pre-fix: both must be 0"

    # Post-fix: reach bonus enabled.
    monkeypatch.setenv("LP_TOPOLOGY_FEATURES", "1")
    monkeypatch.setenv("LP_REACH_BONUS", "1")
    monkeypatch.setenv("LP_DEFENSE_BONUS", "0")
    monkeypatch.setenv("LP_FRONT_PENALTY", "0")
    post_p1 = lo._per_planet_topology_score(10, world, model, sense, my_id=0)
    post_p2 = lo._per_planet_topology_score(30, world, model, sense, my_id=0)
    assert post_p1 > post_p2 + 1e-6, (
        f"p1 (3 nearby neutrals) should score higher than p2 (isolated); "
        f"got p1={post_p1:.2f}, p2={post_p2:.2f}"
    )
    assert post_p1 > 0.0, f"p1 reachability should be positive; got {post_p1}"


# ---------------------------------------------------------------------------
# Test 2 — mutual_defense_bonus rewards planets near my cluster.
# ---------------------------------------------------------------------------

def test_mutual_defense_bonus_rewards_clustered_planets(monkeypatch):
    """Planet near 3 of my OTHER planets scores higher than isolated
    candidate planet. Pre-fix (LP_DEFENSE_BONUS=0): both 0.
    """
    # 4 owned planets clustered at the origin
    cluster = [
        _planet(0, 0, production=2, x=0.0, y=0.0),
        _planet(1, 0, production=2, x=4.0, y=0.0),
        _planet(2, 0, production=2, x=0.0, y=4.0),
        _planet(3, 0, production=2, x=4.0, y=4.0),
    ]
    # neutral_near sits next to the cluster
    neutral_near = _planet(10, -1, production=2, x=2.0, y=2.0)
    # neutral_far is far away
    neutral_far = _planet(20, -1, production=2, x=80.0, y=80.0)

    world = _world_from_planets(
        my_id=0, planets=cluster + [neutral_near, neutral_far])
    model = _model(world)
    from lib.geo.sense import sense_state
    sense = sense_state(world, model)

    # Disable other features to isolate defense.
    monkeypatch.setenv("LP_TOPOLOGY_FEATURES", "1")
    monkeypatch.setenv("LP_REACH_BONUS", "0")
    monkeypatch.setenv("LP_DEFENSE_BONUS", "1")
    monkeypatch.setenv("LP_FRONT_PENALTY", "0")

    score_near = lo._per_planet_topology_score(10, world, model, sense, my_id=0)
    score_far = lo._per_planet_topology_score(20, world, model, sense, my_id=0)
    assert score_near > score_far + 1e-6, (
        f"neutral near my cluster should score higher than far one; "
        f"got near={score_near:.2f}, far={score_far:.2f}"
    )

    # Pre-fix: defense disabled → near and far both 0.
    monkeypatch.setenv("LP_DEFENSE_BONUS", "0")
    pre_near = lo._per_planet_topology_score(10, world, model, sense, my_id=0)
    assert pre_near == 0.0, f"pre-fix: defense disabled → 0; got {pre_near}"


# ---------------------------------------------------------------------------
# Test 3 — recapture_risk PENALIZES planets near opp threats.
# ---------------------------------------------------------------------------

def test_recapture_risk_penalises_frontier_planets(monkeypatch):
    """A neutral close to an opp planet has high recapture risk →
    NEGATIVE topology score (with only the front-penalty feature on).
    """
    me = [_planet(0, 0, production=2, ships=20, x=0.0, y=0.0)]
    # contested neutral between me and opp
    contested = _planet(10, -1, production=2, x=10.0, y=0.0)
    # opp planet just past the contested neutral (close threat)
    opp = [_planet(1, 1, production=2, ships=40, x=14.0, y=0.0)]

    world = _world_from_planets(my_id=0, planets=me + [contested] + opp)
    model = _model(world)
    from lib.geo.sense import sense_state
    sense = sense_state(world, model)

    # Only front penalty on.
    monkeypatch.setenv("LP_TOPOLOGY_FEATURES", "1")
    monkeypatch.setenv("LP_REACH_BONUS", "0")
    monkeypatch.setenv("LP_DEFENSE_BONUS", "0")
    monkeypatch.setenv("LP_FRONT_PENALTY", "1")

    score = lo._per_planet_topology_score(10, world, model, sense, my_id=0)
    assert score < 0.0, (
        f"contested neutral with close opp threat should have NEGATIVE "
        f"topology score (recapture risk); got {score:.2f}"
    )

    # Pre-fix: feature disabled → score is 0.
    monkeypatch.setenv("LP_FRONT_PENALTY", "0")
    pre = lo._per_planet_topology_score(10, world, model, sense, my_id=0)
    assert pre == 0.0, f"pre-fix: front penalty disabled → 0; got {pre}"


# ---------------------------------------------------------------------------
# Test 4 — _topology_bonus is 0 when row.owner_T != my_id.
# ---------------------------------------------------------------------------

def test_topology_bonus_zero_when_opp_owns_after_subset(all_features_on):
    """The LP only credits topology for planets WE end up owning.
    If `row.owner_T` says opp wins this subset, bonus is 0 regardless
    of the planet's topology score. Locks the ownership-gate semantic
    that prevents the LP from picking subsets that LET opp capture
    high-topology planets.
    """
    scores = {5: 999.0}  # huge positive topology score for planet 5
    row_we_lose = _row(subset=(99,), owner_T=1)  # opp wins
    row_we_win = _row(subset=(99,), owner_T=0)   # we win

    bonus_lose = lo._topology_bonus(5, row_we_lose, my_id=0,
                                    topology_scores=scores)
    bonus_win = lo._topology_bonus(5, row_we_win, my_id=0,
                                   topology_scores=scores)
    assert bonus_lose == 0.0, (
        f"row.owner_T != my_id → bonus must be 0 (we don't get to "
        f"compound on planets opp wins); got {bonus_lose}"
    )
    assert bonus_win == pytest.approx(999.0), (
        f"row.owner_T == my_id → bonus = topology_scores[pid]; got {bonus_win}"
    )


# ---------------------------------------------------------------------------
# Test 5 — topology features disabled → no-op (clean A/B baseline).
# ---------------------------------------------------------------------------

def test_topology_disabled_short_circuits_to_zero(monkeypatch):
    """When LP_TOPOLOGY_FEATURES=0, all three sub-flags are False, so
    `_per_planet_topology_score` returns 0.0 regardless of board, and
    `_topology_bonus(scores=None)` returns 0.0. Locks the no-op contract
    that makes pre-Level-1 / post-Level-1 a clean A/B differential.
    """
    monkeypatch.setenv("LP_TOPOLOGY_FEATURES", "0")
    monkeypatch.setenv("LP_REACH_BONUS", "0")
    monkeypatch.setenv("LP_DEFENSE_BONUS", "0")
    monkeypatch.setenv("LP_FRONT_PENALTY", "0")

    me = [_planet(0, 0, production=2, x=0.0, y=0.0)]
    neutrals = [_planet(i + 10, -1, production=2, x=5.0 * i, y=0.0)
                for i in range(5)]
    world = _world_from_planets(my_id=0, planets=me + neutrals)
    model = _model(world)
    from lib.geo.sense import sense_state
    sense = sense_state(world, model)

    for pid in [10, 11, 12, 13, 14]:
        score = lo._per_planet_topology_score(pid, world, model, sense,
                                              my_id=0)
        assert score == 0.0, (
            f"feature disabled: score for planet {pid} must be 0; got {score}"
        )

    # _topology_bonus with None scores → 0.
    row = _row(subset=(99,), owner_T=0)
    assert lo._topology_bonus(10, row, my_id=0, topology_scores=None) == 0.0


# ---------------------------------------------------------------------------
# Test 6 — sense_state being None (defensive guard) returns 0 cleanly.
# ---------------------------------------------------------------------------

def test_per_planet_score_none_sense_returns_zero(all_features_on):
    """The helper short-circuits to 0 if `sense` is None — defensive
    against `sense_state` raising during construction. The solve loop
    sets `topology_scores = None` in that case, which the bonus
    function also short-circuits on.
    """
    me = [_planet(0, 0, production=2, x=0.0, y=0.0)]
    world = _world_from_planets(my_id=0, planets=me)
    model = _model(world)
    score = lo._per_planet_topology_score(0, world, model, sense=None, my_id=0)
    assert score == 0.0
