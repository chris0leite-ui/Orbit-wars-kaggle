"""Unit tests for the CRN opp-trajectory refactor (B.3.1, 2026-05-23).

Three properties this test set asserts:

1. **Determinism.** `compute_opp_trajectory(snap, me, num_seats, h, tier)`
   returns bit-identical action sequences when called twice on the same
   inputs. Required for CRN — if the trajectory itself drifts run to run
   we have no common random numbers across candidates.

2. **CRN identity (kwarg path).** When `opp_traj` is provided to
   `score_action` (composite) and `score_candidate_v4` (trajectory), the
   opp seats see exactly the precomputed sequence. Two candidates with
   IDENTICAL inputs return IDENTICAL Δ — no per-tick reactive divergence.

3. **Backward compatibility.** With `opp_traj=None` (default), behavior
   is byte-equal to the pre-refactor reactive path. Production agents
   that haven't set `BASELINE_OPP_TRAJ_TIER` are unaffected.
"""

from __future__ import annotations

from kaggle_environments import make

from agents.baseline.chooser import (
    build_idle_baseline,
    score_action,
)
from agents.baseline.chooser_trajectory import (
    build_trajectory_baseline,
    score_candidate_v4,
)
from lib.fast_sim import from_obs as fs_from_obs
from lib.opp_model import compute_opp_trajectory
from agents.baseline.proposer import propose
from lib.intent import World
from lib.world_model import WorldModel
from agents.baseline.value import select_favor_fn


def _snapshot_from_seed(seed: int = 42, num_seats: int = 2):
    env = make("orbit_wars", configuration={"seed": seed}, debug=False)
    env.reset(num_seats)
    obs = env.steps[0][0].observation
    return obs, fs_from_obs(obs, num_seats=num_seats)


def test_compute_opp_trajectory_is_deterministic():
    """Run twice on the same snap+tier; sequences must be byte-equal."""
    _obs, snap = _snapshot_from_seed(7)
    traj_a = compute_opp_trajectory(snap, me=0, num_seats=2, max_horizon=20, tier="lite")
    traj_b = compute_opp_trajectory(snap, me=0, num_seats=2, max_horizon=20, tier="lite")
    assert traj_a == traj_b
    assert len(traj_a) == 20
    # Me-slot is always empty (caller injects).
    for step_actions in traj_a:
        assert step_actions[0] == []
        assert isinstance(step_actions[1], list)


def test_compute_opp_trajectory_downgrades_topmix_in_4p():
    """topmix is too expensive for 4P (3 opp seats × 10ms × 10 ticks = 300ms
    one-time, plus per-candidate cost). Auto-downgrade to lite when num_seats>2.
    The contract is behavioral: with num_seats=4 and tier='topmix', the
    returned trajectory must equal what 'lite' would produce.
    """
    _obs, snap = _snapshot_from_seed(7, num_seats=4)
    traj_topmix = compute_opp_trajectory(snap, me=0, num_seats=4, max_horizon=15, tier="topmix")
    traj_lite = compute_opp_trajectory(snap, me=0, num_seats=4, max_horizon=15, tier="lite")
    assert traj_topmix == traj_lite


def test_build_idle_baseline_legacy_vs_crn_path_lite():
    """With tier='lite', the precomputed opp_traj should drive opp seats
    using the same policy that legacy `opp_actions_for_snap` uses (which
    defaults to lite_greedy_policy via BASELINE_OPP_TIER=0). The two
    baselines should match.
    """
    _obs, snap = _snapshot_from_seed(7)
    h = 10
    favs_legacy = build_idle_baseline(snap, me=0, num_seats=2, max_horizon=h, gamma=0.99)
    traj = compute_opp_trajectory(snap, me=0, num_seats=2, max_horizon=h, tier="lite")
    favs_crn = build_idle_baseline(snap, me=0, num_seats=2, max_horizon=h, gamma=0.99, opp_traj=traj)
    assert favs_legacy == favs_crn


def test_score_action_crn_identity_two_calls():
    """The CRN property: with opp_traj provided, two calls with identical
    candidate inputs return identical Δ. (Today's reactive path also gives
    identical Δ because the policies are deterministic — this test is the
    explicit gate that opp_traj wiring doesn't introduce drift.)
    """
    _obs, snap = _snapshot_from_seed(7)
    h = 10
    traj = compute_opp_trajectory(snap, me=0, num_seats=2, max_horizon=h, tier="lite")
    favs = build_idle_baseline(snap, me=0, num_seats=2, max_horizon=h, gamma=0.99, opp_traj=traj)
    # Action params are intentionally trivial — we're testing identity, not
    # that the action does anything useful.
    d1 = score_action(
        snap, me=0, num_seats=2, src_id=999, angle=0.0, ships=1,
        horizon=h, baseline_favors=favs, wait_N=0, gamma=0.99,
        opp_traj=traj,
    )
    d2 = score_action(
        snap, me=0, num_seats=2, src_id=999, angle=0.0, ships=1,
        horizon=h, baseline_favors=favs, wait_N=0, gamma=0.99,
        opp_traj=traj,
    )
    assert d1 == d2


def test_score_candidate_v4_legacy_vs_crn_path_lite_match():
    """For a candidate that the admissibility filter passes, both paths
    (legacy reactive opp / CRN replay with tier='lite') should compute
    identical Δ because lite_greedy_policy is deterministic and the
    snapshot state under candidate-insertion is the same.

    This is the strongest backward-compat assertion: not just "shape
    matches" but "value is bit-equal." If this test fails, the CRN
    refactor changed observable scoring behavior even on the default
    opp tier — that's a regression we'd want to catch immediately.
    """
    obs, snap = _snapshot_from_seed(11)
    me, num_seats = 0, 2
    h = 8
    world = World.from_obs(obs)
    model = WorldModel.from_world(world)
    planets_list = list(world.planets_by_id.values())
    my_planets = [p for p in planets_list if p.owner == me]
    target_pool = [p for p in planets_list if p.owner != me]
    gamma = 0.99
    omega = 0.0
    prerank = propose(
        my_planets, target_pool, world, model, me, omega,
        baseline_len=h + 1,
    )
    if not prerank:
        # No candidates on this seed — test is vacuous, exit cleanly.
        return
    # Take the highest-prerank candidate.
    _cd, src, tgt, ships, angle, _eta, prop_horizon, wait_N = prerank[0]
    favor_fn = select_favor_fn()

    # Build the two baselines (must MATCH for the Δs to MATCH).
    favs_legacy = build_trajectory_baseline(
        snap, me, num_seats, h, favor_fn, gamma,
    )
    traj = compute_opp_trajectory(snap, me, num_seats, h, tier="lite")
    favs_crn = build_trajectory_baseline(
        snap, me, num_seats, h, favor_fn, gamma, opp_traj=traj,
    )
    assert favs_legacy == favs_crn, "baseline values diverge between legacy/CRN path"

    d_legacy, st_l, _ = score_candidate_v4(
        snap, src, tgt, int(ships), float(angle),
        me, num_seats, world,
        favs_legacy, favor_fn, gamma,
        horizon=h, skip_admissibility=False, wait_N=int(wait_N),
    )
    d_crn, st_c, _ = score_candidate_v4(
        snap, src, tgt, int(ships), float(angle),
        me, num_seats, world,
        favs_crn, favor_fn, gamma,
        horizon=h, skip_admissibility=False, wait_N=int(wait_N),
        opp_traj=traj,
    )
    assert st_l == st_c
    assert d_legacy == d_crn, f"Δ diverges: legacy={d_legacy}, crn={d_crn}"
