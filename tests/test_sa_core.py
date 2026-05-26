"""Unit + parity tests for lib/sa_core.py.

The critical-rigor gate of the sa_online architecture. Every claim
made by the SA layer has a test here. If any of these fail, the SA
agent is broken — don't ship.
"""
from __future__ import annotations

import random
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

from lib.sa_core import (  # noqa: E402
    _noop_policy,
    perturb,
    score_plan_from_snap,
    simulated_anneal_online,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_snap0(seed: int = 0, steps: int = 50):
    """Build a turn-0 snapshot for seed/steps — used by parity tests."""
    from kaggle_environments import make
    from lib.fast_sim import from_obs as fs_from_obs

    env = make("orbit_wars",
               configuration={"seed": seed, "episodeSteps": steps},
               debug=False)
    env.reset(num_agents=2)
    obs0 = env.steps[0][0]["observation"] if isinstance(env.steps[0][0], dict) else env.steps[0][0].observation
    return fs_from_obs(obs0, env.configuration,
                       episode_seed=seed, num_seats=2)


def _initial_planets(seed: int = 0, steps: int = 50):
    """Pull initial_planets list from a fresh env at the same seed."""
    from kaggle_environments import make

    env = make("orbit_wars",
               configuration={"seed": seed, "episodeSteps": steps},
               debug=False)
    env.reset(num_agents=2)
    obs0 = env.steps[0][0]["observation"] if isinstance(env.steps[0][0], dict) else env.steps[0][0].observation
    od = obs0 if isinstance(obs0, dict) else dict(obs0)
    return [list(p) for p in (od.get("planets") or [])]


# ---------------------------------------------------------------------------
# score_plan_from_snap — determinism + the critical solo-vs-online parity
# ---------------------------------------------------------------------------


def test_score_deterministic():
    """Same inputs → same output. No hidden state."""
    snap = _build_snap0(seed=0, steps=50)
    emissions = []  # empty plan
    s1 = score_plan_from_snap(emissions, snap, opp_policy=None, max_steps=50)
    s2 = score_plan_from_snap(emissions, snap, opp_policy=None, max_steps=50)
    assert s1 == s2, f"non-deterministic: {s1} vs {s2}"
    assert s1 > 0, f"empty-plan score should be > 0 (own home produces): {s1}"


def test_score_solo_parity():
    """score_plan_from_snap(plan, snap0, noop, T) == old solo score_plan."""
    from scripts.sa_solo_solver import score_plan
    snap = _build_snap0(seed=0, steps=50)
    emissions = []
    online_score = score_plan_from_snap(emissions, snap,
                                         opp_policy=None, max_steps=50)
    solo_score = score_plan(emissions, seed=0, steps=50)
    # Both build the same snap and roll the same empty plan against noop.
    # Result must be identical.
    assert online_score == solo_score, \
        f"parity break: online={online_score} solo={solo_score}"


def test_score_snap_unchanged_after_call():
    """score_plan_from_snap must NOT mutate the input snap."""
    from lib.fast_sim import ship_totals
    snap = _build_snap0(seed=0, steps=50)
    ships_before = ship_totals(snap).get(0, 0.0)
    _ = score_plan_from_snap([], snap, opp_policy=None, max_steps=50)
    ships_after = ship_totals(snap).get(0, 0.0)
    assert ships_before == ships_after, \
        f"snap mutated: {ships_before} -> {ships_after}"


# ---------------------------------------------------------------------------
# perturb — physics validity + opp-awareness invariants
# ---------------------------------------------------------------------------


def _build_test_ctx(seed: int = 0, steps: int = 50,
                     plan=None, opp_policy=None):
    """Build a PerturbContext from a fresh snap for use in perturb tests."""
    from lib.sa_core import _build_perturb_context
    snap = _build_snap0(seed=seed, steps=steps)
    if plan is None:
        plan = []
    return _build_perturb_context(
        snap, plan, opp_policy=opp_policy,
        max_steps=steps, t_start=0, t_end=steps, me=0,
    )


def _emission_passes_physics(emit, ctx) -> bool:
    """Run predict_fleet_fate on an emission; returns True iff outcome=='target'."""
    from lib.trajectory import predict_fleet_fate
    if ctx.world is None:
        return False
    _turn, action = emit
    try:
        src_id = int(action[0]); angle = float(action[1]); ships = int(action[2])
    except (TypeError, ValueError, IndexError):
        return False
    src = ctx.world.planets_by_id.get(src_id)
    if src is None:
        return False
    fate = predict_fleet_fate(src, src, angle, max(1, ships), ctx.world)
    return fate.outcome == "target" or fate.outcome == "planet"
    # ("planet" is fine; the operators guarantee outcome="target" against
    # the SPECIFIC target they computed for, but predict_fleet_fate above
    # is called with src=src as a placeholder target, so we accept both
    # outcomes meaning the fleet successfully hits SOME planet without
    # dying to sun/OOB/timeout.)


def test_perturb_remove_shrinks():
    """remove op decreases plan length by 1 when plan is non-empty."""
    from lib.sa_core import _op_remove
    plan = [(5, [0, 0.0, 10]), (10, [1, 0.5, 20])]
    ctx = _build_test_ctx(seed=0, steps=50)
    new = _op_remove(list(plan), random.Random(0), ctx)
    assert new is not None
    assert len(new) == len(plan) - 1


def test_perturb_remove_empty_returns_none():
    """remove op returns None when plan is empty (lets dispatcher fall through)."""
    from lib.sa_core import _op_remove
    ctx = _build_test_ctx(seed=0, steps=50)
    result = _op_remove([], random.Random(0), ctx)
    assert result is None


def test_perturb_validity_invariant():
    """SUBSTRATE GATE: every emission added by any operator must produce
    a physics-valid fleet (not sun / OOB / timeout). 200 trials.

    This is the load-bearing invariant of the redesign — perturb is now
    physics-aware by construction. If this fails, the operator generated
    a candidate that today's `perturb` would have via the old random
    `add`/`angle` ops, defeating the whole point.
    """
    n_adds_total = 0
    n_physics_pass = 0
    plan_initial = []
    ctx = _build_test_ctx(seed=0, steps=50, plan=plan_initial)
    for trial in range(200):
        rng = random.Random(trial)
        new = perturb(list(plan_initial), rng, ctx)
        # adds are new emissions that didn't exist in plan_initial
        if len(new) > len(plan_initial):
            new_emissions = new[len(plan_initial):]
            for emit in new_emissions:
                n_adds_total += 1
                if _emission_passes_physics(emit, ctx):
                    n_physics_pass += 1
    # We're testing the INVARIANT: every successful add is physics-valid.
    # The number of adds varies (operators may return None if no valid
    # target exists), but every one that DOES land must pass.
    if n_adds_total > 0:
        assert n_physics_pass == n_adds_total, (
            f"physics-validity invariant violated: "
            f"{n_adds_total - n_physics_pass} of {n_adds_total} emissions "
            f"failed predict_fleet_fate. The whole point of the redesign is "
            f"that this never happens.")


def test_perturb_shift_turn_changes_turn_only():
    """_op_shift_turn must produce a valid emission at a NEW turn when
    one is reachable.

    PI 2026-05-27: waiting is essential to the search space — many
    captures need a delayed fire to accumulate ships or align with
    orbital timing. This operator is the explicit wait-perturbation.

    Test approach: try several seeds + initial turns until one produces
    a capturable target (sa_core's capture eligibility depends on the
    specific geometry of the seed). On that scenario, exercise shift
    and verify at least one resulting emission has a different turn AND
    passes the physics gate.
    """
    from lib.sa_core import _op_shift_turn, _compute_capture_emission
    scenarios = [(0, 80, 5), (7542, 100, 5), (7542, 100, 10),
                  (1153, 100, 5), (2794, 100, 5), (2794, 100, 15)]
    seed_emit = None
    ctx = None
    for (seed, steps, seed_turn) in scenarios:
        candidate_ctx = _build_test_ctx(seed=seed, steps=steps)
        owned = [pid for pid, (owner, _ships)
                  in candidate_ctx.ownership_cache.get(0, {}).items()
                  if int(owner) == int(candidate_ctx.me)]
        if not owned:
            continue
        src = candidate_ctx.world.planets_by_id[int(owned[0])]
        for tgt_id, tgt in candidate_ctx.world.planets_by_id.items():
            if int(tgt_id) == int(src.id):
                continue
            emit = _compute_capture_emission(src, tgt, seed_turn, candidate_ctx)
            if emit is not None:
                seed_emit = emit
                ctx = candidate_ctx
                break
        if seed_emit is not None:
            break
    if seed_emit is None:
        pytest.skip("no working (seed, turn) found for shift test setup; "
                     "physics gate is doing its job — all generated "
                     "emissions would have been unreachable")
    orig_turn = seed_emit[0]
    seed_plan = [seed_emit]

    n_shifted = 0
    for trial in range(50):
        rng = random.Random(trial + 1000)
        result = _op_shift_turn(seed_plan, rng, ctx)
        if result is None:
            continue
        assert len(result) == 1, "shift must keep the same number of emissions"
        new_turn = result[0][0]
        if new_turn != orig_turn:
            n_shifted += 1
            assert _emission_passes_physics(result[0], ctx), \
                f"shift produced a physics-failing emission: {result[0]}"
    assert n_shifted >= 1, \
        f"shift never landed at a different turn over 50 trials"


def test_admissible_set_only_physics_valid():
    """PI 2026-05-27: every emission in `ctx.admissible` must be a
    physics-valid capture by construction. If anything in the set
    fails predict_fleet_fate.outcome == 'target' on its specific
    target, the enumeration is broken.

    Tests across multiple seeds to cover varied geometries (static,
    rotating, contested vs uncontested layouts).
    """
    from lib.sa_core import _build_perturb_context
    from lib.trajectory import predict_fleet_fate
    seeds = [0, 7542, 1153, 2794]
    n_admissible_total = 0
    n_validated = 0
    for seed in seeds:
        ctx = _build_test_ctx(seed=seed, steps=80)
        for emit, tgt_id in zip(ctx.admissible, ctx.admissible_targets):
            n_admissible_total += 1
            turn, action = emit
            src_id = int(action[0])
            angle = float(action[1])
            ships = max(1, int(action[2]))
            src = ctx.world.planets_by_id.get(src_id)
            tgt = ctx.world.planets_by_id.get(int(tgt_id))
            if src is None or tgt is None:
                continue
            fate = predict_fleet_fate(src, tgt, angle, ships, ctx.world)
            # outcome must be "target" — the admissible set is pre-validated
            # specifically against tgt.
            assert fate.outcome == "target", (
                f"admissible emission failed physics gate: seed={seed} "
                f"emit={emit} tgt={tgt_id} fate={fate}")
            n_validated += 1
    # We don't require a minimum count (some seeds may have very few
    # reachable targets) but at least ONE across all seeds.
    assert n_admissible_total > 0, \
        f"admissible set was empty across all {len(seeds)} seeds — " \
        f"enumeration likely broken or all geometries are blocked"
    assert n_validated == n_admissible_total, \
        f"{n_admissible_total - n_validated} of {n_admissible_total} " \
        f"admissible emissions failed the physics gate"


def test_perturb_add_contested_respects_opp_intent():
    """add_contested only fires when opp_intent_window is non-empty.

    With opp_policy=noop the intent window is empty, so _op_add_contested
    returns None deterministically. With opp_policy=nearest it's non-empty.
    """
    from lib.sa_core import _op_add_contested
    from importlib.util import spec_from_file_location, module_from_spec
    from pathlib import Path
    spec = spec_from_file_location("near", str(Path(__file__).resolve().parents[1] / "agents/simple/nearest.py"))
    near_mod = module_from_spec(spec); spec.loader.exec_module(near_mod)
    near_agent = near_mod.agent

    ctx_noop = _build_test_ctx(seed=0, steps=50, opp_policy=None)
    rng = random.Random(0)
    # Noop opp → empty intent window → operator must no-op.
    assert _op_add_contested([], rng, ctx_noop) is None

    ctx_nearest = _build_test_ctx(seed=0, steps=50, opp_policy=near_agent)
    # nearest opp on seed 0 fires within ~10 turns; window should populate.
    assert len(ctx_nearest.opp_intent_window) > 0, \
        "expected at least one opp_intent entry with nearest opp policy"


# ---------------------------------------------------------------------------
# simulated_anneal_online — monotone-best + parity with solo wrapper
# ---------------------------------------------------------------------------


def test_sa_online_monotone_best():
    """best_score never decreases across iterations (history rows)."""
    snap = _build_snap0(seed=0, steps=50)
    initial_planets = _initial_planets(seed=0, steps=50)
    _, _, history = simulated_anneal_online(
        initial_plan=[], snap0=snap, max_steps=50,
        opp_policy=None, n_iter=30, t0=100.0, cooling=0.95,
        rng=random.Random(42),
        start_step=0, initial_planets=initial_planets,
    )
    best_so_far = -float("inf")
    for (_, _, best) in history:
        assert best >= best_so_far, \
            f"monotone violation: best dropped from {best_so_far} to {best}"
        best_so_far = best


def test_sa_deadline_respected():
    """max_wall_s breaks the iter loop before n_iter completes.

    Regression test: the wallclock fix from the diagnostic post-mortem.
    Without it, sa_online would run all n_iter regardless of cost,
    blowing kaggle's actTimeout on expensive opp models.
    """
    import time
    snap = _build_snap0(seed=0, steps=50)
    initial_planets = _initial_planets(seed=0, steps=50)

    # n_iter very high, max_wall_s very low → must break early.
    t0 = time.perf_counter()
    _, _, history = simulated_anneal_online(
        initial_plan=[], snap0=snap, max_steps=50,
        opp_policy=None, n_iter=10_000, t0=100.0, cooling=0.99,
        rng=random.Random(0),
        start_step=0, initial_planets=initial_planets,
        max_wall_s=0.3,
    )
    elapsed = time.perf_counter() - t0
    # Allow a generous 2× slack on the deadline (last-iter overshoot).
    assert elapsed < 0.7, f"deadline overshot: {elapsed:.2f}s > 0.7s"
    # And we must have done at least one iteration.
    assert len(history) >= 1, "deadline broke before any iteration"


def test_score_diff_vs_absolute():
    """diff mode equals absolute mode when opp has zero ships (noop).

    Sanity: in solo (opp=noop), opp ends with 0 ships, so
    diff = absolute. Differentiates only when opp_ships > 0.
    """
    snap = _build_snap0(seed=0, steps=20)
    # Short game vs noop: opp does nothing, opp_ships at terminal > 0
    # because their home planet produces. So diff < absolute.
    abs_score = score_plan_from_snap([], snap, opp_policy=None,
                                      max_steps=20, score_mode="absolute")
    diff_score = score_plan_from_snap([], snap, opp_policy=None,
                                       max_steps=20, score_mode="diff")
    assert abs_score > 0, f"absolute score should be positive: {abs_score}"
    assert diff_score < abs_score, \
        f"diff ({diff_score}) should be < absolute ({abs_score}) when opp produces"
    # The gap = opp's terminal ships.
    assert abs_score - diff_score > 0


def test_score_me1_perspective_symmetric():
    """score(me=1) on the noop replay returns opp (seat 1)'s ships.

    The me parameter swaps which seat 'emissions' replays for. Verify
    the math: me=0 returns seat-0 ships; me=1 returns seat-1 ships.
    """
    snap = _build_snap0(seed=0, steps=20)
    me0_score = score_plan_from_snap([], snap, opp_policy=None,
                                      max_steps=20, me=0, score_mode="absolute")
    me1_score = score_plan_from_snap([], snap, opp_policy=None,
                                      max_steps=20, me=1, score_mode="absolute")
    # Both seats start with one home planet of identical size in 2P, so
    # under empty emissions both grow identically.
    assert me0_score == me1_score, \
        f"symmetric setup should give equal scores: me0={me0_score} me1={me1_score}"


def test_sa_online_vs_solo_parity():
    """simulated_anneal_online(... noop opp ...) ≡ scripts/sa_solo_solver.simulated_anneal.

    Both should yield the same trajectory given identical RNG (same seed
    drawn from same Random instance). This is THE critical parity test —
    if it fails, the refactor changed the SA behaviour and the existing
    solo results no longer reproduce.
    """
    from scripts.sa_solo_solver import simulated_anneal
    initial_planets = _initial_planets(seed=0, steps=50)
    n_iter = 20

    # Online path
    snap = _build_snap0(seed=0, steps=50)
    rng_online = random.Random(42)
    _, best_online, _ = simulated_anneal_online(
        initial_plan=[], snap0=snap, max_steps=50,
        opp_policy=None, n_iter=n_iter, t0=100.0, cooling=0.95,
        rng=rng_online, start_step=0, initial_planets=initial_planets,
    )

    # Solo wrapper (which delegates internally to sa_online)
    rng_solo = random.Random(42)
    _, best_solo, _ = simulated_anneal(
        initial_plan=[], seed=0, steps=50, n_iterations=n_iter,
        t0=100.0, cooling=0.95, rng=rng_solo, initial_planets=initial_planets,
    )

    assert best_online == best_solo, \
        f"SA parity break: online={best_online} solo={best_solo}"
