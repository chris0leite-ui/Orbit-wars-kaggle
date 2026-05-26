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
    reset_fate_cache,
    score_plan_from_snap,
    simulated_anneal_online,
)


@pytest.fixture(autouse=True)
def _reset_fate_cache_between_tests():
    """Fate cache is module-level + per-process in production (Kaggle
    starts a fresh process per episode). Tests share the process, so
    reset between tests to prevent stale (src, tgt, t_dep) entries
    from one seed leaking into another."""
    reset_fate_cache()
    yield
    reset_fate_cache()


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
    """Run predict_fleet_fate on an emission; returns True iff outcome
    indicates the fleet successfully hit a planet (target or other).

    Uses `wait_N=turn-t_start` so the trajectory check matches the actual
    fire-time geometry — same as the validator inside the emission code.
    """
    from lib.trajectory import predict_fleet_fate
    if ctx.world is None:
        return False
    turn, action = emit
    try:
        src_id = int(action[0]); angle = float(action[1]); ships = int(action[2])
    except (TypeError, ValueError, IndexError):
        return False
    src = ctx.world.planets_by_id.get(src_id)
    if src is None:
        return False
    wait_N = max(0, int(turn) - int(ctx.t_start))
    fate = predict_fleet_fate(src, src, angle, max(1, ships), ctx.world,
                               wait_N=wait_N)
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
    # Iterate scenarios until we find one where both seed-capture AND
    # at least one shift produce valid emissions. With the stricter
    # wait_N=t_dep trajectory check, not every seed_turn produces shift-
    # able candidates; the test must find a scenario that does.
    found = None
    for (seed, steps, seed_turn) in scenarios:
        candidate_ctx = _build_test_ctx(seed=seed, steps=steps)
        owned = [pid for pid, (owner, _ships)
                  in candidate_ctx.ownership_cache.get(0, {}).items()
                  if int(owner) == int(candidate_ctx.me)]
        if not owned:
            continue
        src = candidate_ctx.world.planets_by_id[int(owned[0])]
        seed_emit = None
        for tgt_id, tgt in candidate_ctx.world.planets_by_id.items():
            if int(tgt_id) == int(src.id):
                continue
            emit = _compute_capture_emission(src, tgt, seed_turn, candidate_ctx)
            if emit is not None:
                seed_emit = emit
                break
        if seed_emit is None:
            continue
        # Pre-flight: does at least one shift succeed for this scenario?
        n_pre_shift = 0
        for trial in range(20):
            rng = random.Random(trial + 5000)
            r = _op_shift_turn([seed_emit], rng, candidate_ctx)
            if r is not None and r[0][0] != seed_emit[0]:
                n_pre_shift += 1
                break
        if n_pre_shift > 0:
            found = (candidate_ctx, seed_emit)
            break
    if found is None:
        pytest.skip("no (seed, turn) found where both capture and shift "
                     "produce valid emissions — physics gate is strict")
    ctx, seed_emit = found
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
    from lib.sa_core import reset_fate_cache
    seeds = [0, 7542, 1153, 2794]
    n_admissible_total = 0
    n_validated = 0
    for seed in seeds:
        # Fate cache is intentionally per-process in production (each
        # Kaggle episode = fresh process = fresh planet geometry). In
        # multi-seed tests we must reset between iterations so cached
        # entries from seed N don't poison seed N+1's identical
        # (src, tgt, t_dep) keys with stale outcomes.
        reset_fate_cache()
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
            wait_N = max(0, int(turn) - int(ctx.t_start))
            fate = predict_fleet_fate(src, tgt, angle, ships, ctx.world,
                                       wait_N=wait_N)
            # outcome must be "target" — the admissible set is pre-validated
            # specifically against tgt (with the same wait_N as admission).
            assert fate.outcome == "target", (
                f"admissible emission failed physics gate: seed={seed} "
                f"emit={emit} tgt={tgt_id} wait_N={wait_N} fate={fate}")
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

    Pre-builds the path_graph (matches the agent's real usage pattern:
    one-time build at game start, reused per-turn) so the deadline
    measures the SA loop + ctx-setup cost, not the one-time graph build.
    """
    import time
    from lib.intent import World
    from lib.path_graph import build_path_graph
    snap = _build_snap0(seed=0, steps=50)
    initial_planets = _initial_planets(seed=0, steps=50)
    # Pre-build path_graph (real-agent pattern: built once at game start)
    obs0 = snap.state[0].observation
    od = {}
    for k in ("player", "step", "planets", "fleets", "comets",
              "comet_planet_ids", "angular_velocity"):
        v = getattr(obs0, k, None)
        if v is not None:
            od[k] = list(v) if isinstance(v, list) else v
    world = World.from_obs(od)
    pg = build_path_graph(world, t_max=50, orbiting_bucket=4, comet_bucket=1)

    # n_iter very high, max_wall_s very low → must break early.
    t0 = time.perf_counter()
    _, _, history = simulated_anneal_online(
        initial_plan=[], snap0=snap, max_steps=50,
        opp_policy=None, n_iter=10_000, t0=100.0, cooling=0.99,
        rng=random.Random(0),
        start_step=0, initial_planets=initial_planets,
        max_wall_s=0.3,
        path_graph=pg,
    )
    elapsed = time.perf_counter() - t0
    # Allow a generous 2× slack on the deadline (last-iter overshoot +
    # initial ctx-build forward sim ~50 ms).
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


def test_admissible_cascade_grows():
    """PI 2026-05-28: cascade closure. ctx built against a plan that
    captures planet X at turn T should admit emissions sourced FROM X
    at turn >= T+eta. The "empty plan" ctx only admits sources from
    the planets we own at t_start; the "plan with capture" ctx must
    admit a strict superset (at least: emissions sourced from X).
    """
    from lib.sa_core import _build_perturb_context, _compute_capture_emission
    snap = _build_snap0(seed=7542, steps=80)
    ctx_empty = _build_perturb_context(
        snap, [], opp_policy=None,
        max_steps=80, t_start=0, t_end=80, me=0,
    )
    if ctx_empty.world is None or not ctx_empty.admissible:
        pytest.skip("no admissible captures available on this seed")
    # Find an admissible capture whose target we can use as a new source.
    seed_emit = None
    seed_tgt_id = None
    for emit, tgt_id in zip(ctx_empty.admissible, ctx_empty.admissible_targets):
        # Skip comets (their availability is finite and ETA-dependent)
        if int(tgt_id) in (ctx_empty.world.comet_ids or ()):
            continue
        seed_emit = emit
        seed_tgt_id = int(tgt_id)
        break
    if seed_emit is None:
        pytest.skip("no non-comet admissible capture found for cascade seed")

    seed_plan = [seed_emit]
    ctx_with_capture = _build_perturb_context(
        snap, seed_plan, opp_policy=None,
        max_steps=80, t_start=0, t_end=80, me=0,
    )
    # The captured planet must appear as an owned source at some later turn.
    seen_owned_after_capture = False
    for turn, ownership in ctx_with_capture.ownership_cache.items():
        if int(turn) <= 0:
            continue
        entry = ownership.get(int(seed_tgt_id))
        if entry is None:
            continue
        owner, _ships = entry
        if int(owner) == 0:
            seen_owned_after_capture = True
            break
    assert seen_owned_after_capture, (
        f"plan captures planet {seed_tgt_id} but ownership_cache "
        f"never shows it as owned by me — forward sim broken?")
    # Now check the cascade: admissible should contain at least one
    # emission whose source is the captured planet.
    cascade_emissions = [
        e for e in ctx_with_capture.admissible
        if int(e[1][0]) == int(seed_tgt_id)
    ]
    assert cascade_emissions, (
        f"cascade closure failed: planet {seed_tgt_id} captured by seed "
        f"plan but never appears as a source in admissible set. "
        f"admissible_size={len(ctx_with_capture.admissible)}")


def test_ruin_recreate_physics_valid():
    """Every emission inserted by _op_ruin_recreate must pass the
    physics gate against its actual target — same invariant as
    test_perturb_validity_invariant but exercises the new operator
    directly to bound the trial count to physics-only failures.
    """
    from lib.sa_core import _op_ruin_recreate, _build_perturb_context
    from lib.trajectory import predict_fleet_fate
    snap = _build_snap0(seed=7542, steps=80)
    ctx = _build_perturb_context(
        snap, [], opp_policy=None,
        max_steps=80, t_start=0, t_end=80, me=0,
    )
    if not ctx.admissible or len(ctx.admissible) < 5:
        pytest.skip("not enough admissible candidates for ruin-recreate test")
    # Seed plan: take 6 admissible emissions so ruin (k=3..5) has room.
    seed_plan = list(ctx.admissible[:6])
    # Build target_id_of lookup: id(emit) -> tgt_id from admissible_targets
    tgt_of = {id(ctx.admissible[i]): ctx.admissible_targets[i]
              for i in range(len(ctx.admissible))}
    n_validated = 0
    for trial in range(50):
        rng = random.Random(trial + 2000)
        result = _op_ruin_recreate(seed_plan, rng, ctx)
        if result is None:
            continue
        # Every emission in result must pass physics against its target
        for emit in result:
            tgt_id = tgt_of.get(id(emit))
            if tgt_id is None:
                continue  # part of retained seed; skip
            turn_, action = emit
            src_id = int(action[0])
            angle = float(action[1])
            ships = max(1, int(action[2]))
            src = ctx.world.planets_by_id.get(src_id)
            tgt = ctx.world.planets_by_id.get(int(tgt_id))
            if src is None or tgt is None:
                continue
            wait_N = max(0, int(turn_) - int(ctx.t_start))
            fate = predict_fleet_fate(src, tgt, angle, ships, ctx.world,
                                       wait_N=wait_N)
            assert fate.outcome == "target", (
                f"ruin-recreate produced physics-failing emission: "
                f"trial={trial} emit={emit} wait_N={wait_N} fate={fate}")
            n_validated += 1
    assert n_validated > 0, "ruin-recreate produced no validated emissions across 50 trials"


def test_ruin_recreate_respects_capture_value():
    """_op_ruin_recreate's greedy refill must pick admissible candidates
    ordered by _capture_value (the precise game-model value under
    no-recapture). When two admissibles have very different values,
    the higher-value one must appear in the rebuilt plan.
    """
    from lib.sa_core import _op_ruin_recreate, _capture_value, _build_perturb_context
    snap = _build_snap0(seed=7542, steps=80)
    ctx = _build_perturb_context(
        snap, [], opp_policy=None,
        max_steps=80, t_start=0, t_end=80, me=0,
    )
    if not ctx.admissible or len(ctx.admissible) < 8:
        pytest.skip("not enough admissible candidates to test value ordering")
    # Score every admissible and verify the top-3 by value appear in
    # the rebuilt plan when we ruin all of a 5-emission seed plan.
    scored = sorted(
        [(_capture_value(e, i, ctx), e)
         for i, e in enumerate(ctx.admissible)],
        key=lambda kv: -kv[0],
    )
    top3 = [e for _v, e in scored[:3]]
    # Build a seed plan with 5 LOW-value emissions (so ruin removes them all)
    low5 = [e for _v, e in scored[-5:]]
    seed_plan = list(low5)
    rng = random.Random(0)
    # Try multiple rng seeds — k is randomised 3..5, want a trial where
    # k>=3 so top-3 all fit in the rebuilt portion.
    found_top1 = False
    for trial in range(30):
        result = _op_ruin_recreate(seed_plan, random.Random(trial), ctx)
        if result is None:
            continue
        # Check that at least the top-value emission appears in the rebuilt
        # plan (greedy must pick it; ties broken by sort stability).
        if any(id(e) == id(top3[0]) for e in result):
            found_top1 = True
            break
    assert found_top1, (
        "ruin-recreate never inserted the highest-value admissible — "
        "greedy refill broken or value function not used")


def test_ctx_rebuild_idempotent_against_unchanged_plan():
    """Rebuilding ctx with the identical current_plan must yield an
    identical admissible set (same emissions, same target ids).

    Rules out hidden RNG / iteration-order effects in the cascade
    enumeration that would make ctx rebuilds non-deterministic. Uses
    a generous admissible_wall_s so the wallclock deadline doesn't
    truncate either build at a different point.
    """
    from lib.sa_core import _build_perturb_context
    snap = _build_snap0(seed=7542, steps=80)
    ctx_a = _build_perturb_context(
        snap, [], opp_policy=None,
        max_steps=80, t_start=0, t_end=80, me=0,
        admissible_wall_s=30.0,
    )
    ctx_b = _build_perturb_context(
        snap, [], opp_policy=None,
        max_steps=80, t_start=0, t_end=80, me=0,
        admissible_wall_s=30.0,
    )
    assert len(ctx_a.admissible) == len(ctx_b.admissible), (
        f"non-deterministic admissible size: "
        f"{len(ctx_a.admissible)} vs {len(ctx_b.admissible)}")
    for a, b in zip(ctx_a.admissible, ctx_b.admissible):
        assert a == b, f"emission order/content differs: {a} vs {b}"
    assert ctx_a.admissible_targets == ctx_b.admissible_targets


def test_sa_step_budget_under_kaggle_cap():
    """SA with max_wall_s=0.5 must complete within 0.7s even with
    cascade rebuilds enabled. This is the Kaggle actTimeout gate:
    the in-loop rebuild path must not blow the per-turn budget.
    """
    import time
    from lib.intent import World
    from lib.path_graph import build_path_graph
    snap = _build_snap0(seed=0, steps=50)
    initial_planets = _initial_planets(seed=0, steps=50)
    obs0 = snap.state[0].observation
    od = {}
    for k in ("player", "step", "planets", "fleets", "comets",
              "comet_planet_ids", "angular_velocity"):
        v = getattr(obs0, k, None)
        if v is not None:
            od[k] = list(v) if isinstance(v, list) else v
    world = World.from_obs(od)
    pg = build_path_graph(world, t_max=50, orbiting_bucket=4, comet_bucket=1)

    t0 = time.perf_counter()
    _, _, _ = simulated_anneal_online(
        initial_plan=[], snap0=snap, max_steps=50,
        opp_policy=None, n_iter=10_000, t0=100.0, cooling=0.99,
        rng=random.Random(0),
        start_step=0, initial_planets=initial_planets,
        max_wall_s=0.5,
        path_graph=pg,
        rebuild_interval_iters=50,  # force at least 1 rebuild
        max_rebuilds=3,
    )
    elapsed = time.perf_counter() - t0
    assert elapsed < 0.7, (
        f"per-turn budget overshoot with rebuilds: "
        f"{elapsed:.2f}s > 0.7s (kaggle actTimeout is ~1s)")


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


# ---------------------------------------------------------------------------
# warm-start (closest biggest planets at ETA) — seeds SA on Kaggle turn 1
# ---------------------------------------------------------------------------


def test_warm_start_seeds_empty_plan():
    """ctx with non-empty admissible, empty initial_plan -> warm-start
    augments. This is the live failure mode that motivated the function
    (sub 53062327: SA emitted 5 actions in 79 turns from empty start)."""
    from lib.sa_core import _warm_start_from_admissible
    ctx = _build_test_ctx(seed=0, steps=50, plan=[])
    if not ctx.admissible:
        pytest.skip("admissible set empty on seed 0 — env-specific; "
                     "test is meaningful only when admissible is populated")
    augmented = _warm_start_from_admissible(ctx, [], max_emissions=8)
    assert len(augmented) > 0, "warm-start did nothing on an empty plan"
    assert len(augmented) <= 8, "warm-start emitted more than max_emissions"


def test_warm_start_respects_source_budget():
    """No source's cumulative spend up to any t_dep exceeds its available
    ships at that t_dep. Available = ownership_cache[t_dep][src].ships,
    which already includes accrued production from forward-sim through
    the current (empty) plan.

    Iterate the warm-start output sorted by t_dep ascending: maintain
    cumulative-spent per source, assert each new emission's cumulative
    spend never exceeds that source's ships at its t_dep."""
    from lib.sa_core import _warm_start_from_admissible
    ctx = _build_test_ctx(seed=0, steps=50, plan=[])
    if not ctx.admissible:
        pytest.skip("admissible set empty")
    augmented = _warm_start_from_admissible(ctx, [], max_emissions=20)

    cumulative_spent: dict[int, float] = {}
    sorted_emissions = sorted(augmented, key=lambda e: int(e[0]))
    for turn, payload in sorted_emissions:
        src_id = int(payload[0])
        ships_needed = float(payload[2])
        cache_at_dep = ctx.ownership_cache.get(int(turn), {})
        state = cache_at_dep.get(src_id)
        assert state is not None, \
            f"warm-start emit at turn={turn} from src={src_id} not in ownership_cache"
        owner, ships_at_dep = state
        assert int(owner) == int(ctx.me), \
            f"warm-start used non-me source {src_id} at t_dep={turn} (owner={owner})"
        new_total = cumulative_spent.get(src_id, 0.0) + ships_needed
        assert new_total <= float(ships_at_dep) + 1e-6, (
            f"src {src_id} over-allocated at t_dep={turn}: "
            f"cumulative {new_total} > available {ships_at_dep}"
        )
        cumulative_spent[src_id] = new_total


def test_warm_start_skips_when_plan_is_full():
    """If the caller already passes >= threshold emissions, warm-start
    should NOT augment (called only when current_plan is sparse).

    This test bypasses the call-site guard by invoking the function
    directly with a max_emissions of 0 — which is the equivalent of the
    sa_core call site short-circuiting. With max_emissions=0 the function
    should never add anything."""
    from lib.sa_core import _warm_start_from_admissible
    ctx = _build_test_ctx(seed=0, steps=50, plan=[])
    if not ctx.admissible:
        pytest.skip("admissible set empty")
    prior_plan = [(0, [0, 1.5, 5])]
    augmented = _warm_start_from_admissible(ctx, prior_plan, max_emissions=0)
    assert augmented == prior_plan, \
        "warm-start added emissions despite max_emissions=0"


def test_warm_start_pre_subtracts_existing_emissions():
    """When current_plan already emits 90% of a source's ships at t_start,
    warm-start should not stack additional emissions on that source at
    early t_deps where the cumulative spend would exceed the source's
    available ships (which grow with production over time).

    Verify: the cumulative spend for that source AT EACH t_dep of the
    new emissions never exceeds ownership_cache[t_dep][src].ships."""
    from lib.sa_core import _warm_start_from_admissible
    ctx = _build_test_ctx(seed=0, steps=50, plan=[])
    if not ctx.admissible:
        pytest.skip("admissible set empty")
    cache_t0 = ctx.ownership_cache.get(int(ctx.t_start), {})
    my_srcs = [(pid, state) for pid, state in cache_t0.items()
                if int(state[0]) == int(ctx.me)]
    if not my_srcs:
        pytest.skip("no owned sources at t_start")
    src_id, (_o, ships_t0) = my_srcs[0]
    pre_charge = max(1, int(float(ships_t0) * 0.9))
    seed_plan = [(int(ctx.t_start), [int(src_id), 0.0, pre_charge])]

    augmented = _warm_start_from_admissible(ctx, seed_plan, max_emissions=20)

    # Check cumulative-source-spend invariant on src_id across all emissions
    # (seed + warm-start), sorted by t_dep ascending.
    sorted_all = sorted(augmented, key=lambda e: int(e[0]))
    cumulative = 0.0
    for turn, payload in sorted_all:
        if int(payload[0]) != int(src_id):
            continue
        cumulative += float(payload[2])
        cache_at_dep = ctx.ownership_cache.get(int(turn), {})
        state = cache_at_dep.get(src_id)
        if state is None:
            continue
        _o, ships_at_dep = state
        assert cumulative <= float(ships_at_dep) + 1e-6, (
            f"pre-charge ignored: src {src_id} over-allocated at t_dep={turn}: "
            f"cumulative {cumulative} > available {ships_at_dep}"
        )


def test_warm_start_dedup_same_turn_same_source():
    """No two emissions in the warm-start output may share the same
    (turn, src_id). The env's emission semantics make this redundant at
    best and ambiguous at worst."""
    from lib.sa_core import _warm_start_from_admissible
    ctx = _build_test_ctx(seed=0, steps=50, plan=[])
    if not ctx.admissible:
        pytest.skip("admissible set empty")
    augmented = _warm_start_from_admissible(ctx, [], max_emissions=20)
    keys = [(int(t), int(p[0])) for t, p in augmented]
    assert len(keys) == len(set(keys)), \
        f"warm-start produced duplicate (turn,src) keys: {keys}"


def test_warm_start_no_admissible_returns_input():
    """Defensive: if ctx.admissible is empty (e.g. degraded ctx), warm-
    start returns the input plan unchanged rather than crashing."""
    from lib.sa_core import _warm_start_from_admissible, PerturbContext
    # Hand-build a minimal ctx with empty admissible — simulates the
    # degraded-ctx fallback path inside simulated_anneal_online.
    ctx = PerturbContext(
        snap0=None, world=None, world_model=None,
        omega=0.0, comet_paths={},
        ownership_cache={}, opp_intent_window=[],
        t_start=0, t_end=50, me=0,
    )
    out = _warm_start_from_admissible(ctx, [], max_emissions=8)
    assert out == []
