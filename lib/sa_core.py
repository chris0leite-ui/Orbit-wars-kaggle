"""SA primitives shared between solo solver and online MPC agent.

Three pure functions, no module-level state:

  score_plan_from_snap(emissions, snap, opp_policy, max_steps)
      Replay emissions for seat 0 against opp_policy for seat 1 starting
      from `snap` (any step). Returns P0 terminal ships. Clones the snap
      internally so the original is untouched.

  perturb(plan, rng, *, initial_planets=None, t_start=0, t_end=200)
      One uniform random local edit. Five ops (remove / modify ships /
      shift turn / nudge angle / add). 'add' only fires when
      `initial_planets` is provided; it samples turn from [t_start,
      t_end), so receding-horizon callers can constrain additions to
      future turns.

  simulated_anneal_online(initial_plan, snap0, max_steps, opp_policy,
                          n_iter, t0, cooling, rng, *,
                          start_step=0, initial_planets=None)
      Metropolis SA loop. Returns (best_plan, best_score, history).

`emissions` is `list[tuple[turn:int, [src:int, angle:float, ships:int]]]`.

Design constraint (PI 2026-05-26): keep this module simple and pure so
every change has a clear correctness story. No global state. No side
effects beyond what the rng draws.
"""
from __future__ import annotations

import math
import os
import random
import time
from dataclasses import dataclass, field
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any, Callable, Optional

from lib.aim import aim_comet, aim_orbiting
from lib.fast_sim import clone as fs_clone
from lib.fast_sim import from_obs as fs_from_obs
from lib.fast_sim import rollout as fs_rollout
from lib.fast_sim import ship_totals
from lib.fast_sim import step as fs_step
from lib.intent import World
from lib.orbit import predict_relative
from lib.trajectory import predict_fleet_fate
from lib.world_model import WorldModel, _comet_paths_by_id, predict_garrison_at


Policy = Callable[[object], list]
Emission = tuple[int, list]


def _noop_policy(_obs) -> list:
    return []


# ---------------------------------------------------------------------------
# Agent + env helpers (used by both scripts/sa_solo_solver.py and
# agents/sa_online/main.py). Kept here so the bundler can pick them up via
# --lib sa_core; the bundler doesn't know about scripts/.
# ---------------------------------------------------------------------------

def load_agent(path):
    """Load a kaggle-style agent function from a .py file path."""
    spec = spec_from_file_location("a", str(path))
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.agent


def build_solo_snap0(seed: int, steps: int):
    """Build a turn-0 snapshot for the solo (vs noop) game on `seed`."""
    from kaggle_environments import make
    env = make("orbit_wars",
               configuration={"seed": seed, "episodeSteps": steps},
               debug=False)
    env.reset(num_agents=2)
    obs0 = env.steps[0][0]["observation"] if isinstance(env.steps[0][0], dict) else env.steps[0][0].observation
    return fs_from_obs(obs0, env.configuration,
                       episode_seed=seed, num_seats=2)


def record_initial_plan(seed: int, steps: int, agent_path,
                        opp_path=None, noop_default_path=None):
    """Run focal-vs-opp via env.run, log every focal emission.

    `agent_path`: focal agent. `opp_path`: opponent agent (default noop).
    `noop_default_path`: path to the noop agent (caller-provided to keep
    this function decoupled from any REPO constant).

    Returns (emissions_list, env_terminal_ships, n_steps, initial_planets).
    """
    from kaggle_environments import make
    agent_fn = load_agent(agent_path)
    if opp_path is None:
        if noop_default_path is None:
            raise ValueError("opp_path or noop_default_path must be provided")
        opp_path = noop_default_path
    opp_fn = load_agent(opp_path)
    emissions: list[tuple[int, list]] = []

    def recorder(obs):
        t = _get_step(obs)
        acts = agent_fn(obs)
        for a in acts:
            emissions.append((t, [int(a[0]), float(a[1]), int(a[2])]))
        return acts

    env = make("orbit_wars",
               configuration={"seed": seed, "episodeSteps": steps},
               debug=False)
    env.reset(num_agents=2)
    obs0 = env.steps[0][0]["observation"] if isinstance(env.steps[0][0], dict) else env.steps[0][0].observation
    od0 = obs0 if isinstance(obs0, dict) else dict(obs0)
    initial_planets = [list(p) for p in (od0.get("planets") or [])]
    env.run([recorder, opp_fn])
    final = env.steps[-1]
    obs_f = final[0]["observation"] if isinstance(final[0], dict) else final[0].observation
    odf = obs_f if isinstance(obs_f, dict) else dict(obs_f)
    planets = odf.get("planets") or []
    fleets = odf.get("fleets") or []
    p0_ships = sum(float(p[5]) for p in planets if int(p[1]) == 0) + \
               sum(float(f[6]) for f in fleets if int(f[1]) == 0)
    return emissions, p0_ships, len(env.steps), initial_planets


def _get_step(obs) -> int:
    if isinstance(obs, dict):
        return int(obs.get("step", 0))
    return int(getattr(obs, "step", 0))


def _emissions_to_plan_dict(emissions: list[Emission]) -> dict[int, list[list]]:
    plan: dict[int, list[list]] = {}
    for t, action in emissions:
        plan.setdefault(int(t), []).append(list(action))
    return plan


@dataclass
class PerturbContext:
    """Shared state passed to every perturbation operator.

    Built once per `simulated_anneal_online` call (not per iteration) via
    `_build_perturb_context`, which runs a single forward simulation from
    `snap0` recording per-turn ownership + opp emissions.

    Operators are pure functions of (plan, rng, ctx) — easy to add new
    ones (e.g. future multi-source `add_wave_*` operators) without
    restructuring the dispatch.
    """
    snap0: Any                                       # fast_sim.Snapshot at SA start
    world: Any                                       # World.from_obs(snap0.state[0].observation)
    world_model: Any                                 # WorldModel.from_world(world)
    omega: float
    comet_paths: dict                                # planet_id -> (path_list, base_index)
    ownership_cache: dict                            # turn -> {planet_id: (owner_id, ships)}
    opp_intent_window: list                          # [(turn, target_planet_id), ...]
    t_start: int
    t_end: int
    me: int = 0                                      # which seat we are


def _target_tuple_from_planet(p) -> list:
    """6-element planet descriptor expected by `aim_orbiting` /
    `aim_comet`: [id, owner, x, y, radius, prod]."""
    return [int(p.id), int(p.owner), float(p.x), float(p.y),
            float(p.radius), float(p.production)]


def _planet_position_at(planet_tuple: list, omega: float, turn: int):
    """Predict (x, y) for a planet at turn `turn` from snap0.

    For turn=0 the current position is correct. For turn>0 we use the
    closed-form orbital prediction (predict_relative). Static planets
    return their current position (predict_relative is a no-op).
    """
    if turn <= 0 or omega == 0.0:
        return (float(planet_tuple[2]), float(planet_tuple[3]))
    return predict_relative(planet_tuple, float(omega), int(turn))


def _compute_capture_emission(src, tgt, turn: int,
                              ctx: PerturbContext) -> Optional[tuple]:
    """Compute a physics-valid emission that captures `tgt` from `src`.

    Returns `(turn, [src.id, angle, ships])` or `None` if no valid
    emission exists. Generation rules (all checked):
      1. `aim_orbiting` / `aim_comet` returns a valid lead (else None)
      2. `predict_garrison_at` says we can capture (owner != me at arrival)
      3. ships = predicted_garrison + 1 (exact capture cost)
      4. `predict_fleet_fate.outcome == "target"` (no sun / OOB / wrong planet)
      5. `src` owns enough ships at turn `t` (from ownership cache; falls
         back to snap0 for t=0)

    Every emission this returns is guaranteed valid by construction.
    """
    src_tuple = _target_tuple_from_planet(src)
    tgt_tuple = _target_tuple_from_planet(tgt)

    src_xy = _planet_position_at(src_tuple, ctx.omega, turn)
    tgt_xy = _planet_position_at(tgt_tuple, ctx.omega, turn)
    tgt_tuple_at_turn = [tgt_tuple[0], tgt_tuple[1],
                         tgt_xy[0], tgt_xy[1],
                         tgt_tuple[4], tgt_tuple[5]]

    # Provisional ships count for aim_orbiting (it just sets fleet speed,
    # which affects eta only weakly). Refine after garrison prediction.
    provisional_ships = max(1, int(tgt.ships) + 1)
    is_comet = int(tgt.id) in (getattr(ctx.world, "comet_ids", None) or ())

    if is_comet and int(tgt.id) in ctx.comet_paths:
        cpath, base_idx = ctx.comet_paths[int(tgt.id)]
        aim_result = aim_comet(src_xy, float(src.radius), tgt_tuple_at_turn,
                                float(tgt.radius), provisional_ships,
                                cpath, float(base_idx) + float(turn))
    else:
        aim_result = aim_orbiting(src_xy, float(src.radius), tgt_tuple_at_turn,
                                   float(tgt.radius), provisional_ships,
                                   float(ctx.omega))
    if aim_result is None:
        return None
    angle, _arrival_xy, eta_float = aim_result
    eta = max(1, int(math.ceil(float(eta_float))))

    # Predicted garrison at arrival. Use the WorldModel's ledger for the
    # incumbent fleets; we don't include this candidate yet (it's what
    # we're computing). predict_garrison_at takes eta = turns from snap0
    # (where the world_model was built). If `turn` > ctx.t_start (we are
    # WAITING before firing), the arrival from snap0's perspective is
    # `(turn - t_start) + eta_local` — which is the absolute number of
    # ticks between snap0 and the fleet arriving.
    arrivals: list = []
    if ctx.world_model is not None:
        try:
            arrivals = list(ctx.world_model.ledger.get(int(tgt.id), []))
        except Exception:
            arrivals = []
    turn_offset_from_snap0 = max(0, int(turn) - int(ctx.t_start))
    arrival_eta_from_snap0 = turn_offset_from_snap0 + int(eta)
    try:
        pred_owner, pred_garrison = predict_garrison_at(
            tgt, arrival_eta_from_snap0, arrivals)
    except Exception:
        pred_owner, pred_garrison = int(tgt.owner), float(tgt.ships)
    if int(pred_owner) == int(ctx.me):
        return None  # we already own the planet at arrival; no capture needed
    ships = max(1, int(math.ceil(float(pred_garrison))) + 1)

    # Confirm src has enough at turn t. Ownership cache stores ships count
    # for turn `t` from the forward sim. Fall back to snap0 for early t.
    cache_entry = ctx.ownership_cache.get(int(turn))
    if cache_entry is not None:
        src_state = cache_entry.get(int(src.id))
        if src_state is None:
            return None
        cur_owner, cur_ships = src_state
        if int(cur_owner) != int(ctx.me) or int(cur_ships) < ships:
            return None
    else:
        # No cache entry for this turn (out of window). Use snap0 (turn 0).
        if int(src.owner) != int(ctx.me) or int(src.ships) < ships:
            return None

    # Final physics validation. Most failures already filtered above; this
    # catches sun / OOB / wrong-planet collisions the orbital lead missed.
    try:
        fate = predict_fleet_fate(src, tgt, angle, ships, ctx.world)
    except Exception:
        return None
    if fate.outcome != "target":
        return None

    return (int(turn), [int(src.id), float(angle), int(ships)])


def _build_perturb_context(snap0,
                            plan: list[Emission],
                            opp_policy: Optional[Callable],
                            max_steps: int,
                            t_start: int,
                            t_end: int,
                            me: int = 0) -> PerturbContext:
    """One forward sim from snap0 records ownership-at-turn + opp's intended
    captures. Cost ~50 ms per refine; amortised across all SA iterations.

    Forward sim uses the CURRENT plan as our policy and `opp_policy` for
    opp. After each step, records:
      - ownership_cache[turn] = {planet_id: (owner_id, ships)}
      - opp_intent_window appends (turn, target_id) for each opp emission
    """
    if opp_policy is None:
        opp_policy = _noop_policy

    plan_by_turn = _emissions_to_plan_dict(plan)

    def _replay(obs):
        t = _get_step(obs)
        return [list(a) for a in plan_by_turn.get(t, [])]

    # Build world / world_model from the snap0 obs.
    obs0_struct = snap0.state[0].observation
    obs0_d: dict = {}
    for k in ("player", "step", "planets", "fleets", "comets",
              "comet_planet_ids", "angular_velocity"):
        v = getattr(obs0_struct, k, None)
        if v is not None:
            obs0_d[k] = list(v) if isinstance(v, list) else v
    try:
        world = World.from_obs(obs0_d)
        world_model = WorldModel.from_world(world)
        omega = float(world.omega)
        comet_paths = (_comet_paths_by_id(world)
                        if getattr(world, "comet_ids", None) else {})
    except Exception:
        # Degraded ctx: opp-aware operators will silently no-op.
        world = None
        world_model = None
        omega = float(obs0_d.get("angular_velocity", 0.0))
        comet_paths = {}

    ownership_cache: dict[int, dict[int, tuple]] = {}
    opp_intent_window: list[tuple[int, int]] = []

    sim_snap = fs_clone(snap0)
    sim_step = max(0, int(t_start))
    # Record ownership AT the start of each turn within the window.
    for offset in range(max(1, int(max_steps))):
        current_turn = sim_step + offset
        if current_turn >= t_end:
            break
        # Record ownership state BEFORE this turn's actions are applied.
        state_p0 = sim_snap.state[0]
        planets_now = list(getattr(state_p0.observation, "planets", []) or [])
        ownership_cache[current_turn] = {
            int(p[0]): (int(p[1]), int(p[5])) for p in planets_now
        }
        # Determine each seat's action for this step.
        if sim_snap.fake_env.done:
            break
        actions = []
        for seat in range(sim_snap.num_seats):
            seat_obs = sim_snap.state[seat].observation
            if seat == me:
                actions.append(_replay(seat_obs))
            else:
                try:
                    actions.append(list(opp_policy(seat_obs)))
                except Exception:
                    actions.append([])
        # Extract opp's intended target from each opp emission.
        if world is not None:
            for seat, action in enumerate(actions):
                if seat == me or not action:
                    continue
                for emit in action:
                    try:
                        src_id = int(emit[0])
                        angle = float(emit[1])
                        ships = int(emit[2])
                    except (TypeError, ValueError, IndexError):
                        continue
                    src_p = world.planets_by_id.get(src_id)
                    if src_p is None:
                        continue
                    try:
                        fate = predict_fleet_fate(src_p, src_p, angle,
                                                   max(1, ships), world)
                    except Exception:
                        continue
                    if fate.hit_planet_id is not None:
                        opp_intent_window.append(
                            (current_turn, int(fate.hit_planet_id)))
        # Advance the snap.
        try:
            sim_snap = fs_step(sim_snap, actions, in_place=True)
        except Exception:
            break

    return PerturbContext(
        snap0=snap0,
        world=world,
        world_model=world_model,
        omega=omega,
        comet_paths=comet_paths,
        ownership_cache=ownership_cache,
        opp_intent_window=opp_intent_window,
        t_start=int(t_start),
        t_end=int(t_end),
        me=int(me),
    )


def score_plan_from_snap(emissions: list[Emission],
                         snap,
                         opp_policy: Policy | None = None,
                         max_steps: int = 200,
                         *,
                         me: int = 0,
                         score_mode: str = "absolute") -> float:
    """Replay `emissions` over `snap` against `opp_policy`; return a score.

    `me`: which seat owns `emissions` (the seat we score for).
    `score_mode`:
        "absolute" -> ships_me at terminal (default, backward compatible)
        "diff"     -> ships_me - max_o(ships_o for o != me)

    The "diff" mode makes denying-opp inherently positive marginal value,
    fixing the MPC pessimism trap where every aggressive plan looked
    equally bad against a strong fixed opp model.

    `me` also controls which seat's policy plays `emissions`. If me==0
    the policy ordering is [replay, opp_policy]; if me==1 it's
    [opp_policy, replay]. This lets co-evolution search from opp's POV.

    fs_rollout(in_place=False) clones the snap internally so the caller's
    snap object is unchanged — safe to reuse across SA iterations.
    """
    plan_by_turn = _emissions_to_plan_dict(emissions)
    if opp_policy is None:
        opp_policy = _noop_policy

    def replay(obs) -> list:
        t = _get_step(obs)
        return [list(a) for a in plan_by_turn.get(t, [])]

    if int(me) == 0:
        policies = [replay, opp_policy]
    else:
        policies = [opp_policy, replay]

    snap = fs_rollout(snap, K=max_steps,
                      policies=policies, in_place=False)
    totals = ship_totals(snap)
    me_ships = float(totals.get(int(me), 0.0))
    if score_mode == "diff":
        opp_ships = max(
            (float(v) for k, v in totals.items() if int(k) != int(me) and int(k) >= 0),
            default=0.0,
        )
        return me_ships - opp_ships
    return me_ships


def _owned_srcs_at_turn(ctx: PerturbContext, turn: int) -> list[int]:
    """Return ids of planets we own at `turn` (from ownership_cache)."""
    entry = ctx.ownership_cache.get(int(turn))
    if entry is None:
        return []
    return [pid for pid, (owner, _ships) in entry.items()
            if int(owner) == int(ctx.me)]


def _op_remove(plan: list[Emission], rng: random.Random,
               ctx: PerturbContext) -> Optional[list[Emission]]:
    """Drop a random emission. Returns None if plan is empty."""
    if not plan:
        return None
    new_plan = list(plan)
    idx = rng.randrange(len(new_plan))
    new_plan.pop(idx)
    return new_plan


_FIRE_TURN_TRIES = 4   # how many random fire-turn draws each add_* operator attempts


def _sample_fire_turn(rng: random.Random, ctx: PerturbContext) -> int:
    """Sample a fire turn from [t_start, t_end). Including t_start lets the
    operator emit NOW; later turns let SA explore wait-then-fire candidates
    (a src that can't afford the capture today may have enough ships at
    t_start + k after k turns of production)."""
    hi = max(ctx.t_start + 1, ctx.t_end)
    return rng.randrange(int(ctx.t_start), int(hi))


def _try_add_for_target(plan, rng, ctx, tgt) -> Optional[list[Emission]]:
    """Try multiple (fire_turn, src) draws to land one valid capture
    emission for `tgt`. Each draw uses closed-form physics + waiting-aware
    ship accounting via `_compute_capture_emission`. Returns the new plan
    on first success, else None.

    The K=4 draws cover the "wait until I can afford it" axis: if no src
    has enough ships at t_start, a later turn may work because production
    accumulates."""
    for _ in range(_FIRE_TURN_TRIES):
        fire_turn = _sample_fire_turn(rng, ctx)
        owned = _owned_srcs_at_turn(ctx, fire_turn)
        if not owned:
            continue
        rng.shuffle(owned)
        for src_id in owned:
            if int(src_id) == int(tgt.id):
                continue
            src = ctx.world.planets_by_id.get(int(src_id))
            if src is None:
                continue
            emit = _compute_capture_emission(src, tgt, fire_turn, ctx)
            if emit is not None:
                new_plan = list(plan)
                new_plan.append(emit)
                return new_plan
    return None


def _op_add_contested(plan: list[Emission], rng: random.Random,
                       ctx: PerturbContext) -> Optional[list[Emission]]:
    """Add a physics-valid emission targeting a planet opp wants to capture.

    Sample a target from `opp_intent_window`. Sample a fire turn from
    `[t_start, t_end)` (including the wait dimension PI flagged 2026-05-27
    as essential — sometimes we need a few turns of accumulation before
    we can afford the capture). Compute aim + exact ships via closed-form
    physics. Returns `None` if no valid (turn, src) combination.
    """
    if not ctx.opp_intent_window or ctx.world is None:
        return None
    _opp_turn, tgt_id = rng.choice(ctx.opp_intent_window)
    tgt = ctx.world.planets_by_id.get(int(tgt_id))
    if tgt is None:
        return None
    return _try_add_for_target(plan, rng, ctx, tgt)


def _op_add_uncontested(plan: list[Emission], rng: random.Random,
                         ctx: PerturbContext) -> Optional[list[Emission]]:
    """Add a physics-valid emission targeting a high-π planet OUTSIDE the
    contested set. Safe production growth. Same wait-aware turn sampling
    as `_op_add_contested`."""
    if ctx.world is None:
        return None
    contested_ids = {int(tid) for _t, tid in ctx.opp_intent_window}
    horizon_remaining = max(0, ctx.t_end - ctx.t_start)
    candidates = []
    for pid, p in ctx.world.planets_by_id.items():
        if int(pid) in contested_ids:
            continue
        if int(p.owner) == int(ctx.me):
            continue
        score = float(p.production) * horizon_remaining
        if score <= 0:
            continue
        candidates.append((score, int(pid)))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    top = candidates[:max(1, min(8, len(candidates)))]
    _score, tgt_id = rng.choice(top)
    tgt = ctx.world.planets_by_id.get(int(tgt_id))
    if tgt is None:
        return None
    return _try_add_for_target(plan, rng, ctx, tgt)


def _op_modify_target(plan: list[Emission], rng: random.Random,
                       ctx: PerturbContext) -> Optional[list[Emission]]:
    """Pick a random emission, swap its target to a contested one,
    recompute aim + ships. Same src and turn; new target."""
    if not plan or ctx.world is None or not ctx.opp_intent_window:
        return None
    idx = rng.randrange(len(plan))
    turn, action = plan[idx]
    try:
        src_id = int(action[0])
    except (TypeError, ValueError, IndexError):
        return None
    src = ctx.world.planets_by_id.get(src_id)
    if src is None:
        return None
    # New target from contested set.
    _opp_turn, new_tgt_id = rng.choice(ctx.opp_intent_window)
    if int(new_tgt_id) == int(src_id):
        return None
    new_tgt = ctx.world.planets_by_id.get(int(new_tgt_id))
    if new_tgt is None:
        return None
    emit = _compute_capture_emission(src, new_tgt, int(turn), ctx)
    if emit is None:
        return None
    new_plan = list(plan)
    new_plan[idx] = emit
    return new_plan


def _op_shift_turn(plan: list[Emission], rng: random.Random,
                    ctx: PerturbContext) -> Optional[list[Emission]]:
    """Move an existing emission to a different fire turn, recomputing
    aim + ships from the new (turn, src, tgt) triple.

    PI 2026-05-27: waiting is essential — many good plans require firing
    LATER (accumulate ships, time orbital alignment, sequence captures).
    This operator perturbs the wait axis explicitly. Target identity is
    recovered from the current emission's angle+ships via predict_fleet
    _fate; new turn is sampled near the current one.
    """
    if not plan or ctx.world is None:
        return None
    idx = rng.randrange(len(plan))
    turn, action = plan[idx]
    try:
        src_id = int(action[0])
        angle_existing = float(action[1])
        ships_existing = max(1, int(action[2]))
    except (TypeError, ValueError, IndexError):
        return None
    src = ctx.world.planets_by_id.get(src_id)
    if src is None:
        return None
    # Recover target id from the existing emission's trajectory.
    try:
        fate_existing = predict_fleet_fate(src, src, angle_existing,
                                            ships_existing, ctx.world)
    except Exception:
        return None
    if fate_existing.hit_planet_id is None:
        return None
    tgt = ctx.world.planets_by_id.get(int(fate_existing.hit_planet_id))
    if tgt is None:
        return None
    # Sample a nearby new turn. Bias toward small shifts but allow larger
    # ones so SA can rearrange the plan's tempo.
    delta = rng.choice([-8, -5, -3, -2, -1, 1, 2, 3, 5, 8])
    new_turn = max(int(ctx.t_start),
                    min(int(ctx.t_end) - 1, int(turn) + int(delta)))
    if new_turn == int(turn):
        return None
    emit = _compute_capture_emission(src, tgt, new_turn, ctx)
    if emit is None:
        return None
    new_plan = list(plan)
    new_plan[idx] = emit
    return new_plan


# Operator dispatch table. Order is just the uniform-random pool; weighting
# is uniform for v1. Adding a future `_op_add_wave_*` is one entry below
# plus the new function — no infrastructure changes.
_PERTURB_OPS: tuple = (
    _op_remove,
    _op_add_contested,
    _op_add_uncontested,
    _op_modify_target,
    _op_shift_turn,
)


def perturb(plan: list[Emission], rng: random.Random,
            ctx: PerturbContext) -> list[Emission]:
    """One physics-valid, opp-aware local edit.

    Dispatches uniformly among 4 operators (remove / add_contested /
    add_uncontested / modify_target). Every emission an operator
    inserts is validated by `_compute_capture_emission` so that the
    plan never contains a physics-failing or under-strength action.

    Operators can return None (no valid candidate this draw); in that
    case we fall back to `_op_remove` if the plan is non-empty, else
    return the plan unchanged.

    Future extensibility: add an `_op_add_wave_*` for multi-source
    coordination by appending it to `_PERTURB_OPS`.
    """
    op = rng.choice(_PERTURB_OPS)
    result = op(plan, rng, ctx)
    if result is not None:
        return result
    # Fallback chain: try remove if the plan has anything to drop.
    if plan and op is not _op_remove:
        fallback = _op_remove(plan, rng, ctx)
        if fallback is not None:
            return fallback
    return list(plan)


def simulated_anneal_online(initial_plan: list[Emission],
                             snap0,
                             max_steps: int,
                             opp_policy: Policy | None,
                             n_iter: int,
                             t0: float,
                             cooling: float,
                             rng: random.Random,
                             *,
                             start_step: int = 0,
                             initial_planets: list | None = None,
                             max_wall_s: float | None = None,
                             me: int = 0,
                             score_mode: str = "absolute",
                             ) -> tuple[list[Emission], float, list]:
    """Metropolis SA from a given snapshot.

    Returns (best_plan, best_score, history). `history` is a sparse list
    of (iter, current_score, best_score) tuples.

    `start_step` constrains add-perturbations so they don't generate
    actions for turns in the past (which would be no-ops anyway, but
    waste SA iterations).

    `max_wall_s`: optional soft wallclock deadline. The loop breaks
    early once the deadline is exceeded. Set to keep per-turn refines
    inside kaggle's actTimeout regardless of opp_policy cost.

    `me` + `score_mode`: passed through to score_plan_from_snap. See
    that function's docstring; "diff" mode (ships_me - max_o ships_o)
    fixes the MPC pessimism trap.
    """
    t_end_perturb = max(start_step + 1, start_step + max_steps)

    deadline = (time.perf_counter() + max_wall_s) if max_wall_s is not None else None

    current_plan = list(initial_plan)
    current_score = score_plan_from_snap(
        current_plan, snap0, opp_policy, max_steps,
        me=me, score_mode=score_mode)
    best_plan = list(current_plan)
    best_score = current_score
    history: list[tuple[int, float, float]] = []

    # Build the perturbation context ONCE — ownership cache + opp-intent
    # window are amortised across all SA iterations. Cost ~20-50 ms.
    try:
        ctx = _build_perturb_context(
            snap0, current_plan, opp_policy,
            max_steps=max_steps, t_start=start_step,
            t_end=t_end_perturb, me=me,
        )
    except Exception:
        # Degraded ctx: no opp-aware operators, fallback to remove-only.
        ctx = PerturbContext(
            snap0=snap0, world=None, world_model=None,
            omega=0.0, comet_paths={},
            ownership_cache={}, opp_intent_window=[],
            t_start=int(start_step), t_end=int(t_end_perturb), me=int(me),
        )

    temp = t0
    for i in range(n_iter):
        if deadline is not None and time.perf_counter() >= deadline:
            history.append((i, current_score, best_score))
            break
        new_plan = perturb(current_plan, rng, ctx)
        new_score = score_plan_from_snap(
            new_plan, snap0, opp_policy, max_steps,
            me=me, score_mode=score_mode)
        delta = new_score - current_score
        if delta > 0 or rng.random() < math.exp(delta / max(1e-9, temp)):
            current_plan = new_plan
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_plan = list(current_plan)
        temp *= cooling
        if i % 50 == 0 or i == n_iter - 1:
            history.append((i, current_score, best_score))
    return best_plan, best_score, history
