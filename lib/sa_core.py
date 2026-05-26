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
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Callable

from lib.fast_sim import from_obs as fs_from_obs
from lib.fast_sim import rollout as fs_rollout
from lib.fast_sim import ship_totals


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


def perturb(plan: list[Emission], rng: random.Random,
            *, initial_planets: list | None = None,
            t_start: int = 0, t_end: int = 200) -> list[Emission]:
    """One uniform random local edit. See module docstring for ops.

    'add' samples turn from [t_start, t_end). Used by online SA to keep
    new emissions inside the receding horizon (no actions in the past).
    """
    can_add = initial_planets is not None and len(initial_planets) >= 2
    ops = ["remove", "ships", "shift", "angle"]
    if can_add:
        ops.append("add")
    if not plan and not can_add:
        return list(plan)
    op = rng.choice(ops)
    new_plan = list(plan)

    if op == "add":
        src_p = rng.choice(initial_planets)
        tgt_p = rng.choice(initial_planets)
        while int(tgt_p[0]) == int(src_p[0]):
            tgt_p = rng.choice(initial_planets)
        sx, sy = float(src_p[2]), float(src_p[3])
        tx, ty = float(tgt_p[2]), float(tgt_p[3])
        angle = math.atan2(ty - sy, tx - sx)
        ships = rng.choice([10, 20, 30, 50, 80, 120, 200])
        turn_hi = max(t_start + 1, t_end)
        turn = rng.randrange(t_start, turn_hi)
        new_plan.append((turn, [int(src_p[0]), float(angle), int(ships)]))
        return new_plan

    if not new_plan:
        return new_plan
    idx = rng.randrange(len(new_plan))
    if op == "remove":
        new_plan.pop(idx)
    elif op == "ships":
        t, action = new_plan[idx]
        src, ang, ships = action
        new_ships = max(1, int(ships * rng.uniform(0.7, 1.3)))
        new_plan[idx] = (t, [src, ang, new_ships])
    elif op == "shift":
        t, action = new_plan[idx]
        new_t = max(0, t + rng.choice([-2, -1, 1, 2]))
        new_plan[idx] = (new_t, action)
    elif op == "angle":
        t, action = new_plan[idx]
        src, ang, ships = action
        new_ang = ang + rng.uniform(-0.2, 0.2)
        new_plan[idx] = (t, [src, float(new_ang), ships])
    return new_plan


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

    temp = t0
    for i in range(n_iter):
        if deadline is not None and time.perf_counter() >= deadline:
            history.append((i, current_score, best_score))
            break
        new_plan = perturb(current_plan, rng,
                            initial_planets=initial_planets,
                            t_start=start_step, t_end=t_end_perturb)
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
