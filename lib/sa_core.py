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
import random
from typing import Callable

from lib.fast_sim import rollout as fs_rollout
from lib.fast_sim import ship_totals


Policy = Callable[[object], list]
Emission = tuple[int, list]


def _noop_policy(_obs) -> list:
    return []


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
                         max_steps: int = 200) -> float:
    """Replay `emissions` over `snap` against `opp_policy`; return P0 ships.

    fs_rollout(in_place=False) clones the snap internally so the caller's
    snap object is unchanged — safe to reuse across SA iterations.
    """
    plan_by_turn = _emissions_to_plan_dict(emissions)
    if opp_policy is None:
        opp_policy = _noop_policy

    def replay(obs) -> list:
        t = _get_step(obs)
        return [list(a) for a in plan_by_turn.get(t, [])]

    snap = fs_rollout(snap, K=max_steps,
                      policies=[replay, opp_policy], in_place=False)
    return float(ship_totals(snap).get(0, 0.0))


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
                             ) -> tuple[list[Emission], float, list]:
    """Metropolis SA from a given snapshot.

    Returns (best_plan, best_score, history). `history` is a sparse list
    of (iter, current_score, best_score) tuples.

    `start_step` constrains add-perturbations so they don't generate
    actions for turns in the past (which would be no-ops anyway, but
    waste SA iterations).
    """
    t_end_perturb = max(start_step + 1, start_step + max_steps)

    current_plan = list(initial_plan)
    current_score = score_plan_from_snap(current_plan, snap0, opp_policy, max_steps)
    best_plan = list(current_plan)
    best_score = current_score
    history: list[tuple[int, float, float]] = []

    temp = t0
    for i in range(n_iter):
        new_plan = perturb(current_plan, rng,
                            initial_planets=initial_planets,
                            t_start=start_step, t_end=t_end_perturb)
        new_score = score_plan_from_snap(new_plan, snap0, opp_policy, max_steps)
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
