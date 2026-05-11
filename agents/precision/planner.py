"""Global joint planner.

Each turn, decide the joint set of launches across all our planets to maximize
long-horizon score margin. v0 uses greedy marginal-value selection over the
shot menu, with a deadline-aware early exit. Depth-1 minimax is deferred to a
follow-up — the precision physics + global greedy is already a strong baseline.
"""
from __future__ import annotations

import time
from agents.precision import intercept, prediction


def plan_turn(
    world: dict,
    deadline: float,
    max_shots: int = 12,
    horizon_steps: int = 200,
) -> list[intercept.Shot]:
    """Return the list of Shots to launch this turn.

    Strategy: greedy add — at each step, pick the shot with highest marginal
    plan-score improvement, respecting per-source garrison budgets. Stop when
    no shot improves the score (or deadline approaches).
    """
    me = world["player"]

    # Spend up to 35% of the budget on shot-menu construction; remainder is search.
    menu_deadline = deadline - 0.65 * max(deadline - time.perf_counter(), 0)
    menu = intercept.build_shot_menu(world, deadline=menu_deadline)
    if not menu:
        return []

    # Flatten + deduplicate.
    candidates: list[intercept.Shot] = []
    for shots in menu.values():
        candidates.extend(shots)
    if not candidates:
        return []

    # Track remaining garrison per source.
    src_remaining: dict[int, int] = {}
    for p in world["planets"]:
        if p.owner == me:
            src_remaining[p.id] = p.ships

    chosen: list[intercept.Shot] = []
    base_score = prediction.plan_score(world, chosen, horizon_steps=horizon_steps)

    for _ in range(max_shots):
        if time.perf_counter() >= deadline:
            break

        # Find the shot with best marginal improvement.
        best_shot = None
        best_score = base_score
        for shot in candidates:
            if time.perf_counter() >= deadline:
                break
            if shot in chosen:
                continue
            avail = src_remaining.get(shot.src_id, 0)
            if shot.ship_count > avail:
                continue
            trial = chosen + [shot]
            score = prediction.plan_score(world, trial, horizon_steps=horizon_steps)
            if score > best_score:
                best_score = score
                best_shot = shot

        if best_shot is None:
            break

        chosen.append(best_shot)
        src_remaining[best_shot.src_id] -= best_shot.ship_count
        base_score = best_score

    return chosen


def emit_actions(plan: list[intercept.Shot]) -> list[list]:
    """Convert Shots to the engine's action format."""
    return [[shot.src_id, shot.angle, shot.ship_count] for shot in plan]
