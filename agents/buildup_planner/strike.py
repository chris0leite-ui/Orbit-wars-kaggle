"""STRIKE phase (Step 3 — STUB).

When Step 3 lands this module will provide:

    step(world, model, me, plan, step_now) -> list[moves]

For each `Shot` in `plan.shots` whose firing tick matches `step_now`
(= plan.arrival_step - shot.eta), re-validate via
`lib.trajectory.predict_fleet_fate`. Emit `[src_id, angle, ships]` iff
outcome == "target". On ANY validation failure the whole strike is
atomically dropped, the dispatcher's strike_plan is cleared, and
control returns to CONSOLIDATION.

Step 1 ships this stub so dispatcher routing is testable. The dispatcher
never calls it because `predicates.evaluate_inflection` always returns
None in Step 1.
"""
from __future__ import annotations

from agents.buildup_planner.predicates import StrikePlan


def step(world, model, me: int, plan: StrikePlan,
         step_now: int) -> list[list]:
    """Step 1 stub — always returns empty. Step 3 will implement."""
    return []
