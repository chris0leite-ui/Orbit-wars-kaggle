"""Goal-directed portfolio planner agent (P0).

PI 2026-05-19 PM pivot:
> "Replace value-maximization with a goal-directed portfolio planner —
> winning-state predicate (prod_advantage × remaining > opp_pool),
> smallest-sufficient-portfolio identifier, backwards-from-goal capture
> sequencing, portfolio-preservation defense. NO forward-projection in
> agent decision path."

Each turn:
  1. Identify opponent. If winning state already holds, skip acquisition.
  2. Find smallest portfolio of not-mine planets to flip the predicate.
  3. Defense first: protect existing mine planets from incoming threats
     (reserves source ships).
  4. Acquisition: schedule backwards-from-goal captures from remaining
     source budgets.
  5. Emit only the turn_offset==0 launches; re-plan next turn.

All decisions are closed-form. No fast_sim.step. Compute is bounded by
O(planets² × portfolio_size).
"""

from __future__ import annotations

from lib.goal_planner.defense import defense_actions
from lib.goal_planner.portfolio import smallest_winning_portfolio
from lib.goal_planner.predicate import is_winning_state
from lib.goal_planner.sequencer import backwards_acquisition_plan
from lib.trajectory_layer import World


def agent(obs, configuration=None):
    world = World.from_obs(obs, configuration)
    my_id = world.my_id
    other_owners = [p.owner for p in world.planets
                    if p.owner not in (-1, my_id)]
    if not other_owners:
        return []
    opp_id = max(set(other_owners), key=other_owners.count)
    if my_id == opp_id:
        return []

    reservations: dict[int, list[tuple[int, int]]] = {}

    # Defense first: protect existing production. Defense reserves ships
    # at wait_offset=0 so subsequent acquisition can't over-commit.
    defense = defense_actions(world, my_id, opp_id, reservations)

    # Acquisition: skip entirely if already winning (defense-only mode).
    acquisition: list = []
    if not is_winning_state(world, my_id, opp_id):
        portfolio = smallest_winning_portfolio(world, my_id, opp_id)
        if portfolio:
            acquisition = backwards_acquisition_plan(world, my_id, portfolio)
            # Roll acquisition reservations into the shared pool — although
            # defense already ran, future-turn re-plans will see this.
            for L in acquisition:
                reservations.setdefault(L.src_id, []).append(
                    (L.turn_offset, L.ships),
                )

    # Emit only turn_offset==0 launches this turn.
    emits: list[list] = []
    for L in defense + acquisition:
        if L.turn_offset == 0:
            emits.append([L.src_id, L.aim_angle, L.ships])
    return emits
