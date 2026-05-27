"""Multi-source assignment for the reach-frontier chooser.

v2 rev (audit/2026-05-27-rf-v1-root-cause.md Bug 2 fix): replaced the
bipartite Hungarian (build_assignment_matrix + extract_moves) with
`lib.joint_solver.lp.solve_multi_turn` so the same target can be hit
by multiple sources ("gang-up"). The bipartite path's
`if tid in used_tgts: continue` dedup was forcing same-target
collisions to silently drop one source per turn — confirmed in the
seed-0 trace where the chooser had two positive-value columns both
targeting tgt=33 and emitted only one.

solve_multi_turn (`lib/joint_solver/lp.py:286`) uses scipy.optimize.milp
with two constraints: (a) per-source per-time-window ship budget
(prevents over-committing one source); (b) per-target gang-up cap
(`max_contesters_per_target`, default 3). Falls back to a pure-Python
greedy when scipy.milp is absent.

Same final pass as before: physics-validate each emitted move via
`predict_fleet_fate` and drop anything that doesn't land
`outcome=="target"`.
"""

from __future__ import annotations

from lib.joint_solver.columns import Column
from lib.joint_solver.lp import solve_multi_turn
from lib.trajectory import predict_fleet_fate

from agents.reach_frontier.hold import per_candidate_reward


# Allow at most this many of our fleets to converge on a single target
# per turn. v1's bipartite assignment hard-banned this (cap=1) which
# left ~30% of our sources idle in turns with concentrated value.
# v2 cap of 3 matches `lib.joint_solver.lp.DEFAULT_MAX_CONTESTERS_PER_TARGET`
# and lets two sources team up plus one optional reinforce.
DEFAULT_MAX_CONTESTERS_PER_TARGET: int = 3


def _columns_from_reach(
    reach_table,
    world,
    me: int,
    hold_times,
    *,
    lambda_risk: float,
    lambda_loss: float,
) -> list[Column]:
    """Materialise one Column per (src, tgt, k) candidate with reward as value.

    Candidates that can't physically flip the target (`k ≤ expected_garrison`
    for non-mine targets) are dropped here so the LP doesn't waste a
    variable on them. Negative-value candidates are KEPT (so the LP can
    weigh them); solve_multi_turn drops `value ≤ 0` internally so they
    don't actually fire.
    """
    cols: list[Column] = []
    next_id = 0
    me_id = int(me)
    for (_src_id, tgt_id), entries in reach_table.items():
        target = world.planets_by_id.get(int(tgt_id))
        if target is None:
            continue
        hold = float(hold_times.get(int(tgt_id), 0.0))
        for entry in entries:
            if entry.target_owner_at_arrival != me_id:
                if float(entry.ships) <= float(entry.expected_garrison):
                    continue
            value = per_candidate_reward(
                entry, target, hold,
                lambda_risk=lambda_risk, lambda_loss=lambda_loss,
                risk=0.0,
            )
            cols.append(Column(
                column_id=next_id,
                src_id=int(entry.src_id),
                tgt_id=int(entry.target_id),
                ships=int(entry.ships),
                wait_N=0,
                angle=float(entry.aim_angle),
                eta=int(entry.arrival_tick),
                owner=me_id,
                value=float(value),
                horizon_hint=0,
                cheap_delta=float(entry.cost_tick),
                is_opp=False,
            ))
            next_id += 1
    return cols


def pick_actions(
    reach_table,
    hold_times,
    world,
    *,
    me: int,
    lambda_risk: float,
    lambda_loss: float,
    max_contesters_per_target: int = DEFAULT_MAX_CONTESTERS_PER_TARGET,
) -> list[list]:
    """Return env-shape `[[src_id, angle, ships], ...]` after MILP solve.

    Each emitted move passes a final `predict_fleet_fate` validate to
    drop closed-form aim candidates that don't survive the env's
    swept-pair check (the trajectory_roi failure mode flagged in
    Rule 47).
    """
    columns = _columns_from_reach(
        reach_table, world, me, hold_times,
        lambda_risk=lambda_risk, lambda_loss=lambda_loss,
    )
    if not columns:
        return []

    result = solve_multi_turn(
        columns, world,
        my_id=int(me),
        max_contesters_per_target=int(max_contesters_per_target),
        max_wait_N=0,         # v1 chooser is single-turn; no wait_N>0 cols
        time_limit_seconds=0.3,
    )

    # solve_multi_turn returns wait_N==0 moves only. Map back to the
    # column for physics-validate (need src + angle + ships + tgt).
    out: list[list] = []
    fired_by_key = {
        (int(c.src_id), int(c.tgt_id), int(c.ships), float(c.angle)): c
        for c in result.fired_columns
    }
    for src_id, angle, ships in result.moves:
        col = fired_by_key.get(
            (int(src_id), -1, int(ships), float(angle)),  # placeholder tgt
            None,
        )
        # Re-find by (src, ships, angle) match — the placeholder above
        # never matches; iterate fired_columns instead.
        if col is None:
            for c in result.fired_columns:
                if (int(c.src_id) == int(src_id)
                        and int(c.ships) == int(ships)
                        and abs(float(c.angle) - float(angle)) < 1e-9):
                    col = c
                    break
        if col is None:
            continue
        src = world.planets_by_id.get(int(src_id))
        target = world.planets_by_id.get(int(col.tgt_id))
        if src is None or target is None:
            continue
        fate = predict_fleet_fate(
            src, target, float(angle), int(ships), world,
        )
        if (fate.outcome != "target"
                or int(fate.hit_planet_id or -1) != int(target.id)):
            continue
        out.append([int(src_id), float(angle), int(ships)])
    return out
