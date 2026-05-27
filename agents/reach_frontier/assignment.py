"""Hungarian assignment over (sources × targets) for the reach-frontier chooser.

Wraps `lib.joint_solver.lp.build_assignment_matrix` + `solve_assignment` +
`extract_moves` to pick the multi-source move set per turn. One launch
per source (matches the env's per-source launch constraint), distinct
targets per launch (lp.py's `extract_moves` de-duplicates).

The doctrine §6 "defend" slot is encoded via lp.py's diagonal noop column
(cost 0). A negative-reward launch loses to the noop; a positive-reward
launch wins. Explicit defend-reward (reinforce mine planet) is deferred
to a v2 axis — for v1 the noop slot is sufficient because the reach
table doesn't propose same-source-as-target candidates.
"""

from __future__ import annotations

from lib.joint_solver.columns import Column
from lib.joint_solver.lp import build_assignment_matrix, extract_moves, solve_assignment
from lib.trajectory import predict_fleet_fate

from agents.reach_frontier.hold import per_candidate_reward


def _columns_from_reach(
    reach_table,
    world,
    me: int,
    hold_times,
    *,
    lambda_risk: float,
    lambda_loss: float,
) -> list[Column]:
    """Materialise one Column per (src, tgt, k) candidate, computing value
    via the doctrine reward function.

    Columns with value ≤ 0 are still emitted; lp.py routes them to the
    noop slot (build_assignment_matrix drops cells with `value <= 0`
    from the pair-column set, so they can't beat the noop).
    """
    cols: list[Column] = []
    next_id = 0
    me_id = int(me)
    for (src_id, tgt_id), entries in reach_table.items():
        target = world.planets_by_id.get(int(tgt_id))
        if target is None:
            continue
        hold = float(hold_times.get(int(tgt_id), 0.0))
        for entry in entries:
            # Skip candidates that physically can't capture an opp/neutral
            # planet — they'd just bounce off without flipping ownership.
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
) -> list[list]:
    """Return env-shape `[[src_id, angle, ships], ...]` after Hungarian.

    Each emitted move passes a final `predict_fleet_fate` validate —
    columns whose closed-form aim doesn't survive the per-step ray-cast
    are dropped. Matches doctrine §3's "physics filter on chosen" stage.
    """
    columns = _columns_from_reach(
        reach_table, world, me, hold_times,
        lambda_risk=lambda_risk, lambda_loss=lambda_loss,
    )
    if not columns:
        return []
    cost_matrix, _src_ids, col_to_column = build_assignment_matrix(columns)
    row_ind, col_ind = solve_assignment(cost_matrix)
    raw_moves = extract_moves(row_ind, col_ind, col_to_column)

    # Physics filter: only emit moves whose predicted fate is "target".
    out: list[list] = []
    for src_id, angle, ships in raw_moves:
        src = world.planets_by_id.get(int(src_id))
        if src is None:
            continue
        # Recover the target this column landed on. extract_moves doesn't
        # return tgt_id; re-derive by matching the Column from the
        # col_to_column map.
        col = next(
            (c for c in col_to_column.values()
             if int(c.src_id) == int(src_id)
             and int(c.ships) == int(ships)
             and abs(float(c.angle) - float(angle)) < 1e-9),
            None,
        )
        if col is None:
            continue
        target = world.planets_by_id.get(int(col.tgt_id))
        if target is None:
            continue
        fate = predict_fleet_fate(
            src, target, float(angle), int(ships), world,
        )
        if (fate.outcome != "target"
                or int(fate.hit_planet_id or -1) != int(target.id)):
            continue
        out.append([int(src_id), float(angle), int(ships)])
    return out
