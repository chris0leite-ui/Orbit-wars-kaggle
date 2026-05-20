"""Chooser: reactive idle baseline + per-candidate Δ, emit greedy non-dogpile.

Pipeline:
  baseline[h] = favor at horizon h with me idle + opp reactive
  candidate Δ = favor(me_action @ wait_N + opp reactive) - baseline[h]
  emit       = candidates with Δ>0, greedy by Δ desc,
               1 launch per source AND 1 per target per turn.
               wait_N>0 candidates are skipped without claiming the
               src/tgt slot, so a positive-Δ wait_N=0 alternate from
               the same source can still fire this turn.

Opp seats play lib.opp_model.lite_greedy_policy reactively inside every
rollout (not a precomputed trajectory), so my captures trigger opp
counter-launches and fragile leaves are correctly penalised.
"""

from __future__ import annotations

import time

from lib.fast_sim import clone as fs_clone
from lib.fast_sim import step as fs_step
from lib.opp_model import lite_greedy_policy as opp_policy

from agents.baseline.value import select_favor_fn

WALLCLOCK_BUDGET_MS = 600.0
N_VALIDATE = 60
PER_CANDIDATE_SAFETY = 1.5
RESERVED_OVERHEAD_MS = 50.0


def opp_actions_for_snap(snap, me: int, num_seats: int) -> list[list]:
    """One reactive lite_greedy action set per non-me seat."""
    actions: list[list] = [[] for _ in range(num_seats)]
    for opp_id in range(num_seats):
        if opp_id == me:
            continue
        try:
            actions[opp_id] = opp_policy(snap.state[opp_id].observation) or []
        except Exception:
            actions[opp_id] = []
    return actions


def build_idle_baseline(snap_base, me: int, num_seats: int,
                        max_horizon: int, gamma: float) -> list[float]:
    """favor at every horizon 0..max_horizon under (me-idle, opp-reactive)."""
    favor_fn = select_favor_fn()
    snap = fs_clone(snap_base)
    out = [favor_fn(snap.state[me].observation, me, num_seats, gamma=gamma)]
    for _ in range(max_horizon):
        if snap.fake_env.done:
            out.append(out[-1])
            continue
        actions = opp_actions_for_snap(snap, me, num_seats)
        snap = fs_step(snap, actions, in_place=True)
        out.append(favor_fn(snap.state[me].observation, me, num_seats, gamma=gamma))
    return out


def score_action(snap_base, me: int, num_seats: int,
                 src_id: int, angle: float, ships: int,
                 horizon: int, baseline_favors: list[float],
                 wait_N: int, gamma: float) -> float:
    """Δ favor at horizon = leaf(my_action@wait_N) − baseline."""
    favor_fn = select_favor_fn()
    snap = fs_clone(snap_base)
    for step_i in range(horizon):
        if snap.fake_env.done:
            break
        actions = opp_actions_for_snap(snap, me, num_seats)
        if step_i == int(wait_N):
            actions[me] = [[int(src_id), float(angle), int(ships)]]
        snap = fs_step(snap, actions, in_place=True)
    leaf = favor_fn(snap.state[me].observation, me, num_seats, gamma=gamma)
    return leaf - baseline_favors[horizon]


def affordable_validate_cap(snap_base, num_seats: int, max_horizon: int,
                            wallclock_ms: float, min_horizon: int) -> int:
    """Probe per-step cost on the current board, derive a safe candidate
    cap that fits inside the wallclock budget. Min cap = 8.
    """
    t0 = time.perf_counter()
    probe = fs_clone(snap_base)
    probe = fs_step(probe, [[] for _ in range(num_seats)], in_place=True)
    per_step_ms = max(0.05, (time.perf_counter() - t0) * 1000.0)
    avg_K = (min_horizon + max_horizon) / 2.0
    per_cand_ms = per_step_ms * avg_K * PER_CANDIDATE_SAFETY
    budget = wallclock_ms - RESERVED_OVERHEAD_MS
    return max(8, int(budget / per_cand_ms))


def choose(snap_base, prerank, baseline_favors: list[float],
           me: int, num_seats: int, wallclock_ms: float,
           min_horizon: int, max_horizon: int, gamma: float,
           migrations=None) -> list[list]:
    """Validate top candidates with fast_sim, emit greedy non-dogpile moves.

    `migrations` (optional) is a list of own→own repositioning candidates
    from `migration_solver.propose_migrations`. They use the solver's
    closed-form value as Δ directly (no fast_sim rollout, since favor
    delta on own→own is zero by construction).
    """
    if not prerank and not migrations:
        return []

    n_aff = affordable_validate_cap(
        snap_base, num_seats, max_horizon, wallclock_ms, min_horizon,
    )
    top = prerank[: min(N_VALIDATE, n_aff)] if prerank else []

    deadline = time.perf_counter() + wallclock_ms / 1000.0
    validated: list[tuple] = []
    for _cheap, src, tgt, ships, angle, _eta, horizon, wait_N in top:
        if time.perf_counter() > deadline:
            break
        delta = score_action(
            snap_base, me, num_seats,
            int(src.id), float(angle), int(ships),
            int(horizon), baseline_favors, int(wait_N), gamma,
        )
        if delta > 0:
            validated.append((delta, src, tgt, ships, angle, wait_N))

    # Migration candidates: closed-form value IS Δ, fire-now (wait_N=0).
    # No fast_sim rollout since own→own moves are favor-neutral.
    for c in (migrations or []):
        cheap, src, tgt, ships, angle, _eta, _horizon, _wait = c
        if float(cheap) > 0:
            validated.append(
                (float(cheap), src, tgt, int(ships), float(angle), 0)
            )

    if not validated:
        return []

    validated.sort(key=lambda c: -c[0])
    used_srcs: set[int] = set()
    used_tgts: set[int] = set()
    moves: list[list] = []
    for _delta, src, tgt, ships, angle, wait_N in validated:
        sid, tid = int(src.id), int(tgt.id)
        if sid in used_srcs or tid in used_tgts:
            continue
        if int(wait_N) > 0:
            # Don't claim the slot — let a positive-Δ wait_N=0 alternate
            # from the same src/tgt fire instead. (btjeK audit: 248/248
            # positive-Δ idle turns had wait_N>0 as the top scorer; the
            # old reserve-without-emit rule was the dominant cause of
            # mid-game under-emission.)
            continue
        used_srcs.add(sid)
        used_tgts.add(tid)
        moves.append([sid, float(angle), int(ships)])
    return moves
