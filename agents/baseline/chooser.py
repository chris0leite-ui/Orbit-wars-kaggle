"""Chooser: reactive idle baseline + per-candidate Δ, emit greedy non-dogpile.

Pipeline:
  baseline[h] = favor at horizon h with me idle + opp reactive
  candidate Δ = favor(me_action @ wait_N + opp reactive) - baseline[h]
  emit       = candidates with Δ>0, greedy by Δ desc,
               1 launch per source AND 1 per target per turn.
               wait_N>0 winners RESERVE source+target but emit nothing.

Opp seats play lib.opp_model.lite_greedy_policy reactively inside every
rollout (not a precomputed trajectory), so my captures trigger opp
counter-launches and fragile leaves are correctly penalised.
"""

from __future__ import annotations

import os
import time

from lib.fast_sim import clone as fs_clone
from lib.fast_sim import step as fs_step
from lib.opp_model import lite_greedy_policy, top_tier_mirror_policy

from agents.baseline.value import select_favor_fn

WALLCLOCK_BUDGET_MS = 700.0
N_VALIDATE = 60
PER_CANDIDATE_SAFETY = 1.5
RESERVED_OVERHEAD_MS = 50.0


def _select_opp_policy():
    """Tier 3 (2026-05-18 PM): asymmetric opp model selection.

    BASELINE_OPP_TIER env var:
      - "0" or unset → lite_greedy_policy (default, ~1-2ms/call).
      - "1" → top_tier_mirror_policy (~5-10ms/call; ladder-realistic
              opp using v3.5.1 aggressive snipe pipeline). Bench gate
              FIRST before A/B — per-call cost is 5-10× lite_greedy.

    Per-call selection (not cached at import time) so env-var overrides
    inside test fixtures take effect without re-importing the module.
    """
    return (
        top_tier_mirror_policy
        if os.environ.get("BASELINE_OPP_TIER", "0").strip() == "1"
        else lite_greedy_policy
    )


def opp_actions_for_snap(snap, me: int, num_seats: int) -> list[list]:
    """One reactive opp action set per non-me seat. Opp policy is
    selected via BASELINE_OPP_TIER — see `_select_opp_policy`."""
    opp_policy = _select_opp_policy()
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


def affordable_validate_cap(snap_base, me: int, num_seats: int,
                            max_horizon: int, wallclock_ms: float,
                            min_horizon: int, gamma: float,
                            ) -> tuple[int, float]:
    """Probe per-step + per-leaf cost on the current board, derive a
    safe candidate cap and the per-candidate cost estimate.

    Returns `(cap, per_cand_ms)`. `cap` is bounded below by 8. The
    `per_cand_ms` value is used by `choose()` to pre-bail before
    entering a candidate that would push past the deadline.

    Probing per-leaf cost matters because the leaf eval cost varies
    by ~50x between value heads (favor ~100µs vs composite_capture_value
    ~2-5ms — it builds a World + ray-casts every fleet). Without the
    leaf probe the cap stayed sized for favor and composite blew the
    1000ms env budget on heavy turns (max 1292ms vs v15 / v9_scavenge,
    2026-05-17 A/B).
    """
    favor_fn = select_favor_fn()
    t0 = time.perf_counter()
    probe = fs_clone(snap_base)
    probe = fs_step(probe, [[] for _ in range(num_seats)], in_place=True)
    per_step_ms = max(0.05, (time.perf_counter() - t0) * 1000.0)

    t0 = time.perf_counter()
    favor_fn(probe.state[me].observation, me, num_seats, gamma=gamma)
    per_leaf_ms = max(0.05, (time.perf_counter() - t0) * 1000.0)

    avg_K = (min_horizon + max_horizon) / 2.0
    per_cand_ms = (per_step_ms * avg_K + per_leaf_ms) * PER_CANDIDATE_SAFETY
    budget = wallclock_ms - RESERVED_OVERHEAD_MS
    cap = max(8, int(budget / per_cand_ms))
    return cap, per_cand_ms


def choose(snap_base, prerank, baseline_favors: list[float],
           me: int, num_seats: int, wallclock_ms: float,
           min_horizon: int, max_horizon: int, gamma: float,
           world=None,
           reserved_srcs: set[int] | None = None,
           reserved_for_new_commits: set[int] | None = None,
           ) -> tuple[list[list], list[dict]]:
    """Validate top candidates with fast_sim, emit greedy non-dogpile moves.

    Returns `(moves, commits)`. See `chooser_trajectory.choose_trajectory`
    for the full ledger-aware contract; this is the parallel composite
    implementation (default chooser is trajectory).
    """
    if reserved_srcs is None:
        reserved_srcs = set()
    if reserved_for_new_commits is None:
        reserved_for_new_commits = reserved_srcs
    if not prerank:
        return [], []

    n_aff, per_cand_ms = affordable_validate_cap(
        snap_base, me, num_seats, max_horizon, wallclock_ms,
        min_horizon, gamma,
    )
    top = prerank[: min(N_VALIDATE, n_aff)]

    deadline = time.perf_counter() + wallclock_ms / 1000.0
    # Pre-bail headroom: don't ENTER a candidate that would push us past
    # the deadline. score_action is uninterruptible (runs the full K-step
    # rollout once entered), so checking AT the deadline is too late.
    # Closes the long-tail max-turn-ms overrun seen in the 2026-05-17 A/B.
    safe_deadline = deadline - (per_cand_ms / 1000.0)
    validated: list[tuple] = []
    for _cheap, src, tgt, ships, angle, _eta, horizon, wait_N in top:
        if time.perf_counter() > safe_deadline:
            break
        sid_ = int(src.id)
        if int(wait_N) > 0:
            if sid_ in reserved_for_new_commits:
                continue
        else:
            if sid_ in reserved_srcs:
                continue
        delta = score_action(
            snap_base, me, num_seats,
            int(src.id), float(angle), int(ships),
            int(horizon), baseline_favors, int(wait_N), gamma,
        )
        if delta > 0:
            validated.append((delta, src, tgt, ships, angle, wait_N))

    if not validated:
        return [], []

    validated.sort(key=lambda c: -c[0])
    used_srcs: set[int] = set()
    used_tgts: set[int] = set()
    moves: list[list] = []
    commits: list[dict] = []
    commit_step = int(world.step) if world is not None else 0
    for _delta, src, tgt, ships, angle, wait_N in validated:
        sid, tid = int(src.id), int(tgt.id)
        if sid in used_srcs or tid in used_tgts:
            continue
        used_srcs.add(sid)
        used_tgts.add(tid)
        if int(wait_N) == 0:
            moves.append([sid, float(angle), int(ships)])
        else:
            commits.append({
                "src_id": sid,
                "tgt_id": tid,
                "ships_planned": int(ships),
                "angle_original": float(angle),
                "wait_remaining": int(wait_N),
                "commit_step": commit_step,
            })
    return moves, commits
