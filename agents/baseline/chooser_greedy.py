"""Sequential-greedy-with-resimulation chooser (Rule 49 / joint-coordination).

The champion (`chooser_trajectory`) scores every launch in its OWN solo
rollout, then greedily emits a deconflicted subset — so the turn it plays is
never simulated as a whole. That overcounts shared-benefit launches (waste)
and is blind to teamwork captures (two fleets that only win together).

This chooser instead scores each launch's marginal value CONDITIONAL on the
set already chosen this turn:

    gain(c | S) = V(S ∪ {c}) − V(S) = delta(S ∪ {c}) − delta(S)

where `delta(T)` is the joint-rollout value of the launch set T (the shared
`− baseline` term cancels in the difference). Because `fast_sim` is fully
deterministic (opp policies are RNG-free), these marginal gains are EXACT —
no seed-averaging, common-random-numbers is free.

Rungs (all default-OFF behind `BASELINE_CHOOSER=greedy`; the champion
`trajectory` path is byte-for-byte unchanged):
  1. Sequential greedy with re-simulation (conditional marginal gain).
  2. CELF / lazy-greedy acceleration (re-evaluate only the heap top).
  3. Coalition atoms (sync pairs as single candidates → teamwork reachable).
  4. Multi-resolution horizon (shallow build + one deep confirm, drop the
     marginal leg if depth flips its sign).

Reuses the tested joint-rollout primitive `score_candidate_v4_joint`, the
idle-baseline builder, the shared `generate_sync_coalitions` generator, and
the wallclock primitives from `chooser.py`. No new physics.

Full framing: knowledge-base/concepts/joint-coordination-planner.md.
"""

from __future__ import annotations

import heapq
import os
import time

from agents.baseline.chooser import (
    HARDCAP_BAIL_SENTINEL,
    WALLCLOCK_HARD_CAP_MS,
    affordable_validate_cap,
)
from agents.baseline.chooser_trajectory import (
    JOINT_SYNC_MAX_PAIRS,
    JOINT_SYNC_SETTLE,
    MIN_DELTA,
    build_trajectory_baseline,
    generate_sync_coalitions,
    score_candidate_v4_joint,
    select_favor_fn,
)

RESERVED_OVERHEAD_MS = 50.0


def _envint(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _envflag(name: str, default: str) -> bool:
    return os.environ.get(name, default).strip().lower() in ("1", "true", "on", "yes")


def _used_tgts_locked() -> bool:
    """Mirror the trajectory emit's `used_tgts` lock semantics: lifted when
    BASELINE_JOINT_AGGR=1 (multi-source-same-target stacking). Read at call
    time so tests/A-B subprocesses can toggle it."""
    return os.environ.get("BASELINE_JOINT_AGGR", "0").strip() != "1"


def _build_pool(prerank, world, model, me, max_horizon,
                reserved_srcs, reserved_for_new_commits, pool_cap,
                want_coalitions):
    """Unified candidate list. Each candidate is a dict:
      {'launches': [(src,tgt,ships,angle,wait_N), ...],
       'srcs': frozenset[int], 'tgts': frozenset[int], 'kind': str}.

    Singletons from `prerank[:pool_cap]` (after the same reserved-src filter
    `choose_trajectory` applies); coalition atoms appended from the shared
    sync generator when `want_coalitions`.
    """
    pool: list[dict] = []
    for row in prerank[:pool_cap]:
        _cheap, src, tgt, ships, angle, _eta, _ph, wait_N = row
        sid = int(src.id)
        if int(wait_N) > 0:
            if sid in reserved_for_new_commits:
                continue
        elif sid in reserved_srcs:
            continue
        pool.append({
            "launches": [(src, tgt, int(ships), float(angle), int(wait_N))],
            "srcs": frozenset({sid}),
            "tgts": frozenset({int(tgt.id)}),
            "kind": "solo",
        })
    if want_coalitions:
        for launches, _tarr in generate_sync_coalitions(
                world, model, me, max_horizon,
                reserved_srcs, reserved_for_new_commits,
                max_pairs=JOINT_SYNC_MAX_PAIRS):
            pool.append({
                "launches": launches,
                "srcs": frozenset(int(L[0].id) for L in launches),
                "tgts": frozenset(int(L[1].id) for L in launches),
                "kind": "coalition",
            })
    return pool


def _delta(snap_base, launches, me, num_seats, world, baseline_favors,
           favor_fn, gamma, horizon, hard_deadline):
    """Joint-rollout value of a launch set (delta = leaf − baseline). Empty
    set → 0 by construction (me-idle rollout reproduces the baseline)."""
    if not launches:
        return 0.0, "scored"
    return score_candidate_v4_joint(
        snap_base, launches, me, num_seats, world,
        baseline_favors, favor_fn, gamma,
        horizon=horizon, skip_admissibility=False, hard_deadline=hard_deadline,
    )


def choose_greedy(snap_base, prerank, baseline_favors,
                  me: int, num_seats: int, wallclock_ms: float,
                  min_horizon: int, max_horizon: int, gamma: float,
                  world, model,
                  reserved_srcs: set[int] | None = None,
                  reserved_for_new_commits: set[int] | None = None,
                  agent_deadline: float | None = None,
                  ) -> tuple[list[list], list[dict]]:
    """Signature-identical to `choose_trajectory`. Returns (moves, commits).

    `baseline_favors` is ignored (the greedy builds its own at the horizon it
    needs) — kept for call-site parity with the trajectory chooser.
    """
    if reserved_srcs is None:
        reserved_srcs = set()
    if reserved_for_new_commits is None:
        reserved_for_new_commits = reserved_srcs
    if not prerank:
        return [], []

    favor_fn = select_favor_fn()

    shallow_h = _envint("BASELINE_GREEDY_SHALLOW_H", 12)
    deep_h = _envint("BASELINE_GREEDY_DEEP_H", 40)
    max_picks = _envint("BASELINE_GREEDY_MAX_PICKS", 8)
    pool_cap = _envint("BASELINE_GREEDY_POOL_CAP", 40)
    lazy = _envflag("BASELINE_GREEDY_LAZY", "1")
    want_coalitions = _envflag("BASELINE_GREEDY_COALITIONS", "1")
    confirm = _envflag("BASELINE_GREEDY_CONFIRM", "0")
    overhead_ms = float(_envint("BASELINE_GREEDY_SAFE_OVERHEAD_MS", 60))
    floor = MIN_DELTA

    # Scale the pool down by opponent count: each rollout costs ~ (n_opps)×
    # the 2P cost, so a fixed candidate budget must shrink in 4P to keep the
    # init pass affordable.
    eff_cap = max(8, pool_cap // max(1, num_seats - 1))

    h_build = min(max_horizon, max(1, shallow_h))
    h_deep = min(max_horizon, max(1, deep_h))
    h_base = max(h_build, h_deep if confirm else 0)
    baseline_favors = build_trajectory_baseline(
        snap_base, me, num_seats, h_base, favor_fn, gamma)

    # Wallclock: probe per-candidate cost, reserve overhead for post-chain.
    _, per_cand_ms = affordable_validate_cap(
        snap_base, me, num_seats, h_build, wallclock_ms, h_build, gamma)
    now = time.perf_counter()
    deadline = now + wallclock_ms / 1000.0
    safe_deadline = deadline - (per_cand_ms + overhead_ms) / 1000.0
    hard_deadline = now + WALLCLOCK_HARD_CAP_MS / 1000.0
    if agent_deadline is not None:
        hard_deadline = min(hard_deadline, agent_deadline)
        safe_deadline = min(
            safe_deadline, agent_deadline - (per_cand_ms + overhead_ms) / 1000.0)

    pool = _build_pool(
        prerank, world, model, me, max_horizon,
        reserved_srcs, reserved_for_new_commits, eff_cap, want_coalitions)
    if not pool:
        return [], []

    used_srcs: set[int] = set()
    used_tgts: set[int] = set()
    tgt_locked = _used_tgts_locked()
    S: list[dict] = []
    S_launches: list = []
    delta_S = 0.0  # delta(∅) == 0 exactly (deterministic me-idle rollout)

    def _feasible(c: dict) -> bool:
        if c["srcs"] & used_srcs:
            return False
        if tgt_locked and (c["tgts"] & used_tgts):
            return False
        return True

    def _commit(c: dict, new_delta_S: float) -> None:
        nonlocal delta_S, S_launches
        S.append(c)
        S_launches = S_launches + c["launches"]
        used_srcs.update(c["srcs"])
        used_tgts.update(c["tgts"])
        delta_S = new_delta_S

    if lazy:
        # --- Rung 2: CELF lazy-greedy ---
        # Heap entries: (-gain, idx, eval_size, cand). gain over ∅ == delta({c}).
        heap: list = []
        for idx, c in enumerate(pool):
            if time.perf_counter() > safe_deadline:
                break
            d, st = _delta(snap_base, c["launches"], me, num_seats, world,
                           baseline_favors, favor_fn, gamma, h_build,
                           hard_deadline)
            if st == "hardcap_bail":
                break
            heapq.heappush(heap, (-d, idx, 0, c))

        while heap and len(S) < max_picks:
            if time.perf_counter() > safe_deadline:
                break
            neg_gain, idx, sz, c = heapq.heappop(heap)
            if not _feasible(c):
                continue
            if sz == len(S):
                gain = -neg_gain
                if gain <= floor:
                    break
                _commit(c, delta_S + gain)
                continue
            # Stale: recompute marginal gain at the current set, then either
            # commit (still best) or re-push (lost the top).
            d_union, st = _delta(snap_base, S_launches + c["launches"], me,
                                 num_seats, world, baseline_favors, favor_fn,
                                 gamma, h_build, hard_deadline)
            if st == "hardcap_bail":
                break
            gain = d_union - delta_S
            if heap and gain < -heap[0][0]:
                heapq.heappush(heap, (-gain, idx, len(S), c))
                continue
            if gain <= floor:
                break
            _commit(c, d_union)
    else:
        # --- Rung 1: exact O(N·k) greedy (A/B oracle for CELF) ---
        remaining = list(pool)
        while remaining and len(S) < max_picks:
            if time.perf_counter() > safe_deadline:
                break
            best = None  # (gain, d_union, cand)
            for c in remaining:
                if not _feasible(c):
                    continue
                if time.perf_counter() > safe_deadline:
                    break
                d_union, st = _delta(snap_base, S_launches + c["launches"], me,
                                     num_seats, world, baseline_favors,
                                     favor_fn, gamma, h_build, hard_deadline)
                if st == "hardcap_bail":
                    best = None
                    break
                gain = d_union - delta_S
                if best is None or gain > best[0]:
                    best = (gain, d_union, c)
            if best is None or best[0] <= floor:
                break
            _commit(best[2], best[1])
            remaining = [c for c in remaining if _feasible(c)]

    # --- Rung 4: deep confirm (drop the marginal element if depth flips it) ---
    if confirm and len(S) >= 1 and time.perf_counter() <= hard_deadline:
        d_full, st_full = _delta(snap_base, S_launches, me, num_seats, world,
                                 baseline_favors, favor_fn, gamma, h_deep,
                                 hard_deadline)
        if st_full == "scored":
            minus_launches: list = []
            for c in S[:-1]:
                minus_launches += c["launches"]
            d_minus, st_minus = _delta(snap_base, minus_launches, me, num_seats,
                                       world, baseline_favors, favor_fn, gamma,
                                       h_deep, hard_deadline)
            if st_minus == "scored" and d_minus > d_full:
                S = S[:-1]  # last add hurts at depth → drop it

    # --- Emit (matches choose_trajectory contract) ---
    moves: list[list] = []
    commits: list[dict] = []
    commit_step = int(world.step) if world is not None else 0
    for c in S:
        is_coalition = c["kind"] == "coalition"
        for src, tgt, ships, angle, wait_N in c["launches"]:
            if int(wait_N) == 0:
                moves.append([int(src.id), float(angle), int(ships)])
            else:
                commit = {
                    "src_id": int(src.id),
                    "tgt_id": int(tgt.id),
                    "ships_planned": int(ships),
                    "angle_original": float(angle),
                    "wait_remaining": int(wait_N),
                    "commit_step": commit_step,
                }
                if is_coalition:
                    commit["sync_joint"] = True
                commits.append(commit)
    return moves, commits
