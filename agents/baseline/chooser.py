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

from lib.config import env_float, env_int
from lib.fast_sim import clone as fs_clone
from lib.fast_sim import step as fs_step
from lib.opp_model import lite_greedy_policy, top_tier_mirror_policy

from agents.baseline.value import select_favor_fn

WALLCLOCK_BUDGET_MS = 600.0
N_VALIDATE = 60
PER_CANDIDATE_SAFETY = 1.5
RESERVED_OVERHEAD_MS = 50.0

# Cost-model defaults (Phase 1; per-call env reads via lib.config so
# A/B harnesses that monkey-patch env vars between fixtures get the
# new value).
_DEFAULT_STEP_BASE_MS = 0.20
_DEFAULT_STEP_PER_FLEET_MS = 0.06
_DEFAULT_LEAF_BASE_MS = 0.80
_DEFAULT_LEAF_PER_FLEET_MS = 0.25
# Phase 3b extra leaf cost per credited my-fleet when
# COMPOSITE_FLEET_SURVIVAL_CHECK is on. predict_fleet_fate is ~1 ms
# per call on a 24-planet board (max_steps=200).
_DEFAULT_LEAF_FATE_MS_PER_FLEET = 1.0
# Smart-opp leaf window (Phase 4). Same per-call read pattern.
_DEFAULT_OPP_SMART_LEAF_WINDOW = 5


def _state_n_fleets(snap_base) -> int:
    """Total fleet count on the board. Read from snap.state[0].observation
    (fleets aren't seat-scoped). Falls back to a PESSIMISTIC default
    (20) on exception so the cost model errs on the side of a smaller
    cap when obs shape is unexpected — preferring fewer candidates over
    a budget blow-up.
    """
    try:
        obs0 = snap_base.state[0].observation
        fleets = (
            obs0.get("fleets", []) if hasattr(obs0, "get")
            else getattr(obs0, "fleets", [])
        )
        if fleets is not None:
            return len(fleets)
    except Exception:
        pass
    return 20


def _state_n_my_fleets(snap_base, me: int) -> int:
    """Count of fleets owned by `me`. Used by the Phase 3b cost-model
    term. Pessimistic-default same as _state_n_fleets."""
    try:
        obs0 = snap_base.state[0].observation
        fleets = (
            obs0.get("fleets", []) if hasattr(obs0, "get")
            else getattr(obs0, "fleets", [])
        )
        if fleets is None:
            return 5
        # Fleet tuple layout: (id, owner, x, y, angle, from_planet_id, ships).
        return sum(1 for f in fleets if int(f[1]) == int(me))
    except Exception:
        return 5


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


def opp_actions_for_step(snap, me: int, num_seats: int,
                         *, smart_leaf: bool = False) -> list[list]:
    """Per-step opp action selector that optionally swaps in
    `top_tier_mirror_policy` at the leaf window (Phase 4, 2026-05-22).

    When `BASELINE_OPP_SMART_LEAF=1` AND `smart_leaf=True`, the policy
    used for THIS step is `top_tier_mirror_policy` (full v3.5.1
    snipe + reinforce pipeline, ~5-10 ms/call). Otherwise behaves
    identically to `opp_actions_for_snap` — the existing selected
    policy (lite_greedy unless `BASELINE_OPP_TIER=1`).

    `smart_leaf=True` is passed by callers for the FINAL
    `BASELINE_OPP_SMART_LEAF_WINDOW` steps of the rollout (default 5)
    so smart-opp launches have time to propagate to their targets
    before the leaf is evaluated — a single last-step smart launch
    doesn't shift the leaf value because the new fleet hasn't yet
    affected any planet's ownership / production.

    Rationale: the leaf state's opp army composition determines its
    value. Running the full pipeline at every step blows the per-turn
    budget; running it ONLY at the leaf window adds N × 5-10 ms per
    candidate. Default OFF preserves pre-Phase-4 behavior bit-identically.

    Used by both `build_idle_baseline` / `score_action` (here) AND
    `build_trajectory_baseline` / `score_candidate_v4` / `_joint`
    (in chooser_trajectory.py via import).
    """
    if smart_leaf and os.environ.get(
        "BASELINE_OPP_SMART_LEAF", "0",
    ).strip() == "1":
        policy = top_tier_mirror_policy
    else:
        policy = _select_opp_policy()
    actions: list[list] = [[] for _ in range(num_seats)]
    for opp_id in range(num_seats):
        if opp_id == me:
            continue
        try:
            actions[opp_id] = policy(snap.state[opp_id].observation) or []
        except Exception:
            actions[opp_id] = []
    return actions


def opp_smart_leaf_window() -> int:
    """Per-call read of BASELINE_OPP_SMART_LEAF_WINDOW. The previous
    module-level read froze the value at import time — tests/A-B
    harnesses that monkey-patched the env var between fixtures got the
    import-time default silently. Per-call read closes that footgun.
    """
    return env_int("BASELINE_OPP_SMART_LEAF_WINDOW", _DEFAULT_OPP_SMART_LEAF_WINDOW)


def build_idle_baseline(snap_base, me: int, num_seats: int,
                        max_horizon: int, gamma: float) -> list[float]:
    """favor at every horizon 0..max_horizon under (me-idle, opp-reactive).

    Phase 4 (2026-05-22): when `BASELINE_OPP_SMART_LEAF=1`, the final
    step (the one that produces `out[max_horizon]`) uses the smart
    opp model. This matches `score_action`'s leaf-step policy so
    `Δ = leaf − baseline[horizon]` stays calibrated.
    """
    favor_fn = select_favor_fn()
    snap = fs_clone(snap_base)
    out = [favor_fn(snap.state[me].observation, me, num_seats, gamma=gamma)]
    leaf_window_start = max(0, max_horizon - opp_smart_leaf_window())
    for step_i in range(max_horizon):
        if snap.fake_env.done:
            out.append(out[-1])
            continue
        is_leaf_step = (step_i >= leaf_window_start)
        actions = opp_actions_for_step(
            snap, me, num_seats, smart_leaf=is_leaf_step,
        )
        snap = fs_step(snap, actions, in_place=True)
        out.append(favor_fn(snap.state[me].observation, me, num_seats, gamma=gamma))
    return out


def score_action(snap_base, me: int, num_seats: int,
                 src_id: int, angle: float, ships: int,
                 horizon: int, baseline_favors: list[float],
                 wait_N: int, gamma: float,
                 baseline_horizon: int | None = None) -> float:
    """Δ favor at horizon = leaf(my_action@wait_N) − baseline.

    Phase 4 (2026-05-22, calibrated 2026-05-23): when
    `BASELINE_OPP_SMART_LEAF=1`, the smart-opp swap fires at the
    ABSOLUTE rollout step `baseline_horizon - WINDOW` onward. This is
    the SAME absolute step the baseline build used, so candidates at
    horizon < baseline_horizon - WINDOW never see smart-opp and their
    Δ = leaf - baseline_favors[h] stays apples-to-apples
    (both endpoints used lite_greedy throughout for steps < the absolute
    window). The pre-2026-05-23 code anchored the window per-candidate
    horizon, which biased every shorter-horizon candidate's Δ.

    `baseline_horizon` defaults to `horizon` for callers that didn't
    pass it (pre-Phase-4 callers); in that case the smart-opp swap fires
    at the candidate's own tail, matching the older behaviour.
    """
    if baseline_horizon is None:
        baseline_horizon = horizon
    favor_fn = select_favor_fn()
    snap = fs_clone(snap_base)
    leaf_window_start = max(0, baseline_horizon - opp_smart_leaf_window())
    for step_i in range(horizon):
        if snap.fake_env.done:
            break
        is_leaf_step = (step_i >= leaf_window_start)
        actions = opp_actions_for_step(
            snap, me, num_seats, smart_leaf=is_leaf_step,
        )
        if step_i == int(wait_N):
            actions[me] = [[int(src_id), float(angle), int(ships)]]
        snap = fs_step(snap, actions, in_place=True)
    leaf = favor_fn(snap.state[me].observation, me, num_seats, gamma=gamma)
    return leaf - baseline_favors[horizon]


def affordable_validate_cap(snap_base, me: int, num_seats: int,
                            max_horizon: int, wallclock_ms: float,
                            min_horizon: int, gamma: float,
                            adaptive_k_extension: int = 0,
                            ) -> tuple[int, float]:
    """Deterministic per-candidate cap. Returns `(cap, per_cand_ms)`.

    State-aware cost model (Phase 1, calibrated 2026-05-23):
        per_step_ms = STEP_BASE + STEP_PER_FLEET * n_fleets
        per_leaf_ms = LEAF_BASE + LEAF_PER_FLEET * n_fleets
                      + LEAF_FATE_PER_FLEET * n_my_fleets   (if Phase 3b)
        per_cand_ms = (per_step_ms * effective_avg_K + per_leaf_ms) * SAFETY
        effective_avg_K = (min_horizon + max_horizon + adaptive_k_extension) / 2

    `adaptive_k_extension`: pass ADAPTIVE_K_BUMP (e.g. 10) when Phase 2
    is enabled, so the cost model sees the higher avg K that critical
    candidates run at. Default 0 = un-bumped.

    Phase 3b leaf cost: added automatically when COMPOSITE_FLEET_SURVIVAL_CHECK
    is on (per-call env read). Each my-fleet incurs one extra
    predict_fleet_fate call inside composite_capture_value.

    Guards:
    - per_cand_ms clamped to >= 0.001 to avoid ZeroDivisionError if a
      diagnostic A/B zeroes all cost constants.
    - `n_fleets` falls back to 20 on obs-shape exceptions (pessimistic).

    `me`, `gamma` retained for signature-compat.
    """
    n_fleets = _state_n_fleets(snap_base)
    n_my_fleets = _state_n_my_fleets(snap_base, me) if me is not None else 0
    del gamma  # signature-compat

    step_base = env_float("BASELINE_STEP_BASE_MS", _DEFAULT_STEP_BASE_MS)
    step_pf = env_float("BASELINE_STEP_PER_FLEET_MS", _DEFAULT_STEP_PER_FLEET_MS)
    leaf_base = env_float("BASELINE_LEAF_BASE_MS", _DEFAULT_LEAF_BASE_MS)
    leaf_pf = env_float("BASELINE_LEAF_PER_FLEET_MS", _DEFAULT_LEAF_PER_FLEET_MS)
    per_step_ms = step_base + step_pf * n_fleets
    per_leaf_ms = leaf_base + leaf_pf * n_fleets
    # Phase 3b: extra per-my-fleet leaf cost when survival check is on.
    if os.environ.get("COMPOSITE_FLEET_SURVIVAL_CHECK", "0").strip() in ("1", "true", "on", "yes"):
        fate_pf = env_float(
            "BASELINE_LEAF_FATE_MS_PER_FLEET", _DEFAULT_LEAF_FATE_MS_PER_FLEET,
        )
        per_leaf_ms += fate_pf * n_my_fleets

    avg_K = (min_horizon + max_horizon + int(adaptive_k_extension)) / 2.0
    per_cand_ms = (per_step_ms * avg_K + per_leaf_ms) * PER_CANDIDATE_SAFETY
    # ZeroDivisionError guard: if a diagnostic sets all cost constants
    # to 0, per_cand_ms is 0. Clamp to a tiny positive value so the cap
    # comes out at N_VALIDATE bound (deterministic, doesn't crash).
    if per_cand_ms < 0.001:
        per_cand_ms = 0.001
    budget = wallclock_ms - RESERVED_OVERHEAD_MS
    if budget <= 0:
        # Caller passed an absurdly small wallclock_ms; floor cap at 8
        # (the explicit minimum) so we don't degrade to 0 candidates.
        return 8, per_cand_ms
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

    n_aff, _per_cand_ms = affordable_validate_cap(
        snap_base, me, num_seats, max_horizon, wallclock_ms,
        min_horizon, gamma,
    )
    top = prerank[: min(N_VALIDATE, n_aff)]

    # Count-based budget only (Phase 1, 2026-05-22): n_aff IS the cap;
    # the prior wallclock pre-bail at `safe_deadline` was a second
    # non-determinism source (loop exited at varying indexes across
    # runs depending on probe timing). With deterministic per_cand_ms
    # and N_VALIDATE bound the budget math is enough.
    validated: list[tuple] = []
    for _cheap, src, tgt, ships, angle, _eta, horizon, wait_N in top:
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
            baseline_horizon=int(max_horizon),
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
