"""cascade_chooser — deterministic plan selector over the cascade-DAG.

Replaces the SA loop in agents/sa_online with iterative ctx-rebuild +
greedy selection. Each rebuild's forward-sim of the accumulated plan
turns prior captures into valid sources for new captures (cascade-aware
admissibility). Pair-joint enumeration adds multi-source coordination
(S1 + S2 → T with same t_arr; combat-rule-1 pools ships at arrival).

Pure functions; no module-level state. Seat-aware via explicit `me`.

Per-turn budget on Kaggle (1.6 CPUs, 1 s actTimeout): three rebuilds
× ~80 ms ctx-build + joint enumeration ≤ 100 ms + greedy ≤ 20 ms
≈ 280 ms, fits comfortably under the timeout.
"""
from __future__ import annotations

import math
import time
from typing import Optional

from lib.sa_core import _build_perturb_context
from lib.sa_core import _capture_value
from lib.sa_core import PerturbContext
from lib.trajectory import predict_fleet_fate
from lib.world_model import predict_garrison_at


def _value_with_horizon(emit, idx: int, ctx: PerturbContext,
                         value_t_end: int) -> float:
    """Like `lib.sa_core._capture_value` but with an explicit ROI horizon.

    Rationale: cascade_greedy's admissibility enumeration uses a small
    horizon (e.g. 25 turns) to bound search cost, while ROI scoring
    should integrate production all the way to game-end. If we use the
    enumeration horizon for ROI, marginal captures with late arrival
    are massively underpriced — `production × 0 − ship_cost < 0` causes
    the `val > 0` gate to skip them, leaving the agent passive. n=16
    vs nearest = 2/16 with that bug (commit 57e7100); this restores
    the proper closed-form `production × game_remaining − ship_cost`.
    """
    turn, payload = emit
    if idx < 0 or idx >= len(ctx.admissible_targets):
        return -math.inf
    if ctx.world is None:
        return -math.inf
    tgt_id = ctx.admissible_targets[idx]
    tgt = ctx.world.planets_by_id.get(int(tgt_id))
    if tgt is None:
        return -math.inf
    eta = 1
    if ctx.path_graph is not None:
        edge = ctx.path_graph.lookup(int(payload[0]),
                                      int(tgt_id), int(turn))
        if edge is not None:
            eta = int(edge.eta)
    t_arr = int(turn) + int(eta)
    remaining = max(0, int(value_t_end) - t_arr)
    ship_cost = float(payload[2])
    return float(tgt.production) * float(remaining) - ship_cost


# ---- public tunables -------------------------------------------------------

DEFAULT_MAX_REBUILDS = 3
DEFAULT_MAX_PICKS_PER_REBUILD = 8
DEFAULT_JOINT_TOP_K_TARGETS = 20
DEFAULT_JOINT_TOP_K_PER_TARGET = 6
DEFAULT_TOTAL_WALL_S = 0.7
DEFAULT_JOINT_WALL_S = 0.1


# ---- internals -------------------------------------------------------------

def _candidate_targets(ctx: PerturbContext, *, top_k: int) -> list[int]:
    """Top-K non-`me`-owned target ids by weighted production × horizon.
    Mirrors `_populate_admissible_set`'s target ranking."""
    if ctx.world is None:
        return []
    contested = {int(tid) for _t, tid in ctx.opp_intent_window}
    horizon = max(1, int(ctx.t_end) - int(ctx.t_start))
    scored: list[tuple[float, int]] = []
    for pid, p in ctx.world.planets_by_id.items():
        if int(p.owner) == int(ctx.me):
            continue
        weight = 1.0 if int(pid) in contested else 0.5
        scored.append((weight * float(p.production) * float(horizon), int(pid)))
    scored.sort(reverse=True)
    return [pid for _s, pid in scored[: int(top_k)]]


def _enumerate_pair_joints(
    ctx: PerturbContext,
    *,
    top_k_targets: int = DEFAULT_JOINT_TOP_K_TARGETS,
    top_k_per_target: int = DEFAULT_JOINT_TOP_K_PER_TARGET,
    max_wall_s: float = DEFAULT_JOINT_WALL_S,
    value_t_end: Optional[int] = None,
) -> list[tuple]:
    """Enumerate (S1, S2) → T joint candidates that solo-capture cannot.

    Strict inclusion rule: BOTH S1 and S2 individually have ships <
    `defender + 1` at their fire turn, but combined ≥ `defender + 1`.
    Same-target same-`t_arr` is required so combat-rule-1 pools the
    fleets at the planet. Solos that ARE individually viable are
    handled by the admissibility-set path; including them as joints
    would be strictly worse (over-pays ships, same value).

    Returns a list of tuples shaped:
        (joint_value, t_dep, src_a, src_b,
         emit_a, emit_b, tgt_id, t_arr, ships_a, ships_b)
    where each `emit_*` is `(t_dep, [src_id, angle, ships])` matching
    the admissibility entry shape.
    """
    if ctx.world is None or ctx.path_graph is None or ctx.world_model is None:
        return []
    pg = ctx.path_graph
    target_ids = _candidate_targets(ctx, top_k=top_k_targets)
    if not target_ids:
        return []

    bucket = max(1, int(pg.orbiting_bucket))
    deadline = time.perf_counter() + float(max_wall_s)
    joints: list[tuple] = []

    for tgt_id in target_ids:
        if time.perf_counter() >= deadline:
            return joints
        tgt = ctx.world.planets_by_id.get(int(tgt_id))
        if tgt is None:
            continue
        for t_dep in range(int(ctx.t_start), int(ctx.t_end), bucket):
            if time.perf_counter() >= deadline:
                return joints
            owned_at = ctx.ownership_cache.get(int(t_dep))
            if owned_at is None:
                continue
            srcs: list[tuple[int, object, int]] = []
            for src_id, (owner, ships) in owned_at.items():
                if int(owner) != int(ctx.me) or int(src_id) == int(tgt_id):
                    continue
                edge = pg.lookup(int(src_id), int(tgt_id), int(t_dep))
                if edge is None:
                    continue
                srcs.append((int(src_id), edge, int(ships)))
            if len(srcs) < 2:
                continue
            by_arr: dict[int, list[tuple[int, object, int]]] = {}
            for s in srcs:
                by_arr.setdefault(int(s[1].t_arr), []).append(s)
            for t_arr, group in by_arr.items():
                if len(group) < 2:
                    continue
                try:
                    arrivals = list(
                        ctx.world_model.ledger.get(int(tgt_id), []))
                except Exception:
                    arrivals = []
                arr_eta_from_snap0 = (
                    max(0, int(t_dep) - int(ctx.t_start))
                    + int(group[0][1].eta)
                )
                try:
                    pred_owner, pred_garrison = predict_garrison_at(
                        tgt, arr_eta_from_snap0, arrivals)
                except Exception:
                    pred_owner = int(tgt.owner)
                    pred_garrison = float(tgt.ships)
                if int(pred_owner) == int(ctx.me):
                    continue
                garrison_plus_one = int(math.ceil(float(pred_garrison))) + 1
                # Sort by ships desc, keep top-K to bound pair enumeration
                group.sort(key=lambda r: -r[2])
                group = group[: int(top_k_per_target)]
                wait_N = max(0, int(t_dep) - int(ctx.t_start))
                for a in range(len(group)):
                    s_a, e_a, n_a = group[a]
                    if n_a >= garrison_plus_one:
                        continue  # solo-viable; skip joints with this src
                    src_planet_a = ctx.world.planets_by_id.get(s_a)
                    if src_planet_a is None:
                        continue
                    for b in range(a + 1, len(group)):
                        s_b, e_b, n_b = group[b]
                        if n_b >= garrison_plus_one:
                            continue
                        if n_a + n_b < garrison_plus_one:
                            # group is sorted desc; further `b` only weaker
                            break
                        src_planet_b = ctx.world.planets_by_id.get(s_b)
                        if src_planet_b is None:
                            continue
                        try:
                            fate_a = predict_fleet_fate(
                                src_planet_a, tgt,
                                float(e_a.angle), int(n_a),
                                ctx.world, wait_N=wait_N)
                            fate_b = predict_fleet_fate(
                                src_planet_b, tgt,
                                float(e_b.angle), int(n_b),
                                ctx.world, wait_N=wait_N)
                        except Exception:
                            continue
                        if (fate_a.outcome != "target"
                                or fate_b.outcome != "target"):
                            continue
                        roi_end = (int(value_t_end)
                                    if value_t_end is not None
                                    else int(ctx.t_end))
                        remaining = max(0, roi_end - int(t_arr))
                        joint_val = (
                            float(tgt.production) * float(remaining)
                            - float(n_a + n_b)
                        )
                        if joint_val <= 0.0:
                            continue
                        joints.append((
                            joint_val,
                            int(t_dep), int(s_a), int(s_b),
                            (int(t_dep),
                             [int(s_a), float(e_a.angle), int(n_a)]),
                            (int(t_dep),
                             [int(s_b), float(e_b.angle), int(n_b)]),
                            int(tgt_id), int(t_arr),
                            int(n_a), int(n_b),
                        ))
    return joints


def _greedy_pick_from_ctx(
    ctx: PerturbContext,
    joint_candidates: list[tuple],
    *,
    max_picks: int,
    spent_per_src: dict[int, float],
    claimed_targets: set[int],
    value_t_end: Optional[int] = None,
) -> list:
    """Greedy non-conflicting selection over solo + joint candidates.

    Candidates are scored by closed-form capture value (solos via
    `_capture_value`; joints via the joint value computed at
    enumeration time). Sorted descending, picked in order subject to:
      - target uniqueness (no double-capture this rebuild)
      - source affordability (cache_ships − running spend ≥ ships_needed)

    Mutates `spent_per_src` and `claimed_targets` in place. Returns
    new emissions to append to the plan."""
    candidates: list[tuple[float, str, object]] = []
    for i in range(len(ctx.admissible)):
        tgt = int(ctx.admissible_targets[i])
        if tgt in claimed_targets:
            continue
        if value_t_end is not None:
            val = _value_with_horizon(ctx.admissible[i], i, ctx, value_t_end)
        else:
            val = _capture_value(ctx.admissible[i], i, ctx)
        if val == -math.inf or val <= 0.0:
            continue
        candidates.append((val, "solo", (i, tgt)))
    for jc in joint_candidates:
        joint_val = jc[0]
        tgt = int(jc[6])
        if tgt in claimed_targets:
            continue
        candidates.append((joint_val, "joint", jc))
    candidates.sort(key=lambda r: -r[0])

    picks: list = []
    for _val, kind, payload in candidates:
        if len(picks) >= int(max_picks):
            break
        if kind == "solo":
            i, tgt = payload  # type: ignore[misc]
            if int(tgt) in claimed_targets:
                continue
            emit = ctx.admissible[i]
            t_dep = int(emit[0])
            src_id = int(emit[1][0])
            ships = float(emit[1][2])
            cache_at_dep = ctx.ownership_cache.get(t_dep, {}).get(src_id)
            if cache_at_dep is None:
                continue
            owner, ships_at_dep = cache_at_dep
            if int(owner) != int(ctx.me):
                continue
            if (float(ships_at_dep)
                    - spent_per_src.get(src_id, 0.0)) < ships:
                continue
            spent_per_src[src_id] = spent_per_src.get(src_id, 0.0) + ships
            claimed_targets.add(int(tgt))
            picks.append(emit)
        else:  # "joint"
            _v, t_dep, s_a, s_b, emit_a, emit_b, tgt, _t_arr, n_a, n_b = payload  # type: ignore[misc]
            if int(tgt) in claimed_targets:
                continue
            ok = True
            for src_id, ships_needed in ((int(s_a), float(n_a)),
                                          (int(s_b), float(n_b))):
                cache_at_dep = ctx.ownership_cache.get(int(t_dep), {}).get(src_id)
                if cache_at_dep is None:
                    ok = False
                    break
                owner, ships_at_dep = cache_at_dep
                if int(owner) != int(ctx.me):
                    ok = False
                    break
                if (float(ships_at_dep)
                        - spent_per_src.get(src_id, 0.0)) < ships_needed:
                    ok = False
                    break
            if not ok:
                continue
            spent_per_src[int(s_a)] = spent_per_src.get(int(s_a), 0.0) + float(n_a)
            spent_per_src[int(s_b)] = spent_per_src.get(int(s_b), 0.0) + float(n_b)
            claimed_targets.add(int(tgt))
            picks.append(emit_a)
            picks.append(emit_b)
    return picks


def cascade_greedy_select(
    snap0,
    *,
    t_start: int,
    t_end: int,
    me: int,
    opp_policy=None,
    path_graph=None,
    max_rebuilds: int = DEFAULT_MAX_REBUILDS,
    max_picks_per_rebuild: int = DEFAULT_MAX_PICKS_PER_REBUILD,
    enable_joints: bool = True,
    total_wall_s: float = DEFAULT_TOTAL_WALL_S,
    value_t_end: Optional[int] = None,
) -> list:
    """Iterative cascade-greedy plan selection.

    For each rebuild (up to `max_rebuilds`):
      1. Build PerturbContext (forward-sim of the accumulated plan
         under `opp_policy` populates `ownership_cache[t_dep]`).
      2. `_populate_admissible_set` runs inside ctx-build, emitting
         physics-validated single-source captures where the source
         is owned-at-t_dep per the cache — captures from earlier
         rebuilds appear as new sources.
      3. Enumerate pair joints (if enabled) from path_graph + cache.
      4. Greedy-pick non-conflicting top-`max_picks_per_rebuild`
         solo+joint candidates.

    Returns the full plan `[(turn, [src, angle, ships]), ...]`. The
    caller filters by turn for the fire-now emit (no plan carryover
    across turns).

    `value_t_end`: ROI horizon override for `_capture_value` scoring.
    If None (default), uses ctx.t_end (= enum horizon). Pass the
    full episode-end to avoid underpricing captures with late arrival.

    Does not mutate `snap0`. Stops early on wallclock deadline or
    when a rebuild adds no new picks.
    """
    plan: list = []
    horizon = max(1, int(t_end) - int(t_start))
    deadline = time.perf_counter() + float(total_wall_s)
    for r in range(int(max_rebuilds)):
        if time.perf_counter() >= deadline:
            break
        try:
            ctx = _build_perturb_context(
                snap0, plan, opp_policy,
                max_steps=horizon, t_start=int(t_start),
                t_end=int(t_end), me=int(me),
                path_graph=path_graph,
            )
        except Exception:
            break
        joints = (_enumerate_pair_joints(ctx, value_t_end=value_t_end)
                   if enable_joints else [])
        if not ctx.admissible and not joints:
            break
        spent: dict[int, float] = {}
        claimed: set[int] = set()
        new_picks = _greedy_pick_from_ctx(
            ctx, joints,
            max_picks=int(max_picks_per_rebuild),
            spent_per_src=spent,
            claimed_targets=claimed,
            value_t_end=value_t_end,
        )
        if not new_picks:
            break
        plan.extend(new_picks)
    return plan
