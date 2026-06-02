"""Exact-oracle REFINER on top of the champion (Rule 49, augment-not-replace).

The 2026-06-02 finding: using the conditional sequential-greedy planner as a
*generator* underperforms the champion's independent solo-delta scoring + locks
(greedy 9/16 vs champion 16/16 vs v7_0). Its marginal gains are passive-self-
pessimistic, so it prunes genuinely-good independent launches → under-commits;
flat capture credit over-corrects. See state/MULTI_BRANCH.md.

The reframe: the conditional machinery's real value is as a *refiner*, not a
generator. Bound it by the champion's own bundle and it can only help:

  • TEAMWORK-ADD — two fleets that only capture TOGETHER each score ≤0 solo, so
    the champion (independent scoring) is structurally blind to them. We
    generate coalition atoms from the sources the champion DIDN'T use and add
    any whose EXACT marginal joint value (deterministic `fast_sim` oracle) is
    positive and that don't conflict with the champion's locks. This NEVER
    removes a champion launch ⇒ it cannot reintroduce the under-commit. Pure
    capability gain.
  • DROP-ONE WASTE (default OFF) — ask the oracle each champion launch's exact
    marginal contribution; drop any that is ≤0 (genuinely redundant). Bounded
    by the champion's set ⇒ worst case a no-op.

Selected via `BASELINE_CHOOSER=refine`. With no atoms added and DROP off, the
emitted (moves, commits) are the champion's verbatim (we append, we don't
rebuild) — so the refiner degrades to the champion exactly.
"""

from __future__ import annotations

import os
import time

from agents.baseline.chooser import WALLCLOCK_HARD_CAP_MS, affordable_validate_cap
from agents.baseline.chooser_greedy import _delta, _envflag, _envint
from agents.baseline.chooser_trajectory import (
    JOINT_SYNC_MAX_PAIRS,
    MIN_DELTA,
    build_trajectory_baseline,
    choose_trajectory,
    generate_sync_coalitions,
    select_favor_fn,
)


def _used_tgts_locked() -> bool:
    """Mirror the champion emit's `used_tgts` lock (lifted by
    BASELINE_JOINT_AGGR=1). Read at call time for test/A-B toggling."""
    return os.environ.get("BASELINE_JOINT_AGGR", "0").strip() != "1"


def choose_refine(snap_base, prerank, baseline_favors,
                  me: int, num_seats: int, wallclock_ms: float,
                  min_horizon: int, max_horizon: int, gamma: float,
                  world, model,
                  reserved_srcs: set[int] | None = None,
                  reserved_for_new_commits: set[int] | None = None,
                  agent_deadline: float | None = None,
                  ) -> tuple[list[list], list[dict]]:
    """Signature-identical to `choose_trajectory`. Returns (moves, commits)."""
    if reserved_srcs is None:
        reserved_srcs = set()
    if reserved_for_new_commits is None:
        reserved_for_new_commits = reserved_srcs
    if not prerank:
        return [], []

    start = time.perf_counter()
    champ_frac = float(_envint("BASELINE_REFINE_CHAMP_PCT", 70)) / 100.0
    overhead_ms = float(_envint("BASELINE_REFINE_OVERHEAD_MS", 60))
    refine_h = min(max_horizon, max(1, _envint("BASELINE_REFINE_HORIZON", 20)))
    max_add = _envint("BASELINE_REFINE_MAX_ADD", 4)
    atom_cap = _envint("BASELINE_REFINE_MAX_EVAL", 8)
    do_drop = _envflag("BASELINE_REFINE_DROP", "0")
    floor = MIN_DELTA

    # --- 1. Champion decides, the full bundle is captured for the oracle. ---
    # Give the champion the bulk of the budget; reserve a slice for refinement
    # so the turn never blows the env cap (the lesson from the flat-credit
    # 1983ms blowup). agent_deadline binds the champion's internal safe cap.
    champ_deadline = start + (wallclock_ms / 1000.0) * champ_frac
    if agent_deadline is not None:
        champ_deadline = min(champ_deadline, agent_deadline)
    champ_launches: list = []
    moves, commits = choose_trajectory(
        snap_base, prerank, baseline_favors, me, num_seats, wallclock_ms,
        min_horizon, max_horizon, gamma, world, model,
        reserved_srcs=reserved_srcs,
        reserved_for_new_commits=reserved_for_new_commits,
        agent_deadline=champ_deadline,
        out_chosen=champ_launches,
    )

    used_srcs = {int(L[0].id) for L in champ_launches}
    used_tgts = {int(L[1].id) for L in champ_launches}
    tgt_locked = _used_tgts_locked()

    # --- 2a. Generate coalition atoms (CHEAP — geometry only, no rollout) and
    # filter to those that don't conflict with the champion's locks. If none
    # survive and DROP is off, return the champion verbatim with ZERO oracle
    # overhead — so a turn the refiner can't improve costs nothing over the
    # champion (the slowness fix + the degrade-to-champion guarantee). ---
    atoms = []
    _raw = 0
    for launches, _tarr in generate_sync_coalitions(
            world, model, me, max_horizon,
            reserved_srcs, reserved_for_new_commits,
            max_pairs=JOINT_SYNC_MAX_PAIRS):
        _raw += 1
        srcs = frozenset(int(L[0].id) for L in launches)
        tgts = frozenset(int(L[1].id) for L in launches)
        if srcs & used_srcs:
            continue
        if tgt_locked and (tgts & used_tgts):
            continue
        atoms.append({"launches": launches, "srcs": srcs, "tgts": tgts})
    if _raw and os.environ.get("BASELINE_REFINE_DEBUG", "").strip() == "1":
        import sys
        champ_waitlegs = sum(1 for L in champ_launches if int(L[4]) > 0)
        print(f"[refine-raw] step={int(world.step) if world else -1} "
              f"raw_coalitions={_raw} feasible={len(atoms)} "
              f"champ_launches={len(champ_launches)} champ_waitlegs={champ_waitlegs} "
              f"used_srcs={len(used_srcs)}", file=sys.stderr)
    if not atoms and not do_drop:
        return moves, commits

    # --- refiner wallclock: bounded by the original turn budget. ---
    deadline = start + wallclock_ms / 1000.0 - overhead_ms / 1000.0
    hard_deadline = start + WALLCLOCK_HARD_CAP_MS / 1000.0
    if agent_deadline is not None:
        deadline = min(deadline, agent_deadline - overhead_ms / 1000.0)
        hard_deadline = min(hard_deadline, agent_deadline)
    favor_fn = select_favor_fn()
    favs = build_trajectory_baseline(
        snap_base, me, num_seats, refine_h, favor_fn, gamma)
    _, per_cand_ms = affordable_validate_cap(
        snap_base, me, num_seats, refine_h, wallclock_ms, refine_h, gamma)
    safe_deadline = deadline - per_cand_ms / 1000.0

    # Champion's launches are the oracle baseline set; delta(∅)=0 by
    # construction, so delta(S) is the exact joint value of the bundle.
    S_launches = list(champ_launches)
    if time.perf_counter() <= safe_deadline:
        delta_S, st = _delta(snap_base, S_launches, me, num_seats, world,
                             favs, favor_fn, gamma, refine_h, hard_deadline)
        if st != "scored":
            return moves, commits  # oracle bailed → champion verbatim
    else:
        return moves, commits

    # --- 2b. TEAMWORK-ADD: score each atom ONCE over the champion base
    # (1 + ≤atom_cap rollouts, not max_add×n) and greedily add the
    # highest-gain non-conflicting ones. The marginal is first-order over the
    # champion bundle — atoms sit on distinct sources/targets, so cross-atom
    # interaction is second-order and not worth the n×picks rollout blowup
    # that made the full game un-runnable. ---
    scored_atoms: list = []  # (gain, atom)
    for a in atoms[:atom_cap]:
        if time.perf_counter() > safe_deadline:
            break
        d_union, st = _delta(snap_base, S_launches + list(a["launches"]),
                             me, num_seats, world, favs, favor_fn, gamma,
                             refine_h, hard_deadline)
        if st != "scored":
            break
        gain = d_union - delta_S
        if gain > floor:
            scored_atoms.append((gain, a))
    scored_atoms.sort(key=lambda t: t[0], reverse=True)
    if os.environ.get("BASELINE_REFINE_DEBUG", "").strip() == "1" and atoms:
        import sys
        best_gain = scored_atoms[0][0] if scored_atoms else float("-inf")
        print(f"[refine] step={int(world.step) if world else -1} "
              f"feasible_atoms={len(atoms)} eval={min(len(atoms), atom_cap)} "
              f"best_gain={best_gain:.4f} floor={floor:.4f} "
              f"pass_floor={len(scored_atoms)}", file=sys.stderr)
    added: list[dict] = []
    for _gain, a in scored_atoms:
        if len(added) >= max_add:
            break
        if a["srcs"] & used_srcs:
            continue
        if tgt_locked and (a["tgts"] & used_tgts):
            continue
        used_srcs |= a["srcs"]
        used_tgts |= a["tgts"]
        added.append(a)

    if added and os.environ.get("BASELINE_REFINE_DEBUG", "").strip() == "1":
        import sys
        print(f"[refine] step={int(world.step) if world else -1} "
              f"added={len(added)} atoms_eval={min(len(atoms), atom_cap)} "
              f"champ_launches={len(champ_launches)}", file=sys.stderr)

    # Append the added coalitions' legs to the champion's emit (no rebuild,
    # so the no-add case is byte-identical to the champion).
    commit_step = int(world.step) if world is not None else 0
    for a in added:
        for src, tgt, ships, angle, wait_N in a["launches"]:
            if int(wait_N) == 0:
                moves.append([int(src.id), float(angle), int(ships)])
            else:
                commits.append({
                    "src_id": int(src.id),
                    "tgt_id": int(tgt.id),
                    "ships_planned": int(ships),
                    "angle_original": float(angle),
                    "wait_remaining": int(wait_N),
                    "commit_step": commit_step,
                    "sync_joint": True,
                })

    # --- 3. DROP-ONE exact waste (default OFF). ---
    if do_drop and champ_launches and time.perf_counter() <= safe_deadline:
        dropped: list = []
        for L in champ_launches:
            if time.perf_counter() > safe_deadline:
                break
            without = [x for x in S_launches if x is not L]
            d_without, st = _delta(snap_base, without, me, num_seats, world,
                                   favs, favor_fn, gamma, refine_h, hard_deadline)
            if st != "scored":
                break
            if d_without > delta_S + floor:  # removing it improves exact value
                dropped.append(L)
                S_launches = without
                delta_S = d_without
        if dropped:
            moves, commits = _strip_dropped(moves, commits, dropped)

    return moves, commits


def _strip_dropped(moves, commits, dropped):
    """Remove emitted entries matching dropped champion launch tuples.
    Fire-now matched by (src_id, ships); wait matched by (src_id, tgt_id)."""
    drop_fire = {(int(L[0].id), int(L[2])) for L in dropped if int(L[4]) == 0}
    drop_wait = {(int(L[0].id), int(L[1].id)) for L in dropped if int(L[4]) > 0}
    moves = [m for m in moves if (int(m[0]), int(m[2])) not in drop_fire]
    commits = [c for c in commits
               if (int(c["src_id"]), int(c["tgt_id"])) not in drop_wait]
    return moves, commits
