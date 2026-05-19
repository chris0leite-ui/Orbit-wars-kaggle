"""Layered chooser — Layer-0 closed-form predicates over a pluggable inner chooser.

Composes the W1/W2 commits + L1/L2 discards (`agents/baseline/predicates`)
with the existing rollout-based chooser of choice. The inner chooser is
selected by `BASELINE_INNER_CHOOSER` (default `"trajectory"`), so the
composition stays valid when the production substrate swaps under us.

Pipeline per turn:
  1. Layer 0 classifies each proposer prerank candidate as one of:
     L1 discard / W1 commit / W2 commit / uncertain.
  2. L2 dominance prune over the uncertain residual.
  3. Layer-0 commits emit greedily (1 launch per source, 1 per target).
  4. Residual (minus sources/targets already used by L0) goes to the
     pluggable inner chooser via `_INNER_DISPATCH`.
  5. Merge: L0 emits + inner emits, with a final src-uniqueness pass
     guarding against any inner chooser quirk.

Adding a new inner chooser is one entry in `_INNER_DISPATCH`; nothing
else in this module nor in `predicates.py` needs to change.
"""

from __future__ import annotations

import os

# Single-line imports below: the submission bundler's per-line
# import-stripping regex leaks continuation lines from a parenthesised
# multi-line import as indented orphans (IndentationError at runtime).
# Friction tag `bundler-modular-agent-namespace-access-breaks-bundle`
# documented in agents/baseline/main.py and proposer.py.
from agents.baseline.chooser import choose
from agents.baseline.chooser_roi import choose_roi
from agents.baseline.chooser_trajectory import choose_trajectory
from agents.baseline.predicates import l1_provably_wasted_launch
from agents.baseline.predicates import l2_dominance_prune
from agents.baseline.predicates import w1_provably_winning_capture
from agents.baseline.predicates import w2_provably_held_reinforce


# Inner-chooser dispatch table. Each entry adapts the kwargs to the
# specific chooser's signature, isolating the divergence between the
# three existing choosers (and any future addition).
def _dispatch_trajectory(k):
    return choose_trajectory(
        k["snap_base"], k["prerank"], k["baseline_favors"],
        k["me"], k["num_seats"], k["wallclock_ms"],
        k["min_horizon"], k["max_horizon"], k["gamma"],
        k["world"], k["model"],
    )


def _dispatch_composite(k):
    return choose(
        k["snap_base"], k["prerank"], k["baseline_favors"],
        k["me"], k["num_seats"], k["wallclock_ms"],
        k["min_horizon"], k["max_horizon"], k["gamma"],
    )


def _dispatch_roi(k):
    return choose_roi(
        k["snap_base"], k["prerank"],
        k["me"], k["num_seats"], k["wallclock_ms"],
        k["min_horizon"], k["max_horizon"], k["gamma"],
        k["world"], k["model"], k["step"],
    )


_INNER_DISPATCH = {
    "trajectory": _dispatch_trajectory,
    "composite": _dispatch_composite,
    "roi": _dispatch_roi,
}


def layer0_classify(prerank, world, model, me, step, gamma):
    """Pure closed-form classification: prerank → (commits, residual).

    Returns:
        commits: list of (candidate_tuple, Verdict) for W1/W2 commits.
        residual: list of candidate tuples for the inner chooser.

    L1 fires first so provably-wasted candidates never reach W1/W2.
    Among the surviving candidates, W1 (capture) and W2 (reinforce)
    are mutually exclusive by their `tgt.owner == me` gating.
    L2 prunes the residual (same-source-same-target dominance).
    """
    commits: list = []
    residual: list = []
    for c in prerank:
        cheap_delta, src, tgt, ships, angle, eta, horizon, wait_N = c

        # L1 first — drop provably wasted.
        v = l1_provably_wasted_launch(
            src, tgt, int(ships), int(wait_N), int(eta), world, model, int(me),
        )
        if v.kind == "discard":
            continue

        # W1 commit?
        v = w1_provably_winning_capture(
            src, tgt, int(ships), int(wait_N), int(eta), world, model, int(me),
            gamma=gamma,
        )
        if v.kind == "commit":
            commits.append((c, v))
            continue

        # W2 commit?
        v = w2_provably_held_reinforce(
            src, tgt, int(ships), int(wait_N), int(eta), world, model, int(me),
        )
        if v.kind == "commit":
            commits.append((c, v))
            continue

        residual.append(c)

    residual = l2_dominance_prune(residual)
    return commits, residual


def _emit_l0(commits):
    """Greedy emit of Layer-0 commits: highest lower_bound first,
    1 launch per source, 1 per target.

    Returns (moves, used_srcs, used_tgts). `wait_N>0` commits reserve
    src+tgt but don't emit a launch (matches the existing chooser pattern
    — wait-then-fire candidates fire on a later turn).
    """
    commits_sorted = sorted(
        commits, key=lambda cv: -float(cv[1].lower_bound),
    )
    used_srcs: set = set()
    used_tgts: set = set()
    moves: list = []
    for c, v in commits_sorted:
        cheap_delta, src, tgt, ships, angle, eta, horizon, wait_N = c
        sid, tid = int(src.id), int(tgt.id)
        if sid in used_srcs or tid in used_tgts:
            continue
        used_srcs.add(sid)
        used_tgts.add(tid)
        if int(wait_N) == 0:
            moves.append([sid, float(angle), int(ships)])
        # wait_N>0: reserve, emit nothing this turn
    return moves, used_srcs, used_tgts


def _resolve_inner_chooser_name() -> str:
    """Resolve the inner chooser name from env, with default fallback."""
    name = os.environ.get("BASELINE_INNER_CHOOSER", "trajectory").strip().lower()
    if name not in _INNER_DISPATCH:
        # Unknown name → fall back to trajectory (current production).
        # Silent fallback rather than raise: the env is shared with
        # bundles that may not understand a new inner chooser yet.
        return "trajectory"
    return name


def choose_layered(snap_base, prerank, baseline_favors,
                   me: int, num_seats: int, wallclock_ms: float,
                   min_horizon: int, max_horizon: int, gamma: float,
                   world, model, step: int,
                   *,
                   inner_chooser_name: str | None = None) -> list:
    """Layer 0 + inner-chooser composition.

    `inner_chooser_name` overrides the env var; if None, reads
    `BASELINE_INNER_CHOOSER`.
    """
    if not prerank:
        return []

    commits, residual = layer0_classify(
        prerank, world, model, int(me), int(step), float(gamma),
    )

    emit_l0, used_srcs, used_tgts = _emit_l0(commits)

    # Filter residual: drop candidates whose src or tgt is already taken
    # by an L0 commit. The inner chooser's own 1-per-src / 1-per-tgt
    # dedup handles intra-residual conflicts; we just need to keep L0
    # commits sovereign.
    residual_filtered = [
        c for c in residual
        if int(c[1].id) not in used_srcs and int(c[2].id) not in used_tgts
    ]

    inner_name = inner_chooser_name or _resolve_inner_chooser_name()
    inner_kwargs = {
        "snap_base": snap_base,
        "prerank": residual_filtered,
        "baseline_favors": baseline_favors,
        "me": int(me),
        "num_seats": int(num_seats),
        "wallclock_ms": float(wallclock_ms),
        "min_horizon": int(min_horizon),
        "max_horizon": int(max_horizon),
        "gamma": float(gamma),
        "world": world,
        "model": model,
        "step": int(step),
    }
    inner_fn = _INNER_DISPATCH[inner_name]
    emit_inner = inner_fn(inner_kwargs) or []

    # Merge with belt-and-suspenders src-uniqueness: in the unlikely
    # event the inner chooser returns a move on a source already used
    # by an L0 commit (shouldn't happen given the pre-filter), drop it.
    final_emit = list(emit_l0)
    for move in emit_inner:
        try:
            sid = int(move[0])
        except (TypeError, IndexError, ValueError):
            continue
        if sid in used_srcs:
            continue
        final_emit.append(move)
        used_srcs.add(sid)

    return final_emit
