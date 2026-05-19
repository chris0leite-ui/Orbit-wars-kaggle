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
import time

# Slice 3 (2026-05-19): minimum wallclock the inner chooser must
# receive even if L0 overran. Prevents zero-budget pathologies; the
# trajectory chooser's own pre-bail logic respects whatever budget
# it's given (down to ~50 ms).
INNER_WALLCLOCK_FLOOR_MS: float = 50.0

# Single-line imports below: the submission bundler's per-line
# import-stripping regex leaks continuation lines from a parenthesised
# multi-line import as indented orphans (IndentationError at runtime).
# Friction tag `bundler-modular-agent-namespace-access-breaks-bundle`
# documented in agents/baseline/main.py and proposer.py.
from agents.baseline.chooser import choose
from agents.baseline.chooser_roi import choose_roi
from agents.baseline.chooser_trajectory import choose_trajectory
from agents.baseline.predicates import UNCERTAIN
from agents.baseline.predicates import Verdict
from agents.baseline.predicates import l1_provably_wasted_launch
from agents.baseline.predicates import l2_dominance_prune
from agents.baseline.predicates import w1_dominance_classify
from agents.baseline.predicates import w1_provably_winning_capture
from agents.baseline.predicates import w2_provably_held_reinforce
from agents.baseline.strategic_lp import compute_lp_assignment


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
    """Pure closed-form classification (Slice 4 — predicates as priors).

    Returns:
        verdicts: list of (candidate_tuple, Verdict) in input order.
                  Every input candidate gets a verdict (commit, discard,
                  or uncertain). The chooser uses `kind == "commit"`
                  entries as a BACKSTOP after the inner chooser runs.
        filtered: list of candidate tuples to send to the inner chooser
                  (input minus L1 discards, with L2 dominance prune
                  applied). W1/W2 commit candidates STAY in `filtered`
                  so the inner chooser sees them and can score them
                  alongside everything else.

    Architectural pivot from Slice 2/3: we no longer preempt the inner
    chooser. Layer 0's commits become a backstop — if the inner chose
    a move that conflicts with a provable commit, the inner's choice
    wins (it had strategic context the closed-form bound lacks). The
    commit is only emitted if the inner left its source unused.

    Reason: Slice 3's audit-replay diagnosis (commit `dcf71e2`) showed
    that L0 preemption was the load-bearing failure mode — even when
    commits were mathematically sound, they preempted source
    allocations the inner would have used more effectively.
    """
    # Slice 5: W1 commits decided by per-source dominance over bounded intervals.
    w1_verdict_by_id = w1_dominance_classify(
        prerank, world, model, int(me), gamma=float(gamma),
    )

    # Slice 6: LP assignment available behind env var. Default OFF
    # because n=16 A/B regressed (8/16 wins vs Slice 5's 9/16, Wlo
    # 0.280 vs 0.332) AND wallclock max blew up (1832ms vs 861ms).
    # The LP picks a static (src → tgt) per source but doesn't see
    # the inner chooser's strategic reasoning about source use; the
    # extra LP commits behave as noise. Module kept for future
    # research (audit-replay analysis, training-data labelling);
    # opt-in via `BASELINE_LP_COMMIT=1` for further experiments.
    if os.environ.get("BASELINE_LP_COMMIT", "0").strip() == "1":
        lp_assignment = compute_lp_assignment(world, model, int(me))
    else:
        lp_assignment = {}
    lp_committed_srcs: set = set()

    verdicts: list = []
    surviving: list = []
    for c in prerank:
        cheap_delta, src, tgt, ships, angle, eta, horizon, wait_N = c

        # L1 first — drop provably wasted from `filtered`.
        v_l1 = l1_provably_wasted_launch(
            src, tgt, int(ships), int(wait_N), int(eta), world, model, int(me),
        )
        if v_l1.kind == "discard":
            verdicts.append((c, v_l1))
            continue

        # W1 dominance verdict (Slice 5).
        w1_v = w1_verdict_by_id.get(id(c))
        if w1_v is not None and w1_v.kind == "commit":
            verdicts.append((c, w1_v))
            surviving.append(c)
            continue

        # W2 reinforce commit (Slice 1).
        v_w2 = w2_provably_held_reinforce(
            src, tgt, int(ships), int(wait_N), int(eta), world, model, int(me),
        )
        if v_w2.kind == "commit":
            verdicts.append((c, v_w2))
            surviving.append(c)
            continue

        # Slice 6: LP-alignment commit. Only one LP commit per source,
        # for fire-now candidates matching (src.id, tgt.id) of the LP's
        # recommended assignment. Lower_bound = LP value (production ×
        # time_remaining); used by the backstop sort to prioritize
        # among multiple LP commits.
        if (int(wait_N) == 0
                and int(src.id) not in lp_committed_srcs
                and lp_assignment.get(int(src.id)) == int(tgt.id)
                and int(tgt.owner) != int(me)):
            lp_value = float(int(tgt.production)) * float(
                max(0, 500 - int(getattr(world, "step", 0) or 0) - int(eta))
            )
            if lp_value > 0:
                verdicts.append(
                    (c, Verdict(kind="commit", lower_bound=lp_value, reason="LP")),
                )
                surviving.append(c)
                lp_committed_srcs.add(int(src.id))
                continue

        verdicts.append((c, UNCERTAIN))
        surviving.append(c)

    filtered = l2_dominance_prune(surviving)
    return verdicts, filtered


def _backstop_emit(verdicts, emit_inner):
    """Slice 4 backstop: append W1/W2 commits the inner chooser didn't
    cover.

    A commit `(src, tgt, ...)` is considered "covered" if the inner
    emitted any move from the same source. We approximate target-side
    coverage as "trust the inner's own 1-launch-per-target dedup"
    because emit moves carry only `[src_id, angle, ships]` (no
    target id), so an exact target-uniqueness check post-emit is
    not free. Source-side dedup is sufficient to prevent
    double-launching from the same planet.

    Returns the appended moves (NOT the full final emit). Caller
    concatenates.
    """
    used_srcs: set = set()
    for move in emit_inner:
        try:
            used_srcs.add(int(move[0]))
        except (TypeError, IndexError, ValueError):
            pass

    # Sort commits by lower_bound desc so the highest-value backstop
    # wins when multiple commits share a free source slot.
    commits = [(c, v) for c, v in verdicts if v.kind == "commit"]
    commits.sort(key=lambda cv: -float(cv[1].lower_bound))

    appended: list = []
    for c, v in commits:
        cheap_delta, src, tgt, ships, angle, eta, horizon, wait_N = c
        sid = int(src.id)
        if sid in used_srcs:
            continue  # inner used this source for something else
        if int(wait_N) != 0:
            # wait_N>0 commits emit nothing this turn (existing chooser
            # convention); reserving src is moot since inner already
            # chose for this source.
            continue
        used_srcs.add(sid)
        appended.append([sid, float(angle), int(ships)])
    return appended


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

    # Slice 4 architecture: classify, then send full filtered prerank
    # to the inner. Commits become a backstop (after inner) rather
    # than a preempt (before inner). Layer 0's role is to inform the
    # final emit set without overriding the inner's source allocation.
    t_l0_start = time.perf_counter()

    verdicts, filtered_prerank = layer0_classify(
        prerank, world, model, int(me), int(step), float(gamma),
    )

    l0_elapsed_ms = (time.perf_counter() - t_l0_start) * 1000.0
    inner_wallclock_ms = max(
        INNER_WALLCLOCK_FLOOR_MS, float(wallclock_ms) - l0_elapsed_ms,
    )

    inner_name = inner_chooser_name or _resolve_inner_chooser_name()
    inner_kwargs = {
        "snap_base": snap_base,
        "prerank": filtered_prerank,
        "baseline_favors": baseline_favors,
        "me": int(me),
        "num_seats": int(num_seats),
        "wallclock_ms": inner_wallclock_ms,
        "min_horizon": int(min_horizon),
        "max_horizon": int(max_horizon),
        "gamma": float(gamma),
        "world": world,
        "model": model,
        "step": int(step),
    }
    inner_fn = _INNER_DISPATCH[inner_name]
    emit_inner = list(inner_fn(inner_kwargs) or [])

    # Backstop: append W1/W2 commits the inner didn't already cover.
    backstop = _backstop_emit(verdicts, emit_inner)
    return emit_inner + backstop
