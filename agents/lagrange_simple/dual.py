"""3-sweep Lagrangian decomposition.

Problem (time-indexed source budgets):
    maximize  Σ_c V(c) · x_c
    subject to
        Σ_{c: src(c)=s, launch_tick(c) ≤ u} ships(c) · x_c  ≤  R_s + P_s · u
                                                            ∀ source s, time u
        Σ_{c: tgt(c)=t}              x_c  ≤  1              ∀ target t
        x_c ∈ {0, 1}

Lagrangian relaxation on the per-source budget. For simplicity v1 collapses
the per-time constraint to a single per-source constraint with effective
budget `R_s + P_s · w̄`, where w̄ is the mean launch_tick of the picked
columns from s (so the dual price is a per-source scalar, λ_s). The
feasibility fix-up then enforces the FULL time-indexed constraint exactly
by sorting picks by launch_tick and dropping the worst (value/ships) until
every cumulative-by-time cell fits.

3 sweeps is enough in practice. Sources that are tight bind quickly,
sources with slack stay at λ_s=0. Math is faithful to the principle:
"each source optimizes its own slot, shadow prices coordinate them."
"""
from __future__ import annotations

from collections import defaultdict

from agents.lagrange_simple.score import Candidate


DEFAULT_SWEEPS = 3
DEFAULT_STEP = 1.0


def _inner_solve(candidates: list[Candidate],
                 lam: dict[int, float]) -> list[Candidate]:
    """Per-target argmax under shadow-price-adjusted score.

    Each target keeps the single candidate with the highest positive score;
    "do nothing" (score 0) is always an option.
    """
    by_target: dict[int, list[Candidate]] = defaultdict(list)
    for c in candidates:
        by_target[int(c.tgt_id)].append(c)
    picked: list[Candidate] = []
    for cands_t in by_target.values():
        best = None
        best_score = 0.0
        for c in cands_t:
            score = c.value - lam.get(int(c.src_id), 0.0) * float(c.ships)
            if score > best_score:
                best_score = score
                best = c
        if best is not None:
            picked.append(best)
    return picked


def _effective_budget(R: int, P: int, picks_for_src: list[Candidate]) -> float:
    """Per-source effective budget for shadow-price update.

    Uses the MEAN launch_tick of picked columns: each pick consumes ships
    "on average" at that tick, when accrued production R + P·w̄ is available.
    For an empty list returns R (no accrual yet).
    """
    if not picks_for_src:
        return float(R)
    mean_w = sum(int(c.launch_tick) for c in picks_for_src) / float(len(picks_for_src))
    return float(R) + float(P) * mean_w


def _time_indexed_feasible(picks_for_src: list[Candidate],
                           R: int, P: int) -> tuple[bool, int]:
    """Check the exact per-time constraint:
        Σ_{c: launch_tick ≤ u} ships(c) ≤ R + P · u  for every u

    Returns (feasible, worst_index): if infeasible, worst_index points to
    the candidate (in original picks_for_src order) with the lowest
    (value / ships); else worst_index = -1.

    Sort by launch_tick ascending, cumsum ships. The first cell where
    cumulative exceeds R + P·u is the binding violation.
    """
    if not picks_for_src:
        return True, -1
    sorted_by_w = sorted(picks_for_src, key=lambda c: int(c.launch_tick))
    cum = 0
    for c in sorted_by_w:
        cum += int(c.ships)
        allowed = float(R) + float(P) * float(c.launch_tick)
        if cum > allowed:
            worst_i = -1
            worst_eff = float("inf")
            for i, c2 in enumerate(picks_for_src):
                eff = float(c2.value) / float(max(1, c2.ships))
                if eff < worst_eff:
                    worst_eff = eff
                    worst_i = i
            return False, worst_i
    return True, -1


def solve(candidates: list[Candidate],
          source_budgets: dict[int, int],
          source_prods: dict[int, int] | None = None,
          *,
          sweeps: int = DEFAULT_SWEEPS,
          step: float = DEFAULT_STEP) -> list[Candidate]:
    """Lagrangian dual + time-indexed feasibility fix-up. Returns picks."""
    if not candidates:
        return []
    source_prods = source_prods or {}

    src_ids = set(int(c.src_id) for c in candidates)
    lam: dict[int, float] = {s: 0.0 for s in src_ids}
    picked: list[Candidate] = []

    for _ in range(int(sweeps)):
        picked = _inner_solve(candidates, lam)
        by_src: dict[int, list[Candidate]] = defaultdict(list)
        for c in picked:
            by_src[int(c.src_id)].append(c)
        for s in src_ids:
            R = int(source_budgets.get(s, 0))
            P = int(source_prods.get(s, 0))
            eff_budget = _effective_budget(R, P, by_src.get(s, []))
            if eff_budget <= 0:
                continue
            used = sum(int(c.ships) for c in by_src.get(s, []))
            over = used - eff_budget
            lam[s] = max(0.0, lam[s] + step * over / eff_budget)

    while True:
        by_src: dict[int, list[Candidate]] = defaultdict(list)
        for c in picked:
            by_src[int(c.src_id)].append(c)
        violated_src = None
        worst_pos = -1
        for s, cs in by_src.items():
            R = int(source_budgets.get(s, 0))
            P = int(source_prods.get(s, 0))
            ok, worst_i = _time_indexed_feasible(cs, R, P)
            if not ok:
                violated_src = s
                worst_pos = worst_i
                break
        if violated_src is None:
            break
        worst_cand = by_src[violated_src][worst_pos]
        picked.remove(worst_cand)

    return picked
