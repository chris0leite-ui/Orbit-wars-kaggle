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

import os
from collections import defaultdict
from dataclasses import replace

from agents.lagrange_simple.score import Candidate, _source_defensive_ok


# Dogpile is OFF by default. Six variants tried this session
# (claude/session-EqJuT, 2026-05-23):
#   (1) naive (all-budget partials, every bucket every turn) → 6/16 ELIM
#   (2) restricted to dominant_endgame                        → 6/16
#   (3) cap at ONE coalition per turn                         → 6/16
#   (4) Phase C variant 1: + per-source rear-defense check    → 6/16
#   (5) Phase C variant 2: + 1-coalition cap                  → 7/16
# All regressed vs Phase B baseline (Phase A opp-into-ledger + Phase B
# rear-defense check on solo path, no dogpile) which gives 14/16 ELIM.
# Common failure mode: even per-source rear-defense isn't enough —
# multiple drained sources can't defend EACH OTHER, opp chain-captures.
# Axis closed permanently per Rule 37; deeper fix needs cross-source
# defensive coordination, not coverable in this session.
DOGPILE_ENABLED = os.environ.get(
    "LAGRANGE_SIMPLE_DOGPILE", "0",
).strip().lower() in ("1", "true", "on", "yes")


DEFAULT_SWEEPS = 3
DEFAULT_STEP = 1.0


def _inner_solve(candidates: list[Candidate],
                 lam: dict[int, float]) -> list[Candidate]:
    """Per-target argmax under shadow-price-adjusted score.

    Each target keeps the single SOLO candidate with the highest positive
    score; "do nothing" (score 0) is always an option. Partial candidates
    are skipped here — they only contribute via `_dogpile_pass`.
    """
    by_target: dict[int, list[Candidate]] = defaultdict(list)
    for c in candidates:
        if c.is_partial:
            continue
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


def _dogpile_pass(all_cands: list[Candidate],
                  picked: list[Candidate],
                  lam: dict[int, float],
                  source_budgets: dict[int, int]) -> list[Candidate]:
    """Multi-source dogpile inner pass.

    Group partial candidates by (tgt_id, arrival_step). For each bucket
    NOT already captured by the solo picks, greedy-add candidates by
    ascending λ_src (cheapest shadow-price first), one per source, until
    cumulative ships > defense_at_arrival. If the committed subset's
    reduced cost (capture_value − Σ λ_src · ships) is strictly positive,
    emit those (right-sized) candidates into the pick list.

    This closes the single-source-per-target ceiling: targets whose
    defense exceeds any single source's budget get coalition coverage.
    """
    captured_targets = {int(c.tgt_id) for c in picked if not c.is_partial}
    residual = {int(s): int(b) for s, b in source_budgets.items()}
    for c in picked:
        residual[int(c.src_id)] = residual.get(int(c.src_id), 0) - int(c.ships)

    buckets: dict[tuple[int, int], list[Candidate]] = defaultdict(list)
    for c in all_cands:
        if c.is_partial and int(c.tgt_id) not in captured_targets:
            buckets[(int(c.tgt_id), int(c.arrival_step))].append(c)

    # Sort buckets by capture_value descending — commit the highest-value
    # coalition first; cap at ONE coalition per turn to bound per-turn ship
    # drain (the rear-defense problem: each commit empties source(s) for
    # eta turns while opp can counter-attack).
    sorted_buckets = sorted(
        buckets.items(), key=lambda kv: -float(kv[1][0].value),
    )

    dogpile_picks: list[Candidate] = []
    for (tgt_id, arrival_step), bucket in sorted_buckets:
        bucket.sort(key=lambda c: lam.get(int(c.src_id), 0.0))
        defense = int(bucket[0].defense_at_arrival)
        capture_value = float(bucket[0].value)

        subset: list[tuple[Candidate, int]] = []
        subset_ships = 0
        subset_cost = 0.0
        seen_srcs: set[int] = set()
        for c in bucket:
            if int(c.src_id) in seen_srcs:
                continue
            avail = min(int(c.ships), residual.get(int(c.src_id), 0))
            if avail < 1:
                continue
            need = max(1, defense + 1 - subset_ships)
            take = min(avail, need)
            subset.append((c, take))
            subset_ships += take
            subset_cost += lam.get(int(c.src_id), 0.0) * float(take)
            seen_srcs.add(int(c.src_id))
            if subset_ships > defense:
                break

        if subset_ships <= defense:
            continue
        if (capture_value - subset_cost) <= 0.0:
            continue

        for c, take in subset:
            dogpile_picks.append(replace(c, ships=int(take)))
            residual[int(c.src_id)] -= int(take)
        break  # cap: at most ONE dogpile coalition per turn (variant 2)

    return picked + dogpile_picks


def _run_time_indexed_fixup(picked: list[Candidate],
                            source_budgets: dict[int, int],
                            source_prods: dict[int, int]) -> list[Candidate]:
    """Drop the worst-(value/ships) entry from any source whose per-time
    cumulative constraint is violated; iterate until all sources feasible."""
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
            return picked
        worst_cand = by_src[violated_src][worst_pos]
        picked.remove(worst_cand)


def solve(candidates: list[Candidate],
          source_budgets: dict[int, int],
          source_prods: dict[int, int] | None = None,
          *,
          sweeps: int = DEFAULT_SWEEPS,
          step: float = DEFAULT_STEP) -> list[Candidate]:
    """Lagrangian dual + dogpile + time-indexed feasibility fix-up.

    Returns the chosen pick list (a mix of solo captures and dogpile-
    coalition fleets). Partials in the input candidate list are routed
    only through `_dogpile_pass`; the per-target argmax skips them.
    """
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

    picked = _run_time_indexed_fixup(picked, source_budgets, source_prods)

    if DOGPILE_ENABLED:
        # Dogpile pass: cover targets the solo path couldn't.
        picked = _dogpile_pass(candidates, picked, lam, source_budgets)
        # Re-run fix-up: dogpile commits use residual, but per-time cumulative
        # could still be violated when partials launch at the same tick as
        # solo picks from the same source.
        picked = _run_time_indexed_fixup(picked, source_budgets, source_prods)

    return picked
