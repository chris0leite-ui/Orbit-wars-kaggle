# PV_ETA is a near-no-op on emitted moves; silence comes from elsewhere

Date: 2026-05-29 PM3 (this session).
Inputs: `scripts/trace_pv_eta_scoring.py` + `scripts/analyze_pv_eta_trace.py`
        → one full game, frozen PV_ETA anchor `submissions/baseline_pv_eta_anchor_1163.py`
        (= sub 53111837, μ=1163.5) vs `submissions/v7_0_drop_one.py`,
        seed=0, P0 = focal, OMP_NUM_THREADS=1, PYTHONHASHSEED=0.
        15,272 single-launch candidates + 2,406 joint candidates scored.

## TL;DR

The `γ^(wait_N+eta)` multiplier in `score_candidate_v4` does very little
to the chooser's actual emit decisions:

| Metric | Value |
|---|---|
| Δ-sign flips across the emission threshold (pre>0 → post≤0) | **0 / 15,272** |
| Top-1 winner identity change pre vs post | **9 / 127 turns** (7.1%) |
| Top-1 winner is a `wait_N>0` PATIENCE candidate (silent for that src) | **44 / 127 turns** (34.6%) — identical PRE and POST |

PV_ETA `delta *= γ^(wait_N+eta)` scales the delta-from-idle, not just
the launch leg, so it can never flip a candidate from emit to no-emit
(γ^k > 0 preserves sign). Among Δ>0 candidates it does re-rank, but
the top-1 only changes in 7% of turns — and in those 7%, the resulting
emit is nearly always similar in src/tgt.

**Implication:** PV_ETA's claimed +62μ lift over the leaf_pv_2p
attempt is not coming from "γ-discount fixes the silent-turns thesis."
The +62μ lift is some leaf-shape difference that has nothing to do
with the γ multiplier.

## What this falsifies

PM2 thought `2026-05-29-half-our-turns-are-no-action-with-ships.md`
proposed the mechanism:

> "PV_ETA discounts the candidate-launch side by `γ^(wait_N + eta)`
> but does NOT discount the no-action side. So 'do nothing' has a
> flat present value while 'launch this candidate' has a discounted
> one."

The code does NOT match that description. The relevant lines in
`agents/baseline/chooser_trajectory.py:543-601` are:

```python
leaf = favor_fn(snap.state[me].observation, ...)   # leaf-with-action at horizon
delta = leaf - baseline_favors[horizon]             # baseline = leaf-without at horizon
...
if PV_ETA_ENABLED and (int(wait_N) + int(eta)) > 0:
    delta *= gamma ** (int(wait_N) + int(eta))      # γ multiplies the DIFFERENCE
```

The multiplier hits the difference `(leaf_with - leaf_without)`, so it
scales BOTH legs by the same factor. It is symmetric in scaling, not
asymmetric in discount. The fix proposed in that PM2 note ("symmetric
PV on both sides") is already the de-facto current behavior.

The "49% no-action-with-surplus" empirical finding from PM2 stands —
that's real and confirmed by my trace's "all candidates Δ≤0" turns
(see turns 14, 15, 16, 19, 20 in the per-turn dump). What changes is
the **mechanism attribution.** It's not the γ multiplier.

## Where silence actually comes from

Two distinct mechanisms in this trace:

**1. Patience-by-dogpile silence.** 34.6% of positive-emit turns
have a `wait_N>0` candidate as the top-1. The chooser at
`agents/baseline/chooser.py:188-203` iterates validated candidates
in delta order, reserves `src+tgt`, and only emits if `wait_N==0`:

```python
if int(wait_N) == 0:
    moves.append([sid, float(angle), int(ships)])
# else: PATIENCE — src+tgt reserved above, no emit, no commit.
```

So when the top-1 from a src is a patience candidate, THAT SRC IS
SILENT THIS TURN, even if it had positive-Δ immediate launches in
positions #2-N. Visible in the trace turns 0, 3, 7, 9 — every one
has `src=12→tgt=20 wait=N>0` at top-1, blocking all other src=12
candidates. PV_ETA is irrelevant to this: the patience candidate
wins by 60+ Δ points, far above any discount magnitude.

**2. Δ≤0-on-everything turns.** 7 of the 22 turns in 14-20 have
all candidates at Δ ≤ 0 (top-1 = 0). This means the leaf says no
candidate beats sitting idle for the horizon window. The Δ=0
candidates are dominated by SELF-REINFORCEMENT moves (src=mine →
tgt=mine), which are valued the same as idle at horizon by `favor`
because moving ships between my planets doesn't change F1
(ship-aggregate) or F2 (prod × pv) at horizon.

This is the leaf-metric blindness:

```python
# value.py:127
return (my_ships - opp_ships) + (my_prod - opp_prod) * pv + elim_bonus
```

`favor` is positionally agnostic. Repositioning ships from rear→front
shows Δ=0 at horizon and is filtered by the `delta > 0` emit gate.

## The fix that's already in tree

`agents/baseline/value.py:177-194` defines `favor_hybrid_spatial`
which adds a positional weight (`1 / (1 + d_min / SPATIAL_DECAY)`,
where `d_min` is distance from each ship to nearest non-our planet)
to the standard leaf. Default `BASELINE_SPATIAL_WEIGHT=0.5`. **It is
shipped, OFF in the live anchor** (the anchor sets
`BASELINE_VALUE_HEAD=hybrid`, not `hybrid_spatial`).

The 2P-only spatial term should make rear→front mobilization show
Δ>0 in the leaf, breaking the "all candidates Δ≤0" turns and
addressing rear-mobilization + small-fleet symptoms directly.

The historical reason this is off: a 4P-spatial A/B regressed 4P
substantially (docstring `bv33jlzwj` reference). The 2P-only path
hasn't been A/B'd against the current PV_ETA anchor. The handover's
deferred-Item-3/Item-4 ("commit-to-hold," opp spatial restriction)
are *modeling-correct* fixes to the same family of symptoms — but
`hybrid_spatial` is the cheapest test (zero new code, one env var).

## Proposed next experiment (cheap, modeling-correct)

A/B `BASELINE_VALUE_HEAD=hybrid_spatial` (single env var) vs the
PV_ETA anchor, n=32, full 500-step games, vs panel of [v7_0,
v4_planner, v15-equivalent]. Bundle path: build a wrapper of the
PV_ETA anchor with the env var flipped, A/B per Rule 43 + Rule 45.

If hybrid_spatial lifts: the leaf-positional gap is the lever and
we have a no-new-code submission candidate.
If it doesn't lift or regresses: the bv33jlzwj 4P-regression was
real in 2P too, and we need the cleaner modeling fix (Item 3 / 4).

## Caveats / what this trace doesn't cover

- n=1 game, single seed, single opponent. Patterns may not hold
  across archetypes — confirm by sweeping the 32-archetype panel.
- Joint candidates (2,406 of them, ~13% of scored) are tracked
  with `wait_N` as a lower bound for `wait_N + eta`; I didn't
  recompute leg-eta. They could contain decision-altering flips
  this analyzer missed.
- "Top-1 identity change" is the most useful proxy for "emit
  change" but is not exact (the chooser can emit multiple moves
  per turn via different srcs).

## Files

- `scripts/trace_pv_eta_scoring.py` (committed 9900211)
- `scripts/analyze_pv_eta_trace.py` (this commit)
- `audit/pv_eta_trace.json` (gitignored — 6.2MB, regenerable in
  ~4 min: `python scripts/trace_pv_eta_scoring.py --seed 0 --opp submissions/v7_0_drop_one.py`)
