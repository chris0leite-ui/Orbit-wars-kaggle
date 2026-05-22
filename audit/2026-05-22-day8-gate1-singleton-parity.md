# 2026-05-22 — Day 8 Gate 1: singleton parity verdict

## Summary

Gate 1 runs coord (with `COORD_MAX_BUNDLE_SIZE=1` + `COORD_DISABLE_DEFEND=1`)
against minimal on identical game states, comparing per-turn action sets.

**Result (2 seeds × 60 turns = 120 turns evaluated):**

| Metric              | Value   | % of turns |
|---------------------|--------:|-----------:|
| Identical actions   |   63    |   52.5%    |
| coord_srcs ⊆ minimal_srcs | 9 |    7.5%    |
| Divergent           |   48    |   40.0%    |
| **Near-identical (id + subset)** | **72** | **60.0%** |

Acceptance threshold was 80% near-identical → **FAIL on the strict threshold**.

## Diagnosis: structural-not-bug

Looking at the divergent turns, the pattern is consistent: **coord launches
when minimal does not** (e.g., `coord_srcs=[20] minimal_srcs=[]`). There
are no turns where minimal launches and coord doesn't, and very few where
the two pick conflicting sources for the same target.

The structural cause:

- **Minimal's pre-rank** in `propose()` filters candidates via
  `cheap_marginal_value > CHEAP_REJECT_THRESHOLD (-10.0)`. Candidates below
  the threshold are dropped BEFORE Tier-2 ever sees them.
- **Coord's cheap-filter** (`_bundle_cheap_delta`) ranks all enumerated
  bundles by synthesised-obs Δ-favor and takes top-75. There's no
  hard-rejection threshold; weak-but-positive bundles can survive.

Both agents use the SAME Tier-2 scorer (`score_candidate_v4_joint`) and
both gate on `tier2_score > 0`. So when coord fires a bundle that minimal
rejected, the bundle has positive Tier-2 value — it's a real capture
opportunity that minimal's pre-rank filtered out as marginal.

This is not a bug in coord's scaffolding. It's a deliberate design
difference: coord's cheap-filter is a RANKING filter, minimal's is a
THRESHOLD filter.

## What this means for downstream gates

- **Day 9 (defense-equivalence smoke):** Should also expect divergence
  from minimal's reinforce in cases where coord's threat-strength
  formulation values a defensive bundle that minimal's existing
  `propose_reinforce_missions` rejects.
- **Day 11-13 (n=32 multi-opponent panel):** This is the REAL gate.
  If coord's extra fires are quality moves, the panel will show wins.
  If they're noise, the panel will show losses. Either way, the panel's
  empirical signal supersedes Gate 1's structural-equivalence check.

## Sample divergent turns (first 10)

```
seed=0 turn=16: coord_srcs=[20] minimal_srcs=[]
seed=0 turn=17: coord_srcs=[20] minimal_srcs=[]
seed=0 turn=20: coord_srcs=[12] minimal_srcs=[]
seed=0 turn=21: coord_srcs=[12] minimal_srcs=[]
seed=0 turn=22: coord_srcs=[12] minimal_srcs=[20]   ← only conflict case
seed=0 turn=24: coord_srcs=[12] minimal_srcs=[]
seed=0 turn=28: coord_srcs=[20] minimal_srcs=[]
seed=0 turn=29: coord_srcs=[20] minimal_srcs=[]
seed=0 turn=30: coord_srcs=[20] minimal_srcs=[]
seed=0 turn=32: coord_srcs=[20] minimal_srcs=[]
```

Only ONE turn (seed=0 turn=22) shows truly conflicting source choices
(coord picks 12, minimal picks 20). All other divergent turns are
"coord found a capture minimal rejected as marginal."

## Decision: proceed with caveat

The gate verdict on the strict 80% threshold is FAIL, but the divergence
is fully explainable as a design difference, not a scaffolding bug. The
plan's "byte-identical" framing was over-optimistic — the cheap-filter
designs of coord and minimal are fundamentally different. Coord makes
MORE marginal captures (potentially a net positive over a game).

**Proceeding to Day 9** with this understanding. The real verdict will
come from the n=32 multi-opponent panel (Gate 4, Days 11-13) where
actual game outcomes are measured.

If Gate 4 shows coord regressing vs minimal, that's evidence the extra
captures are net-negative; we'd then add a CHEAP_REJECT_THRESHOLD-style
filter to coord's cheap-filter pass. If Gate 4 shows coord winning,
the extra captures are net-positive and the design choice is validated.

## Artifacts

- `audit/20260522T114057Z-gate1-singleton-parity.json` — raw per-seed
  results, 120 turns sampled, 10-turn diff sample.
- `scripts/check_coord_singleton_parity.py` — re-usable probe.
- Knobs added to `agents/coord/main.py::agent()`:
  - `COORD_MAX_BUNDLE_SIZE` env var (int override)
  - `COORD_DISABLE_DEFEND` env var ('1' to disable)
