# Session 2026-05-16 — v8_scavenge analytic chooser falsification

Branch: `claude/recover-main-foundations-MV0e2`
Phase 1 outcome: **FALSIFIED** (Rule 37 — 3+ consecutive variants on
the "depth-0 analytic chooser" axis failed the gate).

## TL;DR

Built `agents/v8_scavenge/main.py` per the PI-approved plan:
- Score each candidate launch by `Δ end-state value` using a BASELINE
  `WorldModel` (built once per turn, before any candidate is considered).
- Reference the F2-style formula `capture_weight × production ×
  time_remaining` for captures; `−waste_weight × ships` for bounces.
- Use `lib.aim.aim_orbiting` for the launch angle.
- Greedy non-dogpile emit (one launch per source / per target per turn).

Result: **0/32 vs v7_0** at three variants. Iterated weights, reserves,
multi-launch — none lifted the floor. The chooser is structurally too
passive: it only finds 0-1 positive-Δ candidates per turn and never
accumulates ships fast enough to compete.

## The arc

| # | Variant | vs nearest n=32 | vs v7_0 n=32 |
|---|---|--:|--:|
| v8.1 | analytic + weights (0.5/1.0) + prod_factor 2× for enemy + ship_cost subtract | 62.5% (20/32) | 0/32 |
| v8.2 | + multi-launch per source, + eta-discount 2%/turn | 53.1% (17/32) | n/a |
| v8.3 | revert eta-discount; keep multi-launch | 53.1% (17/32) | n/a |
| v8.4 | revert multi-launch; weights → composite_capture_value's (0.05/0.5); drop prod_factor + ship_cost | n/t | 0/32 |
| v8.5 | + defense reserve = 2 ships on src | n/t | 0/32 |

All v7_0 runs took ~120s wallclock on 8 workers. p95 turn time
2-8ms (vs v7_0's 590ms) — the analytic approach is ~100× cheaper
in compute. So the failure is NOT a wallclock cliff; it's a
strategy/scoring failure.

## Root cause analysis

Diagnosed on seed 0 vs v7_0 via `scripts/diag_v8.py 0 v7_0`:

- Turn 0: my agent has 1 source (planet 12) with 10 ships. Only ONE
  candidate enumerated (src=12 → tgt=16 ships=10 eta=5 Δ=238.5). All
  other nearest-K targets need >10 ships to capture.
- Turns 1-9: source recovers production, 1-9 ships. Smallest reachable
  neutral has ≥10 ships (production-1 outer ring has 27 ships).
  No positive-Δ candidates. Agent holds.
- Turn 10: 10 ships again. Single candidate again → launch.
- Turn 11-12: 0/1 ships, opp launches a 20-ship counter wave.
- By turn 13: opp has 2 planets, I have 1. Game-over by turn 60-130.

The chooser is doing exactly what the analytic model says is rational:
attack the only beatable target each cycle. But:
1. **The model has no opp speculation.** Opp is treated as IDLE — but
   v7_0 is patient: accumulates ships, then launches multi-fleet waves.
   My agent's "capture is positive Δ" predictions never see those waves
   coming until they're in flight (then it's too late).
2. **No ship-cost amortisation across turns.** Each cycle costs me 10
   ships and gains 1-2 captures; opp's cycle is 20+ ships at one wave
   and gains 1-3 captures.
3. **The marginal-value formula doesn't price the COST of holding** —
   captured planets need defenders, but the chooser doesn't credit
   garrisons or penalise "captured but indefensible."

Rule 37 fires: 3+ falsifications on the "depth-0 analytic chooser
with marginal_value formula" axis. STOP iterating.

## What worked

- `lib.fast_sim`, `lib.aim.aim_orbiting`, `lib.world_model.WorldModel`
  are correct and load-bearing. WorldModel's analytic timeline gives
  predicted owner_at / ships_at instantly (~1ms total). Performance
  is not the bottleneck.
- The plan's structure (depth-0, analytic, no K-step rollout) IS
  fast enough to fit in 800ms. The chooser computes p95=8ms vs v7_0's
  590ms. If a correct scoring function existed, the substrate would
  support it.

## What's still hand-rolled vs origin/main equivalent

| Hand-rolled in v8_scavenge | Origin/main has | Notes |
|---|---|---|
| `_marginal_value` formula | `lib/value_heads/composite_capture_value` evaluated at K-step leaf | composite ran at depth-0 doesn't fire (skips own captures); useful only at leaf |
| Per-source nearest-K × {capture, 2x, budget} | `lib/v7_search._enumerate_drop_one` + missions | v7 stack uses richer enumeration via missions |
| Greedy non-dogpile emit | `lib/planner.settle_plan` | settle_plan has arrival ledger |

## Pivots considered

1. **Use K-step `lib.fast_sim` rollout with composite_capture_value at
   leaf + idle-baseline subtraction.** This is essentially the bootstrap
   session's structure (which reached 43.8%) with a richer leaf head.
   The PI said "no K-step rollout, macro moves only" — but the analytic
   macro-move approach is failing. May need to revisit.

2. **Recover the bootstrap branch's main.py + favor.py + diagnostic
   scripts, then refactor incrementally.** PI explicitly said "drop
   the bootstrap branch entirely." But the bootstrap's 43.8% IS evidence
   that the K-step rollout with idle-baseline subtraction can work
   well as a starting point. We could rebuild that code on origin/main
   without cherry-picking commits — re-implement from scratch using
   the wrap-up's documented bug-1/bug-2/bug-3 fixes.

3. **Thin wrapper on `lib.v7_search.choose` with composite_capture_value
   (= v7_4_capture_value in tree).** Already exists; got 40.6% in prior
   sessions. Not a lift.

4. **Build a different enumeration: mission-based via
   `lib.missions.propose_snipe_missions`, score by my marginal_value,
   emit via `settle_plan`.** Uses the v7 stack's mission proposers but
   our own scorer. The proposers have proper sizing + neighbourhood
   logic that handles "no obvious single target" cases.

## Recommendation to PI

The "depth-0 analytic chooser" approach in the plan I wrote does
NOT compete with v7_0 in practice. WorldModel's analytic prediction
is correct but doesn't capture the strategic dynamic of patient ship
accumulation + multi-wave timing that v7_0 wins with.

Three paths forward (in increasing order of deviation from your
guidance):

**A. Stay analytic but enrich the enumeration.** Switch from
per-source nearest-K to `lib.missions.propose_snipe_missions` for
candidate generation (its sizing accounts for production growth +
opp model). Score by our marginal_value, emit via `settle_plan`.
This is "compose more, invent less." Estimated effort: 1-2h.

**B. Add a 1-step lookahead.** For each candidate, simulate ONE step
via `lib.fast_sim` (my action + opp idle), then evaluate
`composite_capture_value` at the resulting state. This captures
the "ships now in flight" effect that the pure-analytic misses
(my fleet was at src; now it's in flight; src has fewer defenders).
Still no K-step rollout — just 1 step to advance my action. Effort: 1h.

**C. K-step rollout (bootstrap session's approach).** Reimplement the
bootstrap's `score_action` with fast_sim + idle-baseline + composite
leaf scorer. This is the "no K-step" rule you set — but the bootstrap
already showed 43.8% is achievable here. Effort: 2-3h.

My recommendation: **A first**, then **B if A doesn't reach 50%**.
Both stay closer to your "macro moves" framing than C does.

## State as of session end

- `agents/v8_scavenge/main.py` checked in at the v8.1-equivalent
  (no defense reserve, no eta-discount, single-launch-per-source,
  weights 0.05/0.5).
- `scripts/diag_v8.py` adapted to load v7_0 / v4_planner / nearest
  as the opponent.
- pytest baseline: 693 passed, 4 skipped, 1 xfailed, 9 warnings,
  698 collected. No regressions from v8_scavenge addition.
- No commits to origin/main. Local working-tree commits only on
  `claude/recover-main-foundations-MV0e2`.
- NO Kaggle submission. State/current.md unchanged.

## Tests / verification

- `python -m pytest tests/ -q --tb=line` → 693 passed (baseline).
- `python fast.py eval agents/v8_scavenge --vs nearest --max-seeds 16`
  → 62.5% (n=32), Wilson lo=0.453.
- `python fast.py eval agents/v8_scavenge --vs v7_0 --max-seeds 32`
  → 0/32, FAIL.
- `python fast.py bench submissions/v7_0_drop_one.py --games 3`
  → p95=508ms, max=1027ms (v7_0 occasionally peaks past 1s).
