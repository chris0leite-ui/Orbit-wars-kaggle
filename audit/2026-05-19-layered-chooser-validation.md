# Layered chooser validation — 2026-05-19

> Validation of Layer-0 closed-form predicates (W1/W2/L1/L2) +
> swap-able `chooser_layered.py`. Commit `671edb1` on branch
> `claude/strategy-framework-design-OyoYR-rebased`.
>
> Plan reference: `/root/.claude/plans/take-the-lens-of-magical-shore.md`
> §10. No source code changes during this validation — only bundle
> generation and `fast.py` runs.

## Bundles under test

- **Control**: `submissions/baseline.py` (built `2026-05-19` from
  `agents/baseline/` HEAD `671edb1`; sha256 `bcf0b238a03cbe04`).
  Default `BASELINE_CHOOSER=trajectory`.
- **Treatment**: `submissions/_ab/layered_w1w2l1l2.py` (one-line `sed`
  override of the control: `setdefault("BASELINE_CHOOSER", "trajectory")`
  → `"layered"`). Default `BASELINE_INNER_CHOOSER=trajectory`.

Diff is exactly one line — verified via `diff`:
```
11553c11553
< os.environ.setdefault("BASELINE_CHOOSER", "trajectory")
---
> os.environ.setdefault("BASELINE_CHOOSER", "layered")
```

## Step 1 — Bench parity (5 games each vs random)

**Command:**
```bash
python fast.py bench submissions/baseline.py --vs random --games 5
python fast.py bench submissions/_ab/layered_w1w2l1l2.py --vs random --games 5
```

**Control (trajectory)**:
- n=837 turns, p50=252, p95=442, p99=565, max=924, over_1000ms=0
- verdict: **PASS** (gate: p95<800ms AND zero ≥1000ms)

**Treatment (layered)**:
- n=525 turns, p50=26, p95=438, p99=643, max=679, over_1000ms=0
- verdict: **PASS**

**Verdict**: Wallclock parity confirmed. Layered's p50 is ~10× lower
than trajectory's (26 ms vs 252 ms) — Layer 0 short-circuits easy
turns. p95 essentially identical (438 vs 442 ms). Max is markedly
lower under layered (679 vs 924 ms) — fewer rollouts on the worst
turns. **No wallclock regression; gentle improvement.**

## Step 2 — Head-to-head: layered vs trajectory, n=32

**Command:**
```bash
python fast.py eval submissions/_ab/layered_w1w2l1l2.py \
    --vs submissions/baseline.py --max-seeds 32 --workers 4
```

**Output** (2026-05-19, ~45 min wallclock):

```
n=64  wins=33/64  (51.6%)  Wlo=0.396  Whi=0.634  elapsed=1404.6s
verdict=INCONCLUSIVE  (max seeds reached)

focal turn-ms  p50=187  p95=622  max=1286   total elapsed 2713.7s
```

**Verdict**: **STOP**.

Two reasons (either alone is sufficient):

1. **Wlo=0.396 is below the 0.45 floor**. Per plan §10 decision
   matrix: "Wlo < 0.45 means Layer 0 is actively regressing — stop
   and diagnose." The point estimate (51.6%) is statistically
   indistinguishable from break-even, but the lower CI bound permits
   a real regression. The single-game loss in Step 4 was NOT
   anomalous.

2. **Wallclock regression**: max=1286ms exceeds the 1000ms env
   actTimeout. Step 1 bench vs random was clean, but contested mid/
   late-game turns push past the cap. The layered chooser's per-turn
   overhead = Layer 0 classification + inner-chooser rollout on the
   residual; on hard turns both costs stack. The chooser-trajectory
   safe_deadline pre-bail probably saves us inside the rollout but
   the outer L0 pre-pass is uninterruptible.

## Step 3 — Panel calibration + h2h, n=32

**SKIPPED** per Step 2 STOP verdict. Per plan §10, Step 3 runs only
after Step 2 clears Wlo ≥ 0.45.

## Step 4 — Predicate-fire diagnostic

Single-game introspection via `scripts/layer0_introspect.py` (wraps
`layer0_classify` to log per-turn verdicts).

### Game A: layered (P0) vs random (P1), seed 42, 113 turns

- Outcome: layered WIN (1/-1)
- Totals: cands=3057  **W1=1301**  W2=0  L1=0  L2p=0  uncertain=1756
- Per turn: cands=27.1  **W1=11.5**  W2=0.00  L1=0.00  emit_L0=11.5
- Interpretation: against a non-counter-attacking opp, ~42% of all
  candidate launches are provably winning captures. Layer 0
  short-circuits 11.5 rollouts per turn — significant CPU saving and
  zero noise from `lite_greedy_policy` mis-rating.
- W2 / L1 silent: no inbound enemy threats (random doesn't reinforce
  or counter), and no provably-wasted launches.

### Game B: layered (P0) vs trajectory baseline (P1), seed 42, 312 turns

- Outcome: layered **LOSS** (-1/+1)
- Totals: cands=4794  **W1=1741**  W2=84  **L1=3**  L2p=0  uncertain=2966
- Per turn: cands=15.4  **W1=5.58**  **W2=0.27**  L1=0.01  emit_L0=5.85
- Interpretation:
  - W1 still fires ~5.6/turn even against a strong counter-attacking
    opp. Layer 0 IS doing real work, not dead weight.
  - W2 fires 84 times — defensive reinforces that the baseline's
    attacks make actionable. Important for defense.
  - L1 fires only 3 times in 312 turns — proposer's existing
    admissibility filters catch most wastes upstream. L1 is essentially
    redundant in this game.
  - L2 fires zero times — proposer's `(src, tgt, wait_band)` dedup
    already catches everything.
- Single-seed loss does NOT prove Layer 0 regresses; Step 2's n=32
  Wilson CI is the authority.

### Concrete trace (Game B, opening 3 turns)

```
step 0: cands=8 W1=8 W2=0 L1=0
   W1: src=16 -> tgt=0  ships=34 wait_N=6  lower_bound=435.48
   W1: src=16 -> tgt=0  ships=46 wait_N=9  lower_bound=422.44
   W1: src=16 -> tgt=14 ships=10 wait_N=0  lower_bound=362.78
   W1: src=16 -> tgt=4  ships=14 wait_N=1  lower_bound=359.12
   W1: src=16 -> tgt=12 ships=10 wait_N=0  lower_bound=331.18
   W1: src=16 -> tgt=8  ships=62 wait_N=13 lower_bound=331.18
   W1: src=16 -> tgt=24 ships=14 wait_N=1  lower_bound=185.10
   W1: src=16 -> tgt=20 ships=10 wait_N=0  lower_bound=183.23
```

Every opening-turn candidate from our home (P16) is a W1 commit.
Lower bounds are `tgt.production × pv_horizon`: high-value targets
(P0, P14) get ~$400; mid-value (P4, P12, P8) get ~$330; far/low-prod
targets (P24, P20) get ~$185. Bipartite matching (greedy v1) picks
the best one for emit; the rest reserve src+tgt for future turns.

### Findings summary

- **W1 is load-bearing**: 5-12 commits per turn depending on opp
  strength. Doing the work the plan promised.
- **W2 is useful but rare**: fires only when there's an in-flight
  enemy threat AND no at-rest opp in counter-reach.
- **L1 is nearly redundant**: proposer's filters catch most wastes
  upstream. Keep but don't optimise.
- **L2 is dead code**: existing dedup is sufficient. Don't extend
  to cross-target until evidence shows it's missing wins.

## Decision

**STOP.** Do not submit the layered chooser to the ladder. Do not
flip the default `BASELINE_CHOOSER`. Keep `chooser_layered.py` as
opt-in research code on the dev branch.

### Diagnosis

The math is sound (every predicate has a passing unit test). What
fails empirically is the **emit-time decision quality** in two ways:

1. **W1's single-nearest-opp bound is too loose under coordinated
   counter-attack.** The bound assumes only the nearest strong opp
   would counter; v15 / trajectory baseline routinely launches from
   2-3 sources concurrently. A capture that the bound certifies as
   provably-held can still flip under gang-up.

2. **Layer 0 preempts the inner chooser's source allocation.** When
   W1 commits src=A → tgt=T, the trajectory chooser's rollout never
   considers src=A's other options. If trajectory would have used A
   for a higher-value (residual-scored) launch, that win is lost.
   Predicate v1 commits naively; bipartite matching across (W1
   commits ∪ residual) would fix this in part.

3. **Wallclock regression** (max=1286ms). Layer 0's pre-pass is
   uninterruptible and stacks on top of the inner rollout. The
   wallclock cap was clean against random but breaks on contested
   high-candidate-count turns.

### Recommended next slice (do NOT execute without PI sign-off)

1. **Strengthen W1's bound to multi-opp coordinated counter**:
   sum-of-all-opp-ships-in-reach instead of nearest-only. Re-run
   Step 2; expect lower W1 firing rate but higher per-commit
   precision.
2. **Wallclock budget**: cap Layer 0's pre-pass at 50 ms; if it
   runs over, fall through to the inner chooser unconditionally.
3. **Optional**: bipartite matching at emit time across (commits ∪
   inner moves) instead of greedy.
4. **Re-validate at n=32** vs trajectory baseline. Need Wlo > 0.45
   to continue.

If after one round of refinement the bound + budget fix STILL
shows Wlo < 0.45, the Layer-0 architecture is the wrong fit for
this game. Rule 37 (3-variant cap per axis) caps refinement at 3
attempts on the W1-bound axis.

## Notes

- The current rolling-pair floor is μ=1118.8 (HANDOVER 2026-05-19;
  subject to ladder drift). Submission decision waits on all 3 gates
  passing.
- This audit doc is the source of truth for the validation; no
  shortcuts.
