# v7 minimax — ablation sweep results (2026-05-12)

> Branch: `claude/game-ai-lookahead-3ucqH`.
> Baseline: `submissions/v3.5.1.py` (live submission #52565976, PENDING).
> Per-variant: 12 seeds × 2 sides = 24 games. Wilson 95% lower bound;
> PASS ≥ 0.55, NEUTRAL ≥ 0.45, FAIL otherwise. Sweep wallclock: 50 min.

## TL;DR for the PI

**v7_0_drop_one wins decisively** at **20/24 = 83.3% vs v3.5.1** (Wilson
lo 64.1%). p95 turn 781 ms — under the 800 ms safety gate (1 s
actTimeout) but with thin margin. Every richer-enumerator variant
fails or only ties. The lift is in **having a rollout-based veto on
v3.5.1's net-negative-EV launches**, not in proposing new actions.

Recommended next step: bundle `v7_0_drop_one`, run 16-seed 4P FFA,
then PI authorises submit. **No `kaggle competitions submit` runs
from this branch** per Rule 1; v3.5.1 stays the live agent until you
approve replacement.

## 2P A/B vs v3.5.1

| Variant | W/L/D | wins/n | winrate | Wilson lo | p95 ms | verdict |
|---|---|---|---:|---:|---:|---|
| `v7_0_drop_one` | 20/4/0 | 20/24 | 83.3% | 64.1% | 781 | **PASS** |
| `v7_1_target_swap` | 14/10/0 | 14/24 | 58.3% | 38.8% | 900 | **FAIL** |
| `v7_2_ship_sweep` | 12/12/0 | 12/24 | 50.0% | 31.4% | 843 | **FAIL** |
| `v7_3_archetype` | 12/12/0 | 12/24 | 50.0% | 31.4% | 877 | **FAIL** |
| `v7_4_hungarian` | 11/13/0 | 11/24 | 45.8% | 27.9% | 431 | **FAIL** |
| `v7_combined` | 17/7/0 | 17/24 | 70.8% | 50.8% | 866 | **NEUTRAL** |

## What this means

The pattern is sharp: **only the strict-subset enumerator wins**.
drop_one can only return a SUBSET of v3.5.1's incumbent action; the
chooser either picks the full v3.5.1 plan (parity floor) or a subset
of it (drops a launch). Every win comes from a turn where one or more
of v3.5.1's launches was *actively bad* — over-committed to a target
that will be ours anyway, sized too small to capture, or destined to
lose its fleet en route. Dropping that launch keeps the ships at home
defending, and the K=10 rollout under the Tier-1 opponent mirror
correctly identifies the case.

Every other variant **adds candidates that DIFFER from v3.5.1** in
target, ship size, or global assignment. Those candidates' rollout
scores are dominated by self-play-policy noise the K=10 horizon can't
filter. The result is a chooser that, on average, picks a candidate
slightly worse than the incumbent — hence the 45-58% range with
Wilson lo well below the 45% NEUTRAL floor for hungarian and
target_swap.

`v7_combined` (70.8%, Wilson lo 50.8% — NEUTRAL) confirms the
hypothesis: the union includes drop_one's good candidates AND the
others' worse candidates, and the chooser splits the difference.

## Tying back to the Phase 2 caveat

The Phase 2 audit
(`audit/2026-05-11-lookahead-phase2-forward-sim.md:131-146`) called
this out explicitly:

> "Sim<K> measures predictive power (can we read the winner from a
> v2-self-play rollout?) not strategic strength (does acting on the
> prediction make us win more?)."

v7_0 sidesteps the predictive/strategic gap: dropping a launch
*reduces our action space* — there's no opposing strategy that
"acting on the prediction" can fail to exploit, because there's
nothing strategic about not-launching one fleet. The rollout's job is
just to identify which of v3.5.1's launches is net-negative, and at
K=10 it does that well.

The richer enumerators violate this implicit constraint: they propose
*different* actions, and the rollout has to predict *strategic*
outcomes — which Phase 2 said it can't reliably do.

## Caveats / risk flags

1. **p95 turn ms is tight for v7_0:** 781 ms in our local bench,
   ~200 ms below the 1 s actTimeout. Live ladder hardware variance
   could push this over. Mitigation: drop K from 10 → 8 (~20% rollout
   cost reduction) for the live bundle if the 4P FFA confirms the
   lift survives.
2. **12 seeds is the lower bound for a Wilson 55% gate:** v7_0 wins
   at p≈0.833, which is well above the threshold even at n=24. But
   v7_1/v7_2/v7_3 results at p≈0.50-0.58 should be re-tested at
   n=64 next session to confirm they're truly at parity / regression,
   not just sample-size-limited.
3. **2P vs 4P transfer is not guaranteed.** The 4P FFA panel
   (next step) is the second gate. If v7_0 plays the same in 4P,
   ship; if it regresses, debug the multi-opponent rollout.

## Decision recommendation for the PI

**Bundle `v7_0_drop_one` and run the 16-seed 4P FFA panel.** If 4P
PASSES (Wilson lo first-place ≥ 0.90 vs the {v3.5.1, v3_snipe_frozen,
weakest} background) → ready to authorise submission. The bundle
would evict v3.4 (#52556866, μ=995.4) from rolling-last-2; v3.5.1
(#52565976, PENDING) stays.

**If 4P FAILS:** drop K to 8, re-test 2P; v7_0 is still likely the
right candidate; just needs the safety margin.

**Submission slot usage:** 1 PI-authorised push consumes 1 slot;
4/5 daily slots remain. v3.5.1 stays as one of the rolling-last-2
regardless.

## Artifacts

- `audit/tournaments/20260512T090300Z.json` — smoke 4-game v7_0.
- `audit/tournaments/20260512T090901Z.json` — v7_0 A/B (24 games).
- `audit/tournaments/20260512T092357Z.json` — v7_1.
- `audit/tournaments/20260512T092920Z.json` — v7_2.
- `audit/tournaments/20260512T094140Z.json` — v7_3.
- `audit/tournaments/20260512T094437Z.json` — v7_4.
- `audit/tournaments/20260512T095320Z.json` — v7_combined (timestamp
  approximate — last in the run).
- `audit/2026-05-12-v7-iteration-summary.json` — machine-readable
  one-row-per-variant summary.
- `audit/2026-05-12-v7-iteration-narrative.md` — pre-sweep planning
  narrative (what each variant was intended to test).

## What ships in this branch regardless of submit decision

- `lib/v7_search.py` — reusable enumerator + scorer + chooser.
- 5 ablation agents + `v7_combined`.
- `scripts/run_v7_ablation.py` — autonomous loop.
- 12 unit tests + the bench script.

## Next-session candidates

- **v7.5 — K=8 + n=64 confirmation of v7_0.** Tighter budget,
  larger sample. Probably the right pre-submit gate before any
  future push.
- **v7.6 — narrowed enumerator.** Test drop_one + add ONLY a
  single Hungarian-style global candidate sized identically to
  v3.5.1's. Probes whether the lift survives controlled
  enrichment.
- **v7.7 — depth-2 minimax with width 3.** Beam search on top of
  drop_one. Each leaf at K=5 (shallow leaves to fit in budget).
- **v8 — Tier 2 trained opp model.** Logistic regression on the
  37k labeled launches in `data/shot_validator/`. May break the
  symmetric-opp gradient problem.
