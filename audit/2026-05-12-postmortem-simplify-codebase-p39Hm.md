# Postmortem — 2026-05-12 simplify-codebase-p39Hm

## What went wrong

- **Stale state files were not reconciled against live μ at session-start.** state/current.md reported v3.5.1 PENDING with predicted μ=1090-1100 and σ-equivariance v1 at μ=976.3. Actual live μ via Kaggle API: v3.5.1=994.7 and σ-equiv=1063.2. PI had to push me to check live scores before I would have planned the v7 merge against wrong priors. Cost: would have framed the whole v7 decision on the assumption that aggressive sizing was winning, when it had regressed. The check itself took <60 seconds once invoked.

- **Misdiagnosed v7's broken self-play.** Observed 8/16 = 50% draws (vs σ-equiv's 16/16). Hypothesised env-RNG drift between processes; applied a `cfg["seed"] = obs.step` "fix" to env_from_obs. Post-fix self-play: bit-identical to pre-fix. Real cause was algorithmic — maximin and minimax responses to the same payoff matrix diverge when there's no pure saddle point. Should have written a 10-line probe (compare P0's and P1's payoff matrices on a single divergent turn) before applying any fix. The diff would have shown they were *identical*, ruling out RNG immediately. Cost: ~30 min eval compute + a stale "fix" commit on the branch.

- **Quoted an upstream structural-property claim without re-deriving it.** Game-theory commit `cbed49e` asserted: "σ-equivariance preserved via the existing lib/planner patches. Two v7 instances in self-play should produce identical payoff matrices → mirror picks → draws." This is false on mixed-strategy turns (max-min vs min-max diverge without a pure saddle point). I quoted the claim into HANDOVER and the v7 merge commit message without checking the math. The user (PI) noticed the mismatch — "I thought v7 was a superset of σ-equiv" — and pushed back.

- **Thin local gates drove a +500 LOC integration decision.** The game-theory branch reported 6W/0D/2L = 75% at 8 games vs both v3.4 and precision_v3. PI authorised "merge v7 in full" based on those. Our 32-seed retest of v7 vs frozen v3_snipe was 17/32 = 53.1% Wilson [36.4%, 69.1%] — statistically a tie with 50%, and **worse** than the σ-equiv-only HEAD's 18/32. The 8-game gate's Wilson interval was [37%, 95%]; that's not enough signal to authorise +500 LOC across two new files. I flagged the noise at decision-time but didn't push back hard enough when the merge was authorised.

## Decisions taken — quality review

- **Codebase simplification (commit `1f69039`):** GOOD. PI-directed, executed cleanly, verified bit-identical via self-play 16/16 draws. Net −13,441 lines.
- **σ-equivariance merge (commit `294da5e`):** GOOD. Live μ data showed +68μ lift over our current submission; merge was a clear win at the time. Tests passed, self-play 16/16, A/B 47% → 56%.
- **v7_minimax merge in full (commit `9114f31`):** QUESTIONABLE. Local A/B vs frozen actually regressed (17/32 vs 18/32 for σ-equiv only). Self-play broke from 16/16 → 8/16 draws. Given the priors at decision-time (8-game gates with wide CI, an unfounded structural-superset claim, and PI direction to merge in full), the merge was a defensible bet but the gate threshold was too loose.
- **Deterministic-seed "fix" (commit `7c1e078`):** BAD. Applied without verifying the hypothesis. Turned out to be a no-op. Should have probed first; would have saved 30 min and a stale commit.

## Frictions logged this session

Cross-linked to today's `audit/friction.md` `## 2026-05-12 (simplify-codebase-p39Hm)` block:

- `state-files-out-of-date-vs-live-mu`
- `local-A/B-not-live-mu-predictor`
- `misdiagnosed-v7-self-play-break`
- `thin-local-gates-drove-500-LOC-merge`
- `accepted-upstream-correctness-claim-at-face`

## Promotion candidates (drafted; PI ratification pending)

### [ ] [CROSS-CUTTING] CLAUDE.md / Rule 32 addendum — pull live submission scores at session-start

**Tag:** `live-mu-check-at-session-start` (state/current.md was off by 60-100μ in both directions today; planned the v7 merge against stale priors until PI prompted a live check).

**Where to insert:** Append to CLAUDE.md Rule 32 ("Session-start git fetch"), after the `git diff HEAD..origin/main HANDOVER.md` line.

**What to add:**

> Additionally, pull live submission scores from the Kaggle API and reconcile against state/current.md / HANDOVER.md claims. State-file μ predictions have drifted off live scores by ±100μ twice in this comp; any session decision that conditions on "what's our best submission" must use the live number, not the state-file number.
>
> One-shot probe:
> ```
> curl -s -u $KAGGLE_USERNAME:$KAGGLE_KEY \
>   https://www.kaggle.com/api/v1/competitions/submissions/list/orbit-wars \
>   | python -m json.tool | head -200
> ```

**Why:** Today's session was about to plan v7 against a stale "v3.5.1 expected 1090-1100" prior when it had actually scored 994.7. Without the live check the entire v7 merge would have been framed against wrong assumptions. Cost: would have wasted 6+ hours of compute and engineering on the wrong question.

### [ ] [CROSS-CUTTING] do-and-dont.md — probe before fix on empirical-symmetry breaks

**Tag:** `probe-before-fix-on-empirical-symmetry-breaks` (today: applied a "deterministic-seed" fix to a v7 self-play break without probing the actual artifact; turned out to be a no-op).

**Where to insert:** Add to do-and-dont.md under "Common antipatterns".

**What to add:**

> **Antipattern:** When an empirical symmetry property breaks (e.g. self-play stops drawing, σ-equiv lock falls below 100%), do NOT immediately apply a fix based on the first hypothesis. Write a 5-10 line probe that examines the actual artifact (payoff matrices, action diffs, intermediate scores) on a single divergent example before changing code. Mismatched hypotheses lead to no-op commits that look like fixes and obscure the real cause.

**Why:** Today's `cfg["seed"] = obs.step` fix was a no-op (env was already deterministic via tournament harness). A 10-line probe comparing P0's and P1's payoff matrices on a single 8W/8D divergent seed would have shown they were *bit-identical*, immediately ruling out RNG drift and pointing at the algorithmic root cause (max-min vs min-max divergence without a pure saddle point). Cost: 30 min eval compute + a stale `7c1e078` commit on the branch.

## PI additions

_pending; appended after PI replies to step 4 of the postmortem skill._

## Framework version at session-end

- Commit SHA: `7c1e078` (pre-postmortem)
- Active rules: 1-36 from CLAUDE.md `## Operating rules — concise`
- Loaded skills this session: postmortem, kaggle-comp (inherited via CLAUDE.md pointers)
