# Postmortem — 2026-06-04 champion-ml-graft-majestic-storm

## Status

**Not done — parked, not killed.** PI explicit: "this is not done yet. Just...
we couldn't transfer the results to our strategy."

The underlying observation — "we don't use all our ships, especially rear
stockpiles that sit idle far from action" — remains a real, PI-verified
problem from live games. What we falsified is **three specific
implementations** of a circulation-style fix as a thin post-pass; we did
NOT falsify that the fix-shape itself (pressure-gradient ship routing) is
the right idea. Biel's "Producer" agent on Kaggle uses a near-identical
mechanism successfully — but his **entire planner thinks in pressure**, so
the regroup's destinations align with where his planner is firing. Our
chooser thinks in (source, target) trades, so a transferred pressure-
gradient post-pass routes ships to destinations our chooser ignores.

## What went wrong

- **Three successive attempts at the same mechanism family before
  abandoning, when v2 → v3 inversion was already a stop-signal.** v1
  (centroid scalar) regressed 5/16 + wallclock blowup 2958 ms. v2 (Biel
  pressure) reached parity 8/16 + max 1424 ms. v3 (pressure + dst-
  usefulness filter) regressed back to 5/16 + max 1396 ms. The fact that
  a STRICTLY TIGHTER filter (v3) made things worse than v2 is the same
  pattern as the prior `chooser-family-saturation` finding from 2026-05-16:
  when local-tightening regresses, the substrate is wrong; further tuning
  of the same lever won't recover it. I could have stopped at v3 trigger
  and would have noticed faster.

- **The diagnosis "Biel's planner thinks in pressure end-to-end" arrived
  too late.** I only surfaced this framing during the code review AFTER
  v2's parity result. If I had read Biel's notebook in full before
  proposing v1, the impossibility of a thin-post-pass transfer might have
  been obvious — Biel's scoring, his garrison forecast, and his regroup
  all reference the same scalar field. Ours doesn't. A planner that
  doesn't share the source signal's representation can't usefully consume
  ships pre-positioned by that signal.

- **PI overrides this session: 0 (consistent direction).** Both my
  "abandon now" calls (after idle_stockpile v1 regression, after
  circulation v1 regression) were overridden in favor of tuning. The
  override turned out to be partly correct for idle_stockpile (fixing the
  safety gate moved it from regression to parity) but not for circulation
  (no tuning version cleared parity). On balance the calibration says PI
  runs more aggressive than I do on "this lever is worth more iteration"
  — not bad signal, but I should weight my abandon calls less heavily
  next time and present them as "weak abandon" rather than "strong
  abandon".

- **Compute budget for the session was reasonable.** Three n=16 A/Bs
  (~12 min each) + several probes + multiple bundles. ~2-3 hours of
  compute on this lever family across the session. Not catastrophic.

## Frictions logged this session

See `audit/friction.md` 2026-06-04 entry:

- `tag: mechanism-family-cross-stack-transfer-needs-planner-compat` — a
  competitor's published lever (Biel's pressure-gradient regroup) cannot
  be transferred to our stack as a thin post-pass when their planner
  internally uses the same scalar field and ours doesn't.

- `tag: tighter-filter-regresses-not-improves` — v2 8/16 → v3 5/16 after
  adding a strictly tighter destination filter. The filter cut off
  launches that were apparently providing incidental value (defensive
  density, pre-positioning) the v3 logic doesn't model.

## Promotion candidates (PI ratified)

**PI 2026-06-04: nothing to promote.** The frictions tagged today are
useful descriptive labels but don't yet rise to a generalisable rule the
framework should enforce. Re-evaluate if a fourth same-family
falsification happens.

## PI additions

PI replied "Nothing to add or to promote."

## Where this is parked for future sessions

- **Live code is preserved** behind `BASELINE_FRONTIER_CIRCULATION=1`
  (default OFF). The shipped champion is unaffected. Three commits on
  the branch: `924b44a` (v1), `24ac0d7` (v2), `b836407` (v3). v3 is the
  HEAD state.
- **Path back to lift, if you want to try again:** either (a) rewrite the
  chooser's scoring to be pressure-aware (large project — essentially
  port Biel's planner), or (b) find a CONCRETE 2-hop attack the chooser
  is missing today, and pre-position ships only for THAT specific play
  (goal-directed, not gradient-driven). Option (b) is closer to "send
  ships only when the chooser would have fired them next turn if it
  could see further".
- **Pivot away from circulation-as-post-pass for now.** Three
  falsifications says this mechanism family doesn't lift in our stack
  with current chooser internals.

## Framework version at session-end

- Commit SHA: `b836407` (`feat(frontier_circulation v3): destination-
  usefulness filter aligns lever with chooser`)
- Branch: `claude/champion-ml-graft-majestic-storm` ahead 119 of
  `origin/main`
- Active rules: 0, 1, 12, 32, 35, 36, 38, 39, 40, 42, 45, 46
  (per `CLAUDE.md`)
- Loaded skills this session: `postmortem`, `kaggle-comp` (background)

---

# Postmortem — 2026-06-04 (second session, sub 53369848)

This session continued the same branch on top of `b836407`. The
producer_plus migration host (planned in `state/MIGRATION_PLAN.md`)
moved from Step 1 (skeleton) through Step 4 (multi-size) and produced
the first hybrid submission.

## What went wrong

- **Step 4 first revision shipped a silent-crash bug**:
  `clamp(min=float, max=Tensor)` is invalid PyTorch syntax.
  `plan_lite_waves` threw, the agent returned `_empty_entries`, and
  the seat-balanced n=32 came in at 1/64. The only clue was wallclock
  — p50=14 ms vs vanilla 22 ms (FASTER than vanilla). Unit tests
  passed (env-gate function pure-Python only); synthetic invocation
  in a one-game smoke also passed under the same gating. The bug was
  a tensor-op edge case needing real game state to trigger. Cost:
  ~45 min A/B compute on what turned out to be a crash, not a
  regression.

- **n=8 P0-only triage gave 6/8 win rate** which was misleading. The
  same methodology flaw flagged in the M1/M2/M3 post-mortem
  (previous session) recurred. Followed up with seat-balanced n=16
  vs adaptive_k (13/16) which looked great — then n=16 vs vanilla
  producer came in at 5/16 (regression). Rule 45 was explicitly
  overridden by PI to submit on n=16.

- **Step 2 (adaptive K) was carried into the multi_size submission
  without ever being directly validated vs producer.** When PI asked
  for the validation A/B, it came back at exactly 8/16 (parity).
  Step 2+4 vs producer was 5/16 (regression); stripping Step 2
  brought Step 4 alone to 10/16 (62.5%). Adaptive K interfered
  destructively with multi-size — discovered only at session-end.

- **Cross-wave over-drain bug was caught only by the deep code
  review.** The PyTorch fix made the agent runnable but greedy's
  source budget was still uncapped at `obs.ships` (not `drain`).
  Single-size's invariant (`cand_send == drain` ⇒ at-most-one wave
  per source) accidentally protected against over-drain; multi-size
  with smaller variants broke it silently. Without the 9-angle code
  review, the n=16 vs producer would likely have come back at parity
  or worse and we'd still be diagnosing.

## Frictions logged this session

See `audit/friction.md` entries dated 2026-06-04 (second session):

- `tag: empty-actions-faster-than-vanilla-is-silent-crash`
- `tag: migration-step-validate-vs-target-not-just-prior-step`
- `tag: cross-wave-over-drain-from-shared-budget`

## Promotion candidates (PI ratified: NO)

Three candidates were drafted (wallclock-as-crash-canary,
migration-step-validate-vs-target, greedy-budget-needs-explicit-cap).
PI replied "Nothing to add or to promote." All three remain as
session-frictions only; no entries added to
`.claude/skills/kaggle-comp/improvements.md`.

## PI additions

PI replied "Nothing to add or to promote."

## What shipped

- **Submission 53369848** — `producer_plus_multi_size_on.py` (commit
  `4005b19`, 211 011 B). First hybrid from the producer_plus
  migration track. Step 4 multi-size enumeration on Producer's
  engine, no Step 2 adaptive K. Local n=16 seat-alt vs vanilla
  producer: 10/6 (62.5%), Wilson [0.39, 0.81], perfectly
  seat-balanced (5W/3L on both P0 and P1). Wilson-lo (0.39) does
  not strictly clear Rule 45's 0.55 gate; PI explicit override.
- Evicts: 53332500 `champ_computeByShips_on` (μ 1150.6).
- Backstop in rolling pair: 53336920 `champ_refine_adaptivek`
  (μ 1148.8).
- Predicted live μ: ~1200-1250 (rough). Warm-up over ~24 h.

## Next steps for the next session

1. **Read live μ once TrueSkill warms up** (~24 h post-submit).
2. **If μ ≥ ~1170:** proceed to Step 5 of `state/MIGRATION_PLAN.md`
   (multi-source coalitions — biggest expected lift, Producer is
   explicitly single-source).
3. **If μ < ~1170:** investigate local-vs-live divergence. Possible
   causes: opponent panel difference (local A/B used producer only;
   live ladder is a mixed panel), seat-bias artefact, or n=16 was
   just lucky.

## Framework version at session-end

- Commit SHA: `6896f2b` (`docs: push-claim board + handover updated
  for sub 53369848`)
- Branch: `claude/champion-ml-graft-majestic-storm`, ahead 138 of
  `origin/main`.
- Active rules: 0, 1, 12, 32, 35, 36, 38, 39, 40, 42, 45, 46
  (per `CLAUDE.md`).
- Loaded skills this session: `postmortem`, `code-review`,
  `kaggle-comp` (background).
