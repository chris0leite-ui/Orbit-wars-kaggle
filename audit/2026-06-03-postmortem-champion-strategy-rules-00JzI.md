# Postmortem — 2026-06-03 champion-strategy-rules-00JzI

Decision-quality framing (not outcome). Two experiment arcs closed, both
negative; the load-bearing lesson is about the *measuring instrument*, not the
mechanisms.

## What went wrong

- **~5 heavy A/B runs burned to timeout (process bug, my decision).** I launched
  n=16 heavy-vs-heavy A/Bs (v7_minimax ×2, champion-vs-champion ×1) without first
  pricing per-game cost. Champion bundles are ~2 min/game; n=16 at workers=8
  never finished. After the FIRST timeout I should have sized n/workers to budget
  immediately rather than re-launching the same shape. Bad decision given priors
  available after timeout #1. → promotion candidate.

- **Refiner built before confirming the opportunity exists (prior session, paid
  off today).** The augment-not-replace refiner was built fully, then found
  INERT because `generate_sync_coalitions` yields zero candidates in real games.
  The cheap "does this structure even occur?" check (raw coalition count) would
  have pre-empted the build. Defensible under the PI's "be ambitious, skip the
  divergence measurement" steer, but the cheapest falsifier should precede the
  build. → rule-gap.

- **ME-defends: good decision, bad outcome (no blame).** Picked the principled,
  built, default-OFF modeling fix on the PI-chosen value-leaf axis; verified
  mechanism (test) and timing (bench) BEFORE the A/B. Result 5/16 vs champion is
  a clean negative — exactly the kind of good-decision/bad-outcome the framework
  says not to retro-blame.

- **Meta: local A/B is a near-dead instrument at the champion's level.** Both
  the refiner and ME-defends died on a testbed that structurally cannot resolve a
  lift — every local opponent is saturated (v7_0 16/16) or too heavy to batch.
  This is the real finding of the session.

## PI-overrides / calibration

- PI reframed my multiple-choice fork into "think how to turn what we have into
  an advantage" → produced the refiner reframe. (Agent defaulted to a menu; PI
  wanted synthesis.)
- PI chose the value-leaf axis; PI surfaced the early-finish/throughput idea
  (correct instinct — it just isn't in the harness); PI called wrap-up.
- No PI correction of a wrong technical claim this session.

## Frictions logged this session
- `heavy-vs-heavy-ab-throughput-wall` (audit/friction.md 2026-06-03)
- `rollout-self-policy-precomputed-at-tick0` (ditto)
- `same-agent-variant-ab-env-collision` (ditto)

## Promotion candidates (PI ratification: PENDING)

### [ ] [CODE-COMP-DISCOVERED] guardrails — size heavy-vs-heavy A/Bs to budget
**Tag:** `heavy-vs-heavy-ab-throughput-wall`
**What to add:** Before any n≥16 A/B where BOTH agents run full search
(champion-vs-champion, vs minimax), price one game first; use workers=4 and a
finish-able n; never re-launch the same n=16 shape after a timeout. Prefer a
lighter NON-saturated opponent for triage.
**Why:** ~1h+ compute wasted today across 5 timed-out runs.

### [ ] [CODE-COMP-DISCOVERED] guardrails — confirm the testbed can resolve a lift before a champion-level local A/B
**Tag:** `local-ab-saturated-at-champion-level`
**What to add:** A local A/B at the champion's level is only informative if the
opponent is non-saturated AND finishes at n≥16. If neither holds (v7_0 saturated;
heavier opponents don't finish), the verdict cannot distinguish neutral from a
small lift — prefer submit-and-measure (PI-gated) over more local A/B.
**Why:** two consecutive sessions (refiner, ME-defends) produced
inconclusive/negative locals the instrument could not have shown as lifts.

## Framework version at session-end
- Commit SHA: 609e1c3 (pre-wrap; wrap commit follows)
- Active rules: CLAUDE.md Rules 0–49.
- Loaded skills this session: Plan, Explore agents; postmortem (this).
