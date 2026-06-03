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

---

# Postmortem (PM session) — 2026-06-03 champion-strategy-rules-00JzI

Second session this day on this branch. PI greenlit building a
consolidation/massed-strike proposer for `holdgrab`; recon flipped it into a
pre-registered inert-check first. See
`audit/2026-06-03-mass-to-hold-consolidation-step1.md`.

## What went wrong
- **Minor decision-quality miss:** the first census smoke used holdgrab's
  full-rollout focal (~800ms/turn) when the measured quantity (opportunity rate)
  is rollout-independent. I had the throughput prior (`config.py` rollout budget
  800ms) and should have reasoned about it before launch. Caught and fixed inside
  one 4-game smoke (→ closed-form focal, ~20×). Low cost; logged.
- **Stale-rule (framework gap, not a session error):** CLAUDE.md Rule 49 still
  presents joint-coordination as "the active multi-session thrust," but its main
  sub-axis (capture-coalition) was closed 2026-06-02. A fresh session reading
  Rule 49 would contradict the closed-tracks list. Surfaced as candidate A.
- **PI-overrides:** none. Direction was delegated ("[No preference]"); no
  mid-session correction.
- **Rule-bypass:** none. Rules 44 (closed-tracks check), 37/47 (inert-check
  before build), 26 (surfaced the contradiction via AskUserQuestion) all applied
  — that discipline is what converted a multi-day build into a 4-game gate.

## What went right (decision quality)
- Recon-before-build caught a 24-hour-old falsification of the same axis, and the
  inert-check-first reframing means the worst case is one committed enumerator,
  not a built-then-falsified mechanism (the 2026-06-02 failure mode).

## Frictions logged this session
- `census-focal-rollout-too-slow` (audit/friction.md 2026-06-03) — instrument
  harness drove the production agent instead of the cheapest faithful policy;
  fixed same-session (Rule 29). Recurs with this morning's
  `heavy-vs-heavy-ab-throughput-wall` — a standing throughput theme.

## Promotion candidates (PI ratified: NONE — declined this session)
- **A — Fix Rule 49 staleness** (CLAUDE.md): note capture-coalition sub-axis
  closed; only mass-to-HOLD residual live. *Not promoted.*
- **B — Inert-check before exploit-build** on falsified-adjacent axes: cheap
  opportunity-census with pre-registered GO/NO-GO before the mechanism. *Not
  promoted.*
- **C — Instrument uses cheapest faithful policy** when the measured quantity is
  policy-independent. *Not promoted.*
  (Recorded for the record; PI selected "none" on the promotion question.)

## PI additions (from step 4)
- None ("." = nothing to add).

## Census disposition
- Full 192-game panel census was in-flight at the prior wrap; the session
  resumed across a boundary and the `/tmp` census output was lost (job did not
  finish). Early 4-game read = NO-GO (0.33% of turns, median 0/game). Tool is
  committed + reproducible; next session re-runs `python
  scripts/probe_consolidation.py` for the frozen verdict.

## Framework version at session-end
- Commit SHA: 2195b0b (pre-wrap; wrap commit follows)
- Active rules: CLAUDE.md Rules 0–49.
- Loaded skills this session: Explore + Plan agents; postmortem (this).
