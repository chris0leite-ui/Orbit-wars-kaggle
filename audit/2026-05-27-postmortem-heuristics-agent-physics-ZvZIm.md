# Postmortem — 2026-05-27 heuristics-agent-physics-ZvZIm

## What went wrong

Three decisions worth flagging in retrospect, all related to the
Phase 4 iteration burn:

- **Iterated Phase 4 variants at n=32 without first nailing down the
  Phase 3 baseline at n=64.** Phase 3's effect-size estimate ranged
  6/32 → 7/32 → 8/32 across re-runs (Wilson [0.09, 0.35] consistently),
  while each Phase 4 candidate landed within the same Wilson band.
  Three Phase 4 variants — joint capture, multi-wave time_to_hold,
  keep-0-ships — produced results indistinguishable from baseline
  noise. ~12 minutes of compute (3 × 220s eval) on what was always
  going to be noise. The correct sequence is: lock the baseline at
  the n that distinguishes the expected effect size, **then**
  iterate. This is essentially Rule 45 (n≥32 minimum) applied
  forwards rather than just at the gate.

- **Idle drain shipped a recipient-safety hole that was foreseeable
  pre-experiment.** The mechanism forwarded leftover ships from a
  deep source to the most-frontier own planet without checking
  whether the recipient was itself predicted to flip in the in-flight
  window. Result: 0/32 (catastrophic). Phase 3 had just spent its
  whole rationale on "don't strip a doomed source"; the symmetric
  "don't reinforce a doomed sink" was sitting right there. I
  implemented the drain in 5 minutes and the audit-of-my-own-design
  took another 5 minutes after seeing 0/32. Mirror Rule 38 (verify
  reproduces the failure state) — for a mechanism that mirrors an
  earlier fix's logic, sanity-check the mirror BEFORE the n=32 run.

- **Recognized Rule 37 axis-cap (3+ same-axis falsifications) on
  the 4th variant, not the 3rd.** Joint capture, multi-wave, idle
  drain were all "modeling fixes that ENABLE more launches" — same
  axis. keep-0-ships was the 4th. By count, Rule 37 was triggered
  after multi-wave; instead I added drain and keep-0 before stopping.
  Axis labelling is the issue — each variant looked different
  surface-syntactically but shared the underlying axis. Fix: when
  iterating fixes-in-a-row, explicitly name the axis the candidate
  sits on and count variants on it.

No PI-overrides this session. PI's only inputs were the question
answers at session start (which Phase to pursue) and the wrap
direction at session end. The mid-session iteration was on me.

## Frictions logged this session

None appended to `audit/friction.md` this session. The three items
above are first-occurrence decision-quality issues; per the friction
template guideline ("new tags get one cycle of grace before
promotion"), they belong here in the postmortem, not yet in
friction.md. If similar patterns recur next session, they should
graduate to friction.md.

## Promotion candidates (PI ratified: pending)

### [ ] `CLAUDE.md` — Rule 45 forward-application

**Tag:** `n32-baseline-not-locked-before-iteration` (Phase 3
direction unconfirmed at n=32; three Phase 4 variants iterated
against a noisy baseline)

**Where to insert:** sub-clause under Rule 45 (`n≥32 minimum for any
A/B lift claim`).

**What to add:**

> Sub-clause: before iterating variant N+1 on top of variant N,
> establish variant N's effect estimate at n where the Wilson CI
> narrows below the expected effect size. The same n=32 evidence
> that is "wide noise" at the gate is also wide noise as an
> iteration baseline. Re-running variant N at n=32 three times to
> get the same wide-CI answer is a session-time bug, not progress.

**Why:** ~12 minutes of compute this session burned on three Phase 4
variants that landed inside Phase 3's own Wilson band. Distinguishing
"variant N+1 helps" from "variant N+1 is noise" requires N+1 to
clear noise relative to N's locked estimate, not relative to N's
single-run point estimate.

### [ ] `CLAUDE.md` — Rule 38 mirror clause

**Tag:** `mirror-mechanism-skipped-pre-experiment-audit` (idle drain
mirrored Phase 3's source-reservation logic on the recipient side
without applying the same predicate)

**Where to insert:** sub-clause under Rule 38 (fix-verification
reproduces failure state).

**What to add:**

> Sub-clause: when a new mechanism is the dual / mirror of an existing
> one (source-side ↔ destination-side, launch ↔ receive, build-up ↔
> tear-down), it MUST inherit the existing mechanism's safety
> predicate before its first n=32 run. The original's rationale —
> "don't strip a doomed planet" — implies "don't reinforce a doomed
> planet" by symmetry; one safety check missing from the mirror
> often shows up as a catastrophic regression.

**Why:** Idle drain (this session) regressed 18.8% → 0% against v7_0
on a missing recipient-safety check that source reservation had
made explicit five minutes earlier in the same agent. Symmetry was
visible at design time, not after the eval.

### [ ] `CLAUDE.md` — Rule 37 axis-naming

**Tag:** `rule-37-axis-count-mislabelled` (4 Phase 4 variants
surface-syntactically distinct but axis-identical; Rule 37 trigger
recognised one variant late)

**Where to insert:** sub-clause under Rule 37 (3-variant axis cap).

**What to add:**

> Sub-clause: when proposing a variant, name the axis explicitly
> (e.g. "modeling fixes that enable more launches"; "ROI bonuses on
> bias direction"; "DEFEND_HORIZON tuning"). If the named axis has
> already absorbed 2+ falsifications, the new variant counts toward
> the same cap regardless of surface differences.

**Why:** Joint capture, multi-wave time_to_hold, and idle drain
shared the underlying axis "modeling fix that ENABLES more launches";
keep-0-ships-behind was the 4th. By count the cap fired on the 3rd
not the 4th. The mislabel cost one extra falsification cycle
(~3-4 min compute).

## PI additions

PI asked at session end: "Anything you'd add to the postmortem?" —
PI replied "Nothing to add" and ratified `Promote none` on the
three promotion candidates. The three sub-clauses above stay
recorded in this artifact for future reference but do not graduate
to `.claude/skills/kaggle-comp/improvements.md` this cycle.

## Framework version at session-end

- Commit SHA: `d564ae4c889141f399420afafaf2d567730152f9`
- Branch: `claude/heuristics-agent-physics-ZvZIm` (ahead 3 of
  origin/main: phase-2b → phase-3 → phase-3-with-falsifications).
- Active rules: 1..47 (CLAUDE.md).
- Loaded skills this session: `postmortem`. The session-start hook
  also surfaced `simplify` (not invoked).

## Session result (decision-quality framing)

- **Phase 3 (source reservation)** — shipped. Direction is positive
  vs Phase 2b (5/32 → 7-8/32 vs v7_0, +5-10pp). Not confirmed lift
  per Rule 45. Decision quality: good — Rule 40 modeling fix, fixed
  a concrete failure mode I'd reproduced in the seed-0 trace (t=69
  src-stripping into doomed garrison). I would re-run this with the
  same priors.
- **Phase 4 — joint capture (later-eta align), multi-wave
  time_to_hold, idle drain, keep-0-ships.** All falsified at n=32.
  Decision quality: mixed — diagnostic before each was thin, three
  ran on a noisy baseline (item 1 above). Joint capture and
  multi-wave were reasonable hypotheses; drain was a foreseeable
  miss.
- **Hand-back.** Agent's projected μ ≈ 800–900 vs rolling pair
  floor 1078. Rule 42 blocks submission. The deliverable for this
  branch is the committed code + the falsification record in the
  docstring, not a leaderboard push.
