# Postmortem — 2026-05-20 audit-workflow-performance-btjeK

Branch: `claude/audit-workflow-performance-btjeK` (ahead 109, behind 21)
Session shape: diagnose → fix-architecture → validate → reject → workflow change.
Submissions used: 0 (PI direction: diagnostic + local-only validation).
Outcome: NEGATIVE result on ledger; NEGATIVE result on panel-anchor;
NET POSITIVE workflow lesson (h2h-vs-production is the only valid
under-emission gate).

## What went wrong

- **Built the ledger on what-if signal alone before h2h.** The
  what-if harness (`scripts/whatif_postmortem.py`) showed +118 vs +15
  final planets in the ledger's favor across 6 episodes. I treated
  this as strong evidence and built `_PENDING_LAUNCHES` + lifecycle
  logic without first running a quick h2h vs current production.
  Real h2h showed the ledger LOSES 2/16 (12.5%, Wlo=0.035). Root
  cause of the false positive: what-if replays opp actions verbatim;
  after divergence the opp is effectively passive. **At decision
  time** I had the data showing this — divergence at turn 24, opp
  actions stale thereafter — but read it as "we look great after
  divergence" rather than "the metric is biased toward whatever
  agent diverges first."

- **Pursued the launch-rate symptom despite an existing friction
  tag.** `launch-rate-is-symptom-not-cause` was promoted to
  improvements.md in the 2026-05-17 fleet-efficiency negative-result
  session. This cycle ran straight into the same pattern: observe
  49% idle → build a fix that emits more → regress. **At decision
  time** I had not consulted improvements.md before designing the
  ledger; the tag was sitting there explicitly warning against this
  pattern.

- **Built the sary-class panel anchor before testing if existing
  anchors were sufficient.** Spent ~1 hour on
  `agents/sary_class/main.py`. Then 10 min testing existing anchors
  (roi, v7_0_drop_one) showed they all lose to `led_on` too. The
  cheap check should have come first. **At decision time** the
  question "do existing anchors already do the job?" was answerable
  in 10 min; I didn't ask.

- **Wrote audit files with date 2026-05-21 on a day that was
  2026-05-20.** `audit/2026-05-21-*.md` should be `2026-05-20-*.md`.
  Pre-existing friction tag `audit-date-must-track-system-currentdate`
  was promoted Day-1 and I still slipped. Not load-bearing for
  correctness but is a recurrence.

## PI-overrides (calibration)

- "Now test if that improves our current strategy" — clear directive
  to validate via h2h. I did test; the negative result was honest.
  No override mid-test; the failure was honest.
- "Inspect few games closely... small IP tests until we're sure we
  have a lift" — PI explicitly framed the workflow as
  inspect-then-iterate-then-A/B. Followed.
- "Ambitious larger improvement, not patching" — PI direction
  shaped the choice toward the ledger over A1 minimal patch.
  Ambitious framing was honored; ambitious doesn't mean correct.
  The ledger was the principled fix; just empirically wrong.
- 0 mid-session overrides. PI was not pulled in to correct course
  this cycle. Decision-quality was self-correcting via the
  validation gate.

## Frictions logged this session

See `audit/friction.md` 2026-05-20 section. Five entries:
- `whatif-static-opp-false-positive` (new)
- `launch-rate-is-symptom-not-cause` (3rd recurrence)
- `panel-anchor-strength-floor` (new)
- `test-existing-tools-first` (new — likely subsumes a generic rule)
- `audit-date-must-track-system-currentdate` (recurrence)

## Promotion candidates (PI ratification pending)

### [ ] [CROSS-CUTTING] CLAUDE.md Rule 42 — h2h-vs-production gate before architectural rebuild

**Tag:** `whatif-static-opp-false-positive` (2026-05-20)

**Where to insert:** `## Operating rules — concise` in `CLAUDE.md`, after Rule 41.

**What to add:**

```
42. **Architectural rebuilds clear h2h-vs-production at n≥8 BEFORE
    full implementation.** Before merging any agent-architecture
    change that touches the chooser shape, value head, opp model,
    or proposer redesign, run a quick h2h vs current production at
    n=8. Wilson-LB ≥ 0.40 is the minimum. What-if rollout against
    recorded opp actions is a useful chooser-behavior debugger
    but NOT a μ-lift predictor — opp's recorded actions become
    stale after divergence and the metric is biased toward
    whichever agent diverges first. Origin: 2026-05-20 ledger
    build — what-if showed +118 vs +15 final planets, h2h showed
    led_on 2/16 (12.5%) vs led_off. ~1.5 days wasted. Friction
    tag: whatif-static-opp-false-positive.
```

**Why:** The validation gate cost zero compute relative to the cost
of building the ledger (1+ days). At n=8 in 5-7 min wallclock the
ledger would have been killed before the full lifecycle code was
written.

### [ ] [CROSS-CUTTING] Session-start: grep improvements.md for the diagnosis tag

**Tag:** `launch-rate-is-symptom-not-cause` (3rd recurrence; auto-promotion).

**Where to insert:** `WRAPUP.md` section A new step 0, OR
`.claude/skills/kaggle-comp/SKILL.md` session-start checklist.

**What to add:**

```
At session start, after the git fetch (Rule 32), grep
.claude/skills/kaggle-comp/improvements.md for the
diagnosis-axis tag in the current `HANDOVER.md::Next-session
first-action`. Example: HANDOVER says "investigate under-emission
in idle turns" → `grep -i "launch-rate\|under-emit\|idle"
.claude/skills/kaggle-comp/improvements.md`. If a tagged warning
exists, READ it before designing any fix. Pinned to session-start
because the cost of skipping is a wasted day on a known-bad axis.
```

**Why:** The launch-rate-is-symptom-not-cause tag was both (a) sitting
in improvements.md explicitly warning against this and (b) part of a
3-recurrence pattern. The fix isn't a new rule; it's an enforcement
mechanism for an existing rule.

### [ ] [CROSS-CUTTING] CLAUDE.md Rule 43 — test existing infrastructure before building diagnostic agents/harnesses

**Tag:** `test-existing-tools-first` (2026-05-20)

**Where to insert:** `## Operating rules — concise` in `CLAUDE.md`, after Rule 42 (or sibling to Rule 22).

**What to add:**

```
43. **Test existing infrastructure before building a new diagnostic.**
    Before writing a new agent, harness, or panel anchor whose stated
    purpose is to expose a specific failure class, spend ≤ 30 min
    running existing infrastructure (panel, h2h, replay-mine, etc.)
    against a known regressed variant to confirm the existing
    infrastructure ISN'T already sufficient. Origin: 2026-05-20
    sary_class build — spent ~1h on a new anchor; testing existing
    anchors took 10 min and showed they were all too weak (and the
    new anchor was too). Friction tag: test-existing-tools-first.
```

**Why:** Could have spared the sary_class build entirely. 10 min of
roi/v7_0/v4_planner-vs-led_on h2h shows none of them catch this
regression. That answers the question without the build.

## Framework version at session-end

- Commit SHA: `87896c2` (sary-class panel anchor: FAILS — last commit
  on this branch before WRAPUP).
- Active CLAUDE.md rules: 1–40 (verified in CLAUDE.md `## Operating
  rules — concise`).
- Loaded skills this session: `postmortem` (now), `kaggle-comp`
  (loaded at session-start by other skills indirectly).
- Branch: `claude/audit-workflow-performance-btjeK` ahead 109 / behind 21.

## PI additions (step 4 of postmortem skill)

Stop-hook closed the loop before PI replied with additions. Three
candidates above are drafted but **NOT promoted** to
`.claude/skills/kaggle-comp/improvements.md` — next session can
review + ratify. The friction entries themselves are committed.

---

# Postmortem (second session, same date 2026-05-20 UTC)

(System date is 2026-05-20; in-session audit files use 2026-05-21
suffix — recurring `audit-date-must-track-system-currentdate` tag,
not renamed per prior PI guidance.)

## What went wrong

- **F-flag false positive in H44 Phase 1.** Added a fleet-destroyed-
  in-flight diagnostic based on "fleet not in fleets list at
  landing−1 OR landing." Didn't account for the env behavior that
  fleets vanish from the list at combat resolution regardless of
  outcome. Published commit `106afbe` claiming 65% F-dominance and an
  aim/`predict_fleet_fate` infrastructure bug. PI reversed it in one
  challenge ("I have not seen fleets getting out of bounds — give me
  an example") by demanding a hand-trace. Spot-check of 5 F-flagged
  launches showed all arrived within target radius. Diagnostic rewritten
  (removed F; added G near-tie-combat); corrected verdict shows A+D
  (chooser sizing) = 46% of failures in lost episodes.
- **PI override**: the one challenge above. Calibration data-point —
  when an audit verdict implies "the agent has a systemic
  infrastructure bug" with no prior evidence, PI's prior weights
  against it correctly; agent should too.
- **Rule-bypass**: `test-existing-tools-first` (5/21 ledger session)
  and `whatif-static-opp-false-positive` both already say
  "validate diagnostic mechanism before drawing conclusions." Both
  were applicable; neither was applied. The friction file already had
  the warning; not consulted before designing the new diagnostic.
- **Rule-gap**: no standing "must hand-trace 3 examples before
  publishing a dominant-mode claim" rule. Candidate promotion drafted
  in friction.md::`fleet-absence-mistaken-for-destruction`; PI declined
  promotion this session.

## What went right

- v2 audit (`scripts/large_to_small_audit_v2.py`) strict-gate held:
  LEAK REJECTED verdict produced before any Phase B fix work.
- Rule 22 public-notebook scan (H45) launched as a background Explore
  agent in parallel with H44, returning cheap candidates (sigmaborov's
  comet-profit gate; rahul's neutral-denial term) without blocking
  main-thread work.
- H44 Phase 1 correction landed in-session, not deferred. Bad verdict
  is now annotated in the audit doc itself.
- Cross-agent push-coordination friction was logged immediately when
  it surfaced (`cross-agent-push-coordination-gap`).

## Frictions logged this session

Three new entries in `audit/friction.md` under the
`## 2026-05-21 (claude/audit-workflow-performance-btjeK — v2 audit +
cross-agent push-coordination friction)` and (later in same file)
`fleet-absence-mistaken-for-destruction`:

- `per-launch-denominator-leaks-frequency-as-quality` — v1 audit's
  per-launch denominator inflated launch-frequency into a phantom
  leak; v2 per-ship normalisation rejected the leak.
- `cross-agent-push-coordination-gap` — submission 52845073 pushed by
  an unrelated agent unaware of the rolling-pair state; evicted the
  1135.1 floor for a probe that settled at 1066.7.
- `fleet-absence-mistaken-for-destruction` — fleet-list disappearance
  taken as evidence of mid-flight destruction; actually it's combat
  resolution. F-flag false positive reversed by PI in one example.

## Promotion candidates (PI ratified: no)

Three candidates drafted and presented to PI in step 4:

1. Rule 41 — per-decision denominators leak frequency-as-quality.
2. Push-authorisation must read `rolling_last_2`.
3. 3-hand-trace gate before publishing a dominant-mode claim.

PI declined ratification this session ("no"). Candidates remain in
`audit/friction.md` for future-session reconsideration.

## PI additions (from step 4)

None ("no" to additions; "no" to promotion).

## Substantive results this session

- A.8 leaf → null. v1 large→small leak was selection-bias + end-state
  bias artefact. Confound-controlled v2 audit confirms.
- H44 Phase 1: corrected verdict says chooser ship-sizing (A + D) is
  46% of failed-landing causes in lost episodes. C (race-condition)
  is 17% in both won and lost. Combat math + other instrumentation
  gaps account for ~30% residual.
- H45: no new top-5 public notebooks since 5/14 scan. Two cheap
  candidates surfaced for future probes (sigmaborov comet-profit gate;
  rahul neutral-denial term).
- H46 (4P weakest-opp targeting) held until next session.
- Rolling pair as of session-end: 52845073 (μ=1066.7, accidental
  old-line probe) + 52827111 (μ=1136.6, current line). Floor 1066.7.

## Framework version at session-end

- Branch: `claude/audit-workflow-performance-btjeK` (ahead 115 of main
  pre-this-commit; will be ahead 116 after wrap-up commit).
- Commit SHA pre-wrap-up: `9994b62`.
- Active rules: CLAUDE.md Rules 1–40. Rule 40 (modeling-correctness
  over restriction-tuning) was applied correctly in the Phase B design
  that ultimately never executed. Rule 38 (fix-verification reproduces
  failure state) was the basis for the PI's hand-trace challenge.
- Loaded skills this session: postmortem (now), kaggle-comp.
- Plans referenced:
  `/root/.claude/plans/let-s-figure-out-how-purrfect-mist.md`.
