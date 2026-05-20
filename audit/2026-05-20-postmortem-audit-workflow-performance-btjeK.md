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
