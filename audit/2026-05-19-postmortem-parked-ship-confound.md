# Postmortem — 2026-05-19 audit-workflow-performance-btjeK (parked-ship research)

Decision-quality-based per `.claude/skills/postmortem/SKILL.md`.

## What went wrong

- **Bad decision (mine):** ran a parked-ship win/loss analysis using a
  distance-to-non-our threshold (`min_dist_to_nonour ≥ 35`) as the
  "rear" definition, reported a 27.9 pp win-vs-loss gap as evidence
  that "parking is not a leak," and started writing an audit doc
  with that conclusion — without first sweeping the metric for
  mechanical confounds. The PI immediately caught the obvious one:
  the rear definition is RELATIVE to non-our planets. When you're
  winning, you own most planets, so "non-our" is rare and far →
  every ship looks "rear" by construction. The correlation I reported
  was tautological with territory share, not informative about
  chooser behavior. Same alarm should have fired earlier on the
  original 43.8 % number (drawn from a pool with 87.5 % winrate).

- **PI override:** PI corrected the analysis after the first results
  message. I should have caught this myself.

- **Rule-bypass failure:** Rule 26 says "every BOTE asks PI: (i) Q6
  metric alignment ... once-per-session devil's-advocate ritual."
  The parked-ship analysis IS a BOTE (back-of-the-envelope) that
  the entire pivot decision was about to hinge on. I did not run a
  devil's-advocate sweep on the metric before reporting the
  conclusion. Rule existed; applicable; not applied.

- **Rule-gap failure:** no explicit rule says "enumerate mechanical
  confounds before drawing causal conclusions from a correlation
  measurement." Rule 24 covers fold-safe label-conditional aggregates
  in tabular comps but doesn't generalise to code-comp metric design.
  Candidate for promotion (see below).

## Frictions logged this session

- `audit/friction.md` § "2026-05-19 ... parked-ship analysis confound"
  - `tag: territory-share-confound-on-distance-metric`

## Promotion candidates (pending PI ratification)

### [ ] CLAUDE.md — new rule: confound-sweep before correlational conclusion

**Tag:** `territory-share-confound-on-distance-metric` (code-comp metric
design; PI override 2026-05-19).

**Where to insert:** `## Operating rules — concise`, after Rule 26 (PI
interaction protocol). New rule 41.

**What to add:**

```
41. **Confound-sweep before correlational conclusion.** Before
    reporting a correlation between a metric and an outcome (win/loss,
    high-LB/low-LB, treatment/control) as evidence FOR or AGAINST a
    hypothesis, enumerate ≥2 mechanical confounds the metric is
    sensitive to. If any confound is plausible AND not controlled for,
    label the result "correlational, not causal" and propose either
    a controlled subset (e.g. restrict the window) or a different
    metric. Distance-to-class-X, ratio-of-Y, and time-to-event metrics
    are the highest-risk family — they shift with the very quantity
    you're testing against. Origin: 2026-05-19 parked-ship analysis;
    "rear = min_dist_to_nonour ≥ 35" grows automatically with territory
    share, so the win/loss split was tautological.
```

**Why:** PI override 2026-05-19; same root cause as the 2026-05-17
43.8 % framing that ran for ~36 h without challenge. Two recurrences
within 48 h on the same axis is enough.

## PI additions (from step 4)

(pending — to be filled when PI responds to the wrap-up prompt)

## Framework version at session-end

- Commit SHA: `eeb8733` (this session's net commits: 0 before wrap-up;
  postmortem + friction + state edits stage into a single wrap commit).
- Active rules: 1..40 per `CLAUDE.md ## Operating rules — concise`
  (rule 41 above is a promotion candidate, not yet active).
- Loaded skills this session: `postmortem`.
