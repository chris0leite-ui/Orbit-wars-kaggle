---
name: postmortem
description: Use at session-end to review what happened and what to learn. Triggers automatically when PI says "wrap up" / "prepare handover" (alongside WRAPUP.md), or directly when PI says "postmortem" / "do the postmortem". Reviews session frictions, surfaces promotion candidates from audit/friction.md to .claude/skills/kaggle-comp/improvements.md, asks PI for additions, and writes audit/YYYY-MM-DD-postmortem-<slug>.md before commit.
---

# Postmortem — what went wrong, what to learn

Run at session-end. Augments `WRAPUP.md` section A (step 4b). One
markdown artifact is produced and staged for commit alongside the
rest of the wrap-up.

Decision-quality-based, not outcome-based — see
`knowledge-base/concepts/decision-quality-vs-outcome-quality.md`.
A bad outcome from a good decision is fine; a good outcome from a
bad decision is still a bad decision.

## When to invoke

- **Automatically** as `WRAPUP.md` section A step 4b (after friction
  one-liners, before file-size guard).
- **Directly** when PI types "postmortem" / "do the postmortem".

## Steps (in order)

### 1. Read the session

- Today's appends to `audit/friction.md` (the `## YYYY-MM-DD` block).
- The most recent `audit/YYYY-MM-DD-*.md` files this session wrote.
- `git log --since="<session start>" --oneline` on the current branch.
- Your own working memory of decisions taken and overrides received.

### 2. Identify what went wrong

Concretely list:

- **Bad decisions** — choices that, on reflection, you would not
  retake **given the same priors that existed at decision-time**.
  (Hindsight refinements don't count here — those produce rules,
  not retroactive blame. See concepts file.)
- **PI-overrides** — moments when PI corrected the agent
  mid-session. Each is a calibration data-point.
- **Rule-bypass failures** — a rule existed, was applicable, was
  not applied. Why?
- **Rule-gap failures** — no rule existed, framework should have
  caught the issue. Candidate for promotion.

If nothing to flag: say so. Do not invent flags.

### 3. Surface promotion candidates

For each friction entry appended this session, ask:

- Does it name a **generalisable rule**, not a one-off?
- Cost: ≥ 1 LB slot, OR ≥ 1h compute waste, OR required PI override?
- Has the **same pattern** been observed earlier in this comp?

Any entry meeting **at least one** of those is a promotion candidate.

Draft a pending entry for
`.claude/skills/kaggle-comp/improvements.md` mirroring the existing
format (see that file's header):

```markdown
### [ ] <target file> — <one-line edit summary>

**Tag:** `<kebab-tag>` (<one-line context>)

**Where to insert:** <section / line ref>

**What to add:**
[content]

**Why:** <citation to friction entry / audit / cost evidence>
```

Do **not** commit candidates to `improvements.md` until PI ratifies
in step 4.

### 4. Ask PI for additions

After presenting (2) and (3), ask verbatim:

> Anything you'd add to the postmortem? Frictions I missed, rules
> you want extracted, decisions worth flagging?

Block until PI replies. Append PI additions to the appropriate
sections. Then ask:

> Promote these candidates to improvements.md? (yes / no / edit each)

Apply the resulting edits to `improvements.md` only after explicit
yes.

### 5. Write the artifact

Output path:
- Default: `audit/YYYY-MM-DD-postmortem.md`.
- On a non-`main` branch: `audit/YYYY-MM-DD-postmortem-<slug>.md`
  where `<slug>` is the part of the branch name after `claude/` —
  follows WRAPUP.md parallel-branch convention.

Required sections:

```markdown
# Postmortem — YYYY-MM-DD <branch-slug>

## What went wrong
- (bullets, one per item from step 2; or "nothing flagged this session")

## Frictions logged this session
- (cross-links to today's audit/friction.md entries)

## Promotion candidates (PI ratified: yes / no / edited)
- (drafted entries from step 3 + PI's decision per candidate)

## PI additions (from step 4)
- (verbatim or paraphrased)

## Framework version at session-end
- Commit SHA: <`git rev-parse HEAD`>
- Active rules: 1..N (cite CLAUDE.md `## Top-level rules`)
- Loaded skills this session: <list>
```

### 6. Stage and let WRAPUP commit

`git add audit/YYYY-MM-DD-postmortem*.md`. If PI ratified promotions,
also `git add .claude/skills/kaggle-comp/improvements.md`. WRAPUP.md
step 6 commits everything in one shot.

## What never to do

- Generate fake frictions to fill the page. Empty postmortems are
  fine and informative — they say "this session ran clean".
- Commit promotions to `improvements.md` without PI sign-off.
- Score sessions on outcomes (LB lift, gate PASS). Score on
  **decision quality** given pre-run information.
- Skip the "anything to add?" step. PI's hand on the wheel is
  load-bearing.
- Re-litigate decisions whose priors you can't reconstruct. Note
  the gap and move on.

## Adjacent

- `WRAPUP.md` — the wrap-up checklist this skill plugs into.
- `audit/friction.md` — input (one-liners written by WRAPUP step 4).
- `.claude/skills/kaggle-comp/improvements.md` — output (only after
  PI ratification).
- `knowledge-base/concepts/decision-quality-vs-outcome-quality.md` —
  the framing this skill operationalises.
- `knowledge-base/concepts/operational-environment.md` — why the
  artifact must commit before /exit.
- `knowledge-base/flags/2026-05-06.md` — standing duty: flag points
  warranting careful PI review (the postmortem is one place those
  surface).
