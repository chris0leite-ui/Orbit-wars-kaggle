# Postmortem — 2026-05-27 session-EqJuT (H44 staleness fix)

## Session summary

Short status-check turn that surfaced a cross-branch stale-claim bug.
PI asked "where are we, what next, how would we score?" My summary
cited the H44 finding as "65% fleet-destroyed-in-flight" from
`state/MULTI_BRANCH.md` + `state/mechanism-ledger.md`. PI challenged
the physics ("how can fleets get destroyed in flight?"). Investigation
revealed the 65% claim is a **false positive PI themselves caught and
corrected on 2026-05-20** on btjeK (commit `9994b62`); the correction
filed an audit doc but the headline number persisted in 4 aggregation
docs on this branch under a "corrected" label.

Delivered: 6 line-edits across `state/MULTI_BRANCH.md`,
`state/mechanism-ledger.md`, `state/TOOLS.md`, `HANDOVER.md` —
replaced "65% fleet-destroyed-in-flight" / "physics-driven defensive
mechanism" with the corrected diagnosis (A+D=46% chooser ship-sizing
in lost episodes: D under-delivered 24% + A source over-drained 22%;
C third-party flip 17%).

## What went wrong

Decision-quality flags (assessed against priors-at-decision-time):

- **Stale-claim parroting from aggregation docs.** When the PI asked
  for status, I delegated a cross-branch state survey to an Explore
  subagent, then lifted "65% fleet-destroyed-in-flight" verbatim into
  my reply without reading the cited audit file
  (`audit/2026-05-21-h44-phase1-CORRECTED.md`). Had I checked the
  file, I'd have seen the audit name didn't exist on this branch
  (actual name is `CORRECTION.md`, only on btjeK) — a smoke signal
  for stale aggregation. Decision quality at delegate-time: BAD.
  Subagent output is leaf data, not source-of-truth; any claim with
  a file citation must be cross-checked against the file before
  relaying to PI. Rule-gap: Rule 44 covers the EDIT side (read
  state-of-truth before editing); no rule covers the SUMMARY side
  (verify audit-doc claims before relaying). Cost: 1 PI catch + 1
  cycle of 4-file state-doc edits + this postmortem.

- **Cross-branch correction propagation gap.** The H44 correction
  landed on btjeK on 2026-05-20 (commit `9994b62`) as a fresh audit
  file. The headline number "65% destroyed-in-flight" persisted in
  `state/MULTI_BRANCH.md`, `state/mechanism-ledger.md`,
  `state/TOOLS.md`, and `HANDOVER.md` on session-EqJuT (and likely
  other branches), wrapped in a "H44 corrected" label that
  superficially looked updated. Decision quality at correction-write-
  time (2026-05-20, not this session): BAD-in-hindsight, but the
  priors-at-decision-time defense is weak — the corrected audit file
  was sitting next to the aggregations it should have updated. Same
  pattern as `same-session-pi-note-amendment-on-axis-closure`
  (drafted-but-not-promoted candidate B from the 2026-05-23
  verification postmortem). PI made the same physics-intuition catch
  on the same finding twice in 7 days. Rule-gap.

- **Same-physics-catch-twice from PI.** PI's challenge today is
  verbatim equivalent to the 2026-05-20 challenge ("I have not seen
  fleets getting out of bounds or missing targets") that triggered
  the original correction. The framework let the same friction
  reach PI's lap twice. This is the diagnostic signal that
  candidate A below is real.

## Frictions logged this session

See `audit/friction.md` § "2026-05-27 (claude/session-EqJuT — H44
staleness fix + wrap)":

- `tag: stale-claim-propagated-across-branches-via-labeled-correction`
- `tag: summary-cites-audit-doc-without-reading-it`

## Promotion candidates (PI ratified: no — both deferred)

PI did not ratify either candidate. Both remain drafted-only below;
not added to `improvements.md`. Friction log + state-doc fix this
session deemed sufficient. Same outcome as the 2026-05-23
verification-pass postmortem's two candidates (gate-score-claim-
needs-saved-artifact + same-session-pi-note-amendment-on-axis-
closure). Revisit if pattern recurs.

### Candidate A — `correction-must-propagate-to-aggregation-docs`

**Where to insert:** `.claude/skills/kaggle-comp/improvements.md`
pending block; sub-clause of Rule 44.

**What to add:**

```markdown
### [ ] CLAUDE.md — corrections must propagate to aggregation docs

**Tag:** `correction-must-propagate-to-aggregation-docs`
(write-side complement to Rule 44)

**Where to insert:** Rule 48 candidate (after Rule 47).

**What to add:**
48. **Corrections must propagate to aggregation docs.** When you
    file a correction in `audit/` that overturns a previously-
    reported headline number or claim, you MUST in the same commit
    update every aggregation site that cites the original claim:
    `state/MULTI_BRANCH.md`, `state/mechanism-ledger.md`,
    `state/TOOLS.md`, `HANDOVER.md`, plus any open
    `knowledge-base/thoughts/` entries. A "[CORRECTED]" label
    without replacing the headline number is INSUFFICIENT and
    actively misleading — it gives downstream readers false
    confidence that the docs reflect current truth. Run
    `grep -rn "<old headline phrase>" state/ HANDOVER.md
    knowledge-base/` before the correction commit. Origin:
    2026-05-27 H44 staleness fix — the same PI physics-catch
    landed twice on the same finding because the 2026-05-20
    correction filed an audit doc but left the "65%
    fleet-destroyed-in-flight" number live in 4 aggregation
    sites under a "H44 corrected" label.

**Why:** PI made the identical "fleets don't fight in flight"
catch on 2026-05-20 (triggering the H44 correction) and again on
2026-05-27 (against my status summary that lifted the stale
headline). Cost: 1 cycle of cross-branch state-doc edits + 1
postmortem.
```

### Candidate B — `summary-must-verify-audit-doc-citations`

**Where to insert:** `.claude/skills/kaggle-comp/improvements.md`
pending block.

**What to add:**

```markdown
### [ ] CLAUDE.md — verify audit-doc citations before relaying claims

**Tag:** `summary-must-verify-audit-doc-citations`
(read-side complement to Rule 44)

**Where to insert:** Rule 44 sub-clause OR fresh Rule 49.

**What to add:**
Any claim with an `audit/...md` or `knowledge-base/...md`
citation that you relay to PI MUST be cross-checked against the
cited file's actual content, not lifted verbatim from a meta-
summary (aggregation doc, subagent reply, prior session
handover). Subagent outputs are leaf data; aggregation docs go
stale (see Rule 48 candidate above). The check costs one Read
tool call; skipping it costs PI's trust + 1 correction cycle.
Origin: 2026-05-27 H44 staleness fix — `state/MULTI_BRANCH.md`
cited `audit/2026-05-21-h44-phase1-CORRECTED.md` (a file that
doesn't exist on this branch; actual name is `CORRECTION.md` on
btjeK only). The non-existent-filename was itself a smoke
signal for stale aggregation that one Read call would have
surfaced.

**Why:** Same incident as candidate A; the read-side angle.
Cost evidence shared with A.
```

Note: candidates A and B together address what is effectively
one bug (correction did not propagate, then propagation gap was
not caught at relay time). PI may prefer to merge them, or
promote one and defer the other.

## PI additions

> "Nothing to add — proceed"

## Framework version at session-end

- Branch: `claude/session-EqJuT` (will be at HEAD after wrap commit)
- Commit SHA pre-wrap: `145ee8d`
- Active rules: CLAUDE.md Rules 1-47 (Rule 48 candidate filed in
  promotion candidates above)
- Skills invoked this session: postmortem (this artifact)
