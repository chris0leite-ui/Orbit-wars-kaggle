# Postmortem — 2026-05-18 reverse-engineer-seat-geometry-BPJKs

## What went wrong

Two distinct frictions, same shape: **code-change proposed before
reading the state-of-the-world docs that index the subsystem in tree.**

1. **`wrong-file-recon-skipped-state-md`** — proposed launch-threshold
   tweaks to `data/main.py` (60-line Kaggle starter example, untouched
   since competition kickoff on 2026-05-01) as if it were our submitted
   agent. Two rounds of analysis on the wrong file. Reading
   `state/current.md` would have surfaced `agents/baseline/` as the
   actual submission in one minute. PI caught: "is that really our
   submission? check again." No compute wasted (caught pre-commit) but
   the recommendation chain was off-target.

2. **`crn-symmetry-broken-without-reading-prior-audits`** — designed
   asymmetric chooser change: `top_tier_mirror_policy` (aggressive Tier-1)
   in `build_idle_baseline`, kept `lite_greedy_policy` (passive) in
   `score_action`. Panel: 0 wins / 32 games, Wilson 0.00-0.11, decisive
   FAIL. Burned ~30 min compute + one full 128-seed panel slot. Reverted
   (commit `f28c9fc`). Root cause: the chooser's Δ requires
   common-random-numbers symmetry — both legs of `leaf(action) -
   baseline` must use the SAME opp trajectory.
   `audit/2026-05-17-state-function-principled-fix-results.md` documents
   the v11 → v12 → v13 progression that fixed exactly this asymmetric-Δ
   failure mode. Reading it before designing the change would have
   flagged the CRN-violation.

**PI overrides this session: 2** (both course-saving)
- "is that really our submission? check again" → prevented committing
  to a wrong-file fix.
- "why should we tune that baseline?" → forced a devil's-advocate read
  which surfaced Rule 40 framing and pushed me to deeper recon
  (eventually identifying the right code: `agents/baseline/chooser.py`
  + `lib/opp_model.py`).

**Rule-bypass:** none. Rule 32 was applied (session-start git fetch,
log, diff). Rule 38 was applied (fix-verification reproduces failure
state — the panel A/B was a real reproduction of the predicted
behaviour change).

**Rule-gap:** there's no current rule that says "before proposing
edits to a subsystem, read the state docs and recent audits that
index it." Rule 32 covers the session-start sweep; Rule 38 covers
fix-verification; nothing covers **edit-proposal time**.

## Frictions logged this session

Both appended to `audit/friction.md` § 2026-05-18:

- `wrong-file-recon-skipped-state-md` — recon on data/main.py instead
  of agents/baseline/ — PI caught.
- `crn-symmetry-broken-without-reading-prior-audits` — asymmetric
  Tier-1 chooser change, 0/32 panel, reverted.

## Promotion candidates (PI ratified)

- **[ ] Read state docs + recent audits before proposing subsystem
  edits** — ratified for promotion to
  `.claude/skills/kaggle-comp/improvements.md`. Drafted under
  "Pending — promotion needed" as the new top entry above the
  stop-hook item. Both frictions cited as cost evidence.

## PI additions

PI directive: "promote, the[n] think, what to push to main."
The promotion is applied (above). The "what to push to main" portion
is handled in the wrap-up note that follows this artifact — see
`HANDOVER.md` (next-session brief) or the wrap-up reply.

No additional frictions or rule extractions surfaced by PI this
session.

## What WAS load-bearing this session (positive)

The audit chain that ran in the first half of the session is real,
high-confidence diagnostic material that survives the chooser-change
failure:

- `audit/2026-05-18-team-archetype-gap.md` — per-archetype winrate
  delta top-10 vs our submission.
- `audit/2026-05-18-archetype-action-audit.md` — top-gap-cell
  behavioural fingerprint diff.
- `audit/2026-05-18-archetype-action-audit-allcells.md` — cross-cell
  summary; the aggression-deficit pattern is universal.
- `audit/2026-05-18-archetype-action-audit-gap-vs-even.md` —
  disambiguation: two universal features
  (`launches_per_turn` d=+1.26, `mean_garrison_at_launch` d=-0.82),
  two conditional features (hoarding-OK-in-even-cells), one feature
  that flips (target distance archetype-dependent).

The diagnosis from those audits stands. The next chooser-side fix
needs symmetric stronger opp (vectorise `top_tier_mirror_policy` or
train the Tier-2 logreg placeholder at `lib/opp_model.py:128-140`)
to act on it.

## Framework version at session-end

- Commit SHA: `89bbdea069e2f66f860812edf8d22fa8afe46a85`
  (about to add the postmortem + thoughts entry as a follow-up
  commit; final SHA will land in the next push).
- Active CLAUDE.md rules: 0-40 (no new rule added this session;
  Rule 41 promotion drafted in `improvements.md`, awaiting next
  cycle).
- Loaded skills this session: `postmortem` (this skill).
