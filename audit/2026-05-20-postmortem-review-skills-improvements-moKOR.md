# Postmortem — 2026-05-20 review-skills-improvements-moKOR

Session: cross-branch doc consolidation pass. No code changed; merge-up
of branch-only artifacts is the follow-up. Wrap-up postmortem per
WRAPUP.md section A step 4b + `.claude/skills/postmortem/SKILL.md`.

## What went wrong

Three decisions I'd retake given the same priors:

1. **Underspecified initial inventory.** First drafts of the cross-branch
   tools registry (`state/TOOLS.md`) were category-grouped prose. PI
   nudged three times to surface what should have been itemized from
   the start:
   - "have you mentioned the ML competition branch? also latest updates
     on other branches?" → exposed missed `precision-physics-engine-ymJkA`
     branch + missing PFhzM third-track status.
   - "have you listed the A B testing tools and the single game diagnose
     tools?" → forced 12-row A/B table + 21-row diagnostic table.
   - "have you listed the validation and testing tools?" → forced the
     40-row test catalog + 6-step consolidation-merge gate.

   Same priors, retake: when PI asks for an inventory / registry /
   catalog / tools list, default to itemized enumeration (one row per
   item), not category-level prose. Categories aren't searchable;
   itemized tables are.

2. **Missed `precision-physics-engine-ymJkA` branch entirely** in the
   initial survey. Filtered the branch list to "recent" (commits in last
   ~5 days). Precision branch is 9 days old but holds a LIVE-PUBLISHED
   submission (#52552139, μ₀=600) and the only guaranteed-landing
   inverse-intercept solver in the repo (`agents/precision/intercept.find_shot()`).
   Surfaced only when OyoYR-rebased's tier-split HANDOVER commit
   (84 min ago) cited it. Same priors, retake: substrate-asset discovery
   should query ALL `claude/*` branches; "recent" filter is appropriate
   for ACTIVE work, not for asset discovery.

3. **Authored Rule 41 without cross-branch grep for existing proposals.**
   btjeK's knowledge-base already had a "Rule 41 candidate"
   (confound-sweep before correlational conclusion). Collision caught
   only when PI asked for latest updates on other branches; renumbered
   to 42-47 + adopted btjeK's proposal as the new Rule 41. Cost: one
   re-edit pass on the plan + improvements-archive draft. Same priors,
   retake: before proposing new rule numbers, grep all dev branches'
   knowledge-base + improvements.md for pre-existing proposals.

**PI overrides this session:** three explicit "have you covered X?"
nudges (logged as item 1). Each was a calibration data-point that
initial summaries underspecified the artifact.

**Rule-bypass failure:** new Rule 44 (state-of-truth read before
subsystem edits) — I authored CLAUDE.md edits while NOT reading other
branches' knowledge-base carefully enough to see the btjeK Rule 41
proposal. The rule I was writing this session would have prevented its
own authoring collision if it had been applied to the meta-task of
writing it.

## Frictions logged this session

Appended to `audit/friction.md` under `## 2026-05-20`:

- `tag: inventory-as-categorical-summary-not-itemized`
- `tag: substrate-asset-discovery-filtered-to-recent-only`
- `tag: rule-number-collision-without-cross-branch-grep`

## Promotion candidates (PI ratified: **NO — do not promote**)

PI explicit verdict 2026-05-20 EOS: "do not promote". The candidates
remain in this postmortem as drafted-not-ratified for future reference.
`improvements.md` is NOT edited for these.

### [DRAFTED, NOT PROMOTED] Inventory/registry artifacts default to itemized enumeration

When PI asks for an inventory / registry / catalog / tools list, the
artifact MUST be itemized (one row per item, scannable table), not
category-level prose. Three instances this session; total doc grew
~3× from first draft to final.

### [DRAFTED, NOT PROMOTED] Substrate-asset discovery scans ALL `claude/*` branches

When discovering reusable substrate (lib modules, primitives, test
fixtures), query all branches; don't filter by recency. Old branches
may hold live-published primitives. `precision-physics-engine-ymJkA`
is 9 days old but holds the repo's only guaranteed-landing solver.

### [DRAFTED, NOT PROMOTED] Cross-branch rule-number grep before authoring new rules

Before authoring a new CLAUDE.md rule number, grep all dev branches'
knowledge-base + improvements.md for pre-existing proposals. Possibly
fold as a sub-clause into Rule 44.

## PI additions (from "anything you'd add?" step)

PI declined to add frictions or rules; verdict was the singular
directive "do not promote".

## Framework version at session-end

- Commit SHA: `de79b02` (consolidate: cross-branch state of truth + tools registry + Rules 41-47).
- Active rules: CLAUDE.md 1-47 + R-defaults (R1-R8). Rules 41-47 added this session.
- Loaded skills this session: `kaggle-comp`, `postmortem` (this artifact).
- Branch: `claude/review-skills-improvements-moKOR`. Ahead 1 / behind 0 vs `origin/main`.

## Session output summary

- 3 new docs: `state/MULTI_BRANCH.md`, `state/TOOLS.md`,
  `.claude/skills/kaggle-comp/improvements-archive-2026-05-20.md`.
- 7 docs edited: `CLAUDE.md`, `HANDOVER.md`, `state/current.md` (deprecated),
  `state/mechanism-ledger.md`, `.claude/skills/kaggle-comp/SKILL.md`,
  `day-loop.md`, `improvements.md`.
- Net: +882 / -394 lines, 10 files.
- Code changes: **zero** (doc-only consolidation pass per PI directive).
- Next-session pickup: HANDOVER.md priority-1 (substrate primitive
  merge-up: `lib/trajectory_layer.py` + `agents/precision/`).
