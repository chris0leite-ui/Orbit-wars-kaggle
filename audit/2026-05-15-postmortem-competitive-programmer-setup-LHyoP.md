# Postmortem — 2026-05-15 competitive-programmer-setup-LHyoP

## What went wrong

- **Built MSP breadth-first instead of probe-first.** Shipped ~250 LOC
  (4 templates + analytical scorer + orchestrator) before testing
  whether ANY template's first-turn action would beat the incumbent
  at analytical Phase-1 score. Trace at completion: 0/40 turns. A
  30-min probe of one template would have surfaced the "first-turn
  collapses to incumbent" failure mode. Same pattern as
  mission-persistence v1 (−42pp). Logged as
  `build-large-architecture-without-probe-of-value-gradient`.
- **Started architecting around PI hypothesis before auditing it.**
  PI claim: "iter starts too late, garrison too high." All three
  sub-claims refuted by 9-game audit (median first launch step 3.3,
  12.9 ships vs opponents 11.5, 1.7 ships at home in steps 0–5).
  Caught mid-plan via AskUserQuestion; reframed cleanly. Logged as
  `pi-hypothesis-unaudited-pre-architecture`.
- **Magnitude-uncalibrated cluster head A/B.** Wired `cluster_value`
  at weight=1.0 without comparing to composite's increment. Cluster
  dominated composite 3:1 in the layered head. First A/B confounded
  by scale; fix attempt (frontier_discount=1.0) regressed identically
  → design was wrong, but the scale confound should have been ruled
  out first. Logged as `value-head-magnitude-uncalibrated`.
- **Scoped 32-seed A/B without timing one game.** Plan said "~12 min";
  single game takes 139s; first attempt timed out with zero data.
  Reduced to 8 seeds + retry. Logged as `a-b-compute-budget-not-pretimed`.

PI overrides: 1 (live-game audit refuting your opening-saturation
hypothesis; accepted cleanly via AskUserQuestion — positive
calibration data-point, Rule 26 working as intended).

Rule-bypass failures: none. Rule 37 (consecutive-falsification cap)
fired correctly on both axes (additive-candidate 3/3; hand-designed-
leaf 2/3 since composite WIN). Rule 26 caught the data-vs-hypothesis
contradiction.

Rule-gap failures: three; one promoted (see below).

## Frictions logged this session

`audit/friction.md` under `## 2026-05-15`:

- `pi-hypothesis-unaudited-pre-architecture` (promotion candidate;
  PI ratified — see below)
- `build-large-architecture-without-probe-of-value-gradient`
  (promotion candidate; PI held)
- `value-head-magnitude-uncalibrated` (promotion candidate; PI held)
- `a-b-compute-budget-not-pretimed` (single event, weak candidate)
- `chooser-leaf-noise-resists-additive-candidates` (Rule 37 cap
  fired on the axis; structurally absorbed, not promoted as a new
  rule)

## Promotion candidates (PI ratified)

- **A — `pi-hypothesis-unaudited-pre-architecture`: PROMOTED** to
  `.claude/skills/kaggle-comp/improvements.md` pending list as a
  CLAUDE.md Rule 26 sub-clause: when PI proposes a hypothesis about
  AGENT BEHAVIOR (distinct from strategy), agent must audit ≥5 recent
  live games before architecting around it.
- **B — `probe-before-build-architectural-additions`: HELD.** PI
  wants more evidence before promoting.
- **C — `value-head-magnitude-comparison-gate`: HELD.** PI wants more
  evidence before promoting.

## PI additions

(from "Anything you'd add to the postmortem?" — PI answered "Nothing
to add". No additions logged.)

## Framework version at session-end

- Commit SHA (pre-wrap-commit): `869c13a`
- Active rules: 1..39 from CLAUDE.md `## Operating rules — concise`,
  plus R1/R2/R5/R7/R8 tabular-only defaults (inert on Orbit Wars).
  Rule 37 (consecutive-falsification cap) and Rule 38 (fix-
  verification reproduces failure state) both fired this session as
  intended.
- Loaded skills this session: postmortem (this skill, end-of-session).

## Outcome quality vs decision quality

Three falsifications this session (MSP dormant, geo −12.5pp, cluster
−25pp) — all OUTCOMES were negative, but the DECISIONS to attempt
each were defensible given priors:

- MSP attempt: ROI-on-trajectories is the PI's repeatedly-stated
  concept; an additive multi-turn plan scorer that the chooser gates
  was the architecturally-safe interpretation.
- Geo allocator: directly addresses joint-action ROI; geo agent
  has the validated allocator code already.
- Cluster leaf head: identified by root-cause analysis as the
  deepest hand-designable attack on the chooser's leaf-scorer noise.

What we learned: the K=10 leaf scorer is the actual bottleneck. Both
the "more candidates" axis and the "smarter hand-designed leaf" axis
are now decisively exhausted (Rule 37 cap fired on each). Next
session's structurally correct swing is imitation learning leaf
head — the only direction that escapes "human-designed leaf scorer"
entirely.

Decision-quality score: defensible. Outcome-quality score:
negative-but-informative.
