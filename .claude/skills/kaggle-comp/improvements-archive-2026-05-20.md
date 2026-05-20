# improvements-archive — rotated 2026-05-20

> Rotated out of `improvements.md` by
> `claude/review-skills-improvements-moKOR` on the cross-branch
> consolidation pass. Items moved here are PROMOTED-to-CLAUDE.md
> (rules 41-47) or SUPERSEDED by the new state architecture
> (`state/MULTI_BRANCH.md` / `state/TOOLS.md`).

## Promoted to CLAUDE.md rules (2026-05-20)

### [PROMOTED → Rule 41] [CROSS-CUTTING] Confound-sweep before correlational conclusion

`tag: large-to-small-leak-rejected-after-confound-sweep` +
`tag: launch-rate-is-symptom-not-cause` +
`tag: per-launch-denominators-unsafe`
(all `claude/audit-workflow-performance-btjeK`, 2026-05-19/20).

When comparing decision-classes that differ in cohort composition
(per-attempt vs per-resource; large-launch vs small-launch; early
vs late game), an aggregate-rate gap is NOT a quality signal until
the confound is controlled. The btjeK v2 confound-controlled re-audit
REJECTED the large→small leak that the v1 aggregate had flagged.

**Applied:** CLAUDE.md Rule 41.

### [PROMOTED → Rule 42] [CODE-COMP-DISCOVERED] Pre-submit cross-branch coordination gate

`tag: cross-agent-push-coordination-gap` (btjeK 2026-05-21).

Five sequential pushes from `claude/strategy-framework-design-OyoYR-rebased`
evicted strong agents (μ ≈ 1149 → 1143 → 1135 → 1136 → 1122) from
sibling branches in 24 h, leaving the rolling-last-2 at μ 806 / 829.
~320 μ floor lost, unrecoverable until the rolling window cycles.

**Applied:** CLAUDE.md Rule 42 + `state/MULTI_BRANCH.md` push claim board.

### [PROMOTED → Rule 43] [CODE-COMP-DISCOVERED] Multi-opponent panel mandatory pre-submit

`tag: local-vs-v7_0-only-misses-ladder-distribution` (original 2026-05-12) +
`tag: panel-pass-without-h2h-vs-current` (4 recurrences) +
`tag: local-AB-not-calibrated-to-live-ladder` (5/19 strategy-framework: 0/16 local → live μ=711.5).

Single-opponent local A/B is BANNED as sole evidence. Gate requires
`--vs-panel` clearing Wilson-lo ≥ 0.55 per opponent AND h2h vs the
current rolling champion at n ≥ 32, Wilson-lo ≥ 0.50.

**Applied:** CLAUDE.md Rule 43.
Supersedes the prior pending item "Make `--vs-panel` mandatory before submission."

### [PROMOTED → Rule 44] [CROSS-CUTTING] State-of-truth read before subsystem edits

`tag: wrong-file-recon-skipped-state-md` +
`tag: crn-symmetry-broken-without-reading-prior-audits`
(both 2026-05-18, claude/reverse-engineer-seat-geometry-BPJKs).

Same-session double recurrence: edits to "our agent" without first
reading the state docs that index the agent or the audit notes
that document the subsystem's design history. Asymmetric Tier-1
chooser change violated CRN symmetry the v11→v13 line had fixed;
panel returned 0/32, reverted (PR #31, commit `f28c9fc`).

**Applied:** CLAUDE.md Rule 44.
Supersedes the prior pending item "Read state docs + recent audits
before proposing subsystem edits."

### [PROMOTED → Rule 45] [CROSS-CUTTING] n ≥ 32 minimum for A/B lift claims

`tag: n16-falsely-shows-parity` +
`tag: small-n-ab-noise-misled-panel`.

Two false-positive lifts shipped on n = 8/16 evidence. n = 8 Wilson
CI is too wide to distinguish parity from a 20-pp regression.

**Applied:** CLAUDE.md Rule 45.

### [PROMOTED → Rule 46] [CODE-COMP-DISCOVERED] Bundle + parity smoke before submission

`tag: composite-a2-hybrid-bundle-import-error` +
`tag: bundle-multi-line-imports-broken` +
`tag: bundle-aliased-imports-broken` +
`tag: cross-agent-imports-not-bundled` +
`tag: tests-pass-bundle-broken` +
`tag: source-bundle-behavior-diverges`.

`composite_a2_hybrid` (sub #52744234) ERROR'd on an absolute import
the local tests didn't catch. 5 separate silent-fail modes
documented across phase-5 strategy-framework work.

**Applied:** CLAUDE.md Rule 46.
NOTE: EpMVP bundler upgrade ("inline agent submodules + explicit-name
imports") addresses 3 of the 5 modes and is a merge-up candidate.

### [PROMOTED → Rule 47] [CODE-COMP-DISCOVERED] Physics-primitive verification before agent design

`tag: physics-gate-and-mvp-confirmation` (PFhzM 2026-05-19).

Entire trajectory_roi line missed `lib.trajectory.predict_fleet_fate`
(6.8% physics waste). 4 A/Bs burned before the substrate problem
was discovered. Synthetic-scenario tests gave false confidence; all
17 unit tests passed but runtime was physically invalid.

**Applied:** CLAUDE.md Rule 47.

## Superseded by new state architecture (2026-05-20)

### [SUPERSEDED] CLAUDE.md / Rule 12 addendum: pre-submit eviction record

`tag: rolling-last-2-tradeoff-needs-explicit-decision-record`.
Original 2026-05-11.

The pre-submit eviction record is now codified as Rule 42's push
claim board in `state/MULTI_BRANCH.md`. No separate sub-clause needed.

**Applied:** subsumed by Rule 42.

### [SUPERSEDED] do-and-dont.md — ISO date convention; never invent Day-N

`tag: day-counter-drift`. Original s6e5 2026-05-08 PM.

The cross-branch state doc uses ISO dates throughout; the per-branch
state files inherit. Day-N counters that don't anchor to comp start
are now caught at consolidation review.

**Applied:** practice in `state/MULTI_BRANCH.md`; not enforced
elsewhere. Re-promote if drift reappears.
