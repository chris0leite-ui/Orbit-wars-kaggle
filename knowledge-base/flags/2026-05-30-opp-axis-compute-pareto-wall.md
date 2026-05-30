# FLAG: opp-axis lift requires compute project, not session-budget A/B

Date filed: 2026-05-30.
Branch: claude/kaggle-submission-review-gZsCu.
Standing duty: surface in next-session HANDOVER if "opp model is the
lever" framing returns without a compute plan attached.

## What this flag is

Today's three asymmetric A/Bs (nearest, top_tier_mirror, v7_0 as opp
model — paired-seed n=4-5) plus the no-launch control establish a
hard Pareto wall: any opp-model upgrade whose per-call cost exceeds
~2× lite_greedy's is search-starved by the chooser's
`affordable_validate_cap` before its strategy quality can show. At
the live 1000ms turn cap, lite_greedy is Pareto-optimal among
feasible variants.

This contradicts the PM3 strategic-direction framing of "opp model
is the most under-explored lever." That framing was correct as a
*direction* but underestimated the budget constraint.

## When to re-surface

If a future session proposes another opp-model A/B *without* one
of these:

- A documented per-call cost <2× lite_greedy's 0.01 ms (i.e.
  sub-0.02 ms per call). Spatial-restricted lite_greedy qualifies.
  Defenders-weighted qualifies. MLP-validated may or may not —
  needs bench.
- A compute project plan (e.g. "port v3.5.1 snipe logic to a sub-0.5
  ms representation"; "distill v7_0's chooser into a small net").
- A wallclock-budget pad (10× via BASELINE_WALLCLOCK_MS for testing
  only, with the understanding the result cannot ship at the 1000ms
  ladder cap).

…then surface this flag and the audit at
`audit/2026-05-29-pm-nearest-opp-model-n5-directional-null.md`
(sections 3a–3e are the data) and re-discuss before burning the
A/B compute.

## How to clear this flag

If any of these become true, this flag can be archived:

- A heavy opp model is shown to lift the live ladder μ by ≥10
  points after addressing the search-starvation confound (via
  budget pad in testing, then a faster port for shipping).
- A spatial-restricted lite_greedy A/B at n=32 shows clean lift
  (Wilson-lo ≥ 0.55) — proves the "expects opponents from everywhere"
  diagnosis is the real lever, and the heavy-opp-model framing was
  unnecessary in the first place.
- A different lever entirely (proposer breadth, value head, joint
  candidate enumeration) ships a μ lift that supersedes this
  question.

## Adjacent

- `knowledge-base/thoughts/2026-05-30-opp-axis-compute-pareto-wall.md`
  — full analysis.
- `knowledge-base/questions/2026-05-30-seed-3493-no-belief-beats-lite-greedy.md`
  — the one specific board where the conjecture might be testable.
- `audit/2026-05-29-pm-nearest-opp-model-n5-directional-null.md` —
  the data underlying this flag.
