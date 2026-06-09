# 2026-05-29 — perf-chain confound + H41 falsification

## The thing I want to remember

When you A/B agent_v_new vs agent_v_baseline, **the build-commit
delta is part of the experimental design.** Not background. Not
inert. Every commit in `git log baseline..new --oneline` is a
mini-experimental-arm whose effect compounds with whatever you're
trying to measure. If the harness doesn't print that delta, you
are running a confounded experiment.

This session I built five "perf" commits, then ran three A/Bs
against a baseline bundle from before the perf chain. The result
was a flat 37.5% across every config: Stage-3 breadth = 37.5%,
no Stage 3 = 37.5%, H41 floor = 37.5%, no floor = 37.5%. I read
that as "the experimental knobs don't help." Wrong read. The
*perf chain* is what was producing the 37.5% floor — every knob
got compared against itself-without-the-knob, but all knobs were
on the same broken substrate.

The flat 37.5% line is the signature of a bad substrate. If
multiple distinct experimental treatments converge on the same
point estimate well below 0.5, the substrate is the binder.

## The structural lesson

Perf commits CAN change behavior. Five plausibly-inert speedups
(vectorize a hot loop, cache a precompute, add a hardcap, add
an agent-level deadline, push the soft-budget up 100ms) cumulated
into a ~12pp regression. Why:

- **Vectorization changes FP rounding.** `predict_fleet_fate`
  decisions depend on positions to ~6 decimal places; a different
  reduction order swings borderline cases differently.
- **Singletons leak state across game boundaries.** The KT
  singleton is reset per-turn but lives at module scope; if any
  consumer reads it stale (across episode boundaries in the
  harness), decisions skew.
- **Hardcap sentinels propagate.** Returning `-1e9` is filtered by
  `if delta > 0`, but if multiple sentinels reach the
  `argmax(delta)` call before the filter, the max is still one
  of them.
- **agent_deadline cuts good late-candidates.** When the chooser
  has 15 unranked candidates and 50ms left, it's better to score
  one heavy candidate than zero.

None of these are bugs in the strict sense. They're behavioral
delta from changes labeled "perf."

## What I'd do differently

1. **Bundle provenance in every A/B.** Stamp the focal and opp
   git-shas into the harness output. Refuse to run if
   `git log opp_sha..focal_sha --oneline` is non-empty without
   `--accept-build-drift`.

2. **Perf commits get the same gate as strategy commits.** Push
   no perf commit without an n=16 sequential A/B vs the immediate
   predecessor. Wilson-lo ≥ 0.45 to pass.

3. **The "flat point estimate across treatments" alarm.** If
   three distinct experimental treatments all measure exactly
   37.5%, stop and ask: what's invariant across them? The
   answer is the substrate, and the substrate is the bug.

## H41 specifically

H41 (late-game pv-discount depreciation) was structurally
plausible:

- pv decays 99.3 → 9.6 from step 0 → 490 (10× collapse)
- EDA Mine 4 said 76% of top-10 winners EXPAND ship-share in
  the last 100 turns; only 1.7% contract

Looked like an open-and-shut case. Floored pv at 50, A/B'd,
landed at 3W/5L. Tempting to retune (try floor=30, gamma=0.97)
but per Rule 37 with a wider "chooser-time scoring" axis
definition this is already iteration #2 in that axis (compute
chain = #1). One more variant and the axis closes.

The original docstring's warning ("future-prod over-weights
captures by ~100× in late game; chooser stops valuing ship
preservation") is empirically real. Seed 2 focal_max dropped
831 → 536 ms post-floor — chooser making fewer validation
calls because all candidates look similarly good.

If a future session re-engages H41, the right experiment is NOT
"tune the floor magnitude" — it's "give late-game opp_prod a
denial weight independent of pv." Captures late game DENY
opp production over the last few turns of game. Currently
modeled as `(my_prod - opp_prod) * pv`, but the denial term is
worth more late, not less.

## For next session

Don't pile a sixth knob on this branch's chooser. The chooser-
axis is saturated. Switch to `btjeK` and engage H44 (physics-
waste mechanism — fleets miss their targets due to orbital
mistiming / sun-route detours / comet expiry; ships survive
but the capture fails). PI flagged 2026-05-29 that the
"destroyed-in-flight" phrasing in `state/MULTI_BRANCH.md:76` is
wrong — fleets cannot be destroyed in flight. Read the actual
H44 audit on the btjeK branch before quoting any magnitude.
Still the highest-EV unfalsified lever in the multi-branch
state, just don't repeat the misquote.
