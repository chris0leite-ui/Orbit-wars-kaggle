# Postmortem — 2026-05-29 game-theory-winning-strategy-SEU7P

The session that built five compute-budget commits on top of a
"scalar champion" baseline that didn't include them — every A/B was
confounded by the build delta. Three measurement loops, two
strategic axes "falsified," and only at the end did the bundle
provenance check expose the confound.

## What went wrong

**Dominant pattern: comparison baseline drift, undetected for the
entire session.**

The A/B opponent `/tmp/baseline_pv_eta.py` was bundled
2026-05-28 14:17 — BEFORE the commit chain this session built on
top of. Specifically:

| Commit | When | In focal bundle? | In opp bundle? |
|---|---|---|---|
| `b4f885d` vec orbital window | 2026-05-29 | YES | **NO** |
| `0f1da5b` KT singleton wire | 2026-05-29 | YES | **NO** |
| `8c6f47c` WC 600→700 | 2026-05-29 | YES | **NO** |
| `357b52d` WC→800 + hardcap | 2026-05-29 | YES | **NO** |
| `bdfe9c7` agent_deadline | 2026-05-29 | YES | **NO** |
| `9ebd311` H41 pv floor=50 | 2026-05-29 | YES | **NO** |

Every measurement in this session was "focal (5 perf commits + 1
strategy commit) vs opp (none)." When focal lost 5/8, I read the
result as "the experimental knob (H41 / Stage-3 breadth) didn't
help." The honest read is "the perf chain + the experimental knob
together don't beat the pre-perf bundle." Different conclusion.

**Why I missed it.** The perf commits were treated as inert speedups
in the design contract — vectorization "shouldn't change behavior,"
KT singleton "produces bit-identical results to predict_relative,"
hardcap "only fires when overshooting." Each individual claim is
defensible, but cumulatively the chain produces ~12pp regression
vs the pre-perf baseline. None of the five commits had a paired
n=16 sequential A/B against the bundle that became the comparison
opp.

**Cost.** Six commits, ~3 sequential A/Bs (n=3, n=8, n=8), one
parallel n=16, and a Stage-3 bundle that I was about to declare
falsified. ~half a session of work measured against a baseline
that was never the right reference.

## What was actually learned

1. **The perf chain regresses ~12pp vs pre-perf baseline.** Specific
   culprit unknown. Candidates: vectorization FP rounding,
   KT singleton state-leak across games, hardcap sentinel
   propagation, agent_deadline cutting short useful late-rollout
   validation.

2. **H41 floor at 50 is not a strategic win at the surface level.**
   Even after accounting for the confound, n=8 sequential split
   5L/3W. The pv floor changes which seeds win — wins different
   games, loses different games. Net: parity with the confounded
   substrate, so probably parity-or-worse with a clean substrate
   too. Original docstring's warning about ship-preservation
   regression was empirically reproduced (seed 2 focal_max dropped
   831→536 ms — chooser making fewer validation calls because
   candidates appear undifferentiable).

3. **Both falsified "axes" this session are actually one falsified
   axis.** Compute-budget and pv-floor are both "chooser-time
   leaf-scoring optimization." Per Rule 37, that's 2 of 3
   consecutive variants in one design axis. One more iteration
   here closes the axis.

4. **Eyeball-level signal of Rule-37 axis exhaustion before
   measurement.** When the doctrine-prescription axis closed
   (2026-05-28), the next session pivoted to "speed up the
   chooser." If the chooser-axis-leaf-scoring family is itself
   saturated (the v9-v15 chooser iteration line was also closed
   on this axis), the next session shouldn't try to coax μ out of
   that family.

## PI overrides

None this session. PI's questions ("how often does the bold model
exceed the 1s cap?", "I want us to really improve. To have a better
strategy.", "wrap up") were directional steers, all correct: each
one would have saved time if I'd internalised it as a hard prior
earlier. The "improve / better strategy" prompt is what triggered
the read of state/MULTI_BRANCH.md and the reach-frontier doctrine
closure note — without it I'd have spent the session on a sixth
compute-budget tweak.

## Friction entries promoted

- `tag: baseline-bundle-provenance-not-checked` — every A/B opponent
  bundle should have its build commit recorded next to its
  filename, and the focal-vs-opp commit-delta listed in the A/B
  output. The 2026-05-28 timestamp on `/tmp/baseline_pv_eta.py`
  was visible from `ls -la` the entire session; nothing in the
  harness surfaced "focal has 6 commits opp doesn't."
- `tag: perf-commit-treated-as-inert` — every perf commit landed
  without a paired n=16 sequential A/B against the prior bundle.
  Speedups can change behavior in non-trivial ways (FP rounding,
  state cache, sentinel propagation). The right gate before
  pushing a perf commit is "n=16 sequential vs the prior bundle,
  Wilson-lo ≥ 0.45." None of the 5 commits had that gate.

## Promotion candidates for `.claude/skills/kaggle-comp/improvements.md`

PI to ratify; staging only:

1. **Bundle-provenance ledger in the A/B harness.** Print
   focal-commit and opp-commit beside their filenames at script
   start; print the commit-delta `git log opp..focal --oneline`.
   Trigger a hard error if the delta is non-empty AND the script
   wasn't invoked with `--accept-build-drift`. Origin: this
   postmortem.

2. **Perf-commit n=16 gate.** Any commit whose subject line
   matches `^perf\(` requires a paired n=16 sequential A/B
   against the immediately-prior commit, with Wilson-lo ≥ 0.45,
   before push. Logged to `audit/perf-commit-gate-<sha>.md`.
   Origin: this postmortem.

3. **Rule 37 "axis" definition tightening.** "Chooser-time leaf-
   scoring optimization" was treated as a different axis from
   "value-head modification" when the two are operationally the
   same (both modify the input to `score_action`'s leaf eval).
   Improvement: explicit axis taxonomy in CLAUDE.md, with
   "chooser-time scoring" as one axis covering both compute-
   budget knobs and value-head functional-form changes.

## Closing read on the strategic landscape

Three of the four "open" strategic candidates I surfaced at the
"I want us to really improve" pivot have known costs:

- **H44 physics-leak fix** (Track B, `btjeK` branch) — un-tried on
  this branch, requires switching branches and reading the
  2026-05-21 corrected audit. Highest EV of the surveyed options.
- **Wrap-baseline portfolio veto** (Track C, `PFhzM` branch) —
  37.5% asymmetry was the only positive signal in 10+ iterations
  there. Different paradigm (veto layer on existing chooser, not
  chooser replacement).
- **H41 late-game expansion** (this session) — falsified at
  floor=50; one more variant allowed under Rule 37 before axis
  closes. Probably not the lever.
- **H40 / H42 (EDA hypothesis board)** — 15 days old, no
  measurement attempted.

Next session entry-point should be H44 on the `btjeK` branch, NOT
another iteration on this branch's chooser axis.
