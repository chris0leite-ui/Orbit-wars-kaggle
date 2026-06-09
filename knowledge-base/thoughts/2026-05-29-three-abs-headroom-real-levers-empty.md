# 2026-05-29 (cont.) — three A/Bs in one session: headroom is real, levers are empty

## The thing I want to remember

When yesterday's session ended I'd handed over a story that read
"perf chain regresses 12pp — figure out which commit." Today's
session inverted that conclusion: the regression was a harness
artifact (KT singleton state-leak across in-process A/B seats).
Once the harness was fixed (subprocess isolation via
`scripts/clean_ab.py`), the post-perf bundle came back to parity
with pre-perf at n=32: **15/32 = 46.9%, Wilson [0.309, 0.636]**.

The headroom was real. The bug wasn't.

## The structural lesson

The KT singleton at `lib/kinematic_table.py:414` is module-global.
In Kaggle, each agent runs in its own process, so the singleton
serves one seat. In our local `fast.py` / `quick_ab.py`, both
seats share a Python process, so the singleton serves whichever
seat called `kt_begin_turn(world)` last. Different worlds; same
cached lookups; one seat reads positions from the OTHER seat's
world.

This is invisible at the test_bundle.py level (single-game smoke,
single agent). It's invisible at the unit-test level (the KT
parity test pins a single world). It is only observable in a 2P
A/B with a non-trivial difference in `world.fingerprint` between
the two agent loads — which is exactly the A/B configuration we
were running.

`scripts/clean_ab.py` was added on 2026-05-15 for env-pollution.
It also (incidentally) cures the singleton-leak. Use it always
from now on.

## Today's three A/Bs

All three focals against the same opp (`/tmp/baseline_post_perf.py`),
subprocess-isolated, n=32:

| Test | Focal | % | Wilson lo |
|---|---|---:|---:|
| Level 0 | pre-perf (WC=600, transitivity) | ~53% | — |
| Level 1 | post-perf + JOINT-expanded (TOP_K=15, MAX_PAIRS=200, AGGR=True, JOINT force-on) | 50.0% | 0.336 |
| Level 2 | post-perf + H44 Phase 3a wait_N filter (cherry-pick `c6a0c80`) | **40.6%** | **0.255** |

Three theories tested for using headroom; three falsified at the
n=32 gate. The chooser-leaf-scoring axis stays closed (Rule 37);
the proposer-admissibility axis just closed at one variant
(strong-enough regression to call it).

## What I'd do differently

1. **Cherry-pick precedent before A/B'ing.** I knew the H44 corrected
   audit had retracted the "65% fleet-destroyed-in-flight" framing
   (PI flagged on 2026-05-29; new commit 92371dc says fleets cannot
   be destroyed in flight). The Phase 3a fix on `extract-physics-
   trajectory-Vjaz9` was written before the retraction. The whole
   premise of the wait_N filter is "cut physics-impossible wait_N
   candidates" — but the premise itself was suspect. I cherry-picked
   anyway because the commit existed and the test was cheap. Cheap
   tests against bad premises are still wasted compute.

2. **Three "spend the headroom" hypotheses, three falsifications.**
   This is the Rule 37 signature: three same-class falsifications
   means the class is dead, not the specific variants. The CLASS
   here is "spend more compute / more breadth on the chooser-or-
   proposer stack." Per Rule 37 the right move was probably to
   STOP after level 1 and pivot off the branch, not run level 2.
   Cost: ~45 min compute on level 2 + commit + revert overhead.

3. **The headroom-spending question is the wrong question on this
   branch.** The post-perf bundle plays at parity with the
   ~μ=1150 PV_ETA champion. Spending the headroom adds zero μ in
   any of the three configurations I tested. The path forward is
   axis-switch (chooser sizing on btjeK, MLP filter on hqNVM, or
   Track-C wrap-baseline) — not finding a fourth knob to spend
   the headroom on.

## For next session

- **Revert commit `8b20b6d`** (the wait_N filter cherry-pick). It
  actively regresses by ~10pp on local A/B; the underlying premise
  was retracted by PI; the original "wait_N would mis-classify"
  bypass was load-bearing.
- **Do NOT cherry-pick Phase 3b (`25589ad`)** on this branch. It's
  the leaf-side mirror of 3a; if 3a regresses, 3b doesn't rescue.
- **Pivot off SEU7P.** The two genuine high-EV pivots are:
  1. **H44 chooser sizing** (corrected audit on btjeK): `D_under_delivered`
     24% + `A_src_lost_pre_landing` 22% in lost episodes = 46% of
     losses addressable by ship-count and source-survival fixes
     in `agents/baseline/chooser_trajectory.score_candidate_v4`
     and `agents/baseline/proposer._source_survives_launch`.
     Both files live on this branch — implementation here is
     possible, but the corrected audit reads better on btjeK.
  2. **Konbu17 shot-validator MLP** on `hqNVM`: Phase 2 v2 GBT +
     39-d features in active development; live precedent +19pp
     panel in the original paper. `baseline_validated` sub
     53131296 is the first cut at μ=1086 (directionally below
     PV_ETA's μ=1154); Phase 2 v2 is the real test.
- **All future local A/Bs go through `scripts/clean_ab.py`.** No
  in-process fast.py eval for decisions.
