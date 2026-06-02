# 2026-06-02 — speed audit + kinematic-table refactor

## The session arc, distilled

PI asked: "look at our speed; we don't have headroom." The thread was
expected to be a sim-perf investigation. It turned into a sequence of
calibration moments where my reported picture kept being wrong by
factors of 3-10×, and the actual load-bearing finding was a structural
bug in a different file.

## What I claimed at each step, and the correction

| Hour | My claim | PI / probe correction |
|---|---|---|
| AM | "215× speedup, sim isn't the bottleneck" | True per-step but irrelevant — bench used idle Snapshot |
| AM | "fast_sim.rollout(K=50) = 12.5 ms/step, big concern" | True for `mirror_self_policy` w/ v3_snipe planner — NOT the production hot path |
| AM | "agent uses 81 ms / 950 ms, 869 ms headroom" | Single seed (42), quiet game, empty late bucket |
| PM | "vs v7_0 over 5 seeds: median 286 ms, p95 701 ms" | Stood. ~3× the morning claim; ~74% of the budget at p95 |
| PM | "swept_pair_hit is O(fleets²)" | Wrong. Fleet-vs-PLANET (O(fleets × planets)). Read the call site. |
| PM | "JAX is easy" | Wrong. Champion's hot loop is policy-bound, lite_greedy_policy is 484 LOC Python, multi-week port to JAX-native; not easy |
| Late | "kinematic table is buggy" | True — singleton contamination across in-process A/Bs, ~9pp phantom regression |

The pattern: **every confident claim I made about the substrate today
was wrong on first pass and corrected by either (a) running more data
or (b) PI reading the function/diff and asking pointed questions.**

## What the speed audit actually produced

Two committed audit docs (`audit/2026-06-01-fast-sim-bench.md`,
`audit/2026-06-01-production-cost-probe-vs-v7_0.md`) with reproducible
numbers; one new tool (`scripts/production_cost_probe.py`) for capturing
realistic per-turn cost distributions; one corrected diagnosis: the
chooser does spend its budget on real opponents (median 286 ms, p95
701 ms), there IS some headroom on the median turn (~660 ms), and the
hard cap at 920 ms exists for real reasons not yet captured.

The 9-percentage-point headroom on the p95 tail (vs the 920 ms hard
cap) is the real constraint on adding chooser features.

## What the refactor actually produced

`lib/kinematic_table.py` no longer routes through a process-global
singleton. Each `World` owns its own KinematicTable. Sibling branches
can cherry-pick commit `40f2614`. The math is unchanged (89/89 parity
+ trajectory + KT tests green). Local in-process A/Bs are now
safe-by-construction for this class of bug.

## The lesson worth keeping

When the substrate has 3+ orthogonal axes (per-step cost, per-turn
step count, policy cost, p95 vs median, opponent-dependence, etc.),
**a single-number summary is almost always wrong.** The bench's
"215× speedup" is the textbook example: technically correct, structurally
misleading. Same for "81 ms / 950 ms headroom." Same for "swept_pair_hit
is the bottleneck."

The PI's role today was specifically to keep pulling me back to a
realistic distribution: "go over budget", "fleets cannot collide", "is
the kinematic table really buggy". Each pull-back closed a 3-10× claim
error. Without those, I would have shipped an audit doc into HANDOVER
that said "sim isn't the bottleneck, look elsewhere" — and the next
session would have inherited that as truth.

## Open question for next session

Where does the 138 ms median non-sim cost (proposer + cheap-rank +
leaf-value computation + bookkeeping) actually go? I have a number but
no breakdown. Without that breakdown we can't tell whether speeding up
the simulator further or pruning the proposer's candidate set is the
bigger lever for getting under the 920 ms hard cap. The
`scripts/production_cost_probe.py` infrastructure is in place to add
this measurement.
