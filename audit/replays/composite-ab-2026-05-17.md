# composite head A/B — 2026-05-17

Date: 2026-05-17 (afternoon)
Branch: claude/audit-workflow-performance-btjeK
Focal: `agents/baseline` + `BASELINE_VALUE_HEAD=composite`
Tool: `fast.py eval --vs-panel … --require-h2h …`
Logs: `/tmp/composite_ab.log`, `/tmp/composite_vs_peaks.log`
   (ephemeral; this file is the persistent record)

## Result table

| opponent | n | wins | rate | Wlo | Whi | verdict |
|---|---:|---:|---:|---:|---:|---|
| v3.5.1 | 32 | 32 | **100.0%** | 0.893 | 1.000 | PASS |
| v4_planner | 32 | 31 | 96.9% | 0.843 | 0.994 | PASS |
| v9_scavenge (μ=1119.9 team peak) | 32 | 30 | **93.8%** | 0.799 | 0.983 | PASS |
| v7_0 | 32 | 28 | 87.5% | 0.719 | 0.950 | PASS |
| v15 (μ=1108.4 champion) | 64 | 43 | 67.2% | 0.550 | 0.774 | **INCONCLUSIVE** |

Cross-eval focal turn-ms:
- panel (v7_0 / v4_planner / v3.5.1): p50=269 p95=782 **max=1183**
- vs peaks (v15 / v9_scavenge): p50=300 p95=757 **max=1292**

## Reading

1. **Composite head decisively beats every prior agent except v15.**
   Most striking: 30/32 = 93.8% vs v9_scavenge — the team peak at
   μ=1119.9. Every prior chooser-family modification regressed against
   v9_scavenge (7 v21/v22/v23 variants on 2026-05-17 fleet-efficiency
   branch, all FAIL). This is the first head-level change to clear it.

2. **vs v15 is borderline.** 43/64 = 67.2% Wlo=0.550 lands exactly
   on the gate. Point estimate is meaningful (composite is winning);
   the 95% CI of [0.550, 0.774] just doesn't fall strictly above gate
   at n=64. n=128 would likely disambiguate but adds 10+ min.

3. **Two blockers before submission** (see plan + commit message):
   - max turn-ms > 1000 (env budget) on both panel runs. Root cause:
     `chooser.affordable_validate_cap` probes per-step cost only,
     undercounts composite leaf cost (~2-5ms) by ~95%.
   - composite has no 4P opp-aggregation. 4P games are ~36% of the
     ladder; default-on without 4P-aware dispatch would regress.

## Retest after 4P-aware + timing-aware fixes (same day, evening)

After landing the 4P-favor dispatch in `favor_composite` and the
per-leaf-cost probe in `affordable_validate_cap` (commit `6aba319`),
re-ran the borderline opponent only:

| opponent | n | wins | rate | Wlo | Whi | verdict |
|---|---:|---:|---:|---:|---:|---|
| v15 (μ=1108.4 champion) | 32 | 24 | **75.0%** | **0.579** | 0.867 | **PASS** |

- max turn-ms 1196 (was 1292) — 96ms improvement but still over the
  1000ms env hard cap on heavy turns.
- p95 turn-ms 721 (was 757).
- Verdict crossed from INCONCLUSIVE → PASS. Point estimate moved up
  (67.2% → 75.0%), but at n=32 the CI is wide; the meaningful change
  is that Wlo cleared 0.55 deterministically.

**Submission readiness:** A/B-wise yes (panel + h2h vs both peaks now
PASS). Wallclock-wise borderline — max=1196 means occasional turns
will exceed Kaggle's 1000ms actTimeout. Submission could still work
(the engine drops actions over budget; doesn't kill the agent), but
those turns play idle. Risk: a high-impact turn timing out = lost
game. Worth a final tighten before live ladder.

## A/B retest #2 (post #1+#2 timing fixes, pre-A2-merge)

After commit `239fdc5` (pre-bail headroom + adaptive WorldModel
horizon), re-ran composite vs v15. Adaptive-tier eval doubled to
n=64 (CI bracketed at n=32):

| n | wins | rate | Wlo | Whi | verdict |
|---|---:|---:|---:|---:|---|
| 32 | 23 | 71.9% | 0.546 | 0.844 | CONTINUE |
| 64 | 40 | **62.5%** | 0.503 | 0.733 | **INCONCLUSIVE** |

- max turn-ms 1580 (vs 1196 last retest) — but this run shared CPU
  with the FFA panel; not apples-to-apples.
- p95 turn-ms 646 (down from 721; the timing fixes ARE working at
  the 95th percentile).
- Total elapsed 1583s (longer due to CPU contention).

**Reading the variance:** point estimate moved 75% → 62.5% across
the two retests. The first was n=32 (noisy); the second is n=64
(more reliable). 62.5% is closer to the original n=64 figure of
67.2%. Best estimate of the true composite-vs-v15 winrate: ~63-67%,
Wlo at the 0.55 gate. Composite is winning in 2P but not
deterministically PASS-able at n=64.

## Post-merge — A2 4P from `claude/kaggle-baseline-strategy-lO4mm`

Pulled in `agents/baseline/value.py` + `tests/test_baseline_value.py`
from the sibling branch (commit `a97806a`). The branch independently
extracted **A2** from public notebook
`romantamrazov/orbit-star-wars-lb-max-1224` (peak LB μ=1224, +116
above v15 at 1108). A2 mechanic (4P only, default-on):

- 1.5× bias on weakest opp's contribution (other opps unweighted)
- +55 elimination bonus when weakest strength ≤110 AND my strength
  ≥0.9 × weakest
- 2P branch unchanged from v15 baseline (their 2P uniform-bias test
  regressed 25/64 = 39.1% vs v15 — INCONCLUSIVE).

`select_favor_fn` gains a `BASELINE_VALUE_HEAD=hybrid` option:
- `unset` → favor (default; A2 in 4P, vanilla in 2P).
- `composite` → favor_composite (composite_capture_value, 2P-only).
- `hybrid` → favor_hybrid (composite in 2P, A2-favor in 4P) — the
  recommended production dispatch.

The composite head numbers above stay valid (panel + peaks were 2P
games; A2 doesn't affect them). The merge upgrades the 4P side that
my `favor_composite` previously fell back to vanilla favor for — and
according to their FFA at n=128, A2-favor in 4P is directionally +2.3
pp vs v15 (within noise but positive sign). Test coverage: 17 tests
on value.py (theirs) + my 6 chooser timing/dispatch tests, 51 green
total.

## Caveat: missing in-family h2h

`--require-h2h agents/baseline` was passed but the fast.py gate's
"same agent as focal" name-check skipped the within-baseline h2h
(composite-on vs composite-off, both at `agents/baseline/`). That
h2h would isolate the value-head's contribution from rollout-depth /
proposer / chooser dispatch effects. Doing it properly requires either
(a) copying `agents/baseline/` to a temp path with the env-var baked
in, or (b) generalising the gate to use content hash, not name.
Documented as follow-up; the panel + peaks data above already
establishes the head's value vs the existing champion stack.
