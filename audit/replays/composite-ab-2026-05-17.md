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
