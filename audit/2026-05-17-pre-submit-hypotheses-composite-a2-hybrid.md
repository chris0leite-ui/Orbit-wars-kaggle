# Pre-submit hypothesis register — composite + A2 hybrid (2026-05-17 PM)

Submission file: `submissions/baseline.py` (286 KB bundle, parity-OK
712 turns, `BASELINE_VALUE_HEAD=hybrid` baked in via
`os.environ.setdefault` in `agents/baseline/main.py`).

Branch: `claude/audit-workflow-performance-btjeK` (HEAD `5ad5daf` at
the time of writing; the submit-state-update commit will extend it).

Composition:
- **2P** (~64% of ladder games): `composite_capture_value` — waste-
  penalty + capture-credit per in-flight fleet.
- **4P** (~36% of ladder games): `favor` with A2 — 1.5× weakest-opp
  bias + +55 elimination bonus. Sourced from public notebook
  `romantamrazov/orbit-star-wars-lb-max-1224` (peak LB μ=1224).

This is a **wholesale architectural change** to the value head only —
proposer + chooser + opp model + emit dedup untouched from v15-line.
The 5/17 fleet-efficiency negative-result session said next iteration
needs exactly this shape ("different value head AND different
proposer AND different chooser, or no change"); we landed the
value-head-only variant — partial coverage of that prescription.

## Local A/B evidence

### 2P — composite head

| opponent | n | rate | Wlo |
|---|---:|---:|---:|
| v3.5.1 | 32 | 100.0% | 0.893 |
| v4_planner | 32 | 96.9% | 0.843 |
| v9_scavenge (μ=1119.9 team peak) | 32 | 93.8% | 0.799 |
| v7_0 | 32 | 87.5% | 0.719 |
| v15 (μ=1108.4 champion) | 64 | 67.2% then 62.5% | 0.550 then 0.503 |

Best estimate of true 2P winrate vs v15: **~63-67% (Wlo at gate)**.
Cleanly beats every prior agent except v15, where the win is
borderline but consistent in sign.

### 4P — A2 (alone, no composite)

Sibling branch's FFA at n=32 seeds × 4 seats = 128 games: baseline
66.4% vs v15 64.1% → **+2.3pp within noise**.

This session's hybrid FFA panel (focal=baseline-with-A2-hybrid,
background v15+v7_0+v4_planner): focal first-place rate **30.1% at
n=103 in flight** (uniform-random expected 25%; +5pp above noise).
v15-as-focal leg not yet run at submit time.

## Calibration history

From `state/current.md`:
> Multiple recent submissions over-predicted live. Local-vs-live
> mapping has been roughly **-20 to -30 pp** on every recent
> submission.

Recent examples:
- v3.5.1 (5/12): -150 μ vs prediction.
- geo v3.1 (5/14): -80 μ floor.
- iter_v1 / iter_v2: composite head on v7_0 chooser → live μ 1035 /
  1036, well below local expectation.

## Hypotheses (pre-registered, with success criteria)

### H1 — Settled live μ ≥ 1080 (HOLD-THE-FLOOR baseline)

The submission does not regress more than v20's level (μ=1094 minus
~14 of allowable downside). Failure means the head swap broke
something the local tests didn't catch.

**Measure:** `kaggle competitions submissions orbit-wars` after
≥ 50 games settle. Cap at 6 h post-submit before reading per the
`early-trueskill-mu-unreliable` friction tag.

### H2 — Settled live μ ≥ 1108 (MATCHES-CHAMPION)

The hybrid head is at least neutral vs v15. Given local 2P 63-67%
and the -20-30 pp calibration historical drift, neutral on live is
the modest target.

**Measure:** same. PASS if settled μ ≥ v15's settled μ at the same
hour-of-day (account for v15 having more games and tighter CI).

### H3 — Settled live μ ≥ 1120 (BEATS-V9_SCAVENGE-CEILING)

The wholesale value-head change clears the ceiling we couldn't pass
in 7 prior fleet-efficiency variants. This is the stretch outcome
the A/B suggests is possible.

**Measure:** same. PASS only if settled μ exceeds 1119.9 (v9_scavenge
peak) by ≥ 1 σ of the new agent's μ.

### H4 — 2P winrate vs v15 in live games ≥ 0.50

The local 63-67% point estimate holds in some form on the live
ladder (even with -20pp calibration drift, we should remain above
50% in 2P head-to-head).

**Measure:** pull live episodes via
`python -m scripts.live_episode_summary <sub_id> --pull`, filter
for 2P games where v15 is the opponent (look up in
`info.TeamNames`), compute winrate. Need ≥ 16 v15 matchups for a
useful read.

### H5 — Max-turn-ms exceedances < 5% of focal turns

The 1000ms env cap is a SOFT cap (engine drops over-budget actions,
doesn't kill agent). Local max was 1196-1580ms. If live shows
> 5% of focal turns exceeding 1000ms, the timing fix (#1+#2) wasn't
enough and the WorldModel-reuse refactor
(`knowledge-base/concepts/worldmodel-reuse-options.md`) becomes
priority-1 before any further submission.

**Measure:** `python -m scripts.episode_postmortem <sub_id>` ships
a per-turn dt_ms field. Compute `(dt_ms > 1000).mean()` across all
focal turns. Threshold 5%.

### H6 — 4P first-place rate > 0.30

Random-uniform 4P baseline = 25%. A2's +2.3pp local lift was within
noise but positive. Live should land ≥ 0.30 if A2 has genuine
4P-side effect.

**Measure:** live episodes filtered to 4P games (`len(TeamNames) ==
4`), focal first-place count / total. Need ≥ 24 4P matchups.

## Eviction risk

Rolling-last-2 currently: v15 (μ=1108.4) + v20 (μ=1094.2). Pushing
this submission evicts **v20**, leaving v15 + this new agent as the
pair Kaggle keeps for final evaluation.

- Best case: new agent settles ≥ 1108 → strict improvement over
  the v20 floor.
- Neutral case: settles 1094-1108 → no improvement vs v15 alone,
  but no worse than the previous rolling pair (since v15 stays).
- Bad case: settles < 1094 → we lose the v20 floor and the rolling
  pair's effective μ shifts down.

Per the 5/16 friction tag `early-trueskill-mu-unreliable`: do NOT
make strategic decisions on the first ~6 hours of settling.

## Submission message

```
composite head 2P + A2 4P hybrid (baseline + bundled). 2P:
composite_capture_value waste/capture credit; n=32 panel vs v7_0
87.5pct, v4_planner 96.9pct, v3.5.1 100pct, v9_scavenge 93.8pct;
n=64 vs v15 62.5-67.2pct Wlo=0.50-0.55 INCONCLUSIVE-at-gate. 4P:
favor + A2 weakness-exploitation (1.5x weakest, +55 elim) sourced
from romantamrazov LB-MAX-1224. Bundle parity 712 turns, max
turn-ms 1196-1580 (over 1000ms cap on heavy turns - engine drops
over-budget actions). Rolling-last-2 evicts v20.
```

## Post-submit checklist (PI / next session)

1. Note submission ID and timestamp in `state/current.md`.
2. Wait ≥ 6 h before any strategic reading of μ.
3. Run `python -m scripts.live_episode_summary <sub_id> --pull`
   when ≥ 20 games settle.
4. Run `python scripts/replay_mine.py <sub_id>` to compare the
   live waste-profile against the 5/17 v15 baseline (47.4% win,
   35.2% defense, 15.7% waste_attack, 0.9% waste_traj, 0.1%
   waste_comet). H4 / H5 / H6 read from this output.
5. Compare settled μ vs H1 / H2 / H3 gates after 24 h.
6. If H1 fails: revert intent before next push (do NOT compound
   the eviction).
