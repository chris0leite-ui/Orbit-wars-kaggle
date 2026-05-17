# Pre-submit hypothesis register — <AGENT NAME> (<DATE>)

> Copy this file to `<sub_id>-<short-name>.md` BEFORE submitting.
> Replace every `<…>` placeholder. Drop sections that don't apply.
> Reference precedent: `audit/2026-05-17-pre-submit-hypotheses-composite-a2-hybrid.md`.

Submission file: `submissions/<bundle>.{py,tar.gz}` (<size>, parity-OK
<turns>, `<key env vars>` baked in).

Branch: `<branch-name>` (HEAD `<commit>` at the time of writing).

Composition: what change is shipping and where in the pipeline.
- proposer: <unchanged / new / tweaked …>
- chooser: <unchanged / new / tweaked …>
- value head: <unchanged / new / tweaked …>
- opp model: <unchanged / new / tweaked …>
- emit/dedup: <unchanged / new / tweaked …>

One-sentence mechanism statement (NOT just "μ goes up" — the
behavioural prediction the change should produce):
**<one sentence; what live-observable thing should change>**

## Local A/B evidence

### 2P

| opponent | n | rate | Wlo | Whi |
|---|---:|---:|---:|---:|
| v3.5.1 | 32 | <pct>% | <wlo> | <whi> |
| v4_planner | 32 | <pct>% | <wlo> | <whi> |
| v7_0 | 32 | <pct>% | <wlo> | <whi> |
| v15 (current rolling champion) | 64 | <pct>% | <wlo> | <whi> |

Best estimate of true 2P winrate vs v15: **<X>%**.

### 4P (if applicable)

FFA panel (focal=<agent>, background <opps>): focal first-place rate
**<pct>% at n=<n>** (uniform-random expected 25%; threshold for
"signal" = ≥ 30% per the H6 precedent).

## Calibration history

From `state/current.md` / `audit/hypotheses/results.md`:
- Local-vs-live mapping has historically been **-20 to -30 pp** on
  heuristic-stack submissions.
- Specific prior examples relevant to this change:
  - <agent X (date)>: local <pct>% → live μ <val>, **delta <Δ>**

## Hypotheses (pre-registered, with success criteria)

> Numbering convention: H1 = HOLD-THE-FLOOR (the no-regress floor),
> H2 = MATCHES-CHAMPION (parity vs current rolling champ),
> H3 = BEATS-CEILING (the stretch outcome). Hn for any mechanism-
> specific behavioural metric (NOT just μ). At least ONE H must be
> a non-μ metric — that's the calibration data point.

### H1 — Settled live μ ≥ <FLOOR> (HOLD-THE-FLOOR)

Reason this floor: the submission cannot cost us our current rolling-
last-2 worse than (current_worst - acceptable_downside).

**Measure:** `kaggle competitions submissions orbit-wars` after
≥ 50 games settle. Cap at 6 h post-submit per `early-trueskill-mu-
unreliable` friction tag.

### H2 — Settled live μ ≥ <CURRENT-CHAMP-μ> (MATCHES-CHAMPION)

The new agent is at least neutral vs the rolling champion.

**Measure:** same. PASS if settled μ ≥ <champ>'s settled μ at the
same hour-of-day (account for champ having more games, tighter CI).

### H3 — Settled live μ ≥ <CEILING> (BEATS-CEILING)

The change clears the prior ceiling. Stretch.

**Measure:** same. PASS only if settled μ exceeds <ceiling> by ≥ 1 σ
of the new agent's μ.

### H4 — <BEHAVIOURAL METRIC name from lib/metrics.py> <op> <threshold>

The mechanism produces a measurable behavioural change. This is the
load-bearing pre-registration: even if μ doesn't move (or moves but
for the wrong reason), this metric says whether the MECHANISM is
firing.

Pre-fix baseline value: <val> (from `<source>` — e.g. v15's
`replay_mine.py` output).
Expected post-fix value: <val>.

**Measure:** `python -m scripts.measure_hypothesis <this file>`. Runs
`lib.metrics.<metric_name>` over live replays pulled via
`scripts/live_episode_summary.py --pull`.

### H5 — Max-turn-ms exceedances < 5% of focal turns

Soft safety. The 1000 ms cap is non-fatal (engine drops over-budget
actions) but exceedances signal a regression in timing budget.

**Measure:** `python -m scripts.episode_postmortem <sub_id>` →
per-turn `dt_ms` field. Compute `(dt_ms > 1000).mean()`.

### H6 — <add additional hypotheses as needed>

…

## Eviction risk

Rolling-last-2 currently: `<champ> (μ=<μ>)` + `<other> (μ=<μ>)`.
Pushing this submission evicts **<which>**, leaving <champ> + this
new agent as the pair Kaggle keeps for final evaluation.

- Best case: new agent settles ≥ <champ-μ> → strict improvement.
- Neutral case: settles <range> → no improvement, no worse than the
  previous pair.
- Bad case: settles < <floor> → we lose the floor and the rolling
  pair's effective μ shifts down.

Per `early-trueskill-mu-unreliable`: do NOT make strategic decisions
on the first ~6 hours of settling.

## Submission message

```
<one-paragraph summary suitable for the Kaggle submit message: what
changed, key local numbers, calibration history, eviction implication>
```

## Post-submit checklist

1. Append a row to `audit/hypotheses/results.md` with sub_id +
   pre-registration timestamp.
2. Note submission ID + timestamp in `state/current.md`.
3. Wait ≥ 6 h before any strategic reading of μ.
4. Run `python -m scripts.live_episode_summary <sub_id> --pull` when
   ≥ 20 games settle.
5. Run `python -m scripts.measure_hypothesis <this file>` to evaluate
   H1..Hn against settled data. Append result row to `results.md`.
6. If H1 fails: revert intent before next push (do NOT compound the
   eviction).
7. If a hypothesis is refuted AND live μ regressed: trigger postmortem
   (Rule 14 strategy-critic-loop).
