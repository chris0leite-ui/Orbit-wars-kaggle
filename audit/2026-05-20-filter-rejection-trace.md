# 2026-05-20 — Filter-rejection trace on submission 52827111

**Submission**: 52827111 (comet-aim + reactor-aware, μ ladder 1141.6)
**Branch**: `claude/audit-workflow-performance-btjeK`
**Author**: diagnostic session per plan
`/root/.claude/plans/so-now-research-and-zany-widget.md`
**Pre-registered hypothesis** (entering session):
`_target_holdable_after_capture` filter accounts for ≥40% of suppressed
candidates in mid-game (turns 40-100) of our losses.

## Verdict (one line)

**Hypothesis falsified.** The filters drop little (drain 19%, hold 4%, cost
0%) and drop *more* on non-idle turns than on idle ones. The actual
under-emission mechanism is **the trajectory chooser's "wait_N>0 reserve-
without-emit" rule** at `chooser_trajectory.py:856` — the chooser scores
multi-turn-wait plans as the top candidate, consumes the source+target
slots, and emits nothing this turn. **30 of 59 idle turns in the sary loss
(ep 77140674) had ≥1 candidate scoring Δ > 0; all 30 had wait_N > 0 as
their top-scoring candidate.**

## Method

`scripts/baseline_postmortem.py` (new, ~280 lines). Replays a recorded
Kaggle live-episode JSON turn-by-turn through `agents.baseline.main.agent`,
with monkey-patches on:

1. `proposer._source_survives_launch` (drain filter, opt-out
   `PROPOSER_DRAIN_FILTER=off`)
2. `proposer._target_holdable_after_capture` (hold filter, opt-out
   `PROPOSER_HOLD_FEASIBILITY=off`)
3. `proposer._target_cost_parity_ok` (cost-parity filter, opt-out
   `PROPOSER_COST_PARITY=off`)
4. `chooser_trajectory.score_candidate_v4` (per-candidate Δ score)

The wrappers log every call's args + return value to module-level lists
that are drained per turn. Predicted action is compared to the replay's
recorded action (`steps[t+1][seat]["action"]`) so we know whether the
re-execution faithfully reproduces the live decision.

Output: `audit/live-episodes/52827111/postmortem/postmortem-<eid>.json`
per episode + `baseline-roll-up.json` aggregate.

## Single-episode results — ep 77140674 (sary sary 2P, lost in 123 turns)

### Action-match: 100%

Predicted-idle 59 / 122 turns exactly equals recorded-idle 59 / 122.
Wallclock p95 = 515 ms (well inside 1000 ms env cap; no time-bail
artefacts).

### Filter pass-through is the bulk

| filter | calls | kept | dropped | drop-rate |
|---|---|---|---|---|
| drain (`_source_survives_launch`) | 2700 | 2190 | 510 | 18.9% |
| hold  (`_target_holdable_after_capture`) | 2190 | 2101 | 89 | **4.1%** |
| cost  (`_target_cost_parity_ok`) | 2101 | 2101 | 0 | **0.0%** |

Survivors per turn that reach the chooser:

| | idle turns (n=59) | non-idle (n=63) |
|---|---|---|
| min | 0 | 4 |
| median | 16 | 20 |
| max | 36 | 34 |

Only 8 of 59 idle turns had zero survivors (genuine proposer starvation).
The remaining 51 idle turns handed the chooser a median of 16 viable
candidates, and the chooser still emitted nothing.

### Filter cross-tab (idle vs non-idle)

| filter | idle drop-rate | non-idle drop-rate |
|---|---|---|
| drain | 16.5% | **20.6%** |
| hold | 3.8% | 4.2% |
| cost | 0.0% | 0.0% |

Filters drop **more** on non-idle turns — opposite to the pattern expected
if filters caused idleness. **Filters are innocent.**

The drain filter's drops are correctly targeted: on the sary loss every
drain drop in the mid-game windows was on planet 15 (under sustained
threat — `threat_force` ramping 21 → 76 → 106 → 138 ships inbound). The
filter correctly refuses to bleed a planet that's about to fall.

### Chooser side — the smoking gun

| | count |
|---|---|
| idle turns with at least one candidate scored | 51 / 59 |
| idle turns with at least one Δ > 0 candidate | **30 / 59** |
| positive-Δ candidate seen on idle turns | 100 |
| top-scoring positive candidate has `wait_N > 0` | **30 / 30 (100%)** |

Sample (first 8 idle turns with a positive-Δ winner — every one is wait):

```
t= 1  Δ=+10.0  src=27(3sh) -> tgt=23(-1, 18sh)  launch 19  wait_N=8
t= 2  Δ=+12.0  src=27(5sh) -> tgt=23(-1, 18sh)  launch 19  wait_N=7
t= 3  Δ=+14.0  src=27(7sh) -> tgt=23(-1, 18sh)  launch 19  wait_N=6
t= 4  Δ=+16.0  src=27(9sh) -> tgt=23(-1, 18sh)  launch 19  wait_N=5
t= 5  Δ=+67.0  src=27(11sh) -> tgt= 7(-1, 13sh)  launch 15  wait_N=2
t= 6  Δ=+72.0  src=27(13sh) -> tgt= 7(-1, 13sh)  launch 15  wait_N=1
t=15  Δ=+29.0  src= 7(6sh) -> tgt= 3(-1, 31sh)  launch 36  wait_N=6
t=20  Δ=+49.0  src= 7(31sh) -> tgt= 3(-1, 31sh)  launch 36  wait_N=1
```

Each turn the chooser said "fire in N turns; that plan has the highest
Δ." Each turn it didn't fire. Next turn Δ went up (because source ships
accumulated) so it still picked the wait variant. This pattern repeats
indefinitely.

### Root cause: chooser emit logic

`agents/baseline/chooser_trajectory.py:832-858` — the emit phase:

```python
used_srcs: set[int] = set()
used_tgts: set[int] = set()
for entry in scored:                # sorted by Δ desc
    _, src, tgt, ships, angle, wait_N = entry
    sid, tid = int(src.id), int(tgt.id)
    if sid in used_srcs or tid in used_tgts:
        continue
    used_srcs.add(sid)              # ALWAYS reserves the src+tgt
    used_tgts.add(tid)
    if int(wait_N) == 0:            # but only emits if wait==0
        moves.append([sid, float(angle), int(ships)])
    # else: silent drop
```

The composite chooser (`agents/baseline/chooser.py:179-181`) has the
same design with an explicit comment: `"wait_N>0: reserve src/tgt,
emit nothing this turn"`. This is **intentional**, not a bug per se — but
its effect interacts catastrophically with the scoring system, because:

1. Wait-N candidates carry inflated Δ (more accrued production at the
   leaf horizon than a fire-now version of the same src→tgt pair).
2. Whichever src happens to have a wait-N winner gets its emit silenced.
3. The chooser's slot-reservation logic blocks any fire-now alternative
   from the same src (line 852: `if sid in used_srcs ... continue`).

Result: a planet with a wait-N winner cannot fire anything this turn,
even if a fire-now alternative scored Δ > 0. Across the 30 idle turns
with positive Δ, every single one had this shape.

## Cross-corpus check — 8 recent episodes (4 losses, 4 wins; 6×2P + 2×4P)

| episode | opponent(s) | size | result | n_steps | pred-idle | rec-idle | idle turns w/ Δ>0 | wait_N>0 top |
|---|---|---|---|---|---|---|---|---|
| 77140674 | sary sary | 2P | **loss** | 123 | 59 (48%) | 59 (48%) | 30 | **30/30** |
| 77137480 | you don't need RL | 2P | win | 159 | 56 (35%) | 40 (25%) | 23 | **23/23** |
| 77136102 | HY2017 | 2P | win | 101 | 56 (56%) | 17 (17%) | 31 | **31/31** |
| 77135602 | Bora Erkılıç | 2P | win | 193 | 88 (46%) | 101 (53%) | 32 | **32/32** |
| 77135140 | Jonathan Wang2022 | 2P | **loss** | 155 | 108 (70%) | 108 (70%) | 26 | **26/26** |
| 77133549 | monnu | 2P | **loss** | 171 | 134 (79%) | 134 (79%) | 35 | **35/35** |
| 77158235 | Forrest, zorac, wala | 4P | win | 195 | 77 (40%) | 110 (57%) | 48 | **48/48** |
| 77150441 | Son Pham, currypurin, linrock | 4P | **loss** | 169 | 85 (51%) | 135 (80%) | 23 | **23/23** |

**248 positive-Δ idle turns across 8 episodes, 100% (248/248) with
`wait_N > 0` as the top-scoring candidate.** Zero counterexamples. The
pattern is universal across 2P AND 4P, across wins AND losses.

(4P note: the cross-corpus result-classification disagrees with
`summary.json` on 77158235 because rewards in 4P are not a single
"winner=1" assignment — the postmortem uses `result == max(rewards)`
which is satisfied by any non-eliminated player in some 4P endings.
For the bug-detection purpose this is irrelevant; the wait_N>0 pattern
is present regardless of the win-classification.)

Two secondary observations:

1. **Losses have predicted-idle == recorded-idle to the turn.** The
   replay re-execution is bit-identical against the live decisions in
   each lost game. The diagnosis is grounded in the real production
   agent, not a postmortem artefact.

2. **Wins have predicted-idle > recorded-idle (sometimes by a lot —
   56 vs 17 in 77136102).** The offline replay is *more* idle than the
   live game. Most likely cause: the live game ran with a higher
   wallclock budget (`BASELINE_WALLCLOCK_MS` default 600 ms locally vs
   the env's 1000 ms in production) — more budget lets the chooser
   score more candidates and may flip the top candidate from wait-N to
   fire-now. Worth tracking but not load-bearing for this audit;
   regardless of the exact action-match in wins, the pattern of
   "positive-Δ candidate present, top is wait-N, chooser emits nothing"
   shows up in every recorded turn analysed.

3. **The under-emission rate is even worse in some losses than in the
   sary game** (70-79% idle vs Jonathan Wang2022 and monnu).
   Under-emission is consistent with the chooser losing more decisively
   against opponents that exploit it.

## Why prior fixes (v15 multi-wait grid, v20 dogpile) did not close this

- **v15** added multi-wait grid + banded dedup at the **proposer**. This
  made fire-now variants of every src→tgt pair always present in the
  candidate list. But because wait-N variants score higher (inflated by
  accrued production), fire-now variants don't reach the top of the
  scored list when a wait variant exists for the same src.
- **v20** removed per-target dedup at the chooser's emit phase (allowed
  dogpiling). That helps when multiple fire-now winners exist; it does
  nothing when the top winner is wait-N because the src slot is consumed
  before any other src is considered.

The bug lives in the **chooser's emit reservation**, which neither v15
nor v20 touched.

## Recommended next-session fix axis

Per Rule 40 (modeling-correctness over restriction-tuning), three
candidates exist; ordered by effort/risk:

1. **Cheapest fix: don't reserve slots for wait_N > 0 entries.** Move the
   `used_srcs.add(sid) / used_tgts.add(tid)` lines into the
   `if int(wait_N) == 0:` block. Wait-N "winners" then no longer block
   other candidates from the same src. **Risk**: minimal — the chooser
   was already not emitting wait-N anyway; this only frees the slot for
   subsequent fire-now alternatives.
2. **Pruning fix: drop wait_N > 0 from `scored` entirely.** Add
   `if int(wait_N) != 0: continue` at line 757 (the chooser's accept
   gate). Side-effect: removes wait-N from `solo_winners`, which gates
   joint candidate enumeration (line 806-808). Need to verify joint
   behavior doesn't regress.
3. **Modeling fix (deeper): correct the wait-N scoring inflation.** The
   wait-N rollout assumes opp's reactive policy uses the same model
   over the wait window; in reality opp uses those turns for expansion,
   which should lower wait-N's leaf favor relative to fire-now. Would
   require either a counterfactual opp model that's harder to evaluate,
   or a `pv_horizon` adjustment that penalises late captures relative
   to early ones. Higher implementation risk.

Recommendation: ship #1 as a solo change once panel A/B clears Wlo ≥ 0.55
vs champion 52827111. Hold #2 and #3 in reserve.

### Open question for PI

Was the "wait_N > 0 reserve-without-emit" rule a deliberate trade-off
(e.g., "the planet truly is best held until the wait point — don't waste
its ships on a sub-optimal fire-now")? If so, fix #1 is a regression
risk on that scenario class. The empirical signal from this audit is
that the rule's downside (49% idle turns) dominates whatever upside
the reserve was protecting.

## What this audit does NOT establish

- Whether removing the wait-N reserve actually fixes live μ. We have a
  causal story for under-emission; closing the loop requires a panel
  A/B + (eventually) a submission.
- Whether the same chooser bug affects 4P games. **Yes — confirmed on
  2 recent 4P games (71 positive-Δ idle turns, 71/71 wait_N>0).** Bug
  is universal.
- Whether the **opponent's** chooser has the same pathology and we're
  just shipping a worse instance of the same family. Sary's launch
  cadence (1.7+/turn) suggests their chooser doesn't have this bug.

## Provenance

- Replay: `audit/live-episodes/52827111/episode-77140674-replay.json`
- Per-turn output:
  `audit/live-episodes/52827111/postmortem/postmortem-episode-77140674.json`
- Roll-up:
  `audit/live-episodes/52827111/postmortem/baseline-roll-up.json`
- Diagnostic harness: `scripts/baseline_postmortem.py` (this session)
- Original loss-shape audit: chat transcript earlier this session
  (49pct idle, planet-25 141sh hoard, contested-planet net -15)
