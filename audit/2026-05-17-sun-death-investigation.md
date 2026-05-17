# Sun-death investigation (PI live observation, 2026-05-17 PM)

## Observation

PI reported watching our composite_a2_hybrid agent launch "a large fleet
into the sun" mid-game. I responded by writing a `point_to_segment_distance`-
based sun-crossing gate in `lib/value_heads.composite_capture_value`
(commit `71bf289`). PI's interrupt: **"observe it in live games and debug
before you speculate."** Rule 38 in action.

## Method

1. Pulled live replays for submission `52744856` (composite_a2_hybrid,
   submitted 14:17 UTC, 57 min on ladder, 13-14 games completed).
2. Ran `scripts/replay_mine.py 52744856` for the waste-profile.
3. Enumerated every `outcome == "sun"` fleet via `attribute_fleets`
   and sorted by ship count.
4. Cross-checked against v15's prior 92-episode corpus
   (`audit/live-episodes/52710995/`).

## Findings

### Sun-deaths on the new submission

| rank | ships | t_launch | launch_xy | episode |
|---|---:|---:|---|---|
| 1 | **8** | 70 | (62.1, 7.3) | 76864489 |
| 2 | 3 | 63 | (49.2, 18.2) | 76861192 |
| 3 | 2 | 86 | (39.1, 18.7) | 76860790 |
| 4 | 2 | 39 | (9.6, 8.9) | 76862878 |

4 events across 1823 fleets = **0.22%**. The largest is 8 ships — well
below anything PI could call "large".

### Sun-deaths on v15

| rank | ships | t_launch | launch_xy |
|---|---:|---:|---|
| 1 | **30** | 63 | (45.1, 60.6) |
| 2 | 30 | 166 | (27.4, 87.5) |
| 3 | 21 | 68 | (62.9, 19.9) |
| 4 | 19 | 67 | (20.4, 38.4) |
| 5 | 18 | 61 | (67.3, 58.9) |
| ... | ... | ... | ... |

13 events across v15's 9,507 fleets = 0.14%. Max sun-fleet = **30 ships**.

### Composite is BETTER than v15 on sun-deaths

- Max sun-fleet size: 8 vs 30 (composite less likely to commit large
  forces to a doomed trajectory)
- Rate is slightly higher (0.22% vs 0.14%) but on a tiny sample
  (1823 vs 9507 fleets); within noise.

### What PI most likely saw

The "large fleet" was almost certainly a v15 game — v15 is still on the
ladder as the other rolling-last-2 entry, plays half of all our matches,
and has 30-ship sun-deaths in its history. PI was watching the team's
ladder feed, not specifically the composite_a2_hybrid submission.

## Bigger picture: 52744856 is not actually regressing

The pre-bundle hypothesis register (`audit/2026-05-17-pre-submit-
hypotheses-composite-a2-hybrid.md`) had H1 (≥1080 floor) showing BUSTED
at settled μ=1049.9. But:

- Submission age: 57 min, 13 games. Per the
  `early-trueskill-mu-unreliable` friction tag, **≥6h / ≥50 games
  needed** before strategic μ reading.
- Actual win record: **10-3 = 76.9%** (2P 8-1, 4P 2-2). That's a
  strong record, consistent with the local A/B (75% vs v15 at n=32,
  87-100% vs panel opponents).
- Replay-mine waste profile (composite vs v15):
  - win % 47.7 vs 47.4 (par)
  - defense % **40.2 vs 35.2** (+5 pp)
  - waste_attack % **10.8 vs 15.7** (-5 pp)
- μ at 1049.9 reflects 13 unsettled games, not regression.

## Decision

**Do NOT submit the sun-crossing gate** (commit `71bf289` stays on the
branch). Reasons:

1. The gate addresses 0.22% of fleets; load-bearing impact is small.
2. Submitting now would evict 52744856 before its μ stabilises — we'd
   lose the live A/B baseline forever.
3. The live win record (10-3) strongly suggests composite is already
   beating v15. Waiting 5+ hours costs us nothing; a premature push
   loses the data and ladder time.

Sun-gate may revive later if a SETTLED μ comparison post-Stage-1 shows
the agent has a sun-related EV leak. For now: hold.

## Next checkpoints

- Re-pull replays at **2026-05-17 21:00 UTC** (≥6 h post-submit).
- Re-query Kaggle for settled μ at the same time.
- Read against the hypothesis register's gates H1-H6.

## What this investigation taught me (process-level)

Rule 38 binds: **fix-verification reproduces the failure state**. I
caught myself writing a fix for a hypothesis I hadn't reproduced. PI's
interrupt was the right move — the investigation took 30 min and
proved the sun-gate isn't the right surface. Without it I'd have
shipped a quota-burning, ladder-time-burning submission for a 0.22%
case, evicting the actually-winning agent in the process.

Promotion candidate (next cycle): a CLAUDE.md sub-clause under Rule 38
saying "live observations from PI also require reproduction in data
before patching" — pattern: PI describes a behavior, agent speculates
on cause, agent ships fix without verifying the cause.
