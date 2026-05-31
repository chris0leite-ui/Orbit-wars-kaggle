# 2026-05-31 — Tier 2 root cause + PI ideas for next iteration

## What landed today

- **Chooser probe-fix bug found and reverted** (commits 05aa624 introduced
  it, 7e8c5dc reverted it). My "fix" had over-estimated `per_cand_ms` by
  ~5× by using `avg_K=32.5` (max horizon) in the cost formula when actual
  candidate rollouts use `prop_horizon ~5-15`. Under-budgeted safe_deadline
  pre-bailed validation after ~3-5 candidates instead of ~15-20 → focal
  emitted nothing most turns → 0/32.

- **Tier 2 architecturally falsified for the chooser as-is, but for a
  DIFFERENT reason than first thought.** After the chooser revert, the
  per-tier numbers (single game, seed=0, default 1000 ms wallclock):

  | Tier | Candidates validated/turn | Positive (score>0) | Emitted |
  |---|---:|---:|---:|
  | 0 (lite_greedy) | ~24 (1209 across 50 turns) | 81 | 14 |
  | 1 (top_tier_mirror) | ~4 (211 across 50 turns) | 43 | 6 |
  | 2 (trained_logreg) | ~3 (155 across 50 turns) | 37 | 5 |

  Heavier opp models eat the wallclock budget. The chooser validates
  ~8× fewer candidates → fewer positives → fewer emits → loses.

- **Baseline anchored.** Bare pv_eta (Tier 0) vs launch_rules at n=16:
  **9/16 = 56.2%** (Wilson [0.332, 0.769], INCONCLUSIVE at n=16). vs.
  pv_eta + Tier 2: 1/16 = 6.2%. 50-pp regression caused by the chooser
  starving candidate budget under heavy opp, NOT by Tier 2 policy
  quality per se.

## PI's three ideas to unblock the structural limit

(Recorded verbatim from voice intent; my elaborations follow.)

### Idea 1 — Event-driven rollout horizon (PI proposed)

> "Instead of thinking in time steps, think in capture steps or capture
> attempt steps. So say, we think three capture attempts in the future."

**What this means in plain language.** The chooser currently advances
the simulation one game-tick at a time and asks the opp policy what to
do at each tick. Over 10 ticks most calls land on boring states where
nothing happens (fleets just travelling). Instead: advance the
simulation until the next *strategic event* — usually the next
fleet-arrival/capture — and only call the opp policy at those points.
Three events of horizon instead of 10 ticks.

**Why it's a real architectural unlock.** It changes the opp-policy
call count per candidate from ~10 to ~3, cutting cost ~3×. It also
keeps the opp REACTIVE (unlike the simpler "cache opp action" option):
the opp only sees the board change at the moments it matters
(captures), but it still gets to react to focal's imagined moves.
Matches how a human reasons: "if I capture B in 7 turns, opp's
counterattack lands in 9, three captures from now the board looks
like X — am I ahead?"

**Hard parts.** Computing "time until next event" cheaply (work out
ETA for all in-flight fleets, take min — should be ~ms).
Fast-forwarding the engine N ticks at once without bugs (production
accrual, orbital movement, comet expiry). Choosing the event boundary
("next fleet arrival" is the cleanest deterministic option).
Horizon comparability across candidates with different event timings
(rollouts get different game-time depths; either extend short ones to
a minimum or discount-by-time).

**Cost.** Maybe 50-80 lines in `chooser_trajectory.py`, ~1-2 hours
write + bundle + smoke. Bigger swing than the cache-only option.

### Idea 2 — Fast learned opp model (PI proposed; reframed)

> "Can we make the opponent model faster by improving the ML algorithm
> that currently implements lite_greedy? Can we learn a more
> sophisticated opp model?"

(Small correction: `lite_greedy_policy` isn't ML; it's a hand-coded
ROI-greedy heuristic. The underlying question is: *can we have a
sophisticated opp model that runs at lite_greedy's speed — ~0.5
ms/call instead of Tier 1/2's 5-6 ms?*)

**Why this matters.** If yes, the whole chooser-affordability problem
dissolves. The 8× per-candidate budget gap closes. Tier 1's strategic
content (or any future heavier opp model) becomes affordable inside
the chooser's existing architecture, no event-driven rewrite needed.

**Two concrete paths.**

- **Path 1: distill `top_tier_mirror` into a tiny model.** Run
  `top_tier_mirror` across a few thousand random-self-play states.
  Record (features, action). Train a small NN or LightGBM to predict
  the action. Inference cost ~0.3-0.5 ms — same as `lite_greedy`,
  strategic shape close to Tier 1. Lose some teacher fidelity. Maybe
  1 day to corpus + train + smoke.

- **Path 2: train directly on real ladder behaviour.** Skip the
  teacher entirely; train the opp model on actual ladder opponents'
  observations and moves. Cleanest theoretical justification — model
  the distribution we actually play against, not Tier 1's
  abstraction.

### Idea 3 — Top-leaderboard Kaggle replays as training data (PI proposed)

> "There are replays available on Kaggle from top of the leaderboard
> places. We have a lot of data to learn from."

**The unlock.** Path 2 above requires a ladder-behaviour corpus.
Kaggle exposes per-game replays for submitted agents — including
high-μ opponents we want to emulate. Pulling a few hundred top-μ
replays, decoding the action sequences, and training a supervised
opp model on `(observation, opp action)` pairs gives us:

- A target that matches the *actual* ladder distribution (not Tier 1's
  selective-strategist abstraction)
- A lot of data — likely 10⁴-10⁵ (obs, action) pairs per replay × N
  replays
- Diversity across opponent styles (the top of the ladder isn't one
  agent; it's a mixture)

**Combined with Idea 2's Path 1.** Could train a tiny model on the
union — distilled Tier 1 outputs *plus* real ladder replays. Probably
a more robust opp model than either source alone.

## How the ideas relate to each other

- **Idea 1 (event horizon) and Idea 2 (fast opp model)** are
  complementary, not competing. Either alone unblocks heavy opp models;
  both together would be the strongest.
- **Idea 3 (top-ladder replays) is the data source** that makes Idea 2
  Path 2 viable and tightens Idea 2 Path 1.
- The fastest single-experiment path forward is probably **Idea 2
  Path 1 (distill Tier 1)** — small corpus from self-play, train tiny
  model, see if chooser regains its lift.

## My recommendation if PI says "do one of these"

Idea 2 Path 1, distillation, with the leaderboard replays from Idea 3
held in reserve to graduate to Idea 2 Path 2 if Path 1 shows lift but
not enough. Idea 1 (event horizon) is the deeper architectural win
but is bigger and riskier; defer until we've ruled out whether cheap-
opp-model alone is sufficient.

## State carry-forward

- `submissions/baseline_pv_eta_vh_opp.py` (~982 KB) — Tier 2 bundle,
  parity-verified, **DO NOT SUBMIT** (1/16 vs launch_rules).
- `agents/baseline_pv_eta_vh_opp/main.py` — wrapper template; reuse
  the env-var preamble for any Tier-2-style follow-up.
- `scripts/bundle_pv_eta_vh_opp.py` — patches `_OPP_BOOSTER_B64`
  blob; structure transfers to any future opp-model wrapper.
- Chooser probe fix is REVERTED in `agents/baseline/chooser.py` —
  current rolling pair is unaffected.
- `data/shot_validator/validator_booster.txt` (PM5, val_acc 0.83) —
  preserved for any classifier-as-filter follow-up; falsified as opp
  policy in chooser rollouts.

## Open questions for next session

- Should we verify the diagnosis by running pv_eta + Tier 1 (`BASELINE_OPP_TIER=1`)
  alone at n=16? Predict 15-30% based on candidate-validation numbers.
  Killed by CPU contention today; rerunnable in 15-20 min.
- For Idea 2 Path 1 (distill Tier 1): what feature set? Reusing the
  45-d `shot_features` would let the same model serve double duty as
  filter and predictor. But it's per-emit not per-state; we may need
  per-state features (similar to the 40-d `value_head_features`).
- For Idea 3 (ladder replays): how does the Kaggle replay API expose
  per-game observations? Does `kaggle competitions submissions` give
  us the JSON, or do we need to scrape?
