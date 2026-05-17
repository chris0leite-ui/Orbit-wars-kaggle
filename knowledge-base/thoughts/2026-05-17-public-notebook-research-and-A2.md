# 2026-05-17 — Public-notebook research reorders the pivot; A2 (4P weakness) lands first

## What prompted this

Mid-session, after the clean modular baseline merged (PR #27, commit
`6ff087e`, μ-parity with v15 at 1112.8), the PI asked: *"do proper
research, 7-step problem-solving, diagnose what we should do
differently."* Then: *"research public notebooks."*

The public-notebook pull surfaced the missing input. Pre-research, the
diagnosis was converging on the CRN refactor (opp_traj + common random
numbers, the state-function principled fix from
`2026-05-17-state-function-principled-fix.md`). Post-research, the
priorities reordered.

## The three public notebooks that reordered priorities

1. **romantamrazov/orbit-star-wars-lb-max-1224** (132 votes, peak LB
   μ=1224). Pure heuristic, ZERO ML, ~3300 LOC of mission-portfolio
   code. Decomposes into: 6+ named mission types (reinforce, eliminate,
   gang-up, crash-exploit, rescue, recapture, generic-attack) competing
   on a unified per-target score; 4P weakness exploitation
   (ELIMINATION_BONUS=55, WEAKEST_ENEMY_MULT_4P=1.5);
   inter-enemy dynamics; indirect wealth map (regional production
   density); adaptive modes (is_behind/is_ahead/is_finishing); phase-gated
   heavy ops; SIM_HORIZON=110 per-target timeline sim.

2. **konbu17/orbit-wars-rule-base-ml-shot-validator-hybrid** (70 votes).
   Tiny MLP (24→64 ReLU→32 ReLU→1 sigmoid, ~5k params, ~15 KB). Trained
   on 8.8k labeled shots with BCEWithLogitsLoss, 40 epochs, val-split
   by game_id. POST-FILTER rejection design at threshold 0.4. **+19 pp
   local lift, +43 pp vs tier4 opponents, no opponent regression.**
   Weights ship as base64-embedded NPZ (Kaggle no-external-data
   compatible). We have 4× konbu17's data (37k examples in
   `data/shot_validator/labels.parquet`) — except the parquet doesn't
   exist; only the schema/README, the actual labels need to be
   regenerated from a replay corpus (which is also absent —
   `audit/external/replays/` is not in the working tree).

3. **aidensong123/lb-highest-1000-search-learned-value-function**
   (63 votes, claimed LB only 1000+ — *lower* than our 1115 ceiling).
   Gradient-Boosted Classifier 500 trees depth 6 (AUC 0.976) trained
   on 267k labeled win/loss pairs (26k top-agent + 240k self-play).
   1-ply forward sim w/ opp counter-move. **PARKED** — the value-head
   approach alone caps at LB 1000+, *below* our existing ceiling. Not
   the breakthrough.

## What this evidence says

The +109 μ headroom above our ceiling (1115 → 1224) is reachable
**without ML**, via heuristic action-space architecture. The
chooser-axis ceiling we hit (Rule 37 cap at 7 variants) is structural
to drop-one enumeration — romantamrazov uses a fundamentally
different action space (named missions, not drop-one). Their +109 μ
is largely 4P-side: weakness exploitation, inter-enemy dynamics,
indirect wealth.

The state-function principled fix (CRN refactor) I designed yesterday
is still good engineering but is NOT the dominant defect. The
empirical evidence is that the chooser action space is the
binding constraint at μ=1115.

## A2 — the cheapest fragment of romantamrazov's pattern

I implemented A2 today as the smallest isolated change extractable
from romantamrazov's pattern:

```python
# agents/baseline/value.py
ELIMINATION_BONUS = 55.0
WEAK_ENEMY_THRESHOLD = 110.0
WEAKEST_ENEMY_MULT_4P = 1.5
WEAKEST_ENEMY_MULT_2P = 1.25
ELIMINATION_GATE_RATIO = 0.9
STRENGTH_PROD_WEIGHT = 15.0
```

- **2P (single opp):** uniform 1.25x bias on opp_ships+opp_prod. Was
  initially 4P-only; revised mid-session when I realized fast.py only
  tests 2P games — a 4P-only change would show INCONCLUSIVE h2h.
- **4P (3 opps):** 1.5x bias on the WEAKEST opp's contribution; the
  other two opps unweighted. Biases leaf valuation toward states
  that further weaken (or eliminate) them.
- **Elimination bonus (both formats):** +55 when weakest's strength
  (ships + 15*prod) ≤ 110 AND my_strength ≥ 0.9 × weakest's. Gate
  prevents elim-then-die bias.

40 lines net delta in `value.py`. 6 unit tests added (3 for 4P; 2 for
2P bias; 1 for constants calibration). 28 baseline tests green; bench
p50=91 p95=261 max=347ms (unchanged envelope); smoke vs random PASS;
smoke vs nearest PASS.

## A 2P revision was load-bearing

Initial implementation was 4P-only ("there's only one opp in 2P, no
weakness choice to make"). I realized after starting the h2h gate
that fast.py's `play_one` uses `env.run([p0, p1])` — 2P only. A
4P-only change would have shown ~50% h2h (INCONCLUSIVE) vs v15 because
the code paths are identical for 2P.

Revised to apply WEAKEST_ENEMY_MULT_2P=1.25 uniformly on the single
opp in 2P (matching romantamrazov's 2P value). This is an enemy
aggression bias — it makes attack trades that were 0-EV at parity
slightly positive-EV in the chooser's Δ. The 2P symmetry test
(balanced state → favor=0) had to change: balanced state now scores
< 0 from either seat, both seats see same magnitude (consistency
check rather than zero check).

Test `test_favor_2p_consistent_from_either_seat` documents the new
property: both seats compute the same favor magnitude in balanced
states. Strategic anti-symmetry is preserved at the action-Δ level
(both biased agents play more aggressively — a symmetric strategic
change).

## What's parked

- **B2 — opp_traj + CRN refactor.** Design is intact in
  `2026-05-17-state-function-principled-fix.md`. The cross-game audit
  (84% v8 losses = mid_economy; emission-rate gap 8-13 vs 19-25) still
  argues for it as a real defect. But the public-notebook evidence
  says action-space architecture (Stage 4 mission portfolio) is the
  dominant gap; CRN is downstream of that.

- **B3 — MLP shot validator.** Blocked on data: `labels.parquet` and
  `audit/external/replays/*.json` are both absent. Need to either pull
  top-LB replays via Kaggle's per-episode-replay API (rate-limited
  and not yet scripted) or generate via self-play with
  `scripts/generate_selfplay_replays.py` (self-mimicking labels, not
  gold-label).

- **Stage 4 — mission portfolio.** Full port is ~3000 LOC; even a
  subset (eliminate + gang-up only) is ~600-1000 LOC. Higher build
  cost than the plan's 400 LOC estimate. Defer until A2 lands; revisit
  scope.

## Hard-won principles re-affirmed

1. **Rule 22 actually moves the needle.** Pulling top public notebooks
   at a plateau was the difference between "do the CRN refactor"
   (~6 hr, +20-50 μ if it works) and "do A2 first then build the
   action-space pivot" (~30 min A2 + multi-day portfolio, +50-100 μ
   evidence-backed). The PI's instinct to ask for public-notebook
   research before committing was load-bearing.

2. **Test what you ship.** The 4P-only A2 would have looked fine in
   unit tests, smoke, and bench — and silently shown ~50% h2h vs v15
   (INCONCLUSIVE) because the gating harness is 2P-only. Mismatch
   between change-axis (4P) and gating-axis (2P) is a friction worth
   logging.

3. **Rule 37 means STOP, not "try one more on the same axis."** Every
   chooser-axis variant since v15 drew at parity. The breakout is on
   a different axis (action-space architecture, per romantamrazov's
   public notebook). The temptation to "tweak the multiplier once
   more" is the same anti-pattern as v16-v20.

## Open questions

- Does A2 clear h2h vs v15 at n=64 Wilson lo > 0.50? (Running as of
  this entry; result TBD.)
- If yes: submit A2 as the next slot? Or stack with B3 (after solving
  the replay-data blocker) before slot-spending?
- If no: tune the multiplier (1.25 → 1.15 / 1.10)? Or skip A2 entirely
  and go directly to Stage 4 subset port?
- For Stage 4: full romantamrazov port (3000 LOC, high risk, high
  ceiling) vs subset port (eliminate+gang-up only, ~800 LOC, lower
  ceiling) vs A2-only + repeated calibration (lowest cost, lowest
  ceiling)?

## Update 2026-05-17 — h2h verdict + 2P-bias rollback

H2H A2 (4P weakness + 2P uniform 1.25x bias) vs v15 (n=64) shipped:

```
n= 32  wins= 13/32  ( 40.6%)  Wlo=0.255  Whi=0.577   CONTINUE
n= 64  wins= 25/64  ( 39.1%)  Wlo=0.281  Whi=0.513   INCONCLUSIVE
turn-ms p50=280  p95=693  max=1340  total elapsed 1468.4s
```

Verdict: INCONCLUSIVE per the plan's hard gate (Wlo > 0.50 required).
Point estimate 39.1% is below the 50% parity line — the 2P uniform
1.25x bias REGRESSES vs v15 in 2P games, even if not statistically
significantly so.

**Diagnosis of why 2P bias regresses:**

The 1.25x multiplier on opp_ships + opp_prod makes the chooser
perceive opp's contribution as 25% MORE significant. For a capture
trade where I lose X ships and opp loses X ships:

- Without bias: delta_F1 = -X - (-X) = 0   (zero-EV trade)
- With 1.25x:   delta_F1 = -X - (-1.25X) = 0.25X   (positive-EV trade)

Effectively the bias makes the chooser more aggressive. v15 is
well-calibrated for the current chooser; over-aggression trades
ships needed later for opp-ships now. The "weakness exploitation"
thesis from romantamrazov is structurally per-WEAKEST (1.5x on
the single weakest opp out of 3), not uniform. Conflating "uniform
2P aggression" with "4P weakness targeting" was the design error.

**Action taken:**

Rolled back the 2P bias path. value.py is now:

  - 2P: UNCHANGED from baseline (max-of-opps, no bias, no bonus).
  - 4P: 1.5x mult on weakest opp + elim bonus (gated).

This means A2 has ZERO effect in 2P games (fast.py default), so we
can't validate via fast.py's 2P harness. Three follow-up options:

1. **Validate via 4P FFA panel** (`scripts/ffa_panel.py`). Compare
   focal=baseline-A2 vs focal=v15 on the same 3-opp background. If
   A2 gives higher first-place rate in 4P, that's evidence. ~30-60
   min wallclock for n=32 seeds.

2. **Submit A2 as a calibration probe.** 4P-only changes don't risk
   2P regression; downside is bounded. But Rule 1 needs PI approval
   and Rule 12 says rolling-last-2 is risk-bearing.

3. **Skip A2 entirely**, go to Stage 4 (mission portfolio subset).
   This is the +109 μ path the plan endorses; A2 was its smallest
   fragment.

Most defensible move: option 1 first (4P FFA validation), then
based on result, option 2 (submit) or option 3 (Stage 4 build).

**Rule reminders surfaced:**

- Rule 37 (3-variant axis cap): the 2P bias rollback is NOT iteration
  on the same axis — it's removing a hypothesis that failed evidence.
  Different from "tune 1.25 → 1.10 → 1.05" which WOULD violate Rule 37.

- Rule 38 (fix-verification reproduces failure): the 2P bias failure
  was caught BY the h2h gate (Rule 27a we just codified). The gate
  worked. The diagnosis WHY would need a replay-level analysis of
  losses (deferred).

- Rule 40 (modeling-correctness over restriction-tuning): the bias
  was an inelegant uniform multiplier; the 4P weakness logic is
  semantically grounded (target the weakest player, the LB-MAX pattern).
  Keeping the modeling-correct piece, dropping the band-aid.
