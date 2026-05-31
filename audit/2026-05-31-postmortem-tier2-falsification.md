# 2026-05-31 postmortem — Tier 2 falsified for chooser-budget reasons; B.3 submitted on PI override

## What was attempted

Implement the PI-ratified Tier 2 opp-emit predictor: filter
`top_tier_mirror`'s candidates with the PM5 shot-validator booster
(45-d features, threshold 0.30) inside `chooser_trajectory`'s
fast_sim rollouts, on the theory that a more realistic opp model
would give the focal chooser better leaf states and unlock the
B.3 head's marginal lift.

## What landed

- `lib/opp_model.py` — `trained_logreg_policy` implementation
  (commit 05aa624). Code is correct; falsified for downstream reasons.
- `agents/baseline_pv_eta_vh_opp/main.py` + `scripts/bundle_pv_eta_vh_opp.py`
  — wrapper + bundler (commit 05aa624). Bundle 982 KB, parity-tested.
  Reusable template for any future opp-model wrapper.
- `agents/baseline/chooser.py` — probe-fix attempt to capture Tier-2
  cost in `affordable_validate_cap` (commit 05aa624) **REVERTED**
  (commit 7e8c5dc) because it caused 0/32. See "What went wrong" #1.
- Two audits + three knowledge-base entries (thoughts / questions /
  flags) capturing the diagnosis.
- **Kaggle sub 53212044** — `baseline_pv_eta_vh_b3smoke` submitted
  on PI override ("submit so we can learn and observe"). Evicts
  `baseline_launch_rules_universal` (μ=1183.7). Predicted μ-loss.

## What went wrong

### 1. Self-inflicted chooser bug, 0/32 caused by my own "fix"

The probe-fix in `affordable_validate_cap` correctly captured the
Tier-2 cost in `per_step_ms` (~7 ms with Tier 2 active). But the
downstream formula
`per_cand_ms = (per_step_ms × avg_K + per_leaf_ms) × 1.5_safety`
multiplies by `avg_K = (MIN_HORIZON + MAX_HORIZON) / 2 = 32.5` —
which is *max* horizon, NOT the typical `prop_horizon` (~5-15)
that `score_candidate_v4` actually uses in its rollout loop. The
estimate over-shoots by ~5×.

`safe_deadline = deadline - 344 ms` then pre-bailed the validation
loop after ~3-5 candidates per turn (instead of the ~15-20 the
chooser needs). With `scored` mostly empty,
`chooser_trajectory.py:1084 if not scored: return [], []` fired —
focal emitted NOTHING most turns → focal did nothing all game →
0/32 vs `launch_rules`. PI flagged the result as suspect; only
then did I diagnose the cause.

**Lesson:** the probe was the right idea (capture true opp cost
in the estimate); the formula needed adjustment in tandem
(smaller `avg_K`, or remove the 1.5 safety multiplier). Don't
ship a fix to one half of a formula and leave the other half
miscalibrated.

### 2. Tier 2 architecturally falsified — but for a different reason than first thought

After reverting the chooser bug, n=16 still gave 1/16 wins. Initial
audit attributed this to "filter design makes opp look passive in
rollouts." That theory was wrong.

The actual cause: even with the chooser unbroken, heavier opp models
(Tier 1 and Tier 2 both) eat the chooser's wallclock budget. Tier 0
(`lite_greedy`, ~0.5 ms/call) lets the chooser validate ~1200
candidates per turn; Tier 2 (~6 ms/call) cuts that to ~155 — an
8× reduction. Fewer validated candidates → fewer with `score > 0` →
fewer emits → focal under-fires against an aggressive launch_rules
opp → loses 94% of games.

The 50-percentage-point regression (56% Tier 0 → 6% Tier 2 vs the
same opponent) is purely a chooser-budget effect, NOT a Tier 2 policy
quality issue. Tier 1 would show the same regression for the same
reason.

**Lesson:** the chooser's wallclock budget is structurally calibrated
for cheap opp models. Substituting a 10×-more-expensive opp model
without changing the chooser is architecturally incompatible.

### 3. Monitor tooling spammed the chat with re-emitted "DONE" lines

My `until both A/Bs are done` Monitor script kept re-printing the
completed Tier 0 result every 30 s for the full 30-min timeout,
because the Tier 1 A/B had been killed by the harness's own 25-min
timeout and never wrote a "wins=" terminal line. The exit predicate
required both logs to have terminal lines. PI saw ~10 system-reminder
notifications after the result was already shared.

**Lesson:** Monitor exit predicates must include
"process is gone AND file is stable for N seconds", not just
"matching line appears" — otherwise a killed sibling task hangs
the monitor.

## What we learned (positive)

- **The PI's diagnosis instinct was right.** "0/32 is too surprising,
  another bug?" surfaced the chooser self-inflicted issue (#1 above).
  Following the suspicion produced a real root-cause and a revert
  rather than a premature axis-closed conclusion.
- **The chooser's wallclock budget is a constraint we now understand.**
  Quantified: 1200 → 155 candidates/turn under Tier 2. Any future
  heavy opp model (RL, IL, distilled) needs to either match
  `lite_greedy`'s ~0.5 ms speed OR the chooser needs an architectural
  change (event-driven horizon, opp-output caching).
- **Three concrete architectural paths logged** with PI's framing:
  event-driven rollout horizon, fast distilled opp model trained from
  top-leaderboard Kaggle replays, or both together. Recommendation
  to next session: distill `top_tier_mirror` into a sub-ms model
  first, since it's the smallest swing with a clear win/lose signal.

## Submission state at session end

| Submit | Agent | Status | μ |
|---|---|---|---|
| **53212044** (just now) | baseline_pv_eta_vh_b3smoke | PENDING | TBD |
| 53197142 | composite_universal_submit (sibling branch) | settling | 1089.6 provisional |
| ~~53182323~~ | baseline_launch_rules_universal | **EVICTED** | (was 1183.7) |

PI sign-off rationale: "submit so we can learn and observe" —
calibration probe for B.3's behavior on the live ladder against
opponents stronger than the local pv_eta-class panel. Worst case
the B.3 head settles ~1140-1180 and is replaced when the next
strong candidate ships.

## Carry-forward for next session

- All Tier 2 artifacts preserved as DO-NOT-SUBMIT reference templates.
- Chooser is reverted to pre-probe-fix state; current rolling pair
  unaffected.
- The three improvement directions (events / distilled opp /
  ladder replays) wait for PI to pick one. Distillation is the
  smallest swing.
- Open question I'd want answered: how does Kaggle expose
  per-game replays for top-leaderboard agents? That access path
  determines whether the ladder-replay corpus idea is feasible.
