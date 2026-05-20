# Phase 5B root-cause analysis — analytical agent loses to trajectory baseline

Session 2026-05-20 PM. Branch `claude/strategy-framework-design-OyoYR-rebased`.
PI directive: don't stop on this axis (Rule 37 cap acknowledged but
overridden); find ROOT CAUSES of failure.

## Surface diagnosis (initially proposed, then falsified)

> "Throughput gap: we fire too few launches per turn."

Falsified by direct measurement: in a single full game at seed 42 vs
trajectory baseline, **both sides fire 56 launches each**. Launch
COUNT is identical. The loss is not throughput-limited.

## Real diagnosis — phased failure with three contributing causes

### Phase 1: We WIN the opening (steps 0-30)

| Phase | me_cap | me_lose | opp_cap | opp_lose |
|-------|-------:|--------:|--------:|---------:|
| 0-30  | 4      | 0       | 3       | 0        |

We capture 4 planets, lose 0. **The opening planner works.**

### Phase 2: We hold parity through step 50

| Phase | me_cap | me_lose | opp_cap | opp_lose |
|-------|-------:|--------:|--------:|---------:|
| 30-50 | 6      | 2       | 7       | 1        |

Net +4 captures vs opp's +6. We're slightly behind but competitive.

### Phase 3: We start LOSING planets faster than capturing (steps 50-70)

| Phase | me_cap | me_lose | opp_cap | opp_lose |
|-------|-------:|--------:|--------:|---------:|
| 50-70 | 3      | 3       | 3       | 1        |

Net 0. Stall.

### Phase 4: COLLAPSE — we lose 5 planets while capturing 1 (steps 70-100)

| Phase | me_cap | me_lose | opp_cap | opp_lose |
|-------|-------:|--------:|--------:|---------:|
| 70-100 | **1** | **5**   | 3       | 0        |

Game ends at step 181 (analytical eliminated).

### Smoking gun — steps 80-88

```
step  | our launches    | our planet count
 79   | 8>46            | 10 planets
 80   | (no launches)   | 10 planets
 81   | (no launches)   | 9 (-1)
 82   | (no launches)   | 9
 83   | (no launches)   | 8 (-1)
 84   | (no launches)   | 6 (-2)
 85   | (no launches)   | 6
 86   | (no launches)   | 5 (-1)
 87   | (no launches)   | 5
 88   | 16>79           | 5
```

**9 consecutive turns of zero launches while losing 5 planets.**

## Root cause #1 — W2 defensive verdict returns "skip" for real threats

At step 80, ALL 10 of our owned planets are threatened
(`model.time_to_enemy_threat` returns non-None for each). The proposer
generates 1-3 defensive reinforce candidates per turn. Every one gets
`value_for_candidate == 0` because `w2_provably_held_reinforce` returns
`verdict.kind == "skip"` — Wald-conservative: it only commits when it
can PROVE the reinforce holds.

When every planet is threatened simultaneously, Wald can't prove
ANYTHING. We sit idle.

### Attempted fix — defensive value fallback

Mirror the Phase 4 W1 mid-bound fix: when W2 returns skip, return a
positive fallback so the LP commits SOMETHING.

| Multiplier | Game length | Total launches | Outcome |
|-----------:|------------:|---------------:|---------|
| 0 (current)|         181 |             56 | LOSE 0/16 |
| 0.5×       |         127 |            ??? | LOSE faster |
| 0.1×       |         160 |            105 | LOSE, fire-spam |

Both positive fallbacks made things WORSE. The defensive launches
don't actually save planets — they're too small or arrive too late.
Reverted.

## Root cause #2 — opp model under-projects threat in early game

`predict_opp_responses` projects 1 launch per opp source. Compared
against actual in-flight enemy fleets in the env:

| step | proj arrivals (ships) | actual in-flight (ships) |
|-----:|----------------------:|-------------------------:|
|   30 |               58      |                     113  |
|   50 |              187      |                     210  |
|   70 |              103      |                     282  |
|   80 |              412      |                     156  |
|  100 |              845      |                     387  |

Early/mid (30-70): **we under-project**. Our value function thinks
captures are safer than they are; we over-commit to offense.

Late (80+): **we over-project** (1-launch-per-opp-source scales
linearly with opp planet count; reality is fleets-already-launched).
Captures look infeasible everywhere; we idle.

## Root cause #3 — LP per-candidate scoring can't trade offense vs defense

The previous-session postmortem (`knowledge-base/thoughts/
2026-05-20-analytical-vs-rollout-architectural-bind.md`) names this:

> "The analytical pieces don't compose into a winning chooser when
> plumbed into a per-candidate scoring loop. The rollout chooser is
> implicitly doing PLANNING — its leaf-favor evaluates a leaf state
> that encodes the joint consequences of the whole turn's move-set.
> No per-candidate analytical score reproduces that."

This session's investigation is empirical evidence of exactly that.
Every "single-knob" fix to the value function either does nothing
(value=0 → idle) or backfires (positive fallback → over-fire +
faster loss). The per-candidate framework cannot encode "if I
commit ships to offense here, am I leaving home undefended" —
that's a GLOBAL property of the whole-turn move-set.

## What's NOT in scope (Phase 5C+)

These would each be multi-session efforts to fix the root cause #3:

- **Outcome-table-based value**: use Phase 1's `enumerate_outcomes`
  to compute JOINT outcomes per turn — what's the planet state if I
  fire columns A, B, C together vs A alone vs nothing. This IS the
  global-property scoring the per-candidate loop can't do. Phase 1's
  outcome table is built but never wired into the value layer.

- **Game-tree search over compressed state**: depth-K alpha-beta with
  the closed-form combat/movement as the transition function. Bounded
  compute, exact within the depth. Same architectural lift the
  previous session named.

- **Phase 5C — rollout for substrate, analytical for input**: the
  candidate Rule 43 says analytical work goes to the SUBSTRATE or
  the INPUT layer. We tried substrate (Phases 3/4/5A — losses). The
  unexplored direction: keep the trajectory baseline's rollout but
  inject analytical primitives (W1 bounds, winning-state predicate,
  opening MILP results) as additional scoring signals INSIDE the
  rollout's leaf evaluator.

## Bottom line for PI

The Phase 5A opening planner works as designed. The defeat is not in
the opening — it's in the mid-game where:
1. Defensive value function is broken (W2 skip → idle).
2. Each attempted defensive fix has unintended consequences.
3. The per-candidate scoring framework is architecturally incapable
   of expressing offense-vs-defense tradeoffs over the whole-turn
   move-set.

The previous session diagnosed this exactly. The analytical-as-
substrate axis has now produced FOUR consecutive A/B losses
(Slice 10 single-turn LP, Phase 4 multi-turn LP, Phase 5A.0 opening
planner strict, Phase 5A.1 opening planner loose). The architectural
bind is real and resistant to per-knob tuning.

PI direction needed.
