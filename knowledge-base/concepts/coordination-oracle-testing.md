# Coordination Oracle Testing — backward-solving from obvious wins

## PI insight (2026-05-18, dekaineko trace session)

> "There are situations that are obviously solvable, so we can backward
> solve from these situations. We can learn from simple situations
> where we can have an easy win by a sort of backwards optimization,
> maybe. We can use these tests to validate a joint modeling approach
> when we do not optimize for each planet individually. At some point
> we will need to learn how to coordinate our actions because
> otherwise we won't get on top of the leaderboard."

## The fundamental architectural limitation

Our chooser scores each candidate launch IN ISOLATION:

1. Pick candidate (src, tgt, ships, wait_N)
2. Run fast_sim rollout with OUR launch injected at wait_N, OPP's
   `lite_greedy_policy` simulated at every step
3. Read leaf favor at horizon
4. Δ = leaf − idle_baseline

**Critical asymmetry**: in the rollout, after our single launch, our
agent does NOTHING for 25 ticks. Only the opponent acts (via lite_greedy).
The chooser is essentially asking "if I make this one move AND THEN STAND
STILL, what's the worst that opp can do?"

This produces a worst-case bound where the chooser refuses launches
that expose ANY source to counter-attack, because we've "agreed" not to
defend in the rollout.

## The "obvious wins" we miss

**Dekaineko game, step 150**: we hold 23 planets, ~3700 ships; opp holds
P8 with 92 ships. Capturing P8 ends the game. Any single launch large
enough to capture P8 leaves the source temporarily exposed. Opp's
counter-attack (predicted at the leaf) recaptures the source. Net Δ ≤ 0
for every candidate. **30 turns of idle.**

But to a human observer the move is obvious: launch from P0 → P8 (122
ships, captures). If opp counter-attacks the empty P0, ONE of our 22
OTHER planets reinforces P0 before opp arrives. P0 stays ours. P8
captured. Game won.

The plan exists. The chooser cannot see it because it models OUR
future moves as "do nothing."

## The principle: oracle testing for coordinated planners

These "obvious win" scenarios are testable propositions. They define
properties a competent planner must satisfy:

1. **Cleanup property**: if `my_ships > K × opp_ships` AND `opp_planet_count
   ≤ 1`, the planner must emit a capture move within ≤ N turns. (For the
   dekaineko scenario: K=10, N=2.)

2. **Coordinated capture property**: if no single source has enough ships
   to capture target T solo, but the SUM of K nearest sources does, the
   planner must consider emitting a coordinated K-source plan.

3. **Defense-by-reinforcement property**: if launching from source S
   creates a counter-attack window, but neighbor N can reinforce S before
   the window closes, the planner must emit the (S→T, N→S-reinforce) plan
   as a unit.

4. **Sequential-buildup property**: if attacking target T requires
   accumulating ships across multiple turns, the planner must commit to
   the multi-turn plan (not re-decide each turn).

Each property maps to a class of game situations. We can extract
concrete oracle scenarios from live replays where these properties
fail.

## The proposed methodology

1. **Mine replays for oracle scenarios.** Build a script that walks
   our live replays and finds situations matching each property's
   antecedent. For each, record:
   - Snapshot obs
   - The "obvious correct" emit (computed by exhaustive search or
     human annotation)
   - The actual emit (what our agent did)
   - The gap (lost μ, missed captures, etc.)

2. **Build a regression suite** (`tests/test_chooser_oracle.py`) where
   each oracle scenario is a test. Test passes iff the planner emits a
   move within the correct class (capture, reinforce, etc.) — not
   necessarily byte-exact.

3. **Iterate planner architecture** against the oracle suite. Each new
   chooser/planner design must pass the existing oracles. Start with
   easy oracles (cleanup property) before tackling hard ones
   (coordinated capture).

4. **Track oracle coverage as a metric**. Today: 0% pass (chooser is
   single-step, doesn't satisfy any coordination properties). Target:
   monotonic improvement.

## Why this matters strategically

Without coordinated planning:

- We plateau at ~μ=1145 (current rolling pair settled point)
- All "more aggressive in 2P" fixes regress 4P (Tier 1 lite_greedy
  21.9%, Joint v1 37.5%, spatial 40.6%) because per-launch decisions
  can't model "I'll defend myself"
- We lose easy wins (dekaineko stall)
- We lose to coordinated opps who DO plan multi-source attacks
  (Roman game pattern: many small coordinated launches > our few
  large isolated ones)

Top-LB agents (top-of-leaderboard at 1300+ μ) presumably DO plan
coordinated actions. To reach that range, we need:

- Architectural change in the planner (multi-step / multi-source aware)
- Oracle test suite to validate the change works on simple cases
- Iterative refinement, each iteration measured against oracles AND A/B

## Concrete next steps (ordered)

1. **Extract first oracle**: dekaineko game step 150 → snapshot +
   correct-emit metadata. Concrete: "given THIS obs, the planner
   must emit at least one launch toward P8 within 2 turns."

2. **Add more oracles** from existing replays:
   - asdf game step 37 (multi-fleet incoming, must reinforce)
   - Roman game step 50 (joint capture of strong neutral)
   - Any "fall-then-recapture" event (preemptive defense)

3. **Design planner architecture** that can pass these:
   - Option A: reactive self-defense in rollouts (mirror lite_greedy
     for ME, not just opp) — bug #14 fix
   - Option B: extended joint candidate enumeration (3-tuples,
     conditional plans)
   - Option C: heuristic overrides for the specific oracle patterns
     (cheapest, but doesn't generalize)
   - Option D: proper multi-turn planner with explicit plan
     representation (biggest rewrite)

4. **Build the oracle test runner**: not just pytest assertions;
   needs a way to load a snapshot, run the planner, check the
   emit set.

5. **Iterate**: planner change → run oracles → measure gap → A/B →
   refine. Each successful oracle pass is a real μ-relevant change.

## Connection to existing bugs

Bugs that share this root cause (in catalog):

- **#4 — drain-frontier**: chooser blind to "I'll defend this source
  later if attacked"
- **#13 — can't finish in dominant positions**: chooser sees no Δ
  benefit because OUR cleanup-attack is unmodeled
- **#14 — asymmetric simulation**: rollout models opp's reactions
  but not ours

All three are symptoms of single-step + me-static-in-rollout. The
oracle methodology lets us validate the FIX without depending on
A/B noise.

## Why oracle tests + A/B together

- **A/B tests** measure aggregate win-rate across noisy games. Slow,
  high-variance, hard to debug.
- **Oracle tests** are deterministic — given input X, emit must
  satisfy property P. Fast, debuggable, surgical.
- Oracle passes are NECESSARY but not SUFFICIENT for A/B wins (a
  planner could pass oracles but lose to specific opp strategies).
- A/B wins without oracle coverage are FRAGILE — a small obs
  perturbation can break the pattern.

Best practice: gate planner changes on BOTH oracle pass AND A/B
non-regression.

## What this displaces

This concept doc deprecates the implicit assumption that A/B is
the only validation mechanism. Going forward, planner changes
should be validated against oracles first (cheap, fast), then A/B
(slow, definitive).

It also reframes the bugs we've catalogued: many are symptoms of
ONE structural limitation (single-step me-static rollout), not
independent issues. Fixing the architecture solves a class of
bugs simultaneously.

## Status

**2026-05-18**: concept documented. Implementation not started.
Pending: extract first oracle scenarios from existing replays,
design oracle test runner.

This is the foundational architectural direction PI identified.
Next session priority.
