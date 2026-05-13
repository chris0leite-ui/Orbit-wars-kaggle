# Lookahead simulator architecture — what we built and how to use it

> Permanent reference for the PI second-brain. Written 2026-05-12
> after the foundation (`d054f18`) + v7 stack iteration (`488383f`).
> Plain English: no abbreviations the PI hasn't already seen.

## The one-line version

We built **a fast offline copy of the game** plus **a decision layer
that runs short "what-if" simulations** before every move. The agent
asks "if I do X this turn, what does the board look like 10 turns
later?" — answers it many times per turn — and picks the move with
the best projected outcome.

## The four-layer stack (bottom to top)

```
                   ┌──────────────────────────────┐
   Layer 4         │   AGENT (agent(obs) entry)   │
                   └──────────────┬───────────────┘
                                  │
                   ┌──────────────▼───────────────┐
   Layer 3         │   CHOOSER (lib/v7_search.py) │
                   │   Enumerate candidates;       │
                   │   score each one; pick max.  │
                   └──────┬───────────────┬───────┘
                          │               │
              ┌───────────▼──────┐ ┌──────▼──────────────┐
   Layer 2    │ OPPONENT MODEL   │ │   FAST SIMULATOR    │
              │ (lib/opp_model)  │ │  (lib/fast_sim.py)  │
              │ What will opp do?│ │  Step the world N×. │
              └──────────────────┘ └─────────────────────┘
                                              │
                          ┌───────────────────▼─────────────┐
   Layer 1                │  PHYSICS (kaggle_environments's │
                          │  orbit_wars.interpreter()       │
                          │  — combat, movement, comets,    │
                          │  production)                    │
                          └─────────────────────────────────┘
```

**Layer 1** is the game's own physics — we don't replace it.
**Layer 2** is the foundation we built: a fast wrapper around the
physics + a model of how the opponent might play.
**Layer 3** is the brain: it generates candidate moves and evaluates
each by simulating ahead.
**Layer 4** is the thin agent file that the Kaggle harness calls.

## Layer 2a — the fast simulator (`lib/fast_sim.py`)

### What it does

Same job as `env.clone() + env.step()` from the kaggle_environments
library: take a snapshot of the game state, apply both players'
actions, return the new state. Bit-exact same outcome as the real
game.

### How it's faster

The real `env.step()` carries a lot of bookkeeping we don't need in a
"what-if" simulation:

- Validates every action against a JSON schema (~1 ms each)
- Wraps the state in proxy objects for general kaggle_environments
  compatibility (~0.3 ms)
- Appends the post-step state to a growing history list (gets slower
  as the game progresses)
- The "clone" path re-validates the schema on every clone (~3 ms,
  growing to ~22 ms mid-game)

`fast_sim` skips all of this. It calls the game's physics function
**directly** on a minimal `Snapshot` object — same physics, no
framework overhead.

### The measured payoff

```
env.clone()+step() per simulated tick:   22.5 ms   (mid-game)
fast_sim.step()    per simulated tick:    0.12 ms
                                          ─────────
                                          ~183× faster
```

### Why this matters

The agent has **1 second per turn** on the live ladder. Before
`fast_sim`, we could afford 2-5 "what-if" simulations per turn
(each ~80-280 ms). After `fast_sim`, we can afford ~50-100 — enough
room to evaluate richer candidate moves OR look further ahead OR
sample multiple opponent guesses.

The catch: `fast_sim` is bit-exact only when we know the random seed
the game used (for the comet spawns at steps 50/150/250/350/450).
On the live ladder that seed is hidden from the agent, so simulations
that cross a comet-spawn boundary drift slightly from reality. Within
a 10-15 step horizon between boundaries the simulation is exact;
across a boundary it's a close approximation.

### Files

- `lib/fast_sim.py` — `Snapshot`, `from_obs()`, `step()`, `clone()`,
  `rollout()`, `ship_totals()`, `delta_us_minus_them()`.
- `tests/test_fast_sim_parity.py` — 10 tests that confirm bit-exact
  match against `env.clone()+step()` over 5 seeds × 100 turns in
  2-player and 2 seeds × 60 turns in 4-player.
- `audit/2026-05-12-fast-sim-bench.md` — the speedup measurement.

## Layer 2b — the opponent model (`lib/opp_model.py`)

### What it does

When we simulate "what if I do X", we have to assume something about
what the opponent does at the same time (the game is
simultaneous-move). The opponent model is a guess.

### The three tiers

- **Tier 0 — `mirror_self_policy`**: assume the opponent plays
  v3_snipe (our older baseline). Cheapest, used by the original
  Phase 2 probe.
- **Tier 1 — `top_tier_mirror_policy`**: assume the opponent plays
  v3.5.1 (our current submitted baseline; aggressive ship sizing
  matching top-10 fingerprints). Better proxy for real ladder
  opponents above μ≈1100. **This is the default.**
- **Tier 2 — `trained_logreg_policy`** (placeholder, not yet wired):
  a small learned classifier on the 37k labelled launches in
  `data/shot_validator/`. The architecture is reserved; the model
  hasn't been trained.

### Why tiers

Real ladder opponents are unknown. If we always assume they play
v3.5.1, our simulations under-estimate stronger opponents
(top-10 sized fleets are bigger, more aggressive). Tier 2 would
break that assumption but needs offline training work.

### Files

- `lib/opp_model.py` — three policies + a `make_opp_policy(tier)`
  dispatcher.
- `tests/test_opp_model.py` — 6 sanity tests.

## Layer 3 — the chooser (`lib/v7_search.py`)

### What it does

On every turn:

1. **Build the incumbent.** Compute the action our existing
   heuristic (v3.5.1) would emit. This is the parity floor — if
   nothing else looks better, we play this.
2. **Generate candidate moves.** A list of plausible alternatives
   to the incumbent (e.g., drop one of the launches).
3. **Score each candidate.** For each candidate, simulate the next
   10 turns with `fast_sim` — we play the candidate on turn 0, the
   opponent plays its mirror policy, both play mirror policies
   thereafter. Compute "our ships − their ships" at the simulated
   final state.
4. **Pick the best.** Return the candidate with the highest score.
   Watchdog: stop scoring at 700 ms; if we run out, return whatever
   was best so far.

### The candidate enumerators built

- `drop_one` — incumbent + each "drop one launch" variant. The
  proven winner (v7_0_drop_one).
- `target_swap` — for each owned source, try the runner-up snipe
  target. Tested, NEUTRAL/FAIL at 12 seeds.
- `ship_sweep` — vary the ship-fraction per launch (0.5 / 0.7 / 0.95).
  Tested, FAIL.
- `archetype` — four sibling action bundles under preset playstyles
  (concentrated / saturation / defensive / baseline). Tested, FAIL.
- `hungarian` — global bipartite source→target assignment. Tested, FAIL.
- `combined` — union of all of the above. Tested, NEUTRAL.

The winning one is `drop_one`: the others propose candidates that
DIFFER from the incumbent, and the K=10 rollout can't reliably tell
them apart from noise. Drop-one wins because it only ever **removes
a bad launch** — the worst case is parity with v3.5.1.

### The variants tested

| Variant | Layer 3 design | Result vs v7_0 |
|---|---|---|
| v7_0_drop_one | drop_one + Tier 1 opp + K=10 | (baseline) |
| v7.1 minimax | 2×N maximin matrix + symmetric scoring | FAIL (-54 pp) |
| v7.5 combined | drop_one + σ-equiv + recapture + 4P-aware | FAIL (-8.3 pp) |
| v7.6 (recapture off) | drop_one + σ-equiv + 4P-aware | PENDING |

### Files

- `lib/v7_search.py` — `enumerate_candidates`, `score_candidate`,
  `choose`, `choose_simple_with_4p`, etc.
- `tests/test_v7_search.py`, `tests/test_v7_1_sigma_equiv.py`,
  `tests/test_v7_4_4p.py` — 23 tests covering chooser invariants.

## How to use this stack for next-step planning

### The basic recipe (already implemented)

```python
# inside agent(obs, configuration):
from lib.v7_search import choose_simple_with_4p

return choose_simple_with_4p(
    obs, configuration,
    K_2p=10,         # how many turns to simulate ahead in 2P
    K_4p=8,          # 4P games — slightly shallower due to 4-seat cost
    wallclock_ms=700.0,
    include_recapture=True,   # off in v7.6 bisect
    value_fn=None,            # default: "our ships − their ships"
)
```

### Knobs to turn

- **Depth (`K`):** how many turns ahead to simulate per candidate.
  More depth = better discrimination but fewer candidates per turn.
  Current default 10 for 2P, 8 for 4P. Phase 2 audit showed AUC ≈
  oracle at K=50, but that's only affordable with cheaper rollout
  policies or learned value heads (see "Next steps" below).
- **Wallclock budget (`wallclock_ms`):** when to stop searching.
  700 ms leaves 300 ms margin under the 1-second turn limit.
- **Opponent tier:** Tier 1 (v3.5.1 mirror) by default. Swap for
  Tier 0 if you want bit-parity with the older Phase 2 probe.
- **Scoring head (`value_fn`):** the function that turns the
  simulated final state into a single number. Default = ship-delta;
  alternative = production-share (`evaluate_value` in
  `lib/lookahead_planner.py`). Different heads reward different
  outcomes (more ships vs. more territory).
- **Candidate set (`enumerator_mode`):** `drop_one` is the only one
  that's ever passed a gate. The others are kept for future
  iteration once we have better discrimination at the leaf.

### Ways to extend the stack (ranked by expected return)

**1. Cheaper rollout policy → bigger K.**
The Tier-1 mirror is ~4 ms per call. Replace it with a stripped-down
"nearest enemy, send half garrison" policy at ~0.5 ms per call, and
the same 700 ms budget supports K=30-40 instead of K=10. Deeper
horizon → better predictive power. Phase 2 audit showed AUC=oracle
at K=50.

**2. Multi-opponent sampling (PIMC).**
For each candidate, run 3 rollouts under different opponent
assumptions (Tier 0 + Tier 1 + a "passive" no-launch policy),
average the scores. Picks robust moves over fragile ones.
Cost: 3× more rollouts per candidate.

**3. Trained value head.**
A small neural network that takes a state and outputs the predicted
"our ships − their ships" 50 turns later. Replace the K-step
rollout with one step + a value lookup. Per-candidate cost drops
from ~80 ms to ~5 ms — opens room for 100+ candidates per turn or
depth-3 tree search.

**4. Defensive lookahead.**
A separate use of the same simulator: each turn, run ONE forward
simulation of "what if I do nothing" under the Tier-1 opponent.
Identify which of my planets fall in the simulated future. Force
the chooser to include "send reinforcements to that planet" as a
candidate. Addresses the "elimination by step 158" loss pattern
that the games-analysis audit flagged as our dominant failure mode.

**5. Real depth-2 tree search.**
Today the chooser is 1-ply: "evaluate each of my moves; pick the
best." Depth-2 would evaluate "my move → opp's likely response →
my best follow-up." Width-3 beam at K=5 leaf rollouts fits the
budget. Maximin overlay at depth 2 didn't work at depth 1 (v7.1
FAIL); a properly-narrow depth-2 might.

## What this architecture is NOT

- **Not a tree search agent** (yet). It's 1-ply: evaluate each
  candidate move under one opponent policy, pick max. No
  recursion into the opponent's response.
- **Not a learned agent.** Every score comes from a forward
  simulation under hand-built policies, not from a neural network.
- **Not a 4P real-game-theory agent.** In 4P, we still just simulate
  3 opponent mirrors and pick our best; no coalition analysis,
  no leader-detection beyond a flat ×1.5 multiplier.
- **Not optimal at the leaf.** The Phase-2-validated AUC of 0.95 at
  K=50 measures "predictive power" (can we read who wins from the
  rollout?). It does NOT mean "acting on the prediction makes us
  win more" — the rollout assumes both players play heuristically,
  and our chooser picks the best move under THAT assumption. Real
  ladder opponents may exploit moves the rollout rated highly.

## Why this matters strategically

Before this session, the project's strongest agents were
hand-tuned heuristics (v3_snipe, v3.5.1, v7_minimax). They top out
around μ=1063 (live) on the ladder. The structural-weaknesses audit
identified ten specific failure modes that no further heuristic
tuning would fix — they're built into the action space being
considered.

This architecture turns a heuristic ("send fleet to highest-ROI
target") into a **veto layer** ("the heuristic suggests these N
moves; the simulator overrules any that look bad in 10 turns").
That's enough to clear v7_minimax in local A/B (79.2% Wilson lo
59.5%, predicted +20-40 μ on the ladder). It's also the substrate
that every richer technique builds on:

- A learned value head replaces the K-step rollout but leaves the
  chooser shape unchanged.
- Depth-2 minimax recursively calls the same chooser.
- Defensive lookahead is a sibling consumer of `fast_sim`.
- 4P-aware extension just changes which `score_candidate_*` is
  called, not the architecture.

The architecture is the load-bearing artifact, not v7_0 itself.
Variant outcomes (v7.1 FAIL, v7.5 FAIL, v7.6 PENDING) churn over
sessions; the substrate stays.
