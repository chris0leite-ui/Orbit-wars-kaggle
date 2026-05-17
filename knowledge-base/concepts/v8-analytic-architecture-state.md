# v8_analytic — what it is, what we learned, what to build on

> Permanent reference for the PI second-brain. Written 2026-05-17
> after the JAX value head was diagnosed as a structural dead-end
> and the value head was replaced with a fast_sim + reactive
> lite_greedy rollout. Plain English, no abbreviations.
> Lives on branch `claude/space-fleet-physics-engine-lrLE6`;
> this doc captures the architecture decisions so they survive
> the merge into `main`.

## The one-line version

v8_analytic is **an offline-simulation beam-style agent** that
enumerates every plausible fleet launch this turn, scores each one
by running a short "what if I do this, what does the board look like
in K turns" simulation, and picks the best non-conflicting set. After
five sessions of dead-end tuning, the value-head got replaced with
the same Python rollout v8_scavenge uses, and that move turned the
architecture from "blind to most actions" into "tunable substrate
worth building on".

## The agent's decision cycle each turn

1. **Build a list of candidate atomic launches.**
   For every planet I own with spare ships, every owned planet
   considers up to 8 nearest non-mine targets at 50 % and 100 %
   ship fractions, plus defensive reinforces aimed at any of my
   own planets a threat is heading for. (Code: `enumerate_capped`
   in `lib/foundation/strategies/analytic_score.py`.)

2. **Filter cheaply down to a manageable set.**
   Each candidate gets a quick approximate score (capture credit
   vs. bounce-and-die-in-combat vs. defend-against-incoming),
   ranked, and the top N are kept. N currently = 25.
   This step is identical to v8_scavenge's "cheap pre-rank" — the
   same formula, same constants. Stays well-tuned; do not touch
   without re-reading why each weight is what it is.

3. **Score each remaining candidate by simulation.**
   For each candidate, take a fast copy of the current board state,
   apply MY launch on turn 0 plus whatever the opponent's
   greedy-nearest policy would do on turn 0, then keep simulating
   for K more turns with the opponent continuing to react each step
   and me sitting idle. After K turns, evaluate the position with
   the F1+F2 favor formula (my ships minus opponent's, plus a
   present-value-discounted multiple of (my production minus
   opponent's production)). Subtract the score we'd have got if I
   had idled the whole time. The result is "how much better is the
   board if I make this launch versus do nothing this turn?".
   (Code: `score_candidates_fastsim` in `analytic_fastsim.py`.)

4. **Greedy non-conflicting selection.**
   Sort candidates by their delta score, keep only positive ones,
   take the highest-scoring, then the next-highest that doesn't
   use the same source planet or aim at the same target planet,
   etc. Cap at 20 launches per turn (the env limit). Pre-committed
   waves from past missions go first, locking out their source
   planets so we don't double-launch from them.

5. **Update the mission/chainer memory** and emit the action tensor
   for the env to apply. (Unchanged from before this session.)

## The structural failure we diagnosed and fixed

The previous value head ran the same kind of K-step simulation but
in JAX with vmap over candidates, K=8, and the formula
`my_ships_total + my_production × time_remaining` evaluated at the
end of the rollout.

The diagnosis (`/tmp/micro_trace.py`, mid-game state seed 1 turn 80,
40 candidate launches):

- **0 candidates** scored better than doing nothing
- **38 of 40** scored EXACTLY the same as doing nothing (zero
  difference to seven decimal places)
- only 2 — the ones whose fleets arrived inside the K-turn window
  AND lost the resulting combat — scored worse

The cause was a quirk of "my ships" accounting: ships in flight
count toward "my ships" just like ships sitting on planets. So if I
launched a fleet, the source planet's ship count dropped by exactly
the amount the in-flight fleet gained. Net zero. The simulation
wasn't run long enough for the fleet to LAND, so my production also
didn't change (the captured planet wasn't yet mine at the end of
the rollout). The scoring formula at K=8 was, for every candidate
whose fleet arrives at turn 9 or later, **mathematically identical
to the no-action case**.

Median fleet ETAs in mid-game are 10-30 turns. K=8 caught almost
none of them. The agent therefore picked "do nothing" almost every
turn — measured at 19% active turns versus the nearest baseline's
49% — and got out-expanded.

This was invisible to every single tuning experiment we tried
across five sessions (eight different cap values, three different
opponent-mirror variants, two beam widths, plus other knobs)
because none of them touched the scoring formula. They all
operated on what candidates entered the value head or on what the
opponent did inside the K-window — never on the way the K-step end
state was being scored.

**The fix that mattered:** stop using the JAX scoring formula
entirely. Use the same fast-Python simulation v8_scavenge uses,
with the opponent re-deciding their action against the rolling
state at every step (rather than picking once and committing), and
score the K-step end state with v8_scavenge's exact F1+F2 favor
formula. K is now 15 (we measured: at K=15 about 5 of 40
candidates score positively versus 1 at K=8).

## Where the architecture stands today

After the pivot:

- **Versus nearest (the trivial baseline) at 8 trials:** 4/8 wins.
  Identical to the previous tuned-JAX baseline in aggregate; the
  set of seeds we win on changed. We now win seed 1 (which we
  lost before — predicted in advance from the K-sweep
  measurement) and lose seed 0 (which we won before). The losses
  are longer "out-produced over a full game" losses rather than
  the previous "eliminated by turn 150" pattern.

- **Versus v7_0 at 8 trials:** 0/8 wins. Regressed from the
  previous tuned-JAX baseline of 2/8. Two consistent
  interpretations: K=15 is still too short for v7_0's
  longer-payoff plays (the simulation can't see the captures land
  before evaluating), AND the "greedy-nearest" model we use for
  the opponent inside the rollout is wrong for v7_0 specifically
  (v7_0 plans further ahead and behaves differently).

- **Per-turn time:** typical 100-300 ms, p95 around 500-700 ms,
  one outlier per 4500 turns above 1000 ms. About 3× cheaper per
  turn than the JAX version we replaced. Leaves about 300 ms of
  headroom under the Kaggle 1000 ms budget.

The architecture is **not currently competitive** in absolute
terms — v15 on the live ladder is around 1117 mu, v20 around 1107,
v8_scavenge around 1089, and the v8_analytic on this branch has
never been submitted because its quality has never cleared the bar.
The decision to KEEP this branch is **not** about its current win
rate. It's about whether the substrate is buildable-on.

## Why we decided this substrate is worth building on

Three independent pieces of evidence that the new value head is
**responsive to interventions in a principled way**:

1. **The horizon knob works the way the math predicts it should.**
   At the diagnosed-broken state, increasing K from 8 to 15 to 25
   to 40 produces 1 → 5 → 8 → 9 candidates scoring better than no-
   op. Monotone. Predictable. The old JAX path had zero positives
   at every value we tried.

2. **A measured-on-one-state prediction held up in real games.**
   The K-sweep predicted that increasing K from 8 to 15 would let
   the value head see at least some captures land. When we ran 8
   full games at K=15, seed 1 specifically (the one we'd most
   recently used the value head to inspect) flipped from 0 wins to
   2 wins out of 2. The mechanism we identified was the
   mechanism that moved the bench.

3. **Timing has headroom for ambitious next steps.**
   ~300 ms of unused budget at K=15 with 25 candidates. Enough room
   to run a small beam search instead of greedy, OR to push K to
   25-40 in light states with a wallclock guard, OR to use a more
   expensive opponent model — without redesigning the substrate.

Contrast with the previous version: five sessions, eight tuning
knob settings, the same 50% win rate against the trivial baseline
every time. That's an architecture stuck at a structural floor.
The new one has a clear gradient and visible levers.

## What to build on it next, ordered by expected value per hour

1. **Wallclock guard plus longer K.** Before each candidate's
   rollout begins, check elapsed wall time; if we're close to
   running out of budget, stop and pick the best score so far.
   With that safety in place, K can be raised to 25-40 in
   light early-game states (when fewer fleets in flight means
   each simulation step is cheaper) and held at 15-20 in mid-game.
   v8_scavenge uses exactly this pattern. About 30 lines of
   Python; expected to lift quality against v7_0 specifically
   (the regressed opponent), since v7_0's payoffs are longer-
   horizon than nearest's.

2. **Beam search over fast_sim.** The per-candidate cost is now
   small enough (~12 ms warm, K=15, mid-game) that a 2-level
   3-wide beam — i.e. consider compound two-launch action sets —
   is affordable inside budget. The infrastructure exists
   (`beam_search.py` was orphaned by the pivot but is functional);
   just point it at the new scorer.

3. **Add "wait then fire" candidates to the enumerator.** Right
   now we only consider launches that fire THIS turn. v8_scavenge
   also generates "if I wait N turns and then fire from a planet
   that will have accumulated more ships by then, this becomes
   feasible" candidates. About 80 lines; copies a working pattern
   from v8_scavenge `agents/v8_scavenge/main.py:316-375`.

4. **Stronger opponent model inside the rollout.** Swap
   `lite_greedy_policy` (which models the opponent as nearest-
   greedy) for `top_tier_mirror_policy` (which models them as a
   smaller copy of us). Already exists in `lib/opp_model.py`;
   single-line change. Costs roughly 10× more per call, so
   requires K or N to drop to fit budget. Expected to specifically
   help against stronger opponents like v7_0 whose behavior
   lite_greedy mispredicts.

5. **Learned value head.** Train a small neural network on
   (state-after-rollout-step-k, F1+F2 favor) pairs sampled from
   actual fast_sim rollouts. At inference time the agent calls the
   network instead of running the rollout — turns the K-step
   simulation into a constant-time lookup. Lets us effectively
   evaluate at K=80 or beyond. Largest implementation effort,
   biggest potential lift.

Item 1 is the single most important first move, because the v7_0
regression is the architecture's biggest visible weakness right
now and the diagnostic above suggests longer K is exactly what
the value head needs to see those games' payoffs.

## Files to know

- `lib/foundation/strategies/analytic_fastsim.py` — new this
  session. Holds the F1+F2 favor leaf, the per-candidate fast_sim
  scorer, and the greedy non-conflicting selection.
- `lib/foundation/strategies/analytic_score.py` — atom
  enumeration plus the cheap pre-rank formula. Untouched logic;
  the only change was adding an option to return
  (atom, target-id) tuples for the greedy selector.
- `lib/foundation/strategies/analytic.py` — the `AnalyticStrategy`
  class. Now calls into `analytic_fastsim` at the two places that
  used to call `beam_search`. The mission and chainer framework
  around it is unchanged.
- `lib/fast_sim.py` — pre-existing fast Python simulator.
  Reused; no changes.
- `lib/opp_model.py` — pre-existing opponent policies.
  Currently using `lite_greedy_policy`; can swap to
  `top_tier_mirror_policy` for stronger opp model.
- `lib/foundation/strategies/beam_search.py` — was the JAX-
  scoring beam. Orphaned by the pivot but still on disk; works
  as a structure, will be re-attached to the new fast_sim
  scorer if we decide to do beam-over-fastsim.

## The audit and postmortem records

- `audit/2026-05-17-v8_analytic-fastsim-pivot.md` — session
  record including all the numbers cited above.
- `audit/2026-05-17-postmortem-fastsim-pivot.md` — decision-
  quality review and three pending rule-promotion candidates.
- `audit/friction.md` 2026-05-17 section — five new friction
  tags including the load-bearing one
  (`K-shorter-than-launch-eta-makes-value-head-blind`).
