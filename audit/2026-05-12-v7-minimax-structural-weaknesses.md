# 2026-05-12 — v7_minimax (#52568317) structural weaknesses

## Why this note exists

PI question: *"Why do we lose some games besides having more ships?
Understand the structural difference by looking at the v7 minimax
submission. Which other systematic weaknesses does it have?"*

This is a diagnostic, not a fix proposal. Every claim below is
backed by source from the parallel branch the submission lives on:
`origin/claude/game-theory-strategy-analysis-0oH4N`.

## What v7_minimax actually is

```
ref:        52568317
file:       v7_minimax.py (81.8 KB)
bundle sha: 1393d32b1f4e691d
pushed:     2026-05-12 06:50:06 UTC
live μ:     1063.0
branch:     origin/claude/game-theory-strategy-analysis-0oH4N
sources:    git show <branch>:agents/v7_minimax/main.py
            git show <branch>:lib/lookahead.py
            git show <branch>:audit/2026-05-12-v7-minimax-submission.md
```

Architecture, verified from `agents/v7_minimax/main.py`:

- **Base = σ-equivariance v1.** lib/planner σ-equiv tie-break +
  lib/geometry `sym_hypot` + score rounding to 6 dp. v3-vs-v3 self-play
  is provably drawn under these patches.
- **Maximin overlay.** Per turn (`agent()`, main.py:207-264): build
  N=2 our-candidates × M=2 opp-candidates, score each cell with
  `score_joint_action_symmetric(env, C[i], O[j], K=3, policy=v3)`,
  then `i* = argmax_i min_j P[i,j]`. Tie-break = lower row index, so
  the v3 incumbent wins all ties.
- **Candidate set.** `C = [v3_incumbent, drop_smallest(v3_incumbent)]`
  (main.py:144-159). `O = [v3_from_opp_POV, drop_smallest(O0)]`
  (main.py:161-178).
- **Symmetric scoring** (lib/lookahead.py:152-185). Averages two
  rollouts (us-as-seat-0, us-as-seat-1) to cancel env's documented
  P1-favoring tie-break. Doubled cost forced K=5→3, with downshift to
  K=2 at 300 ms (main.py:55-60).
- **Hard bail** at 750 ms (main.py:60, 240-261). 4P fallback at
  main.py:212-213 returns pure v3 (no Nash guarantee for n≥3).
- **Deterministic env seed.** `env_from_obs` derives `cfg["seed"]` from
  `obs.step` (commit 7c1e078) so both seats roll out σ-symmetrically.

Local gate (audit/2026-05-12-v7-minimax-submission.md):
v7 vs v3.4 = 6W/0D/2L = 75% W/D over 8 games both-sides; v7 vs
precision_v3 = same. σ-equiv-v1 alone scored μ=1041.4 live, v7 scored
μ=1063.0 — i.e. the maximin overlay contributed **only ~+22 μ** to
the σ-equiv base.

## Part 1 — "We had more ships and still lost"

Source: `audit/2026-05-11-v3-snipe-games-analysis.md` characterized
five distinct v3_snipe loss patterns from 34 live replays. v7 inherits
all five by construction (see §2):

1. **Elimination, not attrition** (lines 9-13 of that audit). 19/20 of
   our 2P losses ended with us at zero planets + zero fleets by step
   158 median. Losses are sudden knockouts, not gradual ship-bleeds.
2. **Recovery deficit** (lines 14-18). 79% of wins lose home at some
   point; 100% of losses do. **Winners recover to median 28 planets
   post-home-loss; losers peak at 6.** The strategic gap is comeback,
   not snipe efficiency.
3. **In-flight volume gap** (lines 19-24). In tied 2P loss phases
   (steps 95-112), opponents held 600-900 ships in flight to our 300-500
   (~2×) — same launch frequency, larger fleets per launch (51.4 vs
   44.6 ships).
4. **Leader-snowball unmapped** (lines 25-29). In 4P, we target the
   leader 58% of the time when we win, 45% when we lose. No leader
   detection beyond a flat LEADER_MULTIPLIER=1.5. **4P live winrate
   35.3% vs 2P 47.1%.**
5. **"One ship too little" bounces** (lines 30-39). 38/518 (7.3%) of
   our bounces were within ±1 of the threshold. v3.3's blanket `eta+1`
   fix regressed (42.2% in 32-seed A/B) because static targets are
   already over-sized by `(r_src + r_target)/v` in the env's ETA
   calculation.

**Why v7's overlay cannot diagnose any of these:**

- The scoring head is `our_ships - opp_ships` at K turns
  (lib/lookahead.py:66-78, `_ship_total_by_owner` sums planet
  garrisons + in-flight fleet ships). At K=3 (or K=2 under
  downshift), almost no fleet has arrived yet — the score is
  dominated by **ships in flight**. Patterns 1, 2, 3, and 5 unfold
  over 30-200 turn windows.
- The search horizon is K=3 turns. Loss decisions happen at step
  ≈158. K=3 ≈ 2% of a typical loss-game length.
- Pattern 5 (one-ship-too-little) is a *single-launch sizing* bug in
  arrival_size. The maximin layer never re-sizes a launch — it only
  picks between v3's incumbent and v3 with one launch removed.
- Pattern 4 (leader-snowball) is structurally outside v7: when
  `_detect_num_players(planets) != 2` (main.py:212-213), v7 returns
  `_v3()(obs)` directly. **Half the ladder is FFA; in those games
  v7 = v3, period.**

The user's question can be answered tersely: **v7 inherits every loss
pattern v3 has, plus three of its own (§2, items d/e/f).**

## Part 2 — Structural weaknesses native to v7

Each item: claim, source, ladder symptom.

### a. Candidate set is a strict subset of v3

`_our_candidates` (main.py:144-159) returns
`[v3_incumbent, _drop_smallest(v3_incumbent)]`. v7 cannot propose any
launch v3 didn't already enumerate. The maximin is a **filter** on v3,
never an **augmenter**.

Whatever class of move v3 misses entirely — proactive garrison
parking, sun-waypoint detour, fleet recall, recapture pre-positioning,
3-source gang-up co-timing — is invisible to v7 by construction.

### b. Opp-model class M=2 is also pure v3

`_opp_candidates` (main.py:161-178) builds the opponent class as
`[v3_from_opp_POV, _drop_smallest(O0)]`. The "worst-case opponent" is
worst-cased over two v3 variants only. Real ladder opponents
(precision_v3, Roman, konbu17 hybrid, bowwowforeach) play outside
the v3 policy class — the maximin guarantee does not apply to them.

**Empirical confirmation** (commit c4b576f,
`audit/2026-05-12-psro-iter1-degenerate.md` on the game-theory
branch): v7 vs precision_v3 = **4-2 over 6 games (67%)** vs v7 vs
v3_snipe = **6-0 (100%)**. The further an opponent sits from the v3
policy class, the smaller v7's edge. Live ledger matches: σ-equiv
alone was 1041.4; v7 = 1063.0; maximin lift = +22 μ against a diverse
ladder (vs 6× wider local margins against v3-class opponents).

### c. K=3 is two orders of magnitude below the strategic horizon

`K_INIT=3` (main.py:57), downshifts to `K_FALLBACK=2` at 300 ms
(main.py:58-59, 254-255). Reference horizons:

- v3 WorldModel arrival-ledger horizon: 250 turns
  (lib/world_model.DEFAULT_HORIZON).
- Median loss-game length: ≈158 steps.
- Median home-loss step in our wins: home is lost then recaptured
  over ~30-60 turns.

A 3-turn window cannot see capture chains, recapture windows, gang-up
arrival co-timing, comet-spawn boundary effects, or the in-flight-
volume race (pattern 3) that decides ship-rich losses.

### d. Scoring head amplifies the elimination pattern

`_ship_total_by_owner` (lib/lookahead.py:66-78) sums:

```python
for p in observation.get("planets", []):
    totals[owner] += p[5]               # planet garrison
for f in observation.get("fleets", []):
    totals[owner] += f[6]               # in-flight ships
```

After K=3 turns from a turn-0 launch, **the launching player's
in-flight pile is at maximum** (nothing has arrived to die yet). A
candidate that launches more this turn always shows a larger ship
count at K=3 than a candidate that holds garrison. So:

- **Maximin systematically prefers the more-aggressive candidate.**
- **Garrison-retention is unrepresented in the scalar.**
- **Bounce-on-arrival risk** (pattern 5) is unrepresented — the
  fleet's still in flight at the scoring horizon.

This *amplifies* pattern 1 (elimination over attrition) and pattern 2
(losers peak at 6 planets) instead of correcting them.

### e. Rollout policy = v3 for both players after the joint first move

`score_joint_action` (lib/lookahead.py:113-150) forces both first-turn
actions, then steps K-1 more turns with `policy=v3` for **both seats**:

```python
for _ in range(max(0, K - 1)):
    a0 = policy(clone.state[0].observation)
    a1 = policy(clone.state[1].observation)
    clone.step([a0, a1])
```

The M=2 opp class therefore only differentiates on **turn 0**. Rounds
1 and 2 are identical v3-vs-v3. Effectively v7 is "best response to a
v3 continuation under two opening-move scenarios," not multi-step
minimax against a multi-step opponent.

### f. Symmetric scorer doubles cost → budget-bail can collapse v7 to v3

`score_joint_action_symmetric` (lib/lookahead.py:152-185) calls
`score_joint_action` twice (us-as-seat-0, us-as-seat-1) and averages.
Doubled cost is why K dropped from 5 to 3.

The budget protocol (main.py:240-261):

```python
for i in range(N):
    for j in range(M):
        elapsed_ms = (time.monotonic() - t0) * 1000.0
        if i > 0 and elapsed_ms > HARD_DEADLINE_MS:
            break
        if i > 0 and elapsed_ms > DOWNSHIFT_MS and K == K_INIT:
            K = K_FALLBACK
```

When env.step latency spikes (mid/late game, many planets and fleets),
rows 1+ skip → those `unfilled` cells become −∞ in `_maximin_pick`
(main.py:182-205) → row 0 wins by tie-break → **v7 plays v3
incumbent**. The minimax layer fails *exactly when the turn is busy
enough to matter*.

### g. Drop-smallest is the only defensive lever

`_drop_smallest` (main.py:98-117) removes one launch by smallest
ship-count, ties broken by index. The candidate set is therefore:

- v3 incumbent
- v3 incumbent minus its single smallest launch

If v3 is wrong in its **largest** launch (e.g., over-committing to a
long-distance snipe that will bounce), drop-smallest cannot represent
the better alternative. If v3 is wrong in **two** launch directions,
drop-smallest can only fix one. There is no "garrison-more,"
"redirect," "recapture-pivot," or "skip-this-turn" candidate.

### h. 4P/FFA falls back to pure v3 — half the ladder

main.py:212-213:

```python
if _detect_num_players(planets) != 2:
    return _v3()(obs)
```

In any 3P/4P game v7 plays exactly v3. Live 4P winrate for v3 is
**35.3% vs 2P 47.1%** (state/current.md). Approximately half of
Orbit Wars ladder games are 4P FFA — v7 has zero minimax presence
there. The PSRO follow-up (audit/2026-05-12-psro-iter1-degenerate.md)
flagged this as an explicit limitation.

### i. Comet-spawn fidelity gap near boundary steps

Commit 7c1e078 fixed σ-equiv self-play by deriving
`cfg["seed"]=obs.step` in `env_from_obs`. Both seats now agree on
comet RNG → σ-symmetric rollouts. **But that seed is not the live
env's seed.** Comets that actually spawn in the next K turns at
steps 50, 150, 250, 350, 450 may differ between Sim<K> and reality.
Decisions made within K-1 turns of a spawn boundary are scored
against a fictitious comet sequence.

The submission audit itself flags this: *"the single fidelity gap is
future comet spawns (steps 50/150/250/350/450) which use the env's
RNG"* (lib/lookahead.py:1-25 docstring).

### j. Indifference resolves to v3 — most turns are indifferent

`_maximin_pick` (main.py:182-205) breaks ties by lowest row index, so
row 0 (v3 incumbent) wins all ties. The submission audit notes:
*"On ~95% of turns the maximin layer doesn't differentiate (no ties)
and v7 plays as σ-equiv."* Combined with item (d)'s scoring-noise
(a 2-rollout average over a 3-turn ship-delta is noisy at the
scale of single-launch differences), many of the "non-trivial"
5% are within-noise indifferences resolved back to v3 anyway.

## Part 3 — Inherited-from-v3 weaknesses v7 cannot fix

By construction (rollout policy = v3, both candidates derived from
v3, both opp models v3), every structural gap in v3 is also in v7:

- **No opponent intent / queued-launch strategy modeling.** WorldModel
  reads the arrival ledger (lib/world_model.py:84-97) but extracts no
  strategic signal — "are they committing to one planet or
  dispersing?" is opaque.
- **Single-hop planet timeline.** `simulate_planet_timeline`
  (lib/world_model.py:100-136) stops at the first ownership flip — a
  planet that goes us→them→us can only see the first transition.
- **Comet ETA against snapshot position.** `propose_snipe_missions`
  computes `d = math.hypot(t.x - src.x, t.y - src.y)` (lib/missions/snipe.py:206)
  to the comet's *current* position, not its future intercept point.
- **No fleet recall / abort.** Once `realize()` is called, the fleet
  is committed (lib/intent.py, lib/mechanism.py validate). World-state
  shifts during flight strand the fleet.
- **No proactive garrison parking.** `reinforce` fires only when
  loss is predicted within horizon=250 (lib/missions/reinforce.py),
  not preemptively at neutral disputed planets.
- **No sun-waypoint routing.** `sun_avoid` (lib/mechanism.py) drops
  intents whose path crosses the sun; it doesn't reroute via a
  waypoint planet.
- **ROI formula assumes solo capture.** `score = priority * value /
  (base_ships + d + AIRTIME_PENALTY_WEIGHT * eta + 1.0)`
  (lib/missions/snipe.py:262-263) with `value = production *
  time_to_hold = production * (500 - step - eta)`. Assumes we own the
  target the whole time; ignores enemy pre-capture during our flight.
- **4P leader logic = flat ×1.5** with no margin/coalition/phase
  awareness (lib/missions/snipe.py:55 + 200-201).
- **End-game (>step 470) no explicit pivot.** `ENDGAME_NEUTRAL_BONUS=1.0`
  (lib/missions/snipe.py:93-94, disabled-by-default identity).

## Part 4 — Where v7's actual lift came from

σ-equivariance v1 alone scored μ=1041.4 (submission #52565034). v7
scored μ=1063.0. The +22 μ delta is small relative to the audit's
predicted range of 1040-1080 and is consistent with §2 a-j:

- Narrow candidate class N=2.
- Narrow opp class M=2.
- Shallow K=3.
- Aggression-biased scoring head.
- v3 continuation policy.
- Half the ladder (4P) gets no minimax.

**Most of v7's score is the σ-equiv tie-break + sym_hypot +
score-rounding patches** at `lib/planner.py` + `lib/geometry.py` on
the game-theory branch. The maximin overlay is a small additive
contribution, not a paradigm shift.

## Caveat

This is a code-and-audit-document analysis. We do not currently have
v7's per-episode live replays pulled locally — `audit/live-episodes/`
contains directories for #52544634 (v3_snipe) and #52532938 (v2) but
not for #52568317 (v7). The "more-ships-still-lost" pattern is
sourced from the v3_snipe replay corpus + the construction argument:
v7 is `σ-equiv-v3 + maximin filter over v3 proposals` → it inherits
v3's loss structure.

A follow-up session could pull v7 replays via the existing Kaggle
pipeline and verify the inheritance empirically. The structural
weaknesses listed in §2 a-j are independent of that — they sit in
the source code regardless of replay evidence.

## Summary — direct answer to the PI's question

We lose games where we have more ships because:

1. **The score is a 3-turn ship-delta**, which biases the maximin
   toward launching more (so we end up with bigger in-flight piles
   that bounce or arrive too late) and against retaining garrison
   (which is how recovery happens — pattern 2).
2. **The 3-turn horizon is blind** to the 30-200 turn loss dynamics
   (sudden elimination by step 158; the 2× in-flight-volume race;
   one-ship-too-little bounces).
3. **The candidate set is v3's incumbent ± a single dropped launch**,
   so v7 can't propose "garrison more" or "recapture" — the very
   moves that win the recovery race.
4. **The opponent model is v3** — so worst-case is worst-case-over-v3,
   not worst-case over precision/Roman/konbu17/bowwowforeach. Against
   diverse ladder opponents the maximin guarantee weakens (PSRO iter
   1: 4-2 vs precision, 6-0 vs v3).
5. **4P games (half the ladder) skip minimax entirely** and inherit
   v3's 35.3% 4P winrate.
6. **Budget-bail under env.step latency spikes collapses v7 to v3**
   exactly when the turn is busiest.

Beyond ship count, v7's other systematic weaknesses are the ten
structural items §2 a-j, plus every v3 weakness in §3.
