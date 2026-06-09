# Reach-frontier v1 — root cause of 0/20 vs baseline

Date: 2026-05-27. Companion to `audit/2026-05-27-rf-v1-b5-triage.md`
(the eval result) and `audit/2026-05-27-rf-v1-triage-share.md` (the
Rule 48 share aggregates).

Single-game trace of `episode-seed0-p0_reach_frontier-vs-p1_baseline-replay.json`
+ profiling + code review. Performance is fine (~36 ms/turn).
**This is not a bug in execution; it is a bug in the doctrine's
operationalisation.** The chooser falls silent for 87 % of turns
because its reward function pathologically zeroes out as the
opponent expands.

## Game shape: what happened

Picked the seed-0 game where reach_frontier was P0 (game ended at
step 265, focal eliminated). Planet count over time:

| Step | focal planets | focal ships+fleet | opp planets | opp ships+fleet | neutral |
|---:|---:|---:|---:|---:|---:|
|   0 | 1  |  10 +   0 |  1 |   10 +   0 | 30 |
|  25 | 3  |  17 +  15 |  2 |   23 +  28 | 27 |
|  50 | 6  |  78 +  25 |  8 |  111 +  44 | 22 |
|  75 | 9  | 244 +  48 | 13 |  247 +  97 | 14 |
| 100 | **4** | 241 +   0 | **17** |  703 +  87 | 11 |
| 150 | 3  | 438 +   0 | 22 | 1963 + 368 | 11 |
| 200 | 1  | 359 +   0 | 24 | 3990 +  33 |  7 |
| 264 | **0** |   0 +   0 | 26 | 5284 + 844 | 10 |

**Activity asymmetry, the dominant fact:**
- focal: 35 launches across 264 turns, **max 1 launch per turn**,
  active only 35 / 264 turns (13 %).
- opp: 153 launches, max 4 / turn, active 100 / 264 turns (38 %).

Opp fires 4.4× as many launches as us. From step 75 onward we
collapse: 9 → 4 planets in 25 turns, then 4 → 1 in another 100.

## The three bugs (single-game trace at the failure steps)

Reproduced the chooser's column set at six key steps. The chooser
emitted 1 move at step 50 and **0 moves at every subsequent
sampled step** (75, 90, 100, 150). Trace excerpt at step 75:

```
step 75: sources=9 reach_pairs=21 cols=29 pos_cols=0 emitted=0
  top cols all have hold=0.0 and value=-0.9 to -1.3 (loss-only penalty)
```

### Bug 1 — primary — `hold = max(0, ρ_opp − ρ_me)` collapses to 0 mid-game

At step 75 every column has `hold = 0`. The chooser is silent for
87 % of the game because of this.

Mechanism, step by step:

1. `WorldModel.time_to_enemy_threat(p, me)` returns the **worst-case**
   minimum over (in-flight + every opp source) of opp's reach to p.
   Once opp owns many planets, the *nearest opp source* to any
   target on the board is close → ρ_opp small.
2. Our ρ_me for the same target includes our actual source-recovery
   cost `+ ships / our_source's_production` (typically `+12/1 = +12`).
3. So ρ_opp = 8 ticks, ρ_me = 25+ ticks, `hold = max(0, 8 − 25) = 0`.
4. With hold = 0, reward = `0 − λ_loss · expected_garrison = −0.9` —
   negative.
5. The lp.py diagonal noop column has cost 0 (i.e. "do nothing").
   The Hungarian picks noop over every negative-value pair column.
6. We emit nothing. Opp keeps expanding. Their ρ_opp shrinks
   further. **Death spiral.**

The empirical fingerprint is unmistakable: hold = 0 for every
reachable target at step 75+, while our garrisons sit at 244 → 438
ships idle.

### Bug 2 — Hungarian forbids multi-source-on-target (no gang-up)

Confirmed by reconstructing the Hungarian at step 50:

```
Step 50: 4 columns
  src=18 -> tgt=33 k=9  val=17.73
  src=18 -> tgt=35 k=18 val=-0.60
  src=22 -> tgt=33 k=12 val=17.73
  src=22 -> tgt=19 k=24 val=-1.30

Hungarian assignment:
  row 0 (src=18) -> tgt=33 val=17.73
  row 1 (src=22) -> tgt=33 val=17.73     <-- SAME target

extract_moves emitted: 1 moves   <-- second pick dropped by `used_tgts`
```

Both sources want to fire at the same high-value target. Hungarian
picks both. But `extract_moves` in `lib/joint_solver/lp.py:151`
drops the second via `if tid in used_tgts: continue` — a doctrine
§4 design choice to forbid same-target gang-up. The losing source
(src=22) falls back to noop because its only other positive option
collides too.

Result: in turns where the "good" captures are concentrated on one
or two targets, multi-source sits idle. The single-launch-per-turn
average even at peak production is the smoking gun.

### Bug 3 — no defensive launches (own planets aren't candidates)

In `agents/reach_frontier/main.py:75`:
```python
targets = [p for p in planets if int(p.owner) != me]
```

Owned planets are EXCLUDED from the target set. Combined with bug 1
silencing offensive launches, this means once opp's fleets land we
have no way to reinforce. Doctrine §6 said defend would "fall out
of the same objective via the noop slot." Empirically it doesn't —
the noop slot at cost 0 only means "don't launch this source," not
"launch this source defensively at one of my own planets to
reinforce its garrison." We never enumerate the reinforce option.

### Bug 4 — asymmetric source-recovery hurts ρ_me

`opponent_reach.py:_max_opp_production` divides opp's ship-recovery
cost by opp's STRONGEST production (currently always 3 if opp owns
a +3 planet, common from mid-game on). Our ρ_me uses our actual
source's production (typically 1). So opp's recovery cost ≈ 1/3 of
ours per ship. This inflates opp's apparent speed by 2-3× on
ρ-comparison. Subordinate to Bug 1 but reinforces the same
direction.

## Profile (rules out perf as a confound)

Ran `cProfile` on 30 turns of self-play:

- Total: 2.39 s for 60 agent calls = **40 ms / call avg**.
- Top cost: `predict_relative` (used by `kinematic_table.begin_turn`)
  at 692 ms cumulative. Expected — orbital positions for 32 planets ×
  500 leads is what the cache builds.
- `predict_fleet_fate`: 125 calls in 30 turns = 4/turn. Pre-filter
  works.
- `aim_orbiting` + `search_safe_intercept`: 2232 calls. Reasonable.

p95 turn time stays well under 1000 ms in self-play. **The agent is
not slow.** The agent is silent.

## Why the doctrine fails as written

Doctrine §4's reward `R = p̃·hold − λ_loss·losses` ASSUMES hold can
be positive. The doctrine implicitly models a *defended* hold-time:
after you capture, you garrison the planet and opp's recapture is
delayed by your remaining defenders. But the formula uses ρ_opp as
the FRESH opp reach against the planet's CURRENT garrison, ignoring
that we'd leave (k − g) ships behind after capture.

Even more fundamentally: in a positional game where opp has more
planets, every target is in opp's Voronoi cell. The doctrine's
"win conditions" (mine cells in §5) become geometrically empty.

The audit `2026-05-27-hold-time-empirical.md` showed share-of-
integral discriminates winners from losers at n=92 — but on already-
played games where the focal HAD held planets. The doctrine is
descriptive of WHAT winning looks like; the implementation tried to
make it prescriptive, and the prescription fails because we never
commit to captures whose hold-time is computed worst-case.

## Three concrete fixes (ranked by tractability)

These are the actionable variants. Each is 30-80 LOC; each is on a
different design axis (Rule 37 applies — can run all three before
saturation).

**Fix A — replace ρ_opp-as-deadline with a probabilistic horizon.**
Change `hold = max(0, ρ_opp − ρ_me)` to e.g. `hold = max(remaining /
N_opps, ρ_opp − ρ_me)` where N_opps is the number of opponents
that could threaten this planet. The "/N_opps" floor encodes "opp
has competing demands; in expectation they only commit a fraction
of their reach to any one of our planets." Closest to the doctrine's
intent.

**Fix B — gang-up via `solve_multi_turn`.** lp.py already has
`solve_multi_turn` with `DEFAULT_MAX_CONTESTERS_PER_TARGET = 3`.
Swap `pick_actions` to use that. Doctrine §4 forbid gang-up by
design; the v1 empirical result falsifies that constraint.

**Fix C — augment, don't replace.** Stack reach-frontier on top of
baseline: baseline picks its move set first; we ADD any positive-
reward launches that don't conflict with baseline's (distinct src
AND tgt). Violates doctrine §6 "framework replacement" but is the
lowest-risk path to a positive A/B.

Fix A is my recommendation for first iteration: it preserves the
doctrine's structure but removes the pathological zero-hold case.
~40 LOC change in `hold.py` and one knob (`N_opps` or a
percentile) for calibration. If A doesn't lift past baseline, B
adds gang-up (~30 LOC change in `assignment.py`). C is the
fallback after B if doctrine§6 hasn't gotten us there.

## Status

Diagnosis complete. Three independent fix paths surfaced; no work
done on them yet (Rule 1, Rule 4). Awaiting PI direction.

The doctrine's mathematical foundation remains valid (the n=92
empirical study confirms share-of-integral discriminates winners).
The operationalisation gap is in the per-turn reward function's
worst-case-ρ_opp assumption. **The v1 chooser is not "wrong about
the integral"; it's wrong about how to act when opp has positional
advantage.** Every fix above is a different relaxation of that
assumption.
