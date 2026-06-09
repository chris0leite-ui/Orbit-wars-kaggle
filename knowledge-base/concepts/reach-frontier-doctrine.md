# Reach-Frontier Doctrine

*A mathematical reformulation of Orbit Wars and a candidate framework
replacement for the saturated K=10 rollout chooser.*

Authored 2026-05-27. Status: **doctrine documented, empirical
verification queued.** Decision gates for the empirical test are in §9.

---

## 1. Why this exists

The chooser-family architecture (analytical/heuristic value head atop a
short rollout, with a target-set selector) is structurally saturated at
roughly μ≈1120 on the live ladder. The saturation thought file
(`knowledge-base/thoughts/2026-05-16-chooser-family-saturation.md`)
documents the v9 Nash-inside-the-family observation: nine sequential
variants on chooser features (H17/H19/H21, ten analytical slices,
trajectory_roi, chooser_roi, decay-coefficient sweeps) all landed in the
1100–1150 μ band, with the strongest variant beating the next-strongest
by less than σ. The closed-tracks list in `state/MULTI_BRANCH.md` records
the same shape from the registry side: the *axis* of "score a fixed
target set with a value head" has been thoroughly explored.

The empirical fingerprint of top-10 (`top-performer-strategies.md` §4)
is also stable across our last two ladder pulls and across @yuriygreben's
public analyses: top-10 launches 3.5× more often, opens 6 turns earlier,
empties garrisons, hits enemy planets 2.3× more often, and ends games
with *fewer* total ships than midpack. The gap between our μ≈1120 ceiling
and the top-10 floor of μ≈1500 is about 25 percentile bands; that is the
size of a different architecture, not a different value-head coefficient.

This doc proposes that the chooser ceiling is real because the
*objective function* the chooser optimises (short-horizon ship-delta) is
the wrong objective. The right objective is the time-integral of owned
production over the full game horizon, and the right algorithm — the one
that the top-10 fingerprint reveals — is a Voronoi-style partition of
planets by *reach time* with explicit hold-time accounting.

The empirical verification of the central premise is the precondition
for any chooser-replacement work; gates in §9.

---

## 2. The mathematical core

### 2.1 Terminal score in closed form

Let `P` be the set of planets (static + orbital + comet). Let `T = 500`
be the horizon. For player `i` and planet `p`, define the *occupancy
indicator*

```
1_p^i(t) = 1 if planet p is owned by player i at time t, else 0.
```

The comp environment's `final_score` for player `i` is the sum of (a)
ships in garrison, (b) ships in flight back to their planets, and (c)
production accrued through ownership. The first two are bookkeeping of
the same currency — ships — that gets produced through ownership.
Integrating the third over the game gives the closed form

```
S_i(T) ≈ S_i(0) + Σ_{p ∈ P} p̃_p · τ_p^i − L_i           (★)
```

where

* `p̃_p` is the per-tick production rate of planet `p` (constant per
  planet, available from `step0.observation.planets[p]["production"]`).
* `τ_p^i = ∫_0^T 1_p^i(t) dt` is the *ownership integral*: total ticks
  player `i` owned planet `p` over the game. For a discretised game with
  per-tick ownership steps it is just the count of ticks owned.
* `L_i` is losses: ships sent and destroyed (sun, ties, failed captures,
  combat). This is the only term that breaks the conservation: every
  produced ship is either still alive or destroyed.

Equation (★) is exact up to two finite-horizon effects: comet planets
that expire mid-game cap their τ at the spawn-to-expiry window, and the
last few turns of in-flight ships may or may not arrive by `T`. Both
effects are small relative to the dominant `p̃·τ` mass.

### 2.2 What the rating system actually measures

TrueSkill (`comp-context.md` §"Turn order" + §"TrueSkill") is
margin-agnostic: a 1-ship win and a 100-ship win move μ identically. So
the rating gradient is over the *probability of finishing 1st* (in a 2P
game) or the *expected finish rank* (in a 4P FFA), not over the score
margin. This is the part that often gets miscommunicated — and the part
that does *not* contradict the production-integral objective.

Claim: against a fixed symmetric opponent strategy, the probability that
`S_me(T) > S_opp(T)` is a monotone-increasing function of
`E[S_me(T) − S_opp(T)]`, and the expected score gap reduces to

```
E[ΔS] = Σ_p p̃_p · (E[τ_p^me] − E[τ_p^opp]) − (E[L_me] − E[L_opp])
```

So the integral structure is *the* control surface that moves win
probability. Margin-agnostic rating doesn't make the integral irrelevant;
it makes the integral the thing whose sign matters more than its
magnitude. The chooser still wants to maximise it; it just doesn't get
extra credit for blowouts.

### 2.3 Three structural consequences

**Consequence A — Enemy captures double-count.** Re-write the score
*gap* (the thing TrueSkill actually scores in 2P):

```
ΔS = S_me − S_opp = Σ_p p̃_p · (τ_p^me − τ_p^opp) − (L_me − L_opp)
```

For a *neutral* planet captured by me at time `t_c` and held to `T`, the
contribution to ΔS is `p̃ · (T − t_c)`. For an *enemy* planet captured
by me at `t_c` (with enemy holding from time 0 say), the contribution is
`p̃ · ((T − t_c) − t_c)` — the same `(T − t_c)` swing for me, plus an
equal-magnitude swing against the opponent. In ΔS, an enemy capture is
*literally* worth twice a neutral capture of the same planet at the
same time. This is the mathematical reason behind the top-10 fingerprint
`targets_enemy_fraction = 0.32` vs midpack `0.14`.

**Consequence B — Early captures compound.** The `(T − t_c)` factor is
linear in `t_c`. Capturing a planet of production `p̃` at `t_c = 50`
contributes `p̃ · 450` to my term; at `t_c = 250` it contributes
`p̃ · 250`. This is the math behind `first_launch_step = 4 (top-10)` vs
`10.5 (midpack)`: six wasted turns at the start is six turns of forfeit
production-integral for every planet I will eventually own.

**Consequence C — Idle garrisons earn nothing.** A ship sitting in a
garrison at planet `p` does not enter the production integral; it
contributes only the final-state +1 ship to `S_me(T)`. A ship launched at
`t = 100` that captures a planet of production `p̃` at `t_c = 110` and
holds it to `T` contributes `p̃ · 390` to the integral, vastly more than
the 1 ship it represents in inventory. This is the math behind
`mean_garrison_at_launch = 10.6 (top-10)` vs `22 (midpack)`. The
counterintuitive empirical observation that midpack ends with *more*
total ships and still loses (top-performer-strategies §4, paragraph
"Counterintuitive finding") is fully predicted by (★): the ships in
midpack garrisons are real, but they didn't enter `Σ p̃·τ`.

### 2.4 What this is *not*

This is not a claim that "production-integral maximisation" is a new
objective that no one has stated. It is a claim that (a) the existing
chooser-family value heads are not directly optimising the integral —
they are optimising short-horizon ship-delta as a *proxy* for it, with
finite K=10 leaves — and (b) the gap between our μ and the top-10 μ is
plausibly the difference between proxy-optimisation and direct
optimisation of the integral via reach-time accounting. The empirical
verification in §9 is the gate that decides whether (a)+(b) is actually
the right diagnosis on *this* ladder.

---

## 3. Why the current architecture has a ceiling

### 3.1 Reading the v9 saturation in this language

The v9-family chooser scores each candidate target by a per-leaf value
function evaluated at K=10 ticks of forward simulation. K=10 measures
the ship-delta over a 10-tick window: did I gain or lose ships relative
to opponent over those 10 ticks?

In the language of (★), K=10 leaf scoring measures a *finite-difference*
approximation of the integral, with window length 10 out of a horizon of
500. It is biased toward two failure modes:

1. **Short-window myopia.** A planet captured at `t_c = 100` of
   production `p̃ = 5` contributes `5·400 = 2000` to my integral if I
   hold to `T`, but to a K=10 evaluator the visible reward is at most
   `5·10 = 50` plus the negative ship-investment cost. The evaluator
   sees a *cost*, not the long-run reward.
2. **Symmetric ignorance of hold-time.** K=10 cannot see whether the
   captured planet will be retaken by the opponent at `t = 200` (in
   which case my `τ_p^me` is 100, not 400). The leaf evaluator's
   estimate `+5·10` is a fixed number regardless of whether the hold is
   secure or transient.

The H17/H19/H21 and chooser_roi and analytical-slice-10 sequence of
falsifications fits this pattern: each variant added a *redundant
signal at the same wrong horizon*. The proposers were comparable; the
leaf scoring window was unchanged. The shape of the result table —
9 variants all in 1100–1150, all within σ — is the shape of "we are at
the achievable ceiling of the proxy, and the proxy is wrong."

### 3.2 What top-10 do that K=10 can't see

The closed-form audit from §2.2–2.3 produces a list of behaviours that
fall out of (★) but are invisible to K=10 ship-delta:

* Open early because `(T − t_c)` is the dominant factor.
* Hit enemy planets because captured enemies double-count in ΔS.
* Empty garrisons because idle ships don't enter the integral.
* Avoid comets because their `τ` is capped by spawn-to-expiry length.
* Use multi-launches because parallel capture *parallelises the τ
  accumulation*; a single 38-ship capture seizes one planet, two
  parallel 19-ship captures (if reach-time-feasible) seize two.

Every line on the top-10 fingerprint table is a direct consequence of
(★). The K=10 leaf cannot see any of them because none of them mature
within the 10-tick window.

---

## 4. The reach-time frontier

### 4.1 Reach time

For player `i`, source planet `s`, target planet `p`, and ship count
`k`, define

```
reach_time_i(s, p, k) = (turn of source-meets-target intercept,
                        given a fleet of size k launched now,
                        accounting for orbital motion and sun avoidance)
```

This is exactly what `lib/trajectory.py:predict_fleet_fate` computes
on the substrate side (`lib/trajectory.py:80`). The argument `k` enters
because a *cost*-bearing reach time is `arrival_tick + cost(k)` where
`cost(k)` is the production-recovery time at the source: launching `k`
ships from a planet of production `p̃_s` costs `k` ships now, which the
source recovers in `k / p̃_s` ticks of post-launch production. The full
reach cost is `arrival_tick + (k / p̃_s)`.

Define the *minimum-cost reach time* for player `i` to capture planet
`p` with sufficient force as

```
ρ_i(p) = min_{s, k}  ( arrival_tick_i(s, p, k) + cost_i(s, k) )
         subject to:  k ≥ garrison_p(arrival_tick_i(s, p, k)) + 1
                      arrival_tick_i feasible (physics, sun)
```

In words, `ρ_i(p)` is the earliest tick at which player `i` can secure
ownership of `p`, amortised by the production-recovery time of the source
that paid for it. This is a single number per (player, planet) per game
state.

### 4.2 The Voronoi-style partition

For each planet `p` and each turn `t`, classify:

* `p ∈ MINE(t)` ⇔ `ρ_me(p) < ρ_opp(p) − δ` (I get there first by `δ`).
* `p ∈ OPP(t)` ⇔ `ρ_opp(p) < ρ_me(p) − δ` (opponent gets there first).
* `p ∈ CONTESTED(t)` ⇔ `|ρ_me(p) − ρ_opp(p)| ≤ δ`.

`δ` is a tolerance band; reasonable default is 2 ticks (one combat
round). This is the *reach-time frontier*. In 4P FFA there are four
ρ values per planet and the classification becomes "min ρ across
opponents," but the structure is identical.

### 4.3 Hold time

For a planet `p` I am the reach-winner of, define

```
hold_time_p = ρ_opp(p) − ρ_me(p)
```

This is the number of ticks I expect to own `p` between my capture and
the opponent's recapture, assuming both players play minimum-cost
reach. The *production* I expect to extract from `p` over that hold is

```
extract_p = p̃_p · hold_time_p
```

For planets I'm *not* the reach-winner of, hold_time is negative and
extract is zero (or negative if I throw ships at it and lose them — the
∆L term in ΔS).

### 4.4 Hold-fraction

A natural unit-free version of hold-time, useful for empirical
verification, is

```
hold_fraction_p = hold_time_p / (T − t_capture)
                = (ρ_opp − ρ_me) / (T − ρ_me)
```

`hold_fraction = 1` means I capture `p` and hold it to game end.
`hold_fraction = 0` means I capture `p` and immediately lose it.
`hold_fraction < 0` is the negative-extract case (we shouldn't even be
launching). Top-10's apparent strategy is to choose targets where
hold_fraction is reliably near 1; midpack's apparent failure is taking
plenty of captures with hold_fraction near 0.

§9 is the gate to verify this empirically before any chooser code.

### 4.5 Tying it back to (★)

The chooser's per-turn problem is now stated cleanly:

```
max  Σ_p∈chosen  p̃_p · hold_time_p  − fleet_cost
```

subject to feasibility (one fleet per source, no double-spending) and
the time horizon `T − t_now`. This is an *assignment* problem (planets
to source-fleet pairs) with closed-form per-(s,p,k) reward
`p̃ · hold_time − cost(k)`, solvable by Hungarian in ms.

---

## 5. The chooser, in pseudocode

```
function reach_frontier_chooser(world, focal):
    # 1. Build per-(s,p,k) reach times for me and each opponent.
    #    Discretise k over a small grid (e.g. {0.25, 0.5, 0.75, 1.0}
    #    of source garrison). predict_fleet_fate gives arrival ticks
    #    for the geometric part; cost(k) is k / p̃_s.
    R_me = compute_reach_matrix(world, focal)              # |S|×|P|×|K|
    R_opp = compute_reach_matrix_min_over_opps(world)      # |P| (min ρ)

    # 2. For each planet p, pick the best (s, k) launch for me:
    candidates = []
    for p in world.planets:
        for (s, k) in argmin-grid(R_me[*, p, *]):
            rho_me = R_me[s, p, k]
            hold = R_opp[p] - rho_me
            if hold <= 0: continue                          # opp owns it
            reward = p.production * hold
            cost  = k                                       # ship cost
            candidates.append( (s, p, k, reward - cost) )

    # 3. Filter physics-infeasible candidates.
    candidates = [c for c in candidates
                  if predict_fleet_fate(c).result == "captured"]

    # 4. Hungarian assignment: one fleet per source, planets distinct.
    assignment = hungarian( build_assignment_matrix(candidates) )

    # 5. Emit actions.
    return [ (c.s, c.p, c.k) for c in assignment ]
```

Wallclock budget: closed-form ρ-table build is dominated by
`predict_fleet_fate` calls. For 24 planets and 4 sources and 4 k-values
that is ~400 trajectory calls per turn. At ~100 µs per call (the
substrate has been tuned for this), that is ~40 ms per turn — well
within the 1s/turn Kaggle budget and inside the chooser-family's
typical 50–80 ms working point. The Hungarian on a 4×24 cost matrix is
sub-millisecond.

Components reused:

* `lib/trajectory.py:predict_fleet_fate` — trajectory physics for both
  reach times and physics filter.
* `lib/kinematic_table.py` — per-tick position cache that makes
  repeated reach queries cheap.
* `lib/joint_solver/lp.py:build_assignment_matrix` (`lp.py:47`) — the
  cost-matrix scaffolding already in tree, used by `chooser_lp` in
  baseline.

Components new:

* The ρ-table builder (closed-form; ~80 LOC).
* The hold-time / extract computation (~30 LOC).
* The candidate-emission + Hungarian assembly (~40 LOC).

Total estimate ~150–200 LOC for the chooser, plus a thin agent
wrapper. Calibrated to fit the existing `agents/<name>/main.py` shape.

---

## 6. Why this differs from chooser_roi (which was 0/32)

The chooser_roi line shipped in 5/19 (sub ~52532938 era; mechanism
ledger row 52) tried a related idea and got falsified at n=32 with
zero-percent win-rate against several baselines. It is important to
explain why this proposal is *not* the same idea, so that the
falsification of chooser_roi doesn't bind this one.

Two specific differences:

**A. Finite, opp-aware hold-time.** chooser_roi computed an unbounded
ROI = `production · remaining_game − cost`. It had no `ρ_opp` term:
captures of opp-cell planets and mine-cell planets and contested planets
all scored identically high if the remaining game was long. The result
was a chooser that over-extended into opp territory.

This proposal computes hold-time as `ρ_opp − ρ_me`. Planets in the
opp Voronoi cell get hold_time ≤ 0 and are filtered. Planets in
the mine cell get hold_time up to `T − t_now`. Planets in the
contested band get small positive hold_times and are treated as the
risky middle. The difference is one term in the cost function and one
filter clause, but it is the difference between "ROI ignoring opp" and
"ROI respecting the reach-frontier."

**B. Replacement of the rollout, not a stacked layer.** chooser_roi was
*stacked on top* of the v9 chooser as an additional value signal, in the
spirit of "more is better." It inherited the K=10 leaf, the existing
target-set selector, and the existing per-turn ranking. So the addition
was a re-weighting, not a re-architecting.

This proposal *replaces* the rollout with the closed-form ρ table. No
K=10 leaf. No legacy value head. The doctrine in §2–§4 is the entire
objective; there is no "+ legacy" term. (CLAUDE.md Rule 40: prefer
modeling-correctness to restriction-tuning. This is the modeling fix; a
stacked re-weight on top of the wrong objective is the restriction
tuning equivalent.)

---

## 7. The trajectory_roi precedent — why physics primitives matter

Trajectory_roi (mechanism ledger row 56) is the other near-miss that
this proposal must not repeat. It was falsified for *substrate* reasons,
not strategy: it used a hand-rolled fleet-fate predictor that didn't
import `predict_fleet_fate`, and ended up with ~6.8% physics waste
(fleets dying to sun, OOB, comet-expiry) in single-game traces. By the
time the waste was visible in head-to-head A/B, four submissions had
been burned.

CLAUDE.md Rule 47 was authored from that postmortem and specifies the
pre-flight: before any chooser using geometric reasoning, run a
single-game trace through `predict_fleet_fate` and confirm sun / OOB /
comet-expiry waste is < 2%. This proposal will honour Rule 47 in the
chooser-build phase (Part C of the plan; see §9 gates). The bullet-list
of components in §5 already uses `predict_fleet_fate` directly, so the
substrate path is the safe one by construction.

---

## 8. Failure modes and honest hedge

A doctrine deserves a list of the ways it can be wrong. The strongest
defences are explicit.

**8.1 ρ_opp accuracy.** The hold-time `ρ_opp − ρ_me` depends on a
reasonable estimate of opponent's reach. In 2P this is doable
(symmetric primitive over the opponent's planets). In 4P FFA "min ρ
over opponents" is more uncertain because three opponents have three
different latent strategies. *Mitigation:* a single-opponent panel
(Rule 43) is mandatory before any submit; if 4P ρ_opp estimation is the
weak link it will show up in pooled-FFA panel performance vs single-opp
panel performance.

**8.2 Combat sequencing in the contested band.** Planets where
`|ρ_me − ρ_opp| ≤ δ` are the high-variance zone: both players arrive
within a few ticks and the combat-resolution outcome depends on the
exact ship counts and arrival sequence. The doctrine treats these as
small-positive hold_time, but in practice the variance around the
expected value is large. *Mitigation:* the chooser should de-rate
contested-band rewards by an opponent-model belief, not score them as if
ρ_opp were a deterministic estimate. This is one place the chooser is
not purely closed-form and may need a probability factor.

**8.3 4P kingmaker.** In 4P FFA, the production-integral objective is
the right one for *maximising my expected rank*, but the rating system
rewards 1st place specifically. There are board states where the
production-maximising action helps the *strongest opponent* by clearing
out a weak third, and a rank-2 finish becomes a rank-3. *Mitigation:*
4P-specific de-rating of opp-cell captures where the opp is the
weakest player. Out of scope for the v1 doctrine implementation, but
flagged.

**8.4 Live-ladder calibration risk (Rules 43, 45).** Local A/B against
self or one external agent is not calibrated to the live ladder. The
saturated chooser family clears local A/B convincingly but loses on the
ladder; the same can happen here. *Mitigation:* the chooser ships only
after (a) the multi-opponent panel gate from Rule 43, (b) the n≥32 A/B
gate from Rule 45 against the current rolling champion, and (c) a
Rule 47 physics-waste trace.

**8.5 The hold_fraction signal might not be discriminative on our
ladder data.** The whole framework depends on top-10 actually winning by
having higher hold-fraction on their captures. If our μ≈1100 data shows
that wins and losses have the *same* median hold_fraction, the doctrine
is falsified on this ladder; either (a) the integral isn't the right
discriminator at our band, or (b) the discriminator is more subtle than
hold-fraction alone. §9 is the empirical gate.

**8.6 Doctrine surface-area discipline.** This doc is 450 lines.
CLAUDE.md Rule 9 keeps the rules file lean precisely because long
docs decay. The mitigation is keeping this file frozen as a *concept
reference* and pushing operational state into `state/MULTI_BRANCH.md`
and the audit ledgers; no day-to-day notes should accumulate here.

---

## 9. Pre-registered empirical gates

Before any reach-frontier chooser code lands, the central premise — that
hold-fraction (= τ_p^me / (T − t_capture)) discriminates winners from
losers — gets a single empirical test. Pre-registering the thresholds
here so the verification cannot move the goalposts later.

**Sample:** local in-tree replay caches at `audit/live-episodes/` (five
sub_id dirs, ~21 games each), plus pulled replays from sub 52744856
(team peak, μ=1149.2 composite_a2_hybrid) and the current rolling pair
(52894340, 52893236). Target ~80 games total, spanning the μ=900–1150
band of our own ladder play.

**Measurement:** for each replay, for each (planet, focal-capture-segment),
compute `hold_fraction = hold_time / (T − t_capture)` weighted by
`p̃_p`. Aggregate per-game; report median hold_fraction split by
won/lost.

**Decision gates** (frozen):

| Outcome | Wins-median hold_fraction | Losses-median hold_fraction | Conclusion |
|---|---|---|---|
| Strong | ≥ 0.70 | ≤ 0.45 | Doctrine confirmed. Build chooser as a single ~200-LOC agent (Part C of the plan). |
| Weak-positive | ≥ 0.60 | ≤ 0.50 | Doctrine directionally correct but hold_fraction alone isn't decisive; build chooser with a second complementary signal (likely opp-density of the captured planet's neighbourhood) in the value function. |
| Falsified | < 0.60 | OR Losses-median ≥ Wins-median | Time-integrated production isn't the differentiator on this ladder. Pause the chooser build; surface to PI with the data. |

**Confirmatory secondary metric.** `Σ p̃·τ_p^me / (Σ p̃·τ_p^me +
Σ p̃·τ_p^opp)` separation between wins and losses > 0.15 adds
confidence regardless of the hold_fraction primary.

**No-goalpost-shift clause.** If the data falls between Strong and
Weak-positive, ship Weak-positive. If between Weak-positive and
Falsified, ship Falsified. Tie-breakers go to the more conservative
outcome.

Empirical-verification deliverables, once gates are evaluated:

* `audit/2026-05-27-hold-time-empirical.md` — table of medians per
  cut (won/lost, 2P/4P, sub_id), the gate verdict, and a written
  diagnosis.
* `audit/2026-05-27-hold-time-empirical.json` — raw per-capture rows
  for re-analysis (do not delete; future audits cite).
* Update to `state/MULTI_BRANCH.md` registry with the outcome and a
  pointer back to the audit.

---

## 10. References

* `knowledge-base/concepts/top-performer-strategies.md` §4 — empirical
  fingerprint of top-10 (the behavioural shape this doctrine predicts).
* `knowledge-base/thoughts/2026-05-16-chooser-family-saturation.md` —
  diagnosis of v9 saturation.
* `state/MULTI_BRANCH.md` — closed-tracks registry; chooser-family
  axis closed.
* `state/mechanism-ledger.md` rows 52, 53, 56 — chooser_roi,
  analytical-slice-10, trajectory-roi-without-physics; the three
  near-misses on this same problem.
* `comp-context.md` §"Turn order" and §"TrueSkill" — game mechanics
  and rating model.
* `lib/trajectory.py:80` — `predict_fleet_fate`, the physics primitive
  ρ_i(s, p, k) calls.
* `lib/kinematic_table.py` — per-tick orbital position cache.
* `lib/joint_solver/lp.py:47` — `build_assignment_matrix`, the
  Hungarian scaffolding.
* CLAUDE.md Rules 40, 43, 45, 47 — modeling-over-restriction,
  multi-opp panel, n≥32 gate, physics primitive verification.
