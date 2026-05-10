# Heuristics research — universal strategy primitives for Orbit Wars

> Research-only note (2026-05-10). Source PI voice-dump:
> `knowledge-base/thoughts/2026-05-10-strategy-heuristics-PI-dump.md`.
> Pairs with `docs/strategies/roadmap.md` (v2 → v4) and the
> strategy/mechanism split defined in `lib/intent.py:21-101`.
>
> Goal: enumerate **universal heuristics** that any future agent
> (heuristic, search, IL, RL, hybrid) can plug in. Each section ends
> with a hand-off pointer to the roadmap mechanism (1–7) or the v2/v3/v4
> version that absorbs it.
>
> **Update 2026-05-10:** §K (Empirical evidence) added at the bottom
> after the sister branch's simple-strategy panel produced 8-seed
> verdicts on five target-selection axes — see also
> `audit/2026-05-10-merge-prep-next-experiments.md` for the planned
> next-axis experiments that build on those results.

---

## A. Game model anchor (what every heuristic must respect)

| Aspect | Value | Source |
|---|---|---|
| Map | 100×100 continuous, sun at [50,50] r=10 (fleets crossing it die) | env spec |
| Planets | 20–40, in 4-fold symmetric groups of 4; mix of orbiting (inner) and static (outer) | env spec |
| Production | 1–5 ships/turn per planet, fixed and observable | env spec |
| Players | 2 or 4 | env spec |
| Step horizon | 500 (or last-standing earlier) | env spec |
| Score | `own_planet_ships + own_fleet_ships` at game end | README §win |
| Rating | TrueSkill (μ₀=600); win/loss/draw matters, not margin | comp brief |
| Action | `[from_id, aim_angle_rad, n_ships]` per launch, multiple per step | env spec |
| Fleet speed | `v(N) = 1 + 5·(log₁₀N / log₁₀1000)^1.5`, clamped at 6 | `lib/fleet.py` |
| Combat | Same-step arrivals at planet: group per owner; largest vs second-largest survives; survivor fights garrison; two-way tie destroys both | README §combat |

**Speed implications (memorise these):**

| Ships | 1 | 10 | 100 | 1000 | 10000 |
|---|---|---|---|---|---|
| Speed | 1.00 | 2.00 | ~3.96 | 6.00 | 6.00 (clamped) |

Bundling matters: a 1000-ship fleet covers ground 6× faster than a
1-ship fleet. Combat-mass groups same-step same-owner arrivals — so
"bundling" can be physical (one fleet) or temporal (timed to land
together).

**Compounding implication:** captured planets emit `production × turns_remaining`
ships. Capturing at step 50 yields ~7× the ship stream of capturing
the same planet at step 400. Early-game ROI dominates.

**Architectural anchor:** strategies emit `Intent(src_id, target_id, ships)`;
the mechanism pipeline (`validate → arrival_size → lead_aim →
comet_aim → sun_avoid`) post-processes (`lib/intent.py:21-101`,
`lib/mechanism.py`). Each heuristic below is tagged
**[strategy layer]** or **[mechanism layer]**.

---

## B. Backwards reasoning — deterministic-win predicates

The PI's question: *are there scenarios where we already know we have
won?* Yes — three predicates of increasing cost and increasing power.

### B.1 Trivial-win predicates *(O(1), always cheap)*
- **Last-player-standing:** all enemies own zero planets AND have zero
  in-flight fleets.
- **All-our-planets, no enemy fleets:** all planets owned by us AND no
  enemy fleet objects exist. Score is monotonic from here.

These are diagnostic only — the env will end the episode shortly anyway.
Useful as a **switch to defensive-only mode** to avoid wasted launches.

### B.2 Production-dominance lock *(O(planets), per-step)*

Given `P_us`, `P_them` (sum of production over owned planets), `S_us`,
`S_them` (current ship totals incl. in-flight), `T = 500 − step`:

```
projected_us  = S_us  + P_us  · T
projected_them = S_them + P_them · T
```

If `projected_us > projected_them` AND **no enemy fleet can capture an
undefended planet of ours within T steps**, the game converges to a
ship-count win without further offense — switch to defense-only.

Caveats:
- Enemy can launch from owned planets; "no enemy fleet can reach" must
  account for `dist(any_enemy_planet, our_planet) / v(garrison + P·t)`.
- Captures change `P_us` and `P_them`; refresh per step.

Cost: O(planets²) for the reachability bound; O(planets) for the
production sum. Affordable every step.

### B.3 Reachability cover — forced sweep *(O(planets² log planets))*

For each enemy planet `e`, find the cheapest of our planets `u` that can
deliver `≥ garrison_e(t+Δt) + 1` ships at arrival time `t+Δt`, where
`garrison_e(t+Δt) = current + production_e · Δt − any_incoming_attacker`.
Build the bipartite matching (each of our source planets used once,
each enemy target covered once) — if a feasible cover exists AND the
enemy cannot intercept any of our launches within the same horizon,
**we have a forced sweep plan**: a sequence of launches that
guarantees elimination.

Solver: Hungarian algorithm, O(n³) for n ≤ 40 → microseconds.

This is the strongest predicate and the most useful one — it's not just
a "we've won" detector, it's also the **plan that wins**. Enemy
reactions (counter-launches) reduce its purity to a probabilistic claim
in practice; treat it as a high-confidence guide, not a proof.

### B.4 Hand-off
- v2: surface B.1 / B.2 as **diagnostic telemetry** (log when they
  fire; no policy change).
- v3 / v4: B.3 becomes a mission class `endgame_sweep`. Sits naturally
  in the Roman-1224 mission framework (roadmap mechanism 7).
- Roadmap line: `docs/strategies/roadmap.md:18-44` mechanism 1
  (arrival-time ownership forecasting) is the prerequisite — B.2 / B.3
  consume the same arrival-ledger data.

---

## C. Planet prioritization — target ranking

### C.1 Cost at arrival *(uses arrival ledger)*

```
ships_needed(target, eta) =
    garrison_owner_at_arrival(target, eta)
  + 1                                          # tiebreak margin
  + Σ enemy_reinforcements_arriving_in_[eta-k, eta+k]
```

Where `garrison_owner_at_arrival` is computed by the v2 arrival-ledger
forward simulator: start from current garrison, add `production · eta`
if currently owned by some player at all relevant times, subtract
incoming attackers, etc.

### C.2 Value of capture

```
value(target, eta) = production(target) · max(0, 500 − step − eta)
```

This is the future ship stream the planet emits if we hold it to
game-end. Equivalently: the BOTE compounding number from §A.

### C.3 ROI score (the basic ranker)

```
roi(target) = value(target, eta) / ships_needed(target, eta)
              · time_discount(eta)
```

`time_discount(eta) = 1 / (1 + α·eta)` with `α ≈ 0.02` — late arrivals
are penalised because (i) more enemy reactions, (ii) we lose interest
on the ships if held in flight.

### C.4 Alternative scorings (different agent personalities)

- **Marginal-production-gain:** rank by `Δ(my_total_production)
  / ships_spent`. Equivalent to C.3 in early game; differs late when
  remaining-step factor flattens.
- **Threat-reduction:** value = enemy production *lost* on capture
  (only positive for enemy-owned planets). Aggressive denial agent.
- **Denial value:** value = production opponent would have gained next
  turn if they captured it. Useful for blocking races to neutrals.
- **Centrality / reachability bonus:** add `+β · |reachable_planets_within_K_steps|`
  — favours planets that open up the map.

These alternatives are the seed of the **agent-personality axis** —
different agents in an ensemble can pick different scorings.

### C.5 Risk multipliers

- **Orbit-prediction error:** for orbiting targets, ω·eta has growing
  error as eta grows. Multiply ROI by `e^(−γ·ω·eta²)` to penalise
  long-range orbital snipes.
- **Sun-crossing detour:** if direct line crosses the sun, replace
  `eta` with `eta + detour_cost` (estimate via tangent-line length).
- **Contested-target failure probability:** P[mission fails] estimated
  from incoming enemy fleets; multiply ROI by `(1 − P_fail)`.

### C.6 Hand-off
- C.1–C.3 are the scoring core inside v3 mission classes `snipe`,
  `reinforce`, `recapture` (roadmap mechanism 7).
- C.4 alternatives become **agent variants** for a future ensemble
  (mechanism-ledger family "Ensemble").
- C.5 risk multipliers absorb the work already done by `lead_aim` and
  the planned `sun_avoid` mechanism.

---

## D. N-step-ahead production maximization

### D.1 The compounding insight

A planet of production `p` captured at step `t` contributes `p · (500 − t)`
ships by game-end (ignoring loss). At fixed `p`:

| Capture step | Ship stream emitted by step 500 |
|---|---|
| 50 | 450p |
| 100 | 400p |
| 200 | 300p |
| 300 | 200p |
| 400 | 100p |

Earlier captures dominate by 3–7×. This corroborates Roman-1224's
aggressive early-spread observed in the public top.
**Implication:** the best universal heuristic is "capture as much
production as possible before step ~150."

### D.2 Greedy 1-step formulation (v1.1 baseline)

For each owned planet, pick the best target by C.3 ROI, send
`arrival_size` ships. Already what `arrival_size` mechanism does.
Computation: O(planets²) per step.

### D.3 Beam search over dispatch plans (a v3/v4 candidate)

- Enumerate top-K candidate dispatch plans per step (K ≈ 5–8).
- Forward-simulate H steps using the arrival-ledger + combat resolver
  (no opponent model — assume they hold).
- Pick the plan maximising `own_total_production + own_total_ships` at
  step `t+H`, H ≈ 20.
- Tractable: O(K · H · planets) per step.

This is a strict generalisation of D.2.

### D.4 Bellman framing (pedagogical)

```
V(s) = max_a [ r(s, a) + γ V(T(s, a)) ]
```

State space is large (~planets × ships × angles), so we don't solve
this exactly. Beam search (D.3) is a 1-deep approximation; depth-2
MCTS (roadmap v4 path-B) is a 2-deep approximation.

### D.5 Tractable approximation — per-planet timeline simulator

The v3 plan (`lib/world_model.py`, roadmap line 79) is a
forward-only simulator: per planet, a sorted list of `(step,
event_kind, ships, owner)` events. Replaying from `t` to `t+H` under a
fixed action plan is fast (each event is a constant-time update). This
is the substrate for D.3.

### D.6 Heuristic anchor — phase-segmented play

A robust heuristic that doesn't require search:

| Phase | Steps | Behaviour |
|---|---|---|
| Land grab | 0 – ~60 | Race-to-neutral; ignore enemy contact; production-weighted ROI |
| Frontier | ~60 – ~200 | Switch to contested-ROI with combat forecasting |
| Consolidation | ~200 – ~400 | Reinforce frontier, deny opponent expansion |
| Endgame | ~400 – 500 | Convert garrison to in-flight ships (counted at end) |

Phase boundaries are tunable; a sensor (e.g., "first contested arrival") can switch dynamically.

### D.7 Hand-off
- D.2 is v1.1 (shipped).
- D.5 is v3 `lib/world_model.py`.
- D.3 beam search is a v4 path-A or path-B candidate (roadmap line 95).
- D.6 phase segmentation is **strategy layer** scaffolding, applicable
  in any version.

---

## E. Ship bundling — speed via mass

### E.1 Speed lever (the core PI insight)

From §A: `v(N)` jumps from 1 (N=1) to 6 (N=1000). A 1000-ship fleet
covers ground 6× faster. The PI's intuition is correct: **bundling
travels faster**.

There are three distinct flavours of bundling, with very different
trade-offs.

### E.2 Same-source bundling *(trivial win)*

For a given source planet `S`, given step `t`, given destination `T`:
**always launch one fleet, never multiple.** Splitting strictly loses
on speed (each smaller fleet is slower than one combined fleet of the
sum) and combat (same-step same-owner arrivals merge anyway by the
combat rule, so splitting offers zero gain). This is a free heuristic
to enforce in a `merge_intents` mechanism.

**[mechanism layer]** — sits between `validate` and `arrival_size` in
the pipeline.

### E.3 Multi-source simultaneous-arrival timing *(the cheap, reliable bundling)*

The combat rule already groups same-owner same-step arrivals. So we
don't need to physically merge fleets — we just need to **time launches
so all arrivals land on the same step**.

Algorithm:
1. Pick target `T`, required mass `M`.
2. Identify a set of source planets `{S_i}` with sufficient combined
   garrison.
3. Compute each `eta_i = ceil(dist(S_i, T) / v(n_i))` where `n_i` is
   that source's contribution.
4. Set the global arrival step `step* = step + max_i(eta_i)`.
5. For every source `S_j` with `eta_j < eta_max`, **delay its launch
   by `eta_max − eta_j` steps** so all fleets land at `step*`.

This is **strictly better than uncoordinated launches** for any target
that requires combined mass — combat resolver does the rest.

Cost: arrival ledger needs to track planned-future launches, not just
in-flight ones.

**[strategy layer]** — encoded as a mission class `gang_up` per Roman.

### E.4 Multi-source physical staging *(expensive, sometimes wins)*

Forward planet `S*` receives ships from peripheral planets, then
re-launches the combined pile to `T`.

Direct vs staged ETA:

```
ETA_direct = max_i ( dist(S_i, T) / v(n_i) )
ETA_staged = max_i ( dist(S_i, S*) / v(n_i) ) + dist(S*, T) / v(Σn_i)
```

Worked example (PI-asked: "send to nearby planet first, then bundle"):
- Two peripheral planets `A`, `B` each with 100 ships, both 30 units
  from a forward staging planet `S*`. `S*` is 50 units from target
  `T`.
- **Direct:** each fleet of 100 at speed 3.96 → ETA = ceil(80/3.96)
  = 21 steps.
- **Staged:** A and B reach S* in ceil(30/3.96) = 8 steps. Combined
  fleet of 200 (assume S* keeps its garrison) at speed ~v(200) ≈ 4.65
  → ceil(50/4.65) = 11 steps. Total = 8 + 11 = 19 steps. **Saves 2
  steps.**

Now scale up: A and B each have 500 ships:
- **Direct:** speed v(500) ≈ 5.4 → ceil(80/5.4) = 15 steps.
- **Staged:** 30/5.4 = 6 steps; combined 1000 at speed 6 → 50/6 = 9 steps.
  Total = 15 steps. **Tie.**

So staging wins when (i) staging planet is on-path to the target,
(ii) peripheral fleets are small enough that the speed gain on
combination outweighs the detour, (iii) staging planet stays friendly
through the handoff.

Staging is also a **risk amplifier**: one captured fleet vs many.
Recommend deferring to v4 unless an opponent-modelling agent can
predict interception.

**[strategy layer]** — special mission class `stage_then_strike`.

### E.5 Risks across all flavours

- Staging planet flips before re-launch (E.4).
- Single-fleet failure point if intercepted (E.4 worse than E.3 here).
- Opponent reads the pile-up at `S*` and counter-attacks (E.4).
- Co-arrival timing (E.3) blocked if any fleet's eta changes due to
  garrison changes mid-flight (it shouldn't — fleet speed is set at
  launch).

### E.6 Hand-off
- E.2 → mechanism `merge_same_source` (v2).
- E.3 → v3 mission class `gang_up`; roadmap mechanism 4 ("fleet
  coordination").
- E.4 → v4 advanced option, parked behind a flag until risk model
  exists.

---

## F. Compete-relative vs absolute objective

The PI's distinction: some agents optimise **own** ships; others
optimise **(own − opponent)**, or **rank**, or **deny opponent
production**. These are not equivalent.

### F.1 Why this matters: TrueSkill

TrueSkill is a Bayesian skill rating that updates μ based on
**ordinal outcome** (who finished where), not score margin. Once we
clear the win threshold by 1 ship, a 100-ship surplus and a 10000-ship
surplus update μ identically.

So **maximising own ships past the win threshold is wasted optimisation.**
The marginal compute is better spent on robustness (probability of
winning) than on margin.

### F.2 2-player: distinction collapses

In 1v1, the score is approximately zero-sum (every ship one player
gains is, in some sense, a ship the other could have made). Own-max
and beat-opponent point in nearly the same direction. Distinguish in
2P is a minor correction.

### F.3 4-player FFA: kingmaker dynamics

Four-player TrueSkill rewards 2nd place over 3rd. So when we cannot
catch the leader, attacking the leader to demote them — even at our
own ship-cost — can yield a better expected μ-update than continued
expansion.

Concrete: suppose at step 350 we are 3rd, leader is far ahead, 2nd is
within reach. Two options:
- **Absolute (own-max):** continue expanding into neutrals. Likely
  finish 3rd.
- **Relative (spoiler):** divert ships to attack leader. Either we
  drag leader to 2nd while ourselves stay 3rd (no rating change vs
  expand), OR we drag leader to 3rd and we move to 2nd (rating
  +Δμ ≈ 30–60).

Expected value depends on the probability of each outcome; the
heuristic is worth invoking when own-rank-improvement probability is
≥ 30% and current-rank lock-in is high.

### F.4 Behavioural differences (compete-relative agent)

- Prefers `deny_enemy_production` over `extend_own_production` at
  similar ROI.
- In 4P, switches to **spoiler mode** when self-rank locked: target
  the current leader's most exposed planets, even at unfavourable ROI.
- Skips low-marginal captures once a B.2 deterministic-win predicate
  fires — frees ships for defense.
- Treats the opponent's projected production gradient as a cost in
  every action: every neutral capture by us is also a denial of
  capture by opponent.

### F.5 Coalition / Nash sketch

In 4P repeated play, rational agents temporarily coalesce against the
leader. Modelling coalition equilibria is intractable; a **static rule
("always attack current leader when self-rank-locked")** captures most
of the gain at zero modelling cost. Upgrade later to a tracker that
learns opponents' launch patterns.

### F.6 Hand-off
- F is **not currently on the v2/v3 critical path.** v2/v3 build the
  arrival-ledger and mission framework — the substrate F lives on top
  of.
- Promote F as a v4 path-A candidate ("lightweight opponent
  modelling," roadmap line 99).
- Add hypothesis H6 (see §J) to `state/hypothesis-board.md`.

---

## G. Brainstorm — 15 additional heuristics (one paragraph each)

**G.1 Lazy garrison reserve.** Don't ship 100% of garrison. Reserve =
`max(largest_incoming_enemy_in_K_steps − production·K, 0)`. Defends
against telegraphed attacks for free.

**G.2 Threat-aware launching.** Before launching from `S`, check
incoming enemy fleets at `S`. If `S` is about to be hit, either
(a) reinforce instead of launch, or (b) launch the surplus only.

**G.3 Frontier zoning.** Partition planets into near / contested / far
zones around our home cluster. Launch policy differs per zone: near =
own-side reinforcement, contested = combat ROI, far = production ROI.

**G.4 Drain-low-prod-first.** Low-production owned planets have low
marginal value as engines; drain them first, preserve high-production
planets as ship factories.

**G.5 Inverse — launch from high-prod.** High-production planets
accumulate large garrisons → bigger fleets → higher v(N). Tension
with G.4. Resolve empirically: at what production rate does the
factory-vs-feeder logic flip?

**G.6 Bipartite assignment source→target.** Cast each turn's
launch decision as an assignment problem: minimise total weighted ETA
across all (source, target) pairs subject to one-per-source. Hungarian
algorithm; optimal for the linear formulation.

**G.7 Multi-target single-source split.** A high-garrison planet may
profitably split into 2–3 fleets to small targets — but only when
each fragment's `arrival_size` clears the target's needed-ships.
Anti-pattern of E.2 only because the targets differ.

**G.8 Risk-adjusted ROI.** Multiply C.3 ROI by `P[mission_succeeds]`,
estimated from contested arrivals and orbit-prediction error.

**G.9 Race-to-neutral first phase.** Steps 0–~60: ignore enemy
entirely, race for max neutrals by production-weighted ROI. The
compounding insight (D.1) makes this dominate.

**G.10 Endgame burn-through.** Last ~30 steps: ships in flight count
for the launcher's score. If a planet is already enemy-locked (we
can't capture by step 500), still launch — the in-flight ships count
in score; the unsuccessful capture is irrelevant. Only abort if the
launch would fail combat at a still-relevant planet.

**G.11 Symmetry-breaking randomness.** Already in v1: per-step seeded
RNG for tie-breaks, prevents mirror-strategy lock. Universal — every
deterministic agent should have it.

**G.12 Opponent-pattern exploit.** Track opponents' typical launch
pattern (e.g., always launch `floor(garrison) − 1`). Attack just after
their launch when they're most depleted. Cheap with a small per-opponent
ring buffer.

**G.13 Cluster lead planets.** Group own planets into clusters;
designate "lead" planets that aggregate ships from cluster peers via
E.4-style staging; only leads launch externally. Reduces decision
dimensionality.

**G.14 Comet ROI gate.** Comets exist for a few-dozen steps (per env
spec). Only chase a comet if `production · expected_lifetime −
ships_to_capture > 0`. v1's negative ablation on `comet_aim` (−22.5%)
suggests the current chase logic is too eager — gate it.

**G.15 Counterfactual-defense check.** Before reinforcing planet `P`,
simulate "do nothing." If `P` survives the projected enemy assault
under existing garrison + production, save the reinforcement for
offense. The reverse of B.3 thinking.

---

## H. Heuristic → roadmap mapping table

| # | Heuristic | Layer | Roadmap mech | First version |
|---|---|---|---|---|
| B.1 | Trivial-win predicate | strategy | – | v2 (telemetry) |
| B.2 | Production-dominance lock | strategy | mech 1 | v2 (telemetry), v3 (policy) |
| B.3 | Reachability cover sweep | strategy | mech 1+2 | v3 / v4 mission class |
| C.1–C.3 | ROI scoring core | strategy | mech 1+7 | v3 (snipe/reinforce/recapture) |
| C.4 | Alternative scorings | strategy | mech 7 | v4 ensemble |
| C.5 | Risk multipliers | mechanism | mech 3+5 | v3 (sun-safe + comet) |
| D.1 | Compounding (early-rush priority) | strategy | – | v0+; phase logic in v3 |
| D.2 | Greedy 1-step (arrival_size) | mechanism | – | **v1.1 (shipped)** |
| D.3 | Beam search dispatch | strategy | mech 1+2 | v4 path-A or B |
| D.5 | Per-planet timeline simulator | substrate | mech 1 | v3 `lib/world_model.py` |
| D.6 | Phase-segmented play | strategy | – | any version |
| E.2 | Same-source merge | mechanism | mech 4 | v2 |
| E.3 | Simultaneous-arrival timing | strategy | mech 4 | v3 mission `gang_up` |
| E.4 | Physical staging | strategy | mech 4 | v4 advanced |
| F.1–F.6 | Compete-relative objective | strategy | mech 7 (new) | v4 path-A |
| G.1 | Garrison reserve | strategy | mech 6 | v3 |
| G.2 | Threat-aware launching | strategy | mech 6 | v3 |
| G.3 | Frontier zoning | strategy | – | v3 / v4 |
| G.4 / G.5 | Source-selection policy | strategy | – | v3 |
| G.6 | Bipartite assignment | strategy | mech 7 | v4 path-A |
| G.7 | Multi-target split | strategy | mech 7 | v3 |
| G.8 | Risk-adjusted ROI | strategy | mech 7 | v3 |
| G.9 | Race-to-neutral phase | strategy | – | v2 / v3 |
| G.10 | Endgame burn-through | strategy | – | v3 |
| G.11 | Symmetry-breaking RNG | mechanism | – | **v1 (shipped)** |
| G.12 | Opponent-pattern exploit | strategy | mech 7 (new) | v4 path-A |
| G.13 | Cluster lead planets | strategy | mech 4 | v4 |
| G.14 | Comet ROI gate | mechanism | mech 5 | v3 (re-enable comet_aim with gate) |
| G.15 | Counterfactual-defense check | strategy | mech 6 | v3 |

Roadmap mechanisms 1–7 are defined in `docs/strategies/roadmap.md:18-44`.
Items tagged "(new)" are mission classes not yet present in the v3
spec — F and G.12 in particular extend the Roman taxonomy.

**Gaps in current v2/v3 plan vs this research:**
- Compete-relative objective (F) — entirely new for v4.
- Opponent-pattern exploit (G.12) — entirely new for v4.
- Bipartite assignment (G.6) — sharper than v3's per-source-best
  greedy; consider before v4.
- Reachability-cover sweep (B.3) — a new endgame mission class.

---

## I. Open questions for future sessions

1. **Bundling break-even.** §E.4 algebra is illustrative; real ETA
   depends on integer ceil's and discrete launch steps. Need a
   simulator sweep over (n_peripheral, dist_to_staging, dist_to_target,
   production) → cost matrix. Defer to post-v2 when world_model.py
   exists.
2. **B.2 / B.3 firing frequency.** How often does the production-lock
   predicate fire in self-play? Replay analysis once `state/replays/`
   exists. Defer.
3. **4P kingmaker probability.** What fraction of 4P games have a
   "spoiler-can-promote-self-from-3rd-to-2nd" window? Needs replay
   analysis. Defer to v4 path-A scoping.
4. **G.4 vs G.5 break-even.** At what production rate does drain-low
   flip to launch-high? Empirical question, needs simulation.
5. **Comet ROI gate threshold.** What lifetime estimate should the
   `comet_aim` gate use? `obs["comets"][].path_index` hints at it —
   needs measurement.

---

## J. Hypotheses to promote to `state/hypothesis-board.md`

(Copy-paste-ready when PI is ready to commit.)

- **H4 (E.3 simultaneous-arrival):** A multi-source arrival timer in v3
  beats v3-without-timing by ≥55% over 24 seeds × both sides.
- **H5 (B.2 dominance-lock):** The production-dominance predicate
  fires before step 200 in <10% of self-play games but, when it does,
  switching to defense-only loses no further games (≥95% retention).
- **H6 (F.3 spoiler mode):** A spoiler-vs-leader rule in 4P FFA
  improves μ by ≥30 vs the always-expand baseline over a 60-game panel.
- **H7 (D.1 early-rush priority):** Front-loading neutral capture by
  re-weighting ROI in steps 0–60 by `(500 − step − eta)^1.5` (vs
  linear) gains ≥3% winrate over v3 baseline.
- **H8 (G.6 bipartite):** Replacing v3's greedy per-source-best with a
  Hungarian-assignment solver gains ≥2% winrate at <100 µs added
  per-step cost.

---

## Suggested next-session follow-ups (out of scope for this note)

- Promote H4–H8 to `state/hypothesis-board.md`.
- Refresh `state/mechanism-ledger.md` (currently shows "none yet"
  but v0/v1/v1.1 are deployed).
- Add the four "gaps" identified in §H to `ISSUES.md` as new B.* leaves
  for future claim.

---

## K. Empirical evidence — cross-reference (last refreshed 2026-05-10 post-merge)

Now-on-main sources:
- `agents/simple/*` (5 target-selection ablations) and
  `scripts/strategy_panel.py` (round-robin runner).
- `audit/2026-05-10-simple-strategy-panel.md` — 8-seed panel.
- `audit/tournaments/20260510T140907Z.json` — 32-seed panel
  (1568 games, the load-bearing capture).
- `audit/2026-05-10-phase1-manifold-verdict.md` — Phase 1 fingerprint
  gate result; introduces the AlphaStar-discrete-basins framing.
- `audit/2026-05-10-meta-strategy-prior-art.md` — Grover 2018 / DRON /
  AlphaStar / Pluribus / Ganzfried — names §F.6's architecture.
- `lib/fingerprint.py` (15 hand-designed features, FEATURE_VERSION=1)
  and `scripts/manifold_check.py` (CV-by-seed RF/LR diagnostic).
- `state/current.md` — `roi` shipped as v1.2 (#52518060, PENDING).
  v1.1 settled at μ=597.4 (~500 μ short of top-5%).

### K.1 Per-section cross-reference (32-seed, post-Phase-1)

| Research-note section | Empirical finding (32-seed) | Status |
|---|---|---|
| §C.3 ROI score `value/cost · time_discount` | `roi = production/(dist+1)` 97.1% mean panel WR / 100% (64/64) vs `v1_orbitfix` | **validated and shipped** as v1.2 |
| §C.4 marginal-production-gain | `production` argmax 67.7% panel WR (regressed from 75.0% at 8 seeds) | **validated** but ROI dominates the same axis better |
| §C.4 threat-reduction / denial-value (as primary axis) | `enemy_first` 32.3% panel WR (8-seed); 8/8 self-play draws | **falsified for primary use** — see §K.3 for how this reshapes §F |
| §D.1 compounding insight (early production captures > late) | Production-aware variants beat `nearest` decisively; gap widens with game length | **validated** indirectly |
| §F.4 "prefers `deny_enemy_production` over `extend_own_production`" | `enemy_first` falsification implies this rule, run as a *primary scoring axis*, is bad pre-contact | **scope-tightened to override** — see §K.3 |
| §F.5 / §F.6 opponent-modelling architecture | Phase 1 confirms broad-class basins (`weakest` 89.7%, `enemy_first` 83.4%, `baseline` 95% in 7-class) but ROI-family is one basin (12-17% mutual confusion) | **partially validated**; 3-class router buildable today, see §K.4 |
| §G.7 multi-target single-source split | `weakest` (always cheap snipes) 15.6% panel WR; 0% vs `roi` | **falsified** — confirms the §G.7 caveat |
| §G.11 symmetry-breaking RNG | Per-step seeded RNG works for non-symmetric strategies; mirror-symmetric scorings still draw on mirrored seeds (`roi` 7/8 self-play draws; `enemy_first` 8/8) | **partially validated** — RNG ≠ symmetry-breaking-of-strategy |
| §G.14 comet ROI gate | `comet_aim` ablation lost 22.5% — off by default in `lib/mechanism.py`; `simple-roi.md` confirms ROI's lifetime-blindness | **strengthened** — gate change required before re-enable |
| §J H4 (multi-source simultaneous-arrival) | not yet tested | **next experiment Axis 3** |
| §J H5 (B.2 dominance-lock) | not yet tested | **deferred to v2** |
| §J H6 (F.3 spoiler in 4P) | not yet tested; panel is 1v1 only | **deferred** — needs 4P-FFA infra |
| §J H7 (D.1 early-rush re-weighting) | partially covered by Axis 4 (phase segmentation) | **next experiment Axis 4** |
| §J H8 (G.6 bipartite assignment) | not yet tested | **future** — heavier than current panel agents |

### K.2 What changed in our priors

1. **§C.3 ROI is the live champion (v1.2 shipped).** Future axis
   experiments build on top of `roi` target selection unless explicitly
   testing target selection.
2. **§F (compete-relative) is scope-tightened to rank-aware override
   behaviour.** "Deny enemy production" as a *headline rule* is
   falsified by `enemy_first`'s 32% panel WR. F-axis only fires when
   self-rank is locked or a basin-detection fires (see §K.3 below) —
   not as a turn-1-onward primary scoring lever.
3. **§D.6 phase segmentation rises in priority.** `enemy_first`'s
   8/8 self-play draws (no-one expanding pre-contact) is consistent
   with the land-grab-then-frontier phase model. A
   `landgrab → roi → endgame_burn` schedule is the cheapest extension.
4. **§G.14 comet gate required before re-enabling `comet_aim`.** The
   22.5% ablation hit + ROI's lifetime-blindness imply a single explicit
   gate change before the mechanism returns.

### K.3 How the Phase 1 manifold verdict reshapes §F

The Phase 1 fingerprint gate failed for the **5-class** problem (RF
80.5% at K=100, target 90%). The verdict is structurally informative:
`nearest` / `production` / `roi` collapse into ONE basin
("production-aware-greedy") while `weakest`, `enemy_first`, and
`baseline` sit in their **own** basins. This is the AlphaStar
"discrete basins" framing the prior-art audit predicted (Vinyals
2019).

Three §F implications follow:

1. **The 3-class manifold IS the right granularity for §F.**
   §F.4–F.5 always treated compete-relative play as **basin-level
   override behaviour** (kingmaker, spoiler, leader-attacker), not
   as fine-grained per-strategy adaptation. Phase 1's failure to
   discriminate inside the ROI-family is consistent with §F.1's
   tournament-rating insight: TrueSkill rewards win/loss, so once
   we're inside the ROI basin our default action (run ROI ourselves)
   is correct regardless of which ROI-family member the opponent is.
   *No §F-derived policy distinguishes* `nearest`-style from
   `roi`-style opponents — and Phase 1 confirms that distinction is
   noise from a behavioural-fingerprint angle as well.

2. **The submission-incentive argument from the verdict matches §F.**
   The verdict says "there is no submission incentive to distinguish
   ROI-family members because ROI dominates them all." That mirrors
   §F's claim that the F-axis is rank-aware override, not a primary
   scoring lever. They reach the same conclusion from different
   directions.

3. **§F.6's opponent-modelling architecture is now buildable today.**
   The verdict shows `weakest` (89.7%) and `enemy_first` (83.4%) are
   already cleanly separable at K=100 with the existing
   `lib/fingerprint.py` v1 features. A 3-class meta-router
   (`production_aware_greedy / weakest / enemy_first`) is one
   relabel-script call away — see §K.4.

### K.4 Recommended path: Phase 1 path-A (3-class meta-router)

The Phase 1 verdict offers three paths (A: coarsen ROI-family labels;
B: bump fingerprint to v2 with distribution-shape features;
C: learned embedding via Grover 2018 protocol). On §F's
strategic-policy grounds, **path A is the correct immediate next
step.** Two reasons:

1. **§F is rank-aware override, not fine-grained scoring.** The BR
   table at basin granularity is exactly what §F prescribes. There's
   no §F-derived policy that requires distinguishing `nearest` from
   `production` — both are "production-aware-greedy" and our best
   response is the same (continue running ROI). Path B would buy us
   discrimination we don't actually use at the policy level.

2. **Path A is a one-line cheap experiment.** Per the verdict:
   `--label-merge nearest=production_aware_greedy
   production=production_aware_greedy roi=production_aware_greedy`
   re-runs `manifold_check.py` at predicted ≥92% RF. Compare to
   half-day for path B and several days for path C. Path A also
   *unblocks Phase 2/3* immediately while paths B/C are open
   research questions.

Path B becomes worthwhile only if §F.5 gets developed into per-strategy
customisations (e.g., learning that `nearest` opponents are
specifically vulnerable to a counter that `production` opponents
aren't). That's a v4-grade refinement, not a v3 critical-path item.
Path C is the right last resort if even path B fails to lift
fingerprint discrimination on the new basin labels.

### K.5 Initial best-response table (proposed; populates Phase 2)

The 3-class meta-router needs a BR-table column. Populated from §F
+ the panel data:

| Detected basin | Our response | Rationale |
|---|---|---|
| `production_aware_greedy` (nearest / production / roi) | run `roi` (current v1.2) | symmetric ROI-vs-ROI converges to RNG-tie-break; panel showed 7/8 draws self-play; default to our champion |
| `weakest` (cheap-snipes opponent) | `roi` + Axis-4 endgame-burn schedule | weakest's 15.6% panel WR shows they leave high-prod neutrals open; we keep ROI expansion + flush garrison in-flight at step 470+ for free score points |
| `enemy_first` (pressure-on-opponent) | `roi` + Axis-1 sizing-overshoot on home cluster | they besiege our planets; production-aware overshoot at home absorbs incoming + we continue ROI expansion. Their 32.3% panel WR confirms attrition loses to economy |
| (4P-FFA only) **leader-locked, self at rank 2** | invoke §F.3 spoiler-mode | attack leader's exposed planet; net rank gain ≥30 μ |
| (B.2 deterministic-win predicate fires) | switch to defense-only (§B.4) | save garrison; let production lock close the game |

This table is **a starting point, not an endpoint** — `weakest` and
`enemy_first` rows are extrapolations from §F + panel data,
validatable on the existing `audit/replays/20260510T132957Z/` corpus
without new replay capture.

### K.6 Hand-off to the next-experiments plan

The companion forward-looking plan
`audit/2026-05-10-research-driven-next-experiments.md` enumerates
**six axes** to pursue in priority order:

- **Axis 0 (NEW; top priority):** Phase 1 path-A 3-class meta-router
  + §K.5 BR-table — implements §F.6 with infra that already exists
  on main. ~1–2 days.
- **Axis 3:** multi-source simultaneous-arrival timing (§E.3) — the
  biggest novel claim from this note, applicable on top of `roi`.
- **Axis 4:** phase segmentation, especially `endgame_burn` (likely
  cheapest free %).
- **Axis 1:** sizing variants (overshoot, garrison-reserve).
- **Axis 2:** source-selection variants (drain-low vs launch-high,
  safe-only).
- **Axis 5:** defense-heuristic (lift uncertain pre-v2).

Compete-relative 4P (§F.3 spoiler-mode) is parked behind 4P-FFA
panel infrastructure — out of scope for v2/v3 critical path; v4
candidate.
