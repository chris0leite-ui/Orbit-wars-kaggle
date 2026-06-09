# Reach-Frontier Chooser — implementation design (v1)

Authored 2026-05-27. Status: **design committed, not yet implemented.**
Companion to `knowledge-base/concepts/reach-frontier-doctrine.md` (the
mathematical framing) and `knowledge-base/concepts/evaluation-metrics.md`
(how the resulting agent will be evaluated, Rule 48).

This doc is the implementation specification. PI review before any code
lands. The build sequence (§10) is structured so each step has a hard
exit condition; if any step fails its gate, work stops and the data
returns to PI.

---

## 1. Mission, scope, boundary

**What ships:** a new agent at `agents/reach_frontier/`, structured as a
*full replacement* for the v9 chooser-family — not a layer stacked on
baseline. Action selection is governed by the closed-form objective
`max Σ p̃·hold_time − cost`, evaluated over a Voronoi-style partition of
planets by reach time. No K=10 rollout. No legacy value head.

**Scope of v1:** 2P-first. The agent functions in 4P (won't crash), but
its decision quality is expected to clear the eval gate in 2P and
underperform in 4P. The 4P-specific branch lands in v2.

**Out of scope for v1:** opponent-model learning, ML value functions,
multi-turn LP planning, commitment ledger. The per-turn problem stays
purely closed-form. Those features add as v2 axes if v1 saturates.

**Hard boundary:** Rule 47 (physics primitive trace), Rule 43 (multi-opp
panel), Rule 45 (n ≥ 32 Wilson-lo), Rule 46 (bundle + parity smoke),
Rule 1 (single-shot PI-approved submits) all apply before any submission.

---

## 2. The math, made implementation-concrete

For player `i ∈ {0, …, P−1}`, source planet `s`, target planet `p`,
ship-count `k`:

**Arrival time `a_i(s, p, k)`** is computed by
`lib.trajectory.predict_fleet_fate(src, target, aim_angle, k, world)`.
The function returns `FleetFate(outcome, hit_planet_id, step)`. We accept
the candidate iff `outcome == "target"` and `hit_planet_id == p.id`; the
`step` field is the arrival tick offset.

**Source recovery cost.** Launching `k` ships from a source of production
`p̃_s` removes `k` ships of garrison; the source needs `k / p̃_s` ticks of
production to recover the launch cost. So the *amortised reach cost* is

```
ρ_i(s, p, k) = a_i(s, p, k) + k / p̃_s
```

(doctrine §4.1, made concrete).

**Capture feasibility.** At arrival tick `a`, planet `p` has `g_p(a)`
ships of whoever owned it. Capture requires `k > g_p(a)`. For neutral /
mine planets `g_p(a)` is straightforward to forward-simulate (production
builds garrison). For opp-owned planets we estimate; see §5.

**ρ_i(p)** — minimum reach cost for player `i` to capture `p` — is the
min over `(s, k)` of `ρ_i(s, p, k)` subject to feasibility. We discretise
`k` over `K = {0.25, 0.5, 0.75, 1.0} × garrison(s)`, capped at the
ceiling that makes capture feasible (rule of thumb:
`min(0.95·garrison, g_p(a) + buffer)`).

**ρ_opp(p) — opponent's minimum reach.** In 2P it's the single opponent's
ρ. In 4P FFA it's `min_j ρ_j(p)` over all opponents `j ≠ me`. The v1
doctrine treats this as a deterministic worst-case; v2 may switch to a
probability-weighted expectation.

**Hold time.** For a planet I can capture: `h(p) = ρ_opp(p) − ρ_me(p)`.
For a planet I currently own: `h(p) = ρ_opp(p) − t_now`. Positive `h`
means I can extract `p̃ · h` worth of production-integral before opp
threatens; negative means opp gets there first and the planet is a bad
target (or already lost).

**Per-candidate reward.** For a candidate launch `(s, p, k)`:

```
R(s, p, k) = p̃_p · h(p)
             − λ_loss · expected_losses(s, p, k)
             − λ_risk · risk(s, p, k)
```

where `risk` is the probability the fleet dies en route (sun, OOB, comet
expiry) — collected from `predict_fleet_fate` — and `expected_losses` is
the expected ship-count consumed by combat at the target. Default knobs:
`λ_risk = 50` (heavy — fleets dying is a doctrine §8.4 critical failure)
and `λ_loss = 1` (price one lost ship at one ship).

**Assignment objective.** Pick `A ⊆ candidates` subject to:
- one candidate per source (matches the env's per-source launch
  constraint),
- planets distinct (no two of our candidates target the same planet —
  gang-ups in v1 are a single "fat" candidate, see §4),
- maximise `Σ_{(s,p,k) ∈ A} R(s, p, k)`.

Solved by Hungarian on a rectangular cost matrix;
`lib/joint_solver/lp.py:build_assignment_matrix` + `solve_assignment`
already does this for the joint-LP chooser. Reuse.

---

## 3. Turn-level pipeline

Per-turn flow with timing budget:

```
agent(obs) →
  [  5 ms] parse obs → World
  [  2 ms] kinematic_table.begin_turn(world)        # cache positions
  [ 10 ms] for each (my_src, target, k) ∈ grid:
              closed-form ρ via kinematic_table
              → ρ_me_candidates (UNVALIDATED)
  [ 15 ms] for each (opp_src, target, k') ∈ opp-grid:
              closed-form ρ via kinematic_table
              → ρ_opp_candidates
  [  1 ms] ρ_me[p] = min over (s,k); ρ_opp[p] = min over (s,k,j)
  [  2 ms] hold_time[p] = ρ_opp[p] − ρ_me[p]   (or − t_now if already owned)
  [  5 ms] for each candidate, compute R(s, p, k)
  [  3 ms] build_assignment_matrix + solve_assignment (Hungarian)
  [ 20 ms] for each picked candidate: full predict_fleet_fate
            physics validate; drop anything with outcome != "target"
  [  5 ms] emit [[src_id, angle, ships], …]
```

Total budget: ~70 ms typical, ~150 ms in pathological 4P (many sources ×
many targets). Well inside the 1000 ms turn timeout. The dominant cost
*was* `predict_fleet_fate` calls; the §9 mitigations keep them to ~10
final-validation calls per turn rather than ~400.

---

## 4. Module layout and function signatures

```
agents/reach_frontier/
├── __init__.py          # makes the package importable
├── main.py              # agent() entrypoint (~80 LOC)
├── reach.py             # ρ-table builder (~120 LOC)
├── hold.py              # hold_time + reward computation (~50 LOC)
├── opponent_reach.py    # ρ_opp estimator (~80 LOC) — §5 below
├── assignment.py        # Hungarian wrapper over lib/joint_solver (~40 LOC)
└── README.md            # what this agent is + doctrine pointer
```

Key function signatures:

```python
# reach.py
@dataclass
class ReachEntry:
    src_id: int
    target_id: int
    ships: int
    aim_angle: float        # closed-form aim
    arrival_tick: int       # from FleetFate.step (or kinematic estimate)
    cost_tick: float        # arrival_tick + ships / source_production
    fate: FleetFate | None  # populated only after final validate pass

def build_reach_table(
    sources: list[Planet],
    targets: list[Planet],
    world: World,
    *,
    k_grid_fractions: tuple = (0.25, 0.5, 0.75, 1.0),
    max_arrival_lead: int = 200,
    validate_physics: bool = False,    # False = closed-form fast path;
                                       # True = full predict_fleet_fate
) -> dict[(int, int), list[ReachEntry]]:
    """Returns (src_id, target_id) -> list of feasible reach entries
    sorted by cost_tick. Physics-infeasible entries are filtered when
    validate_physics=True. Empty list = unreachable pair."""

# opponent_reach.py
def estimate_opp_reach(
    world: World, me: int, num_seats: int,
) -> dict[int, float]:   # target_id -> min ρ over opponents
    """Mirror of my reach table built over each opponent's sources."""

# hold.py
def compute_hold_times(
    world: World, me: int,
    my_reach: dict[(int, int), list[ReachEntry]],
    opp_reach: dict[int, float],
    t_now: int,
) -> dict[int, float]:
    """target_id -> hold_time.
       - if I own it: hold = opp_reach[p] - t_now
       - if I can reach: hold = opp_reach[p] - min cost_tick
       - else: hold = 0 (skip)"""

def per_candidate_reward(
    entry: ReachEntry, target: Planet, hold_time: float,
    *, lambda_risk: float = 50.0, lambda_loss: float = 1.0,
) -> float: ...

# assignment.py
def pick_actions(
    candidates: list[ReachEntry],
    rewards: dict[(int, int), float],   # (src_id, target_id) -> R
    world: World,
) -> list[list]:   # [[src_id, angle, ships], ...]
    """Hungarian over (sources × targets). Returns env-ready move list."""

# main.py
def agent(obs, configuration=None) -> list[list]:
    # 1. parse obs to World
    # 2. build my reach table (closed-form)
    # 3. estimate opponent reach (closed-form)
    # 4. compute hold times
    # 5. emit candidates with rewards
    # 6. assignment
    # 7. final predict_fleet_fate validate-and-filter on chosen
    # 8. return
```

Target: v1 readable end-to-end in ~400 LOC.

---

## 5. The opponent reach estimator (the hardest sub-problem)

This is where v1 will be wrong most often, so the design is explicit
about it.

**The naive approach:** treat opponent symmetrically. For each opponent
`j`, enumerate their (src, k) grid the same way as mine. This is correct
*if* the opponent plays its best reach for each target. Top-10 do roughly
that. Midpack does worse — but we are not modelling midpack, we are
modelling the *strongest threat* to our hold time. The naive approach is
the right starting point.

**Implementation sketch:**

```python
def estimate_opp_reach(world, me, num_seats):
    opp_seats = [s for s in range(num_seats) if s != me]
    per_opp = {}
    for j in opp_seats:
        their_sources = [p for p in world.planets if int(p.owner) == j]
        if not their_sources:
            continue
        per_opp[j] = build_reach_table(
            their_sources, world.planets, world,
            max_arrival_lead=200, validate_physics=False,
        )
    opp_reach = {}
    for p in world.planets:
        best = float('inf')
        for j, table in per_opp.items():
            for src in [t for t in world.planets if int(t.owner) == j]:
                entries = table.get((src.id, p.id), [])
                if entries:
                    best = min(best, entries[0].cost_tick)
        opp_reach[p.id] = best
    return opp_reach
```

**Cost:** doubles closed-form ρ-table calls (mine + opps). 4P with all
opponents alive: ~4× per-turn cost. Per §3 that pushes pathological 4P
to ~150 ms; still well within 1 s.

**Three known biases of this estimator, documented in the README:**

1. **Symmetric-strength assumption.** If opp is weak and won't actually
   play best-reach, ρ_opp underestimates my hold time — we leave value
   on the table by being too defensive. Safe direction.
2. **No opponent commitment.** Opp fleets already in flight are ignored
   when computing ρ. *v1 fix:* when computing ρ_opp(p), if opp has a
   fleet en route to `p`, set
   `ρ_opp(p) = min(ρ_opp(p), fleet_arrival + cost(fleet_ships))`.
3. **No opponent collaboration in 4P.** Three opponents potentially gang
   up; treated independently here. *v1 fix:* none — flagged for v2.

---

## 6. The defensive / already-owned case

A point the doctrine pseudocode glossed: what to do with my home (and
any planet I already own). Two cases:

**Case A — opp ρ to my planet is far in the future.** No threat. I
should *not* defend; I should launch from this planet. The reach
frontier naturally handles this: my planet is a *source*, and its
hold_time is large positive, contributing to the integral passively.

**Case B — opp ρ to my planet is soon.** I'm being threatened. Options:
(i) reinforce from a closer planet, (ii) accept the loss, (iii) keep
launching elsewhere and rebuild. v1 makes this a *no-launch* candidate
that competes in the assignment: if the reward of launching from this
planet elsewhere is `R₁` and the reward of NOT launching (so defenders
repel opp) is `R₂ = p̃ · current_hold + expected_combat_outcome`, the
Hungarian picks the better one.

Concretely: every owned planet contributes a "no-launch / defend"
candidate to the assignment with reward
`p̃ · max(0, ρ_opp(p) − t_now) − idle_penalty`. The Hungarian then
chooses launch-vs-defend per source organically.

This is the right modelling fix (Rule 40) — defence falls out of the
same objective as offence, not as a separate rule.

---

## 7. The 4P branch (designed now, ships in v2)

Empirically (audit/2026-05-27-hold-time-empirical.md), 4P winners differ
from 2P winners in two ways:
- They launch the first capture *later* (median t_capture 137 vs 102).
- They run higher capture volume (66 segments/game vs 48).

So v2's 4P branch adds:

1. **Minimum production cushion gate.** Before the first launch, require
   `Σ_owned p̃ × cushion_ticks > k_launched`. cushion_ticks defaults to
   20 in 4P, 0 in 2P. This *delays* the first launch when garrisons are
   thin, matching the empirical "wait longer" pattern.
2. **Kingmaker de-rating.** When computing `ρ_opp(p) = min_j ρ_j(p)`,
   multiply the reward by `(1 − strongest_opp_share)` if the closest-
   reach opp is the strongest. Captures from the strongest opp help the
   third opp more than us; de-rate.
3. **Mine-cell preference in 4P.** Add a 4P-only bias
   `+ λ_mine · 1[ρ_me(p) < ρ_opp(p) − δ_4p]` with `δ_4p = 5`
   (vs `δ_2p = 2`). Shifts the chooser toward planets in *my* Voronoi
   cell when more opponents might recapture.

All three are knob-tunable. v1 ships 2P-first; we measure 2P-only A/B
before turning on the 4P branch.

---

## 8. Physics-primitive trace (Rule 47, mandatory)

Before any A/B, the agent gets a single-game trace:

```
python -m scripts.episode_postmortem agents/reach_frontier --seed 42
```

Goal: confirm sun + OOB + comet-expiry waste < 2%. If above, the
`λ_risk = 50` coefficient gets bumped or the aim-angle computation gets
re-examined. This is the failure mode that killed trajectory_roi — we
do NOT skip.

---

## 9. Performance and budgeting

The `predict_fleet_fate` call count dominates. Mid-game, P=2, 24
planets, 4 sources each side, 4 k-grid values:

- My reach (naive full validate): 384 calls
- Opp reach (naive full validate): 384 calls
- Final physics filter: ≤ 8 calls
- **Total naive**: ~776 calls × 1.5 ms = **1.16 s — over budget**

Two mitigations baked into the design:

**(a) Coarse-to-fine ship-count grid.** Try one ship-count first (50% of
garrison). Only refine for the top-N targets by closed-form ratio. Cuts
calls by ~4×.

**(b) Kinematic-table-only reach approximation.** For the initial ρ-
table sweep, use closed-form fleet-speed × distance plus orbital position
from `kinematic_table.lookup_relative` and *skip* per-step sun-and-planet
collision check. Only the top-N final candidates get the full
`predict_fleet_fate` validation. Cuts calls by ~10×.

The combination gets us to ~80 `predict_fleet_fate` calls per turn —
comfortably ~120 ms. Design assumes (b) from the start; the kinematic
table exists for exactly this.

**Realistic per-turn breakdown:**
- Closed-form ρ-table sweep (kinematic-table only): ~30 ms
- Top-N candidate selection: ~5 ms
- Full `predict_fleet_fate` filter on top ~10 candidates: ~20 ms
- Hungarian: ~3 ms
- Misc: ~10 ms
- **Total: ~70 ms — within budget with 10× headroom**

---

## 10. Build sequence

Six steps, in order. Each step has a hard exit condition; no skipping.

1. **Skeleton + smoke (≤ 30 min).** `agents/reach_frontier/main.py`
   returning `[]` (no launches). Confirm env-shape works:
   `python -m fast.py play agents/reach_frontier` runs a full game
   without crashing. Bundle test:
   `python scripts/bundle_agent.py reach_frontier &&
    pytest tests/test_bundle.py`.

2. **My-reach-only (≤ 1 h).** Implement `reach.py` + `main.py` calling
   it. Agent picks the single highest-reward candidate per turn,
   ignoring opponents (set `ρ_opp = ∞`). Smoke: agent doesn't crash,
   launches make sense (target planets get captured).

3. **Opponent-reach + hold-time (≤ 1 h).** Implement
   `opponent_reach.py` + `hold.py`. Agent picks based on full
   `R = p̃ · hold_time − costs`. Single-game trace: confirm physics
   waste < 2% (Rule 47).

4. **Assignment (≤ 30 min).** Wire up `assignment.py` for multi-source.
   Agent emits multiple launches per turn.

5. **Triage A/B (~ 30 min wallclock).** n = 16 vs
   `baseline_joint_aggr_consolidated_orbitfix` (current rolling
   champion). Wilson-lo report. If ≥ 0.55, move to step 6. If
   0.45-0.55, iterate on the reward function. If < 0.45, stop and
   surface to PI.

6. **Full panel + n = 32 (~ 1 h wallclock).** Multi-opponent panel
   (Rule 43) + n = 32 A/B (Rule 45). If both clear, the agent is
   *eligible* for submission — the submission decision is the PI's,
   not mine (Rule 1).

Total v1: ~3 hours of focused work end-to-end including smokes.

---

## 11. Test plan

- `tests/agents/test_reach_frontier_smoke.py` — one game with random
  opponent, asserts no crash, no out-of-bounds actions, all launches
  pass `predict_fleet_fate` validation.
- `tests/agents/test_reach_frontier_reach.py` — synthetic 2-planet
  world; asserts `ρ_me` matches a hand-calculated value.
- `tests/agents/test_reach_frontier_hold.py` — synthetic 3-planet
  world with known positions; asserts `hold_time` correctly positive /
  negative per cell.
- `tests/agents/test_reach_frontier_assignment.py` — 3 sources, 5
  targets, known rewards; asserts Hungarian picks optimal triplet.
- `tests/test_bundle.py` (existing) — runs for `reach_frontier`
  automatically when included.

Total new-test runtime: ~10 s (synthetic-world unit tests, not full-
game integrations).

---

## 12. Open design questions flagged for PI input

1. **`λ_risk` and `λ_loss` calibration.** Proposed `λ_risk = 50`,
   `λ_loss = 1` from first principles ("price a dead fleet 50× the cost
   of one ship lost to combat"). Will need empirical tuning. *Suggest:*
   hold at defaults for v1 A/B; tune in a v1.1 ablation if v1 clears
   the gate.
2. **Garrison growth on opp-owned planets.** When computing my ρ to
   their planet, I need their garrison at my arrival tick. *v1
   placeholder:* `g_p(a) = initial_garrison + p̃_p · (a − t_now)`,
   flagged in README as the most likely v1 estimation error.
3. **Ship the 4P branch in v1 or v2.** Designed for v2. If 2P A/B
   looks clean, the next session may want to add 4P immediately rather
   than ship a 2P-only agent and risk a μ regression on 4P-heavy draws.
   *Suggest:* ship 2P-only v1 to get a calibration data point, then
   iterate.
4. **Wire the commitment ledger?** Baseline agents use a multi-turn
   commit ledger to avoid duplicate-target dogpiles across turns.
   Reach-frontier doesn't naturally need one (re-evaluates per turn
   from current observation), but if per-turn decisions oscillate
   (chase / abandon / re-chase), a ledger would smooth them. *Suggest:*
   ship v1 without, observe the trace for oscillation, add only if
   seen.

---

## 13. What I would NOT do

- **Stack this on top of the baseline.** Doctrine §6 explicitly says
  *replace*, not stack. Adding the reward as one signal in the existing
  chooser dilutes it back to the K=10 leaf objective.
- **Build the 4P branch first.** Empirical signal is much cleaner in
  2P; ship the cleaner case first to get a real calibration data point.
- **Use ML for the opp reach estimator.** Closed-form symmetric reach
  is the right v1; ML can come later if v1 saturates.
- **Skip the kinematic-table approximation.** §9 shows the budget
  doesn't work without it. Substrate exists; we use it.
- **Submit before Rules 43, 45, 46, 47 all pass.** Doctrine §8.4
  explicitly warns about local-A/B-not-ladder-calibrated; we don't
  sidestep the gates.

---

## References

* `knowledge-base/concepts/reach-frontier-doctrine.md` — the math this
  implements, especially §4 (reach frontier definition) and §5
  (pseudocode this expands).
* `knowledge-base/concepts/evaluation-metrics.md` — Rule 48 eval
  defaults; the panel that scores the resulting agent.
* `audit/2026-05-27-hold-time-empirical.md` — the empirical study
  that motivates the 2P-first / 4P-deferred sequencing (§7).
* `audit/2026-05-27-between-band-stratification.md` — the band-
  stratification check that sets the μ ceiling expectation.
* `lib/trajectory.py:80` — `predict_fleet_fate`, the substrate.
* `lib/kinematic_table.py` — the closed-form position cache that makes
  §9's mitigation (b) work.
* `lib/joint_solver/lp.py:47, 114` — `build_assignment_matrix` and
  `solve_assignment`, reused for §4's Hungarian.
* CLAUDE.md Rules 40 (modelling over restriction), 43 (multi-opp
  panel), 45 (n ≥ 32 gate), 46 (bundle smoke), 47 (physics trace),
  48 (production-share metric).
