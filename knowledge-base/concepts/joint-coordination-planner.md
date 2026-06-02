# Joint-Coordination Planner

*A reframing of per-turn action selection in Orbit Wars as constrained
combinatorial maximization with a simulation oracle, and a five-rung
implementation path toward genuinely coordinated multi-planet launches.*

Authored 2026-06-02. Status: **framing ratified by PI as a multi-session
strategic thrust; Rungs 1–4 scheduled for implementation (default OFF).**
This doc is the durable anchor — operational state lives in
`state/MULTI_BRANCH.md`, not here (Rule 9 discipline).

---

## 1. Why this exists

The champion chooses launches one planet at a time. Each candidate launch is
scored in its **own** forward rollout that assumes it is the only thing we do
this turn; the chooser then emits a greedy, deconflicted subset under two
locks (one launch per source, one fleet per target). See
`agents/baseline/chooser_trajectory.py:score_candidate_v4` (the per-candidate
rollout) and the emit loop at `chooser_trajectory.py:1438-1504`.

The consequence: **the turn we actually play is never simulated as a whole.**
We rehearse each piece solo and then perform them as an ensemble. The
ensemble's true value can drift from the sum of the solo rehearsals in two
opposite directions.

### 1.1 Subadditive waste (we overcount)

Two launches each receive full credit for a benefit that only needed to
happen once:

- **Shared rescue.** Enemy fleet inbound to planet X. Planet A's "reinforce
  X" is scored alone and credited with saving X; planet B's "attack the
  enemy's launch base" is *also* scored alone and *also* credited with saving
  X. Summed, we think we saved X twice; one fleet was spent for nothing.
- **Redundant pile-on.** A 30-ship neutral. A privately decides "send 31,
  capture"; B decides the same. Both rehearsals show a clean capture. Fire
  both → 62 ships at a 30-ship planet: captured once, 31 ships wasted.

The two per-turn locks blunt the crudest version (same source / same target)
but miss shared-benefit cases where two different launches chase one payoff.

### 1.2 Superadditive blindness (we undercount)

A defended planet has 40 ships. A can send 30; B can send 30. Asked
privately, each says "30 < 40, my fleet bounces — score ~0, don't bother." So
neither fires. Arriving **together on the same tick**, 60 beats 40 and we own
it. Because we only ever rehearse solo, the chooser is structurally **blind to
plans that are only good as a team**, except through the narrow, bounded
synchronized-pair path (`BASELINE_JOINT_SYNC`, default OFF).

---

## 2. The formal frame

Choose the launch set `S` to maximize the *true joint value*

```
V(S) = favor(simulate ALL of S together, opponent reacting) − baseline
```

where each evaluation of `V` costs one forward simulation, under a
~600 ms/turn working budget. This is **constrained combinatorial
maximization with a simulation oracle**: get close to the best `S` while
spending as few oracle calls as possible.

The decisive structural fact is whether `V` is **submodular** — i.e. whether
each added launch helps *less* as the set grows (diminishing returns).

- The **waste** cases (§1.1) are exactly diminishing returns → submodular.
- The **teamwork** cases (§1.2) are *increasing* returns → **not** submodular.

This single mixed character is why neither textbook tool works alone:

- **Greedy marginal-gain** is near-optimal (`1 − 1/e ≈ 63%` of optimal) for
  monotone submodular maximization, but it can never *discover* teamwork —
  each leg looks worthless until its partner is already chosen, and greedy
  adds one element at a time.
- **Hungarian / linear-sum assignment** (the method top Halite bots used to
  assign every ship to a target in one optimal matching) assumes each
  `(source → target)` value is **fixed and independent** — precisely the
  assumption interference and teamwork both violate. It is optimal for a
  *linearized* version of the problem.

The solution must therefore do proper **conditional** evaluation (kills the
waste) **and** seed **teams as atomic candidates** (makes teamwork reachable).

### 2.1 Two facts that make this tractable and low-risk

1. **The simulator is fully deterministic** — the opponent policies in
   `lib/opp_model.py` (`lite_greedy_policy`, `top_tier_mirror_policy`,
   `me_defensive_action`) contain no RNG, and `lib/fast_sim.py` injects no
   stochasticity outside the fixed comet-spawn steps. So two rollouts from
   the same snapshot with the same injected actions are **bit-identical** →
   marginal gains are **exact**, no seed-averaging, common-random-numbers is
   free.
2. **The joint-rollout primitive already exists and is tested.**
   `chooser_trajectory.score_candidate_v4_joint` injects an arbitrary *list*
   of launches into one rollout and returns `leaf − baseline`. Sequential
   greedy is an **orchestration layer on tested physics**, not new physics.

---

## 3. The five rungs

### Rung 1 — Sequential greedy with re-simulation

Maintain a chosen set `S` (starts empty) and `leaf_S` (the joint-rollout leaf
of `S`). Each iteration, for each feasible candidate `c`, compute the
**conditional** marginal gain

```
gain(c | S) = V(S ∪ {c}) − V(S) = delta(S ∪ {c}) − delta(S)
```

(the shared `− baseline` cancels, so we compare deltas from
`score_candidate_v4_joint`). Pick the argmax; commit if above a floor, else
stop. Emit `S`.

This fixes waste automatically: once "reinforce X" is in `S`, a second
launch that also saves X is scored against a board where X is already saved →
~0 marginal gain → never picked. And it degrades gracefully — the **first**
pick's gain over the empty set equals today's solo score, so on a turn with
no interaction the planner reproduces current behavior.

### Rung 2 — CELF / lazy-greedy acceleration

Naive sequential greedy is `O(N·k)` rollouts (N candidates, k picks). Exploit
diminishing returns: keep candidates in a max-heap by last-computed gain;
re-evaluate only the **top** candidate at the current `|S|`; if it stays on
top, commit it without touching the rest (valid under submodularity). The
CELF / "Lazier than Lazy Greedy" literature reports order-of-magnitude
fewer evaluations. The lazy bound can be mildly violated by superadditive
candidates — Rung 3 mitigates by making teams atomic; an exact O(N·k) mode
stays as an A/B safety net.

### Rung 3 — Coalition atoms

Extract the synchronized-pair generator into a pure function that yields
**coalition atoms** (each = a multi-leg launches-list), and feed them into the
candidate pool as single items the greedy can pick **atomically** (an atom
occupies all its sources and its target). Now superadditive teamwork is
reachable in one greedy step, and the residual function on the augmented set
`{singletons} ∪ {coalitions}` is much closer to submodular — so lazy greedy is
both fast and approximately correct.

### Rung 4 — Multi-resolution horizon

Run the greedy *build* at a **shallow** horizon (cheap, broad search), then
run **one deep-horizon** confirmatory rollout of the final `S`. If depth
reveals the last-added element nets negative (e.g. the opponent recaptures by
tick ~30), drop it and re-confirm once. Keep the whole loop **anytime**:
process in priority order so a valid set is always emittable, and leftover
budget deepens the confirm. This respects the hard 1 s cap.

### Rung 5 — (later) Hungarian/LP seed + beam width

Use the closed-form move as a **proposer**, not a decider: build the
`(source × target)` matrix of hold-time values (`ρ_opp − ρ_me`, via
`predict_fleet_fate`) and solve the linear-sum assignment in ~1 ms
(`lib/joint_solver/lp.py`, scipy with greedy fallback) to seed the greedy with
a globally-coordinated starting set; the simulation-greedy then only refines
where independence breaks. **Beam search** (keep the top-W partial sets) is
the "spend more compute → better coordination" dial, for when greedy proves
to leave value on the table.

---

## 4. Compute budget

Per-rollout cost ≈ `H × (1–2 ms) × (num_seats − 1)`, dominated by the
reactive opponent policy per tick. The first full heap-init pass is `|pool|`
rollouts at the shallow horizon. The reallocation insight: we already pay for
~200 independent rollouts per turn today; spending roughly the same budget on
~80 **conditional** rollouts solves the coordination problem instead of
re-answering independent questions. Primary levers: shallow horizon (~12 vs
trajectory's ~25–40), anytime truncation of the init pass, and a pool cap
scaled down by `num_seats` in 4P.

---

## 5. Falsification / validation gates

This is a *new joint search*, distinct from the sync-coalition
*operationalisation* that underperformed live (panel-winner but μ≈1150,
below champion — `state/MULTI_BRANCH.md`). The math frame is sound; the
empirical question is whether better coordination moves our μ-band.

- Ships **default OFF** (`BASELINE_CHOOSER=greedy`); champion
  (`trajectory`) byte-for-byte unchanged.
- **Bench** (Rule 2): p99 under the working budget, no turn over the
  agent deadline.
- **n ≥ 32 A/B vs the rolling champion** (Rule 45) with Wilson-lo ≥ 0.50,
  **and a multi-opponent panel** (Rule 43) before any submission decision.
  (An n=8 smoke is triage only — never a lift claim.)
- **Calibration humility**: local-panel winners have repeatedly failed to
  transfer to the live ladder; treat any local lift as unconfirmed until the
  ladder agrees.

If, after the n≥32 + panel gates, the greedy planner does not beat the
trajectory champion, the honest read is that the coordination seam is not the
binding constraint at our μ-band, and the conclusion is to pivot — but the
machinery (conditional greedy, coalition atoms, deep-confirm) and the
deterministic-oracle insight remain durable substrate.

---

## 6. References

- `agents/baseline/chooser_trajectory.py` — `score_candidate_v4` (solo
  rollout), `score_candidate_v4_joint` (set rollout, REUSED), the inline
  sync-coalition generator (lines ~1266-1422, to be extracted), emit
  template (lines 1438-1504).
- `agents/baseline/chooser.py` — `affordable_validate_cap`,
  `opp_actions_for_snap`, `WALLCLOCK_HARD_CAP_MS`, `HARDCAP_BAIL_SENTINEL`.
- `lib/fast_sim.py` — deterministic `clone`/`step`/`from_obs` oracle.
- `lib/opp_model.py` — RNG-free reactive opponent policies.
- `lib/joint_solver/lp.py` — `build_assignment_matrix` + linear-sum
  assignment (Rung 5).
- `knowledge-base/concepts/reach-frontier-doctrine.md` §4.5 — the
  assignment-problem framing of the per-turn chooser.
- Research grounding: Halite top bots' Hungarian / linear-sum ship→target
  assignment; submodular maximization theory (CELF, "Lazier than Lazy
  Greedy") for the lazy acceleration and near-optimality framing.
- CLAUDE.md Rules 2, 40, 43, 45, 47 — bench discipline,
  modeling-over-restriction, multi-opp panel, n≥32 gate, physics-primitive
  reuse.
