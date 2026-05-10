# Strategy roadmap — v2 → v4

This file pairs with `ISSUES.md::B` (agent class) and the plan at
`/root/.claude/plans/flickering-tinkering-horizon.md`. Public research
touchstones are in `external/kernels/`.

## Where we are

- **v0 (shipped):** Nearest Planet Sniper, μ=303 on the live ladder.
- **v1 (orbitfix):** orbit-aware aim + tie-break randomisation. 40/40
  vs baseline locally. Live μ pending.
- **Public top:** Roman Tamrazov μ=1224 (single-file ~3300-line heuristic).
  Public top-5% threshold ≈ μ 1100-1200.
- **Our target:** top-5% by 2026-06-23 → μ ≥ ~1100.

## The mechanism gap (v1 → top-5%)

Public-cluster agents share these mechanisms that v1 lacks:

1. **Arrival-time ownership forecasting** — for every (planet, future_step),
   maintain a ledger of incoming friendly + enemy fleets and forecast who
   will own the planet at each arrival. Used to skip targets that will
   already be ours, attack targets that need a bigger commitment, and
   defend planets about to flip.
2. **Same-turn combat resolver** — multiple fleets landing on the same
   step are grouped by owner, summed, and resolved per the README rules
   (largest vs second-largest, garrison combat, two-way-tie destruction).
   Lets us send the *exact* tying count to neutralise enemy assaults.
3. **Sun-safe pathing** — continuous segment-vs-sun check; if a direct
   shot crosses the sun, route via a waypoint (a third planet's
   approach angle, or an angular detour).
4. **Fleet coordination** — multiple owned planets contribute to the
   same target with arrival timing aligned so the cumulative force
   exceeds garrison + reinforcements.
5. **Comet path harvesting** — comets have known trajectories
   (`obs["comets"][].paths` + `path_index`). Predict where a comet
   will be at our fleet's arrival, capture it, exploit its production
   for the few-dozen turns it remains on board.
6. **Defence & home protection** — keep enough garrison to absorb the
   largest incoming enemy fleet expected within `K` turns.
7. **Mission classification** (Roman): snipe / rescue / recapture /
   reinforce / crash-exploit / gang-up / elimination as named action
   classes, each with its own scoring rule, with a solver picking
   the best mission per source-planet under a global budget.

## v2 — `arrival_ledger`

**Reference:** `external/kernels/structured-baseline/` (Pilkwang, 197v).

**Adds:** mechanisms 1, 2, partial 4 (avoid double-targeting via the
ledger).

**Outline:**

- New `lib/combat.py`: pure function
  `resolve(garrison_owner, garrison_ships, arrivals: list[Arrival]) -> (new_owner, new_ships)`
  matching README rules 1-4. Three TDD scenarios (single attacker, two
  same-owner reinforcement, two-way tie).
- New `lib/actions.py`: `validate(launch, obs) -> (ok, reason)` — owner,
  garrison sufficiency, no double-spend on the same source-planet turn.
- New `agents/v2_arrival_ledger/main.py`: per turn, build a per-planet
  timeline using known fleet positions + speeds + projected angles.
  For each owned planet, propose launches that flip targets the
  ledger says are otherwise lost. Skip targets that will already be ours.
- **Local gate:** v2 vs v1 ≥ 60% over 24 seeds × both sides;
  p95 turn < 300 ms.
- **Predicted live μ:** 700-900.

## v3 — `missions`

**Reference:** `external/kernels/roman-1224/` (Roman, 124v, μ=1224 outlier).

**Adds:** mechanisms 3, 4 (full), 5, 6, partial 7 — only `snipe`,
`recapture`, `reinforce` initially; `gang_up`, `crash_exploit`,
`elimination` deferred to v4.

**Outline:**

- New `lib/world_model.py`: arrival ledger + planet timeline simulator
  + exposed-enemy detection.
- Refactor v2's combat resolver into the timeline simulator.
- New `agents/v3_missions/main.py`:
  - For each (source planet, target candidate), build one mission
    proposal per applicable class.
  - Score each by expected value + cost (ships + turns out of garrison).
  - `settle_plan` solver assigns ships to mission proposals under a
    no-overlap constraint.
- Add sun-safe pathing here (mechanism 3) since several mission classes
  need it.
- **Hard gate (E.2):** `evaluate(env, [v3, v3], 10)` zero crashes.
- **Local gate:** v3 vs v2 ≥ 58% over 30 seeds × both sides;
  p95 turn < 600 ms; 100-seed soak no timeouts.
- **Predicted live μ:** 1000-1150.

## v4 — branch on data

After v3 lands a real μ:

- **Path A (μ ≥ 950):** v4 = v3 + remaining Roman missions
  (`gang_up`, `crash_exploit`, `elimination`) + lightweight opponent
  modelling (track enemy launch latency; react faster than mean).
- **Path B (μ < 950):** v4 = depth-2 MCTS over top-K mission combos
  from v3's solver, budget-capped 400 ms.

## The RL kill-switch (parked unless triggered)

Open `B.4` only if **both** (i) v3 lands μ < 950 **and** (ii) v4 path-A
delta < +50 μ. Trigger date ≈ day 18 (around 2026-05-28). Roman 1224
proves heuristic alone reaches top-5%, so RL is a fallback, not the plan.

Public RL reference if needed: `external/kernels/rl-tutorial/` (YumeNeko,
PPO self-play, per-planet policy). Underperforms the public heuristic
top, so the path of least resistance is finishing the heuristic stack
first.

## Submission economy reminder

- 5 submits/day; **rolling-last-2** for final.
- Submit only when local panel winrate ≥ 58% vs current ladder
  champion AND p95 turn < 500 ms.
- Never push speculative variants on the same UTC day as a known-good
  submit (CLAUDE Rule 12 caveat — third submit evicts the second).
- Run `scripts/pre_submit_diff.py` (TODO; v3 onward) — 10-game
  head-to-head vs previous submit; abort if < 55%.
