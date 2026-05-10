# ISSUES.md — problem decomposition / claim board

> Live problem-tree per Rule 18. A leaf must be claimed before any
> probe ≥10 min CPU/GPU. Status values: `open`, `wip`, `done`, `null`,
> `parked`. Owner is the branch/agent currently working it.

## Top-level

**Goal:** Build an Orbit Wars agent that finishes top-5% on the
TrueSkill ladder by 2026-06-23 23:59 UTC. Initial μ₀=600; target
μ ≈ public-LB-top-5% threshold (TBD; agent fills after first
`kaggle competitions leaderboard orbit-wars -s` call).

## Active leaves

### A. Env dynamics — understand the game before building agents

- **A.1 Orbit prediction**: given `initial_planets` + `angular_velocity`,
  predict orbiting-planet positions at step t with <0.5 unit error
  over 100 turns. Verify against the live env. Load-bearing for any
  non-myopic agent. `[owner: orbit-wars-bootstrap-irewT | status: done]`
  → see `scripts/orbit_prediction_check.py` + `audit/2026-05-10-day-1-data-inventory.md`.
  **Off-by-one finding (load-bearing):** the naive absolute formula
  `angle = init_angle + omega * N` is WRONG for `env.steps[N]` — it
  predicts one rotation too many, miss is ~1.27 units on inner planets
  (orb_r≈31, omega≈0.041). Two correct alternatives:
  (i) absolute with `omega * (N - 1)` for `N >= 1`;
  (ii) relative — read planet from current obs and project forward by
       `omega * lead_turns`. Recommend (ii) for agents (no step counter
       to track). Verified to 0.0 error on seed 42 across 4 inner planets
       at step 100, and against env.steps[N] for N in {1,5,10,100,200,499}.
- **A.2 Fleet speed + travel time**: validate
  `speed = 1 + 5·(log(ships)/log(1000))^1.5` against the env.
  Tabulate (ships → speed → turns-to-cross-board). `[owner: review-competition-handover-0pGNc | status: done]`
  → see `lib/fleet.py` + `tests/test_fleet.py` (12 tests). speed(1)=1.0,
  speed(1000)=6.0, speed(500)≈5.0 (matches README "~5"), monotonic in
  ships, clamps at 1000+. `eta_turns` ceils partial turns.
- **A.3 Combat resolution**: walk through the README rules with 3
  hand-built collision scenarios (single attacker, two same-owner,
  two-way tie). Confirm the env matches. `[owner: unclaimed | status: open]`
- **A.4 Comet timing**: spawn at steps 50/150/250/350/450; group of 4;
  ship-count is min-of-4-rolls (heavy-skew low). Worth grabbing? `[owner: unclaimed | status: open]`
- **A.5 Sun collision geometry**: continuous path-segment check, not
  endpoint. What's the maximum safe angle from a planet near the sun
  to another planet beyond it? `[owner: review-competition-handover-0pGNc | status: partial]`
  → `lib/geometry.path_clears_sun(src, dst, safety)` implements the
  point-to-segment distance check with an optional safety margin
  (10 tests in `tests/test_geometry.py`). The "max safe angle"
  question itself is not yet answered analytically — that lands when
  v2 needs it for re-targeting around the sun.
- **A.6 Deterministic self-play P0/P1 asymmetry**: shipped baseline
  vs itself wins **4/6 for P1** (seeds 42, 7, 31, 100), 1/6 P0 win
  (seed 1), 1/6 ties (seed 13). Hypothesis confirmed by v1 fix:
  randomising tie-breaks with `random.Random(step ^ player_id_salt)`
  collapses the asymmetry to 5 P0 / 4 P1 / 11 draws over 20 seeds —
  |P0-P1|=5%, well within the ±15% gate.
  `[owner: review-competition-handover-0pGNc | status: done]`
  → `agents/v1_orbitfix/main.py` line ~63 (rng seeded per turn);
  `audit/tournaments/20260510T075217Z.json` records the 80-game grid.

### B. Agent class — pick the simplest class that beats baselines

- **B.1 Heuristic v0**: 1-step improvement on shipped Nearest Planet
  Sniper. Variants: send 110% of garrison instead of +1; weight target
  selection by production rather than distance; ignore home-planet
  defence. `[owner: review-competition-handover-0pGNc | status: done]`
  → folded into v1 (A.6 tie-break randomisation was the live lever;
  overshoot/production-weight not yet tried — keep as v2/v3 ablations).
- **B.2 Heuristic v1**: production-aware + orbit-aware (fire at where
  the planet WILL be at impact, not where it is now). Uses A.1 + A.2.
  `[owner: review-competition-handover-0pGNc | status: done]`
  → `agents/v1_orbitfix/main.py::_aim_angle` does one-step fixed-point
  iteration (arrival_time → lead_position → arrival_time') for
  orbiting non-comet targets; static targets and comets aim at
  current position. v1 vs baseline 40/40 wins on 20×2 seed grid.
- **B.3 Search-based**: minimax / MCTS over short horizons (5-10
  turns) of fleet-launch decisions. Branching factor is huge —
  needs heuristic-pruned action space. `[owner: unclaimed | status: open]`
- **B.4 RL**: PPO/IMPALA self-play with opponent-pool curriculum.
  Heavy compute; defer until heuristic plateau. `[owner: unclaimed | status: open]`
- **B.5 Hybrid**: heuristic policy with learned value head, OR IL
  warm-start on top-LB replays then RL fine-tune. `[owner: unclaimed | status: open]`

### C. Reward / value signal — Q6 metric alignment (Rule 16)

- **C.1 Local proxy choice**: winrate vs panel = [random, shipped
  baseline, our v0..vN]? Or expected-final-ship-count? Or expected
  μ-gain via rough TrueSkill simulation? `[owner: unclaimed | status: open]`
- **C.2 TrueSkill-aware target**: the live ladder matches by similar
  μ — what beats random at μ=600 will lose at μ=900. The local
  opponent panel must scale with our μ. `[owner: unclaimed | status: open]`
- **C.3 Reward shaping for RL** (deferred until B.4 active): dense
  reward (planet captures, fleet kills) vs sparse (terminal win).
  Bias risk on dense rewards. `[owner: unclaimed | status: open]`

### D. Training / eval infra

- **D.1 Local-tournament fixture**: `kaggle_environments.evaluate()`
  wrapper that runs N agents × M seeds × pairs, returns winrate
  matrix. Persistent JSON output for trend tracking. `[owner: review-competition-handover-0pGNc | status: wip]`
- **D.2 Replay logging**: capture `env.steps` + episode metadata for
  every local game. Disk usage: ~1-5 MB per episode JSON. Plan for
  ≥1000 episodes. `[owner: unclaimed | status: open]`
- **D.3 Seed budget**: how many seeds before winrate ±2pp confidence?
  Bernoulli at p=0.5 → ~625 games for ±2pp at 95% CI. Cheaper:
  bootstrap-CI on smaller sample. `[owner: unclaimed | status: open]`
- **D.4 Hold-out opponent**: opponents reserved for end-of-cycle
  eval, never seen during agent design. Prevents overfit-to-panel. `[owner: unclaimed | status: open]`

### E. Submission packaging

- **E.1 Single-file vs tar.gz**: when does a single `main.py` stop
  being sufficient? (e.g. when we ship learned weights or a
  dependency.) `[owner: unclaimed | status: open]`
- **E.2 Validation-episode dry-run gate (G13)**: run
  `kaggle_environments.evaluate(env, [agent, agent], 10)` locally
  before every submit. Kaggle runs the same self-vs-self check on
  upload — if it errors there, the submit is wasted. `[owner: unclaimed | status: open]`
- **E.3 Compute budget per turn**: `actTimeout=1` second. Profile
  worst-case `agent(obs)` wallclock; flag any branch >500 ms. `[owner: unclaimed | status: open]`
- **E.4 Rolling-last-2 cadence**: never push a speculative variant
  on the same UTC day as a known-good submit. The known-good gets
  evicted as soon as a third lands. `[owner: unclaimed | status: open]`

## Falsified or dead

(empty)

## Re-decomposition triggers

- 3 nulls in a row on the same leaf → re-decompose that subtree.
- 50% of comp budget elapsed (≈22.5 days, around 2026-05-31) →
  review tree against current LB shape.
- Plateau ≥2 days on PRIMARY μ → research-loop + re-decompose.
- TrueSkill σ stable but μ stuck → opponent-pool diversity is the
  likely culprit; revisit C.2 + D.4.
