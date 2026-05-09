# comp-context.md — settled-once facts (Orbit Wars)

Filled in once on Day 1 by the kickoff agent (Kaggle CLI auto-fill +
batch confirm with PI). Loaded on every session start. **Never re-asked.**

## Confirmed via `kaggle competitions list -s orbit` + comp files

```yaml
slug: orbit-wars
url: https://www.kaggle.com/competitions/orbit-wars
title: Orbit Wars
category: Featured
task: real-time strategy game (continuous-2D, 2 or 4 players); code/agent submission
submission_format: main.py with agent(obs) function — single file OR tar.gz with main.py at root
metric: TrueSkill-style Gaussian rating N(μ, σ²); μ₀=600 on submit
deadline: 2026-06-23 23:59 UTC
team_size_limit: TBD                     # confirm via comp rules page
submission_budget: 5                     # per day, per team
final_submissions: rolling last 2        # NOT PI-selected; auto-replaced as you submit
data_license: TBD
external_data_allowed: TBD
prize_pool_usd: 50000
total_teams_at_kickoff: 2382
local_simulator: kaggle_environments>=1.28.0   # `make("orbit_wars")`; verified working
```

## Comp-shipped files (downloaded by `bootstrap.sh` into data/)

```yaml
files_provided:
  - README.md       # full game spec — board, planets, fleets, comets, combat, observation/action format
  - agents.md       # getting-started guide; CLI workflow; submit examples
  - main.py         # working baseline agent ("Nearest Planet Sniper" heuristic)
```

## Game spec (from README.md + agents.md)

```yaml
board:
  size: 100x100 continuous (origin top-left)
  sun_center: [50, 50]
  sun_radius: 10                         # fleets that cross the sun are destroyed
  symmetry: 4-fold mirror around center  # ensures fairness regardless of starting position

planets:
  count: 20-40 (5-10 symmetric groups of 4)
  radius_formula: 1 + ln(production)
  production_range: [1, 5]               # ships generated per turn when owned
  starting_ships_range: [5, 99]          # skewed low
  rotation_rule: orbital_radius + planet_radius < 50  # inner = orbits sun; outer = static
  angular_velocity_range: [0.025, 0.05]  # radians/turn; randomized per game
  static_groups_minimum: 3
  orbiting_groups_minimum: 1
  home_planets:
    count: 1 group (4 planets) chosen randomly each game
    starting_ships: 10
    placement_2p: diagonally opposite (Q1 + Q4)
    placement_4p: one planet per player

fleets:
  representation: [id, owner, x, y, angle, from_planet_id, ships]
  speed_formula: 1.0 + (maxSpeed - 1.0) * (log(ships) / log(1000)) ** 1.5
  max_speed_default: 6.0
  collision: continuous (entire path segment checked, not just endpoint)
  destroyed_if: out_of_bounds | crosses_sun | collides_with_planet (→ combat)

comets:
  spawn_steps: [50, 150, 250, 350, 450]
  group_size: 4                          # one per quadrant; symmetric
  radius: 1.0
  production: 1
  speed_default: 4.0
  ships_starting: min of 4 rolls from [1, 99]   # heavily skewed low
  exit_behavior: removed when off-board (with garrison)

combat:
  rule_1: arriving fleets grouped by owner; ships summed
  rule_2: largest attacker fights second-largest; difference survives
  rule_3a: surviving attacker == planet owner → ships join garrison
  rule_3b: surviving attacker != planet owner → fights garrison; if attackers > garrison, ownership flips
  rule_4: two-way tie among attackers → all destroyed (no survivors)

scoring:
  termination: step_limit_500 | only_one_player_remains
  final_score: total_ships_on_owned_planets + total_ships_in_owned_fleets
  win: highest_score
```

## Agent IO (from main.py + README.md observation reference)

```yaml
observation:
  player: int                             # your player ID, 0-3
  planets: list of [id, owner, x, y, radius, ships, production]   # owner=-1 means neutral
  fleets: list of [id, owner, x, y, angle, from_planet_id, ships]
  angular_velocity: float                 # radians/turn (use to predict orbiting planet positions)
  initial_planets: list                   # starting positions (for orbit prediction)
  comets: list of {planet_ids, paths, path_index}   # full trajectories; path_index = current pos
  comet_planet_ids: list of int           # IDs in `planets` that are comets
  remainingOverageTime: float             # seconds remaining in overage budget

action:
  format: list of [from_planet_id, direction_angle_radians, num_ships]
  empty_action: []
  constraints:
    - from_planet_id must be a planet you own
    - num_ships ≤ planet's current garrison
    - multiple launches per turn allowed
  spawn: fleet appears just outside the planet's radius in the given direction

named_tuples:
  import: from kaggle_environments.envs.orbit_wars.orbit_wars import Planet, Fleet, CENTER, ROTATION_RADIUS_LIMIT
  Planet: (id, owner, x, y, radius, ships, production)
  Fleet:  (id, owner, x, y, angle, from_planet_id, ships)
```

## Turn order (READ CAREFULLY — load-bearing for orbit-prediction agents)

1. Comet expiration (off-board comets removed with garrisons).
2. Comet spawning (at the 5 designated steps).
3. Fleet launch (your action processed; new fleets created).
4. Production (all owned planets/comets generate ships).
5. Fleet movement + collision check (out-of-bounds / sun / planet hits).
6. Planet rotation + comet movement (moving planets can sweep stationary fleets into combat).
7. Combat resolution.

## Configuration defaults

```yaml
episodeSteps: 500
actTimeout: 1                             # seconds per turn — tight; budget your agent accordingly
shipSpeed: 6.0
sunRadius: 10.0
boardSize: 100.0
cometSpeed: 4.0
```

## Evaluation (from `kaggle competitions pages orbit-wars --content --page-name evaluation`)

```yaml
on_submit:
  step_1: validation episode (self-vs-self) — must pass or submission marked Error
  step_2: μ₀ = 600 initial skill rating
  step_3: matchmaking by similar μ; new submits play more frequently
ranking:
  family: TrueSkill-like Gaussian (μ, σ²)
  win: μ↑ for winner, μ↓ for loser; σ shrinks with each game
  draw: both μ pull toward their mean
  margin: ignored (only win/loss/draw counts; ship-count score does NOT affect rating updates)
final_evaluation:
  deadline: 2026-06-23 23:59 UTC          # additional submits locked at deadline
  post_deadline_period: ~2 weeks of continued ladder play to finalize
  visible_lb: best-scoring of your bots; track all via Submissions page
```

## Strategic decisions (PI-answered, batched on Day 1)

```yaml
external_data_strategy: TBD              # likely none; confirm from comp rules
time_budget_total_days: 45               # 2026-05-09 → 2026-06-23
compute_budget:
  local_sandbox: cpu_only
  kaggle_notebook: gpu_available         # P100 (single) or T4 x2 — for RL training only
gpu_workflow: kaggle_notebook            # heavy training on Kaggle
agent_class_preference: TBD              # heuristic / search / IL / RL / hybrid
```

## Pre-baseline gate (Day 1, code-comp variant — see pre-baseline-gate.md)

```yaml
gate_artifacts:
  brief: data/README.md                  # comp-shipped game spec (DO NOT modify)
  io_spec: audit/YYYY-MM-DD-day-1-data-inventory.md
  replay_summaries: audit/YYYY-MM-DD-day-1-data-inventory.md
  baseline_opponent_panel: audit/YYYY-MM-DD-day-1-data-inventory.md
  reference_kernel_replication: audit/YYYY-MM-DD-day-1-data-inventory.md
gate_status: TBD                         # cleared once PI signs off
baseline_opponents:
  - random                               # ships with kaggle_environments
  - main.py                              # comp-shipped Nearest Planet Sniper heuristic
  - <our_v0>                             # first heuristic variant (mirror baseline + tweak)
```

## Anti-patterns

- Don't re-ask any field above. Read this file instead.
- Don't add new strategic-decision fields after Day 1 without flagging
  it as a friction event in `audit/friction.md`.
- Don't put per-experiment results here. They go in
  `audit/YYYY-MM-DD-*.md` and the calibration ladder.
- Don't overfit to the random or shipped-baseline opponent — TrueSkill
  matchmaking will surface stronger opponents as μ rises; what beats
  random at μ=600 will lose at μ=900.
- Don't submit late in the day expecting to swap it out — the
  rolling-last-2 rule means your "second-to-last" submit is locked
  the moment you push another one.
