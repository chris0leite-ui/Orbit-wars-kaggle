# Day-1 data inventory + baseline probe

**Branch:** `claude/orbit-wars-bootstrap-irewT`
**Date:** 2026-05-10 UTC (T-44 days; deadline 2026-06-23 23:59 UTC).
Kickoff began 2026-05-09 local time but the submission timestamp and
the bulk of the audit work crossed 00:00 UTC on 2026-05-10; the
submission-day stamp is the canonical one and matches `state/current.md`.
**Scope:** Step 0 prerequisite checks → Step 1 seed import → Step 2 day-1
discovery (a)–(g). **One submission today (calibration probe — shipped
baseline, ID 52497828).** No RL training run. No external notebooks
pulled (deferred per `agent-handover-prompt.md`).

## Environment + creds

- **GitHub:** authenticated as `chris0leite-ui` via MCP.
- **Kaggle:** auth via `KAGGLE_USERNAME` + `KAGGLE_KEY` (KGAT_… form).
  Initial `kaggle competitions list -s orbit` returned 401 with only
  `~/.kaggle/kaggle.json`; resolved by exporting `KAGGLE_API_TOKEN=$KAGGLE_KEY`
  alongside the file. **Friction → see `audit/friction.md`.**
- **Python deps:** first `pip install -r requirements.txt` failed on
  uninstalling system `blinker`; resolved with `pip install --ignore-installed blinker`.
  Final pinned versions: `kaggle 2.1.2`, `kaggle-environments 1.29.1`,
  `numpy`, `pandas`, `pytest`, `ipykernel`.
- **Local simulator:** `make("orbit_wars")` smoke-test PASSES
  (random-vs-random, seed 42 → P0 reward 1, P1 reward -1, both DONE).

## Comp registration (verified)

- `kaggle competitions list -s orbit` →
  `userHasEntered: True`, deadline `2026-06-23 23:59:00`,
  category `Featured`, reward `$50,000`, `teamCount=2413`
  (was 2382 at seed-write time on 2026-05-04).
- Joined competition: ✓.

## Comp-context TBDs filled (rules + evaluation pages)

Source: `kaggle competitions pages orbit-wars --content --page-name {rules,evaluation}`.

| Field | Resolved value |
|---|---|
| `team_size_limit` | 5 (rules §2.1.a) |
| `data_license` | Apache 2.0 (rules §1.7 + §2.4.a.2) |
| `winner_license` | CC-BY 4.0 (rules §1.6 + §2.5.a.1) |
| `external_data_allowed` | yes — public + equally-accessible OR Reasonableness Standard (rules §2.6) |
| `ingress_egress` | **prohibited during evaluation** (rules §2.12) — runtime models/data MUST ship inside submission. No live API calls in `agent(obs)`. |
| `prize_pool_usd` | $50,000 (10 prizes of $5,000 — 1st through 10th place) |
| `sponsor` | Google LLC |
| `post_deadline_eval` | ~2 weeks of continued ladder play |
| `final_submissions` | rolling last 2 — evaluation page: "we only track the latest 2 submissions"; rules §2.2.b boilerplate phrases this as "select up to 2", but the sim-comp evaluation page is the operative spec |

## Comp-shipped data inventory

`data/` (3 files, 16,806 bytes total; downloaded 2026-05-10 via
`bootstrap.sh`):

| file | size | sha-ish (md5) | role |
|---|---:|---|---|
| `data/README.md` | 8,241 B | `ef043d6f54c00693e38b0ea75a2a0110` | full game spec — board, planets, fleets, comets, combat, observation/action format |
| `data/agents.md` | 6,486 B | `a10f6ac24f08902c317892c53998b9ff` | getting-started; CLI workflow; submit examples |
| `data/main.py` | 2,079 B | `28c902ccca072aff2313be5270b5eaa4` | shipped baseline — Nearest Planet Sniper |

## Shipped-baseline rollouts (6 seeds)

Driver: `scripts/run_day1_rollouts.py`. Seeds: `[42, 1, 7, 13, 31, 100]`.
Raw JSON: `audit/2026-05-10-day-1-rollouts.json`.

### Baseline (P0) vs `random` (P1) — 6/6 wins for baseline

| seed | rewards | n_steps | final_ships (P0/P1) |
|---:|---|---:|---|
| 42  | [1,-1] | 163 | 2264 / 0 (P1 eliminated) |
| 1   | [1,-1] | 136 | 1531 / 0 |
| 7   | [1,-1] | 111 | 1844 / 0 |
| 13  | [1,-1] | 227 | 3532 / 0 |
| 31  | [1,-1] | 147 | 1604 / 0 |
| 100 | [1,-1] | 309 | 2208 / 0 |

**Read:** the shipped baseline beats `random` by elimination (well
before the 500-step cap) on every seed tested. Random is incoherent;
this is a calibration ceiling, not an indicator of ladder strength.
Per the kickoff prompt: "what beats random at μ=600 will lose at μ=900."

### Baseline (P0) vs Baseline (P1) — validation-gate test, 6 seeds

| seed | rewards | n_steps | P0 ships | P1 ships | outcome |
|---:|---|---:|---:|---:|---|
| 42  | [-1, 1] | 500 |  661 |  855 | **P1 win** |
| 1   | [ 1,-1] | 269 | 4420 |    0 | P0 win (P1 eliminated) |
| 7   | [-1, 1] | 299 |    0 | 5380 | P1 win (P0 eliminated) |
| 13  | [ 1, 1] | 500 | 1307 | 1307 | exact tie |
| 31  | [-1, 1] | 500 |  828 | 1026 | P1 win |
| 100 | [-1, 1] | 500 |  847 | 1023 | P1 win |

**Validation-gate result:** ✅ all 6 self-play episodes reach `DONE`,
no errors. Kaggle's pre-submit validation episode (self-vs-self)
will pass.

**Asymmetry finding:** P1 wins 4/6, P0 wins 1/6, ties 1/6. All four
long (500-step) games end with P1 ahead by 20–30% on ship count
(seeds 42, 31, 100). Hypothesis: tie-breaking on equidistant targets
routes both players to the same neutral planet; lower player-id
launches first; P0's fleet arrives first; P0 captures and P1 reroutes
to a fresher target. Logged as **ISSUES.md A.6**.

## Orbit-prediction math (load-bearing — A.1)

Driver: `scripts/orbit_prediction_check.py`. Seed 42, target step 100.
4 orbiting planets (id 12,13,14,15; orb_r≈31.10; ω=0.040986 rad/turn).

| formula | max error (units) | verdict |
|---|---:|---|
| absolute, `omega * N`              | 1.27468 | **WRONG** — off by exactly one step's rotation |
| absolute, `omega * (N - 1)` for N≥1 | 0.00000 | exact |
| relative (current obs + `omega * T`) | 0.00000 | exact, no step counter needed |
| static planets over 100 steps       | 0.00000 | confirmed no drift |

**Why it matters:** on inner planets at orb_r≈31, the naive `omega*N`
miss (1.27 units) is roughly 25% of an inner planet's diameter
(r=2.39, diameter 4.77) — a fleet aimed at the predicted intercept
can miss the planet entirely on small targets, even ignoring radius
shrinkage. Recommend agents **always** project from current observed
positions, never from `initial_planets` + `step`.

**Source-of-truth:** `kaggle_environments/envs/orbit_wars/orbit_wars.py`
lines 519–548 — `step = get(obs0, "step", 1)`, but `env.steps[N]` is
the snapshot before that step's rotation, so empirically the rotation
count at `env.steps[N]` is `N - 1`.

## Pre-baseline gate (per `comp-context.md`)

| artifact | path | status |
|---|---|---|
| brief | `data/README.md` | ✅ on disk (DO NOT modify) |
| io_spec | `audit/2026-05-10-day-1-data-inventory.md` (this file) + `comp-context.md` | ✅ |
| baseline_opponent_panel | `random` + `data/main.py` (Nearest Planet Sniper) | ✅ — both runnable; `our_v0` deferred to next session |
| reference_kernel_replication | deferred (comp ships its own working baseline; pull external kernels only on plateau) | (skipped per kickoff prompt) |
| gate_status | TBD — awaiting PI sign-off | open |

## Open items for next session

1. **B.1 — Heuristic v0**: simplest 1-step improvement on shipped
   baseline. Options ranked by simplicity / expected lift:
   (i) overshoot — send `garrison * 1.10 + 1` instead of `garrison + 1`,
   to absorb production growth during fleet travel;
   (ii) production-weighted target choice — pick max
   `production / (distance + 1)` instead of nearest;
   (iii) randomise tie-breaks on equidistant targets to defeat the
   P0/P1 asymmetry from A.6.
2. **D.1 — local-tournament fixture**: thin wrapper around
   `kaggle_environments.evaluate()` that returns a winrate matrix
   for `[random, baseline, our_v0..vN] × M seeds`.
3. **A.6 — investigate self-play asymmetry**: confirm or refute the
   tie-break hypothesis with a 1-line patch (random tie-break) and a
   re-run.
4. **First submission shipped** (PI-approved): submission ID 52497828
   pushed at 2026-05-10 00:09:54 UTC, file `data/main.py` (shipped
   baseline, unmodified), status PENDING. Submission slot used today:
   1/5. Read μ-rating + leaderboard once status flips to COMPLETE
   (validation episode + first ladder games typically settle within a
   few hours).
