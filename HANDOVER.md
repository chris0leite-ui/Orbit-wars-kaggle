# HANDOVER.md — next-session brief

> Last written: 2026-05-10 (Day 1) by the bootstrap agent on branch
> `claude/orbit-wars-bootstrap-irewT`. Format budget ≤150 lines.

## Where we are

- **Comp:** Orbit Wars (slug `orbit-wars`). Deadline 2026-06-23 23:59 UTC →
  **44 days remaining**. Sponsor Google. $50k pool, 10×$5k prizes (1st–10th).
- **Submitted agent:** comp-shipped `data/main.py` (Nearest Planet Sniper),
  unmodified, as a calibration probe. **Submission ID 52497828**, pushed
  2026-05-10 00:09:54 UTC, status `PENDING` (validation episode running).
  **Submission budget used today: 1/5. Total: 1.**
- **Gap to top-5%:** unknown — leaderboard not yet read this session.
  Once submission flips to `COMPLETE`, read μ + LB to populate
  `state/current.md::tournament_rank_today` and
  `comp-context.md::headroom_to_top5pct`.
- **Repo state:** seed imported from
  `chris0leite-ui/Kaggle-playground-may-2026 @ claude/orbit-war-setup-KbeKq:orbit-wars-seed/`,
  pushed to `origin/claude/orbit-wars-bootstrap-irewT`. Day-1 audit
  + scripts added. `data/` populated by `bootstrap.sh`.
- **Pre-baseline gate:** all artifacts present (`our_v0` not yet built,
  not gating). Submitted agent serves as the calibration baseline.
- **Local environment:** Python 3.11, `kaggle 2.1.2`,
  `kaggle-environments 1.29.1`. Kaggle CLI auth requires
  `KAGGLE_API_TOKEN="$KAGGLE_KEY"` to be exported each shell — see friction.

## Today's progress

Load-bearing only; full detail in `audit/2026-05-10-day-1-data-inventory.md`.

1. **Comp-context TBDs filled** (rules + evaluation pages):
   team_size_limit=5, data_license=Apache-2.0, winner_license=CC-BY-4.0,
   external_data permitted (subject to Reasonableness), **NO ingress/egress
   during evaluation** (rules §2.12 — runtime models/data MUST be embedded
   in the submission). Prize structure clarified: 10 × $5k.
2. **Shipped baseline beats `random` 6/6** in 6 seeds, by elimination
   (n_steps 111–309). Calibration ceiling, not ladder strength.
3. **Self-play P0/P1 asymmetry — new finding (ISSUES.md A.6).**
   Baseline-vs-baseline: P1 wins 4/6, P0 wins 1/6, exact tie 1/6.
   All four 500-step games end with P1 ahead by 20–30% on ship count.
   Validation gate (Kaggle's self-vs-self pre-submit check) PASSES
   regardless — no crashes, all 6 reach `DONE`.
4. **Orbit-prediction math verified — A.1 done.**
   `scripts/orbit_prediction_check.py` proves the absolute formula
   `init_angle + ω·N` is **off by exactly one step's rotation**
   (~1.27 units on inner planets at orb_r≈31). Two correct alternatives:
   `ω·(N-1)` for the absolute case, or — preferred — relative projection
   from current obs by `ω·lead_turns`. Static planets confirmed
   non-drifting over 100 steps.
5. **First submission shipped (PI-approved):** ID 52497828, calibration
   probe = shipped baseline. Anchors μ-rating before any variant.
6. **Frictions logged** (`audit/friction.md`): KGAT_-token vs `kaggle.json`
   401, blinker pip conflict, seed-repo out-of-MCP-scope.

## Falsified-or-dead

- (none yet — no levers tried beyond the shipped baseline rollouts)

## Next-session first-action

Ranked. EV is qualitative on Day 1 (no calibration ladder yet); cost
is wallclock on local CPU.

1. **Read submission status + leaderboard** (cost: <1 min, EV: high).
   `kaggle competitions submissions orbit-wars` to confirm 52497828
   went through validation, then `kaggle competitions leaderboard
   orbit-wars -s` for the top-5% μ threshold and bot-name landscape.
   Update `state/current.md::tournament_rank_today` +
   `comp-context.md::headroom_to_top5pct`.
2. **B.1 heuristic v0 — overshoot variant** (cost: ~30 min coding +
   ~10 min self-play eval × M seeds, EV: medium-high). Send
   `garrison * 1.10 + 1` instead of `garrison + 1`; this absorbs
   one or two production ticks during fleet travel. Check the
   winrate against the shipped baseline on D.1's panel; goal ≥55%
   to clear the validation-gate analogue (Rule 3 / G13). **Do not
   submit on the same UTC day as a known-good submit lands** —
   rolling-last-2 means a third submit evicts the calibration probe
   before it has accumulated ladder games.
3. **D.1 local-tournament fixture** (cost: ~30 min, EV: high — every
   later experiment depends on it). Thin wrapper around
   `kaggle_environments.evaluate()` that returns a winrate matrix for
   `[random, baseline, our_vN] × M seeds`. Persist JSON output to
   `audit/`. Use 32 seeds for ±9pp 95% CI bootstrap to start; scale
   later per D.3.
4. **A.6 confirm/refute self-play asymmetry** (cost: ~10 min, EV:
   medium — could be load-bearing or could be variance with N=6).
   Patch the shipped baseline with random tie-breaks on equidistant
   targets (one-line change), re-run baseline-vs-baseline 32 seeds,
   measure P0 vs P1 winrate. If it equalises, asymmetry is the
   tie-break; if it doesn't, dig into turn-order semantics.
5. **(Defer)** RL training (B.4), reference-notebook pull (deferred
   per kickoff prompt; only on plateau).

## Pointers

- `audit/2026-05-10-day-1-data-inventory.md` — comp data + baseline
  probe + orbit-math verification + first-submission record.
- `audit/2026-05-10-day-1-rollouts.json` — raw rollout rewards/ship-counts.
- `scripts/run_day1_rollouts.py` — driver for the 6-seed × 2-pairing rollouts.
- `scripts/orbit_prediction_check.py` — absolute (off-by-one) and
  relative formula verification on seed 42.
- `data/README.md` — full game spec, comp-shipped (DO NOT modify).
- `comp-context.md` — settled facts, now with team_size, licences,
  ingress/egress policy, prize structure.
