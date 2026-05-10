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

---

## Day-1 PM simple-trading-strategies-QS0xV

> Per WRAPUP parallel-branch convention. The next morning's scribe
> consolidates this into the synthesis sections above and removes
> this block.

### Where we are (PM update)

- **v1.2 simple/roi submitted as ID 52518060** (PENDING; validation
  episode running). Pushed 14:59 UTC after one Kaggle 503 retry.
- Rolling-last-2 now: `[v1.1 (μ=597.4), v1.2/roi (PENDING)]`. v1
  (μ=568.0) evicted. **4/5 submissions used today; 1 slot left.**
- v1.1 settled at μ=597.4 earlier in the session (gain +89 over v1).
- Predicted live μ for ROI: 700–1000 (based on 100% local vs v1 and
  v1's +205 vs baseline). Top-5% threshold ≈ 1100 → roi could close
  ~half the gap in one push if predictions hold.

### Today's progress (load-bearing only)

1. **Simple-strategy panel** — five target-selection strategies under
   `agents/simple/` sharing v1.1's mechanism stack. 32-seed result
   (audit/tournaments/20260510T140907Z.json):
   - **roi: 97.1% mean panel WR, 100% (64/64) vs v1_orbitfix.**
   - production 67.7%, nearest 56% (ties v1 by construction),
     enemy_first 33%, weakest 19% (last two falsified).
2. **Phase 1 meta-strategy infrastructure** — replay capture
   (`scripts/tournament.py`), `lib/fingerprint.py` (15-feature
   behavioural fingerprint), `scripts/manifold_check.py` (RF + LR
   with GroupKFold-by-seed CV), prior-art survey.
3. **Phase 1 manifold gate ❌ NOT cleared.** Best K ≤ 100 result on
   the 5-strategy zoo: RF 80.5%, LR 80.6%; target was 90%. ROI-family
   (nearest/production/roi) collapses into one basin (12-17% mutual
   confusion). Cleanly-separated basins: weakest (89.7%), enemy_first
   (83.4%), baseline (95% in 7-class). Verdict + paths forward in
   `audit/2026-05-10-phase1-manifold-verdict.md`.
4. **Bundler extended** to accept flat-file agents
   (`agents/simple/<n>.py`); 151 tests still green.
5. **Plan rewritten** to phased 5-step roadmap (replay capture →
   manifold check → zoo expansion → BR table → meta-router →
   submission). See `/root/.claude/plans/read-the-handover-next-imperative-whisper.md`.

### Falsified-or-dead

- **simple/weakest** (argmin target.ships, snipe-cheap hypothesis):
  19% mean WR at 32 seeds; 0/16 vs nearest, production, roi, v1.
  Kept as opponent-panel diversity (D.4) — strategically distinct
  weak agent is useful for hold-out eval.
- **simple/enemy_first** (pressure-on-opponent hypothesis): 33% mean
  WR at 32 seeds; 8 of 8 self-play seeds → all draws (when both
  sides ignore neutrals, games stalemate). Kept for D.4 same as above.
- **Linear small-dim manifold** as the framing for opponent
  classification — refuted by AlphaStar-style basin clustering at
  the 7-class level. The "strategies live in discrete basins" version
  of the hypothesis still holds.

### Next-session first-action

1. **Read v1.2/roi μ once validation completes** (cost: <1 min,
   EV: high — calibration data point for the rest of the comp).
   `kaggle competitions submissions orbit-wars` should show
   #52518060 flip from PENDING to COMPLETE with a publicScore.
   Append the actual μ to `state/calibration-ladder.md` and reconcile
   against the +200-500 prediction. **Critical:** if μ < 597.4
   (regression vs v1.1), investigate before pushing any further
   variant — could be a bundling bug we missed (parity gate was
   only 4 seeds), or genuine ROI weakness on the live ladder vs
   a class of opponents our local panel didn't surface.
2. **PI choice on Phase 2 path** (no compute cost; sets next batch).
   - **A) Coarsen labels** (cheap; merge ROI-family → 3-class router;
     gate likely clears at ~92%). Recommended first.
   - **B) Richer fingerprint** (half-day; distribution-shape +
     temporal-split + target-id Shannon entropy; FEATURE_VERSION→2;
     gate likely clears full 5-class).
   - C) Learned embedding (Grover et al.) — last resort.
3. **Phase 2 zoo expansion** — once path A or B clears the gate,
   add ~12 more strategies (sizing / coordination / defence / hybrid
   axes per the plan §"Strategy zoo expansion shape"). Triggers
   the parallel-runner deferred infra (`ISSUES.md::D.5`) when the
   ~9k-game panel becomes a wallclock blocker.
4. **(Defer)** Phase 3 BR table + meta-router; Phase 4 bundle-with-
   classifier-weights; B.4 RL fallback.

### Pointers (added this session)

- `audit/2026-05-10-simple-strategy-panel.md` — Phase 0 result.
- `audit/2026-05-10-meta-strategy-prior-art.md` — Grover, DRON, PSRO,
  AlphaStar, Pluribus, Bayes-Bluff (1-page survey).
- `audit/2026-05-10-phase1-manifold-verdict.md` — gate failure
  diagnosis + paths forward.
- `audit/2026-05-10-postmortem-simple-trading-strategies-QS0xV.md` —
  this session's postmortem (PI ratification deferred).
- `audit/tournaments/20260510T140907Z.json` — 32-seed × 7-agent panel.
- `audit/replays/20260510T132957Z/` (gitignored, 404 MB) — 1568
  replays for Phase 2 fingerprint reuse.
- `submissions/roi.py` — staged single-file bundle (PI approval pending).
