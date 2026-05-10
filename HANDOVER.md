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

---

## Day-1 evening improve-strategy-ab-testing-jYA2R

### Where we are

- **v1.2/roi μ=1104.9 confirmed live.** Top-5% threshold ≈1100 (we're
  at the bracket). #1 = 1641.7. Rolling-last-2 holds `[v1.1 (587),
  v1.2/roi (1105)]`. **No new submit this session — PI gated.**
  4/5 submission slots used today; 1 left.
- **lead_aim ETA bug found and fixed** (commit `cbf142b`). Env spawns
  fleet at `src + (r_src + 0.1)·dir`, captures when entering
  `target.radius` — `lead_aim` used center-to-center distance for ETA,
  putting the lead too far ahead → systematic miss in the orbit-forward
  direction. Fix subtracts the offset from flight distance. 160 tests
  pass. A/B vs live: 47/0/53 (tied within Wilson 95% noise) — neutral
  on self-vs-self, expected to lift against varied ladder opponents.
- **Strategic direction set (PI):** Take Kaggle simulation as correct
  and stable (no version-drift hedging). From here, two halves:
  (1) ensure strategy execution is physically accurate (no ships lost
  to wrong modelling); (2) move from per-source greedy to look-ahead
  with joint global decisions, fleet-in-flight awareness, and ROI-
  threshold pruning. Full voice-dump:
  `knowledge-base/thoughts/2026-05-10-pi-direction-physics-then-lookahead.md`.

### Today's progress (load-bearing only)

1. **Physics audit + ETA correction (cbf142b).** Read env source
   (`/usr/local/lib/python3.11/dist-packages/kaggle_environments/envs/
   orbit_wars/orbit_wars.py`); verified fleet-launch, planet-rotation,
   sun-collision, swept-pair-collision semantics. Empirical probe
   confirmed src does NOT rotate between obs step N and the transition
   to N+1 (consequence of the A.1 off-by-one).
2. **Tournament loader fix.** `scripts/tournament._load_agent` now
   registers the module in `sys.modules` before `exec_module` so
   `@dataclass` resolves inside single-file bundled agents like
   `submissions/roi.py` (was breaking the A/B harness against the live
   submission).
3. **Parallel runner discovered already in place.** `scripts/tournament
   .run_tournament` accepts `workers: int` and `scripts/strategy_panel
   .py` exposes `--workers N`. The plan's D.5 is shipped; we were just
   running default `workers=1`. With `--workers 4` on this 4-core box:
   32-seed × 4-agent panel in 1m12s vs ~6m sequential.
4. **Deterministic-correctness ablations** (audit/tournaments/19*.json).
   sun_avoid in DEFAULT_MECHANISMS + strategy pivot, 32-seed vs live:
   regresses (42/6/52 with 3-iter; 16/75/9 with 2-iter — heavy
   stalemate). ETA-fix alone: 47/0/53. Diagnosis: `sun_avoid` checks
   `target.x, target.y` (current) instead of the lead-predicted
   arrival point — wrong for orbiting targets. Same flaw in any
   strategy-side sun pivot.

### Falsified-or-dead

- **sun_avoid in current form** (checks current target position). Two
  ablation attempts regressed vs v1.2/roi anchor. Mechanism stays
  EXCLUDED from `DEFAULT_MECHANISMS` until the arrival-point fix
  lands (punch #7).
- **3-iter lead_aim, standalone** (without the ETA fix). Regressed
  42/52. Re-test queued with the ETA fix on top.

### Next-session first-action

PI-ratified order; dependency-respecting:

1. **Sun-avoid arrival-aware fix** (punch #7, ~30 min). Reuse
   `predict_relative(target, omega, eta)` to compute the arrival point,
   then `path_clears_sun(src.center, arrival_xy, safety=1.0)`. Both
   the mechanism and any strategy-side pivot share this helper. Re-
   promote `sun_avoid` to `DEFAULT_MECHANISMS` and re-A/B.
2. **3-iter lead_aim retest** combined with the ETA fix (punch #8,
   ~5 min A/B). Quick check; cheap.
3. **Capture-success probe** (~20 min). Instrument a roi run; count
   per-fleet (declared target reached? died in sun? out of bounds?
   still in flight at episode end?). Quantifies the accuracy story —
   tells us if #7-#8 are μ-relevant or only theoretical.
4. **Read `obs.fleets` everywhere.** Today's strategies all ignore
   in-flight fleets. First use-case: don't double-commit a source's
   garrison to a target that already has our fleet arriving with
   enough ships. Lives in a new `arrival_ledger` mechanism — propose
   only if `mine.ships > target.ships_at_arrival − our_already_arriving`.
5. **ROI-threshold pruning.** Cheap; isolates action space for later
   work. Top-K filter on the target list + per-owner-class threshold.
6. **(Defer)** Joint global decisions (bipartite assignment, gang-up
   timing). Builds on #4 + #5.
7. **(Defer)** Look-ahead search (beam / mini-MCTS) — only after the
   above are stable.
8. **(Defer)** ROI scoring variants (V1-V5 from earlier plan) — fold
   into the joint global solver, not as standalone per-source variants.
9. **(Defer)** Submission cadence: only after the capture-success probe
   shows the local fixes change game outcomes meaningfully.

### Pointers (added this session)

- `cbf142b` — ETA-offset fix + tournament loader fix + PANEL_OBS
  redesign + 9 ablation JSONs.
- `knowledge-base/thoughts/2026-05-10-pi-direction-physics-then-lookahead.md`
  — PI voice-dump (Rule 35); the north-star reasoning behind the
  next-session ordering above.
- `audit/tournaments/20260510T192940Z.json` — ETA-fix A/B vs live
  (47/0/53, 0 draws, --workers 4 in 19s).
- `audit/tournaments/20260510T19{0047,0224,0605,0823,1113,1630,1745,1836}Z.json`
  — the bisection trail through deterministic-fix combinations.

---

## Day-1 night competition-strategy-brainstorm-ZK6XT

> The morning scribe consolidates this into the top-of-file blocks.

### Where we are (night update)

- **Live submission v1.2/roi μ settled at 978.7** (not the evening
  1105 reading — the LB moved overnight). Rolling-last-2:
  `[v1.1 (μ=570.2), v1.2/roi (μ=978.7)]`. 4/5 submission slots used
  today; 1 left but **no submission was pushed this session** (Rule 1,
  PI did not approve a same-day push of an untested variant).
- **LB cliff data refreshed:** top-10 cutoff is **μ = 1460** (sash).
  #1 = bowwowforeach μ=1663.4. Roman public top μ=1224 ≈ rank 70.
  Gap from us to prize cliff = **+481 μ**. Plan calibration updated
  in `state/current.md`.
- **Strategy plan ratified by PI**:
  `/root/.claude/plans/you-are-a-champion-sprightly-sunset.md`. Target =
  rank 1-10 prize (μ ≥ 1460). Architecture = adapt Roman 1224 into our
  Strategy → Intent → realize(mechanisms) pipeline. 2P AND 4P matter
  (Roman tunes both).
- **Two ship-ready bundles staged, awaiting PI submit approval:**
  - `submissions/v1_3_roi_physics.py` — Block A only. h2h vs frozen
    pre-physics roi = 56% (36/64). Marginal on the panel-WR gate.
  - `submissions/v2.py` — Block A + C + D. Mean panel WR **64.1%**;
    h2h vs frozen pre-physics roi = **69% (44/64)**. **Clears both
    gates.** PRIMARY submit candidate.

### Tonight's progress (load-bearing only)

1. **Strategic-direction plan** (`/root/.claude/plans/you-are-a-champion-sprightly-sunset.md`).
   Five PI decisions locked: top-10 prize target, capture-probe first,
   4P matters (Roman has FOUR_PLAYER_ROTATING_* constants), eviction
   rule ≥60% panel + ≥55% h2h, adapt-Roman architecture.
2. **Capture-success probe** (`scripts/capture_probe.py`,
   `audit/2026-05-10-capture-success-probe.md`). 32 seeds roi self-play,
   24,431 launches: **reached 77.2%, collided_other 10.7%, oob 7.6%,
   sun 2.1%, alive_at_end 2.4%.** Inverted the punch list: sun-avoid
   demoted (smallest bucket), path-clears-other-planets + OOB guard
   promoted.
3. **Public-kernel teardown** (`audit/2026-05-10-public-kernel-teardown.md`).
   Pulled `external/kernels/{roman-1224, structured-baseline,
   physics-accurate-planner, sun-dodging}/` via kaggle CLI. Roman and
   Pilkwang share the same Strategy/WorldModel/Mission skeleton; Roman
   adds 4 mission classes (rescue / recapture / reinforce / gang_up /
   elimination) over Pilkwang's 3. 4P confirmed: Roman tunes
   `FOUR_PLAYER_ROTATING_*` separately.
4. **Block A — physics-module upgrade**. New `lib/aim.py`
   (5-iter aim_with_prediction + search_safe_intercept). Upgraded
   `lib/mechanism.py`: lead_aim_v2, sun_avoid arrival-aware, new
   `path_clears_other_planets` + `oob_guard`. Capture probe lift:
   reached **77.2% → 83.9% (+6.7pp)**, collided_other halved.
   Bundle: `submissions/v1_3_roi_physics.py`. Audit:
   `audit/2026-05-10-block-a-physics-upgrade.md`. 160 tests green.
5. **Block C — arrival-ledger substrate.** New `lib/combat.py`
   (resolve_arrivals, 11 tests), `lib/world_model.py`
   (build_arrival_ledger + simulate_planet_timeline +
   state_at_timeline + WorldModel.from_world, 10 tests). New
   `arrival_ledger` mechanism implemented but EXCLUDED from
   DEFAULT_MECHANISMS — regressed WR 56% → 50% without a planner to
   re-allocate freed ships. 181 tests green.
6. **Block D — v2 strategy.** New `agents/v2/main.py` — ROI target
   selection + per-target `WorldModel.owner_at` lookup to drop
   targets predicted to be ours with surplus garrison.
   **64.1% mean panel WR, 69% h2h vs frozen pre-physics ROI.**
   Bundle: `submissions/v2.py` (50 KB, E.2 self-play 5/5 DONE,
   p95 turn 3.7 ms). Audit:
   `audit/2026-05-10-block-c-d-arrival-ledger-and-v2.md`.

### Falsified-or-dead (tonight)

- **`arrival_ledger` mechanism in DEFAULT_MECHANISMS** — regressed.
  Per-source greedy strategies can't re-pick after a mechanism drops
  their intent, so sources go idle. Mechanism stays in `lib/mechanism.py`
  for v3 planner use. Trail: `audit/tournaments/20260510T215332Z.json`.
- **v2 with ship-bumping per WorldModel-predicted defense.** 0/64 WR.
  Made the strategy prefer low-ROI affordable targets over high-ROI
  ones the mechanism layer would have allowed through. Rolled back to
  "skip already-ours only, no bumping". Trail:
  `audit/tournaments/20260510T215806Z.json`.
- **Punch #7 (sun-avoid) as the headline physics fix.** Capture probe
  shows sun = 2.1% — the smallest bucket. Reframed: sun_avoid stays in
  DEFAULT (low cost, easy to ship) but is no longer load-bearing.

### Next-session first-action

PI-aware order; rolling-last-2 still active.

1. **Submit v2** (cost: <1 min, EV: high). `kaggle competitions submit
   -c orbit-wars -f submissions/v2.py -m "v2 — worldmodel-aware
   roi (skip already-ours dedup); 64% mean panel WR, 69% h2h vs frozen
   v1.2-equiv"`. Expected live μ band 1050-1200 (Wilson 95% CI on
   44/64 = [57%, 79%]). **Evicts v1.1** (μ=570); rolling pair becomes
   `[v1.2/roi (978.7), v2 (PENDING)]`.
2. **Read v2 μ after validation** (~5-30 min for validation; ~24h for
   ladder games to accumulate). Update `state/calibration-ladder.md`.
3. **Block E start: mission framework.** With v2 baseline ladder data
   in hand, begin `lib/mission.py` + `lib/missions/snipe.py` +
   `lib/planner.py` (settle_plan solver). Snipe-only first; reinforce
   / recapture / gang_up after that. Predicted μ band for v3 missions
   1100-1300.
4. **(Defer)** v1.3 submission. If v2 underperforms expectations,
   v1.3 (Block A only) is the fallback at `submissions/v1_3_roi_physics.py`.
5. **(Defer)** Endgame burn-through accounting (cheap; ~2.4pp probe
   bucket).

### Pointers (added tonight)

- `/root/.claude/plans/you-are-a-champion-sprightly-sunset.md` — the
  strategic-direction plan + Block A-F implementation outline.
- `scripts/capture_probe.py` — per-fleet outcome probe; rerunnable.
- `audit/2026-05-10-capture-success-probe.{json,md,v2.json}` — pre +
  post-upgrade probe runs.
- `audit/2026-05-10-public-kernel-teardown.md` — Roman / Pilkwang /
  sigmaborov / sun-dodging notebooks mapped onto our pipeline.
- `audit/2026-05-10-block-a-physics-upgrade.md` — physics-module
  audit; lift = +6.7pp reached.
- `audit/2026-05-10-block-c-d-arrival-ledger-and-v2.md` — substrate +
  v2 audit; lift = +64.1% mean panel WR.
- `external/kernels/{roman-1224, structured-baseline,
  physics-accurate-planner, sun-dodging}/` — gitignored; re-fetch via
  `kaggle kernels pull <author>/<slug> -p external/kernels/<slug>/`.
- `submissions/v2.py` — PRIMARY submit candidate for tomorrow morning.
- `submissions/v1_3_roi_physics.py` — backup if v2 has unexpected
  ladder behaviour.
- `agents/v2/main.py` — v2 source. Strategy = roi + WorldModel-aware
  dedup. Uses `DEFAULT_MECHANISMS` from `lib/mechanism.py`.
- `agents/simple/roi_baseline.py` — local A/B control arm; uses
  `DEFAULT_MECHANISMS_PRE_PHYSICS` to mirror the live v1.2 stack.
- `audit/tournaments/202605102{14455,15332,15806,20206}Z.json` — the
  v1.3 / arrival_ledger / v2-bumped / v2-final A/B chain.
