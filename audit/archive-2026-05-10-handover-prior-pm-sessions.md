# Archived HANDOVER sections — 2026-05-10 prior PM-branch session-end blocks

> Moved out of `HANDOVER.md` on 2026-05-11 wrap-up (branch
> `competition-strategy-brainstorm-ZK6XT`) because HANDOVER had grown
> to 439 lines vs the 150-line WRAPUP cap. These sections are the
> per-branch session-end blocks for two completed PM sessions
> earlier on 2026-05-10. Their load-bearing findings are absorbed
> in the morning synthesis sections (`Where we are`, `Today's progress`,
> `Falsified-or-dead`, `Next-session first-action`, `Pointers`) at the
> top of HANDOVER.md. Kept here verbatim for audit-trail continuity.

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
