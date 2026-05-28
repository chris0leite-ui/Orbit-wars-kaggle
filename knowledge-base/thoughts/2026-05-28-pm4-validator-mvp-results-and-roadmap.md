# 2026-05-28 PM4 — H14 validator MVP A/B + Phase 2 roadmap

Status: MVP built, trained, A/B'd. **Directional lift confirmed at +10.9 pp (Wilson [0.487, 0.719] @ n=64, INCONCLUSIVE by formal gate).** Phase 2 sequenced.

Companion docs:
- `2026-05-28-pm-distillation-action-rank-collapse.md` — Phase A failure diagnosis
- `2026-05-28-pm2-feature-sufficiency-probe.md` — Stage 1-3 probes (rank-collapse mechanism, 35/40 features dead)
- `2026-05-28-pm3-h14-recipe-locked-from-konbu17.md` — research synthesis + recipe lock

## Headline

| Stage | Result |
|---|---|
| MVP corpus (30 games, 3 opps) | 5366 labeled shots, pos_rate **0.529** (healthy) |
| 3-MLP ensemble training | val_acc **0.777** @ 0.5; @ 0.30 thr: precision 0.72, recall 0.89 |
| A/B baseline_validated vs agents/baseline, n=64 | **39/64 = 60.9 %**, Wilson 95 % CI [**0.487**, **0.719**] |
| Verdict | INCONCLUSIVE by Wilson-lo ≥ 0.50 gate, but +10.9 pp directional lift |
| Latency | focal p50=185 ms, p95=282 ms, max=692 ms (env actTimeout 1000 ms — comfortable) |

Tier breakdown shows the swing:
- Tier 1 (n=32): 16/32 = 50.0 % — flat
- Tier 2 (n=32): 23/32 = 71.9 % — strong lift

Variance per-tier is high; one tier looks like parity, next looks like clear lift. The MVP gives us a working substrate, not a verdict. Phase 2 tightens the CI.

## What this means

1. **The filter is doing real work.** Going from 50 % at n=32 to 60.9 % at n=64 isn't drift; it's variance. The base rate is meaningfully above 50 %.
2. **Konbu17's +19 pp is the upper-bound, not the expectation, for our case.** His rule-base was weaker; ours already vetoes most bad shots via the chooser's own logic (sun avoidance, target ROI, OOB filter, joint coalition). The validator only catches the *marginal* cases the chooser misses. Marginal cases are fewer, so the lift is smaller — but still real.
3. **n=64 is at the edge of the formal gate.** Wilson-lo = 0.487 is 0.013 below the 0.50 gate. ~10-20 more games would likely settle it either way — but a richer corpus + features is higher EV than just adding games to the same model.

## Phase 2 plan — feature expansion + opponent diversity + recompute corpus

Two axes that the MVP under-explored:

### Axis A — feature expansion (the missing arrival-time / physics / coordination signals)

Current 24-d feature set (konbu17 spec) describes STATE AT LAUNCH TIME only. The chooser argmax depends on what the target looks like AT ARRIVAL TIME — production accrued, opp fleets en route, predicted post-combat ownership. konbu17's notebook never had these because his proposer didn't either; ours does.

**Tier 1 (top 6 from PM4-evening research synthesis — every formula uses substrate already in `lib/`):**

| # | New feature (dim) | Formula → existing primitive | Why |
|---|---|---|---|
| F2 | `combat_margin_at_arrival` (1) | `(ships_sent − ships_arr) / max(1, ships_arr)`, clip [-1, +1]. `ships_arr` from `lib/world_model.py:239 predict_garrison_at(tgt, eta, ledger[tgt.id])` | The single number "did we send enough to beat predicted defenders" — *literally the binary label*, made explicit. Highest expected lift. |
| F6 | `path_fate_one_hot` (4) | `lib/trajectory.py:80 predict_fleet_fate(src, tgt, angle, ships, world, max_steps=eta+5)`; one-hot over {target, planet, sun, oob} | Encodes H44 (65 % fleet-destroyed-in-flight). audit/2026-05-21-h44-phase1-CORRECTED.md maps directly to this. |
| F3 | `owner_at_arrival_one_hot` (3) | `predict_garrison_at(...).owner`. Replaces current launch-time owner one-hot — same slot count | Roman/Pilkwang/oddshrimp all gate on arrival-time owner, not launch-time. Free swap, strictly more predictive. |
| F10 | `same_target_friendly_inflight_{count, ships}` (2) | `[(e, s) for (e, o, s) in ledger[tgt.id] if o == focal_seat]`; count + ship sum | Closes the redundant-swarm failure mode — pooled in-flight stats can't see "I already have 200 ships landing here." |
| F8 | `src_safe_departure_ratio` + `shot_drains_safely` (2) | `WorldModel.incoming_enemy_eta(src.id, focal_seat)` (`world_model.py:333`) + `ledger[src.id]` + `WAVE_LOOKAHEAD=12` (`world_model.py:53`) — replicates oddshrimp/melisgl `safe-departure` | Source-emptying discipline is the top-10 differentiator (mean garrison-at-launch 11 vs midpack 22). |
| F4 | `pv_capture` (1) | `pv_horizon(step, eta, gamma=0.99, t_total=step + eta + expected_hold(tgt.id, eta, world)) × tgt.production`. Both helpers in `lib/scoring.py:89-140` (`PV_GAMMA = 0.99`) | Late-game scoring asymmetry: γ=0.99 over `expected_hold`-truncated horizon penalises captures we'll lose quickly. |

**Total: 24 d → ~30 d.** F3 replaces 3 d, others net-add. All 6 are 3-5 lines apiece in `lib/shot_features.py`; the substrate functions are parity-tested.

**Tier 2 (slots 7-10 if budget permits):**

| # | Feature | Source |
|---|---|---|
| F11 | `joint_arrival_count_at_eta` (1) | `sum(1 for (e, o, _) in ledger[tgt.id] if o == focal_seat and abs(e − eta) ≤ 1)` — same-step same-owner combat stack |
| F7 | `intercept_enemy_eta` (1) | `WorldModel.incoming_enemy_eta_after(tgt.id, focal_seat, after=0)` (`world_model.py:349-372`) |
| F13 | `target_growth_field_diff` (1) | zvold's electrostatic: `Σ_my_planets prod/dist² − Σ_opp_planets prod/dist²`, clipped |
| F9 | `src_time_to_nearest_enemy_threat` + `src_is_frontier` (2) | `WorldModel.time_to_enemy_threat(src.id, focal_seat, world, arrival_eta=0)` (`world_model.py:374-480`); binary `< 25` (= `DANGER_RADIUS`) |

**Tier 3 (defer, exploratory):**

| # | Feature | Source |
|---|---|---|
| F5 | `uncertainty_at_arrival` | 0Zeta Halite IV: `min(1.0, abs(ships_arr − ships_now) / ships_now × eta / 100)` |
| F12 | `target_indirect_wealth` | oddshrimp: planet value boosted by high-growth opp neighbours |
| F14 | `target_dominance_3nn` | 3-NN ownership signal in [-1, +1] |
| — | `comet_remaining_lifetime` | `lib/world_model.py:comet_remaining_lifetime` — for comet-targeting shots |
| — | `mission_type_onehot` | One-hot from chooser's mission tag (snipe/reinforce/capture/drain/joint/opening) |

Implementation: extend `lib/shot_features.py`. Bump `FEATURE_DIM` from 24 to ~32 for Tier 1, ~38 for Tier 1+2. Update `data/shot_validator/schema.json` to v2. Wire `lib.world_model.WorldModel` for arrival-time predictions (already exists; just import).

Implementation cost: ~2-3 hours of careful coding + tests. The substrate (`predict_garrison_at`, `pv_horizon`, `path_clears_sun`) is already in `lib/`; we just plumb it through.

Inference latency: per-emit feature build was ~10 µs (24-d, no sim). With WorldModel arrival predictions: ~5-50 µs per emit × ~5 emits/turn ≈ 250 µs/turn. Comfortably under budget.

### Axis B — expanded opponent pool

MVP corpus used only 3 opponents: baseline self-play, vs baseline_full, vs v3_snipe. All in-tree baseline-lineage. Phase 2 broadens to 17 opponents stratified by strength:

| Tier | Opponent | Source | Games |
|---|---|---|---:|
| Weak | `agents/simple` | local | 5 |
| Weak | `agents/geo` | local | 5 |
| Weak | `agents/v1_orbitfix` | local | 5 |
| Moderate | `agents/analytical` | local (ANALYTICAL track lineage, closed-form ROI/LP) | 5 |
| Moderate | `agents/v3.5.1` | local | 5 |
| Moderate | `agents/v3_lookahead` | local | 5 |
| Moderate | `submissions/v4_planner.py` | local (μ historical) | 5 |
| Moderate | `submissions/v7_0_drop_one.py` | local (v7 ablation) | 5 |
| Strong | `agents/baseline_full` | local | 5 |
| Strong | `agents/baseline_joint_aggr_consolidated_orbitfix` | local (μ=1124 EVICTED) | 5 |
| Strong | `submissions/baseline_hybrid.py` | local (favor_hybrid head) | 5 |
| Strong | `submissions/baseline_favor.py` | local (favor head — Phase A teacher) | 5 |
| Strong | `submissions/baseline_learned.py` | local (Phase A — distinct policy, adversarial diversity) | 5 |
| Strong | `submissions/v7_minimax.py` | local (search-based) | 5 |
| **Live** | `submissions/_imported/baseline_pv_eta.py` | pulled from sibling, **μ=1154.8 LIVE CHAMP** | 5 |
| **Live** | `submissions/_imported/baseline_leaf_pv_2p.py` | pulled from sibling, μ=1105.4 just submitted | 5 |
| **Live** | `submissions/_imported/baseline_peak_1165_anchor.py` | pulled from sibling, peak μ=1149 ref | 5 |
| Self | `agents/baseline` × `agents/baseline` | local | 5 |
| **Total** | **18 cells** | | **90 games** |

Both seats labeled per game ≈ 16-20k labeled shots (vs MVP's 5,366).

Pull mechanics: the three "live" submissions were pulled this session via `git show origin/claude/kaggle-submission-review-gZsCu:submissions/X.py > submissions/_imported/X.py`. `submissions/_imported/` is `.gitignore`'d (per `submissions/*` rule); if we want them tracked in a follow-up session, move to e.g. `submissions/imported/*.py` and update gitignore.

Compute estimate: 90 games × ~50 s/game / 8 workers × 1.5 wallclock factor ≈ **15-20 min** with `BASELINE_WALLCLOCK_MS=100`.

Pos_rate calibration: weak opps lift pos_rate (we win shots easily); live champion drops it (we lose). The 18-cell ratio is tuned to balance — predicted pos_rate 0.50-0.65. If it lands outside [0.40, 0.85], `scripts/train_validator.py` aborts with explicit error.

### Axis C (defer, only if A+B insufficient)

- **4P training data.** konbu17 documents 2P/4P split AUC lift (+0.011/+0.001 over combined). Add 4P self-play + 4P-cross-opp games. Build separate 4P validator, pick by `obs.num_seats` at inference.
- **Synthetic emit augmentation.** For each real emit, generate counterfactual variants (different ship counts, ±5° angle perturbations) and label via short rollouts. Closes the "negative examples we never see in real play" gap. ~3× compute on gen.
- **Per-candidate score head + ranking loss.** Move from filter to re-ranker. Higher integration risk; defer until filter line has clear ceiling.

## Phase 2 session sequencing (next session)

| Stage | Time | Output | Gate |
|---|---|---|---|
| 1. Feature expansion code | ~2-3 h | `lib/shot_features.py` v2 (≈32-d), updated schema, updated `agents/baseline_validated/main.py`, updated unit tests | All tests green |
| 2. Expanded corpus gen | ~15-20 min | `data/shot_validator/labels_v2.jsonl` (~16-20k labeled shots from 18 opp cells, both seats) | Filtered pos_rate in [0.40, 0.85] |
| 3. Train 3-MLP ensemble | ~1 min | `data/shot_validator/validator_ensemble_weights_v2.npz` | val_acc ≥ 0.80 (vs MVP 0.777) |
| 4. Threshold + topk sweep | ~30 min | held-out per-game eval at 5 (threshold × topk) cells | Pick best operating point |
| 5. A/B vs `agents/baseline` | ~30-60 min | n=32 → n=64 adaptive | Wilson-lo ≥ 0.50 |
| 6. A/B vs `baseline_pv_eta` (live champ) | ~30-60 min | only if step 5 cleared | Wilson-lo ≥ 0.50 (Rule 43) |
| 7. Bundle + parity (Rule 46) | ~10 min | `submissions/baseline_validated.py` | bundle imports + `fast.py play` no crash |
| 8. Submission gate | depends | PI sign-off (Rule 1) | Rolling-pair claim (Rule 42) |

Total: ~5-7 hours. Single session if focused.

## Risk register

| Risk | Probability | Mitigation |
|---|---|---|
| Feature expansion increases inference latency past 100 ms wallclock | Low | Tier 1 features all <1 ms each; bench-verify with `scripts/bench_value_head_inference.py` pattern |
| pos_rate calibration breaks with weak opps spiking it | Medium | `train_validator.py` aborts if outside [0.40, 0.85]; tunable by adjusting ratio (drop a weak opp) |
| Live-champion games dominate corpus (focal loses lots → label noise) | Medium | Cap live-champ games at 5 each (current allocation); both seats labeled means our seat as opp also generates data |
| Bundler doesn't handle the wrapper-style import correctly | Known issue | The MVP A/B used non-bundled path (fast.py loads main.py directly). Bundle path is for the SUBMISSION session; design TBD (see PM3 note on baseline_full bundle workaround) |
| n=64 A/B too slow at 100 ms wallclock | Confirmed (43 min) | Acceptable for n=64; consider workers=8 if CPU available |
| Validator dropping good shots in late game (recall=0.89 means 11% FN) | Medium | Mission-tag feature (Tier 2) would let the filter learn late-game/early-game-specific patterns |
| Phase 2 falsifies the filter direction entirely | Low | If 60.9 % regresses to <55 % after feature expansion, something is wrong with the new features. Roll back to MVP and try different features. |

## What's deferred to later sessions

- **Synthetic emit augmentation** (Axis C). Higher compute, useful if Phase 2 plateaus.
- **4P training + inference split** (Axis C).
- **Per-candidate score head** (replace filter with re-ranker).
- **Bundle workflow for submission.** The wrapper-pattern bundling is non-trivial (see PM3 footnote on `baseline_full` workaround). Design a clean approach when we have a submittable candidate.
- **GNN over planet graph.** Speculative; only if all of the above hits a ceiling.

## Pointers

- `agents/baseline_validated/main.py` — filter wrapper (deployed for MVP A/B; weights embedded as base64)
- `lib/shot_features.py` — 24-d encoder; **target of Phase 2 expansion**
- `scripts/gen_validator_corpus.py` — corpus generator (extend to support 18-pair config)
- `scripts/train_validator.py` — trainer (no changes needed for Phase 2)
- `scripts/embed_validator_weights.py` — embedder
- `data/shot_validator/schema.json` — feature schema v1 (24-d); **bump to v2**
- `submissions/_imported/{baseline_pv_eta,baseline_leaf_pv_2p,baseline_peak_1165_anchor}.py` — strong sibling-branch agents pulled this session (gitignored)
- `lib/world_model.py:predict_garrison_at` — single-tick combat prediction (O(eta), use for arrival-time features)
- `lib/scoring.py:pv_horizon, PV_GAMMA` — present-value substrate
- `lib/geometry.py:path_clears_sun` — sun-collision check
- `knowledge-base/concepts/top-performer-strategies.md` §H14 — original H14 hypothesis (this session's MVP execution)

## What if Phase 2 also lands in the inconclusive band?

If feature expansion + opp diversity still gives Wilson [0.45, 0.65] at n=64, the bottleneck is one of:
- **Threshold mis-tuning** — the threshold sweep should catch this
- **Validator is filtering what the chooser already does** — investigate via `--debug-validator` flag (log every veto + post-game outcome, see what the validator catches that baseline missed)
- **n=64 isn't enough for ±10pp lift detection** — Rule 45 allows scaling to n=128 with PI override
- **The architecture is wrong (filter vs re-ranker)** — pivot to per-candidate score head

In that order. Cheap diagnostics first.
