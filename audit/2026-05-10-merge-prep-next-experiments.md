# 2026-05-10 — merge prep + next-experiments plan

> Branch: `claude/game-strategy-research-8w7EO`.
>
> Companion docs:
> - Research note: `docs/strategies/heuristics-research.md` (this branch).
> - PI source dump: `knowledge-base/thoughts/2026-05-10-strategy-heuristics-PI-dump.md`.
> - Sister branch (target-selection ablations + panel infra):
>   `claude/simple-trading-strategies-QS0xV`.
>
> This file is forward-looking: it assumes the sister branch lands on
> `main` first, then this branch rebases on top. All paths under
> `agents/simple/`, `scripts/strategy_panel.py`, `lib/fingerprint.py`,
> `audit/2026-05-10-simple-strategy-panel.md` reference files that
> arrive **with that merge** and become available after rebase.

---

## 1. Two streams, one merge target

Two branches were spun out from the same Day-1 main:

| Branch | Output | Status |
|---|---|---|
| `claude/simple-trading-strategies-QS0xV` (sister) | 5 `agents/simple/*` target-selection ablations, `scripts/strategy_panel.py` panel runner, `lib/fingerprint.py` + `scripts/manifold_check.py` Phase-1 opponent-modelling infra, 8-seed panel results | runnable, awaits 32-seed confirmation |
| `claude/game-strategy-research-8w7EO` (this branch) | `docs/strategies/heuristics-research.md`, PI dump | research-only |

The two streams are file-disjoint (different files added; no edits to the same file) so the merge is mechanical. Logical reconciliation is what this doc plans.

## 2. Merge readiness

### File-level conflicts: predicted NONE

Sister adds (28 files): `agents/simple/*`, `scripts/{strategy_panel.py,manifold_check.py}`, `tests/test_{simple_strategies,fingerprint}.py`, `lib/fingerprint.py`, `audit/2026-05-10-simple-strategy-panel.md`, `audit/tournaments/*.json`, `docs/strategies/simple-*.md`.
Sister edits: `state/{hypothesis-board.md,mechanism-ledger.md,calibration-ladder.md}`, `docs/strategies/README.md`, `scripts/tournament.py`, `ISSUES.md`.

This branch adds: `docs/strategies/heuristics-research.md`, `knowledge-base/thoughts/2026-05-10-strategy-heuristics-PI-dump.md`.
This branch edits: none.

Disjoint. Rebase on post-merge `main` should be clean.

### Logical conflicts: 2, both resolvable

1. **`state/hypothesis-board.md`** — sister populated `H-roi-32`, `H-production-32`, `H-nearest-vs-v1`. Our research note's §J proposes H4–H8. After merge: append H4–H8 to the same Open list. **No collision** (different topics — sister is target-selection confirmation; ours is multi-source coordination, deterministic-win, spoiler mode).

2. **`state/mechanism-ledger.md`** — sister added the `simple-greedy-target-selection-variants` row. Our planned axes (sizing, source-selection, coordination, phase, defense) extend the same family-table. After merge: add one row per axis we run. No collision.

### Sister-branch open follow-ups we inherit

From `audit/2026-05-10-simple-strategy-panel.md` "Open follow-ups":
- **H-roi-32 / H-production-32:** rerun panel with `--seeds 32` to tighten Wilson CIs. Blocking gate for any submission decision.
- **PI-deferred 6th axis:** sister flagged that PI will name a 6th axis (sizing / coordination / defence) after the 32-seed result lands. **This doc nominates the candidates** (§5).
- **Hedge-ladder construction:** populate `state/hypothesis-board.md` Hedge-ladder section once `roi`/`production` confirm. Out of scope for this prep — final-window-only per Rule R2.

## 3. Post-merge action checklist (executable, in order)

1. **Rebase this branch** onto post-merge `main`:
   ```
   git fetch origin && git rebase origin/main
   ```
   Expect zero conflicts. If any appear, they'll be in `state/*` — both branches append to lists.

2. **Cross-link `heuristics-research.md` § "Empirical evidence (panel)"** (added in this commit) to the now-resolved paths:
   - `audit/2026-05-10-simple-strategy-panel.md`
   - `audit/tournaments/20260510T123059Z.json`
   - `agents/simple/*.py`
   - `docs/strategies/simple-*.md`
   - `scripts/strategy_panel.py`

3. **Promote H4–H8** from `heuristics-research.md` §J into `state/hypothesis-board.md` Open list. Group under heading "2026-05-1X — coordination / endgame / objective extensions" so they sit visibly alongside sister's target-selection block.

4. **Run the 32-seed confirmation panel** (sister's open follow-up; gates everything downstream):
   ```
   python -m scripts.strategy_panel --seeds 32
   ```
   Expected ~30 min CPU. Then update `state/hypothesis-board.md` verdicts on H-roi-32 / H-production-32 / H-nearest-vs-v1.

5. **Decide submission cadence** based on (4):
   - If `roi` 32-seed Wilson-lo ≥ 0.85 vs v1_orbitfix AND v1.1's live μ has settled → `roi` becomes a v1.2 submission candidate (subject to rolling-last-2; never push speculative same UTC day as known-good).
   - Otherwise → defer submission, continue panel work.

6. **Begin Axis 3 (coordination) work** (§5 below) — highest expected value of the new axes, biggest novel claim from research note.

7. **Refresh `state/mechanism-ledger.md`** after each new axis lands a panel result.

## 4. Findings synthesis — what we know after sister + research

### Validated (sister panel, 8 seeds, awaiting 32-seed confirm)

- **Travel-adjusted ROI dominates.** `roi = production / (dist+1)` beats every panel agent including v1_orbitfix (96.9% mean panel WR; 100% / 16-of-16 vs v1). Aligns with research note §C.3.
- **Production-aware target selection is a clear win.** `production` argmax 75% panel WR; 69% vs v1. Aligns with §D.1 compounding insight (early production captures dominate).

### Falsified (sister panel, 8 seeds — leaning, awaits 32-seed confirm)

- **Naive "attack enemies first" is bad.** `enemy_first` → 32.3%. Refines §F.4 of research note: "deny enemy production" as a *primary scoring axis* doesn't work — enemy planets are too well-defended early; opponents grab the cheap neutrals while we besiege one fortress. Compete-relative play (§F) must wrap enemy-pressure inside cost-aware scoring, not run as the headline rule.
- **Sniping the weakest is bad.** `weakest` → 15.6%. Confirms §G.7 caveat: cheap snipes ignore production yield; capturing a 1-ship rock is worse than a closer 50-ship producer.

### What our research adds that sister hasn't tested

The sister panel is **target-selection only**: it holds source-selection, sizing, coordination, defense, and phase fixed. Five axes from our research note remain unexplored:

| Axis | Research-note section | Sketch |
|---|---|---|
| 1. Sizing | §C.1, §G.8 | varies fleet size, not target |
| 2. Source selection | §G.4, §G.5 | varies WHICH owned planet launches |
| 3. **Coordination** | **§E.3** | times multi-source launches for same-step arrival |
| 4. Phase segmentation | §D.6 | different scoring per game phase |
| 5. Defense / reserve | §G.1, §G.2, §G.15 | leaves garrison vs strips fully |
| 6. Compete-relative | §F | rank-aware (parked: needs 4P-FFA panel) |

Axis 3 is the single biggest novel claim in the research note — combat groups same-owner same-step arrivals (README rule), so **timing achieves combat mass without speed loss**. That's a lever neither sister nor any public-top-5 heuristic explicitly exploits in our reading.

## 5. Next-experiment plan — five axes, drop-in compatible with `strategy_panel.py`

Each axis follows the sister-panel template:
- N agents under `agents/simple/<axis>_<variant>.py`, sharing `DEFAULT_MECHANISMS` (`[validate, arrival_size, lead_aim]`).
- One score function or one piece of logic differs per variant; everything else fixed.
- Tests under `tests/test_<axis>.py` (mirror `tests/test_simple_strategies.py`).
- Panel run via `scripts/strategy_panel.py --strategies <list>`.
- Audit at `audit/<date>-<axis>-panel.md` summarising verdicts.
- Decision gate: ≥58% beat over current panel champion at 32 seeds with Wilson-lo ≥ 0.55.

### Axis 1: Sizing (≤ ½ day)

**What it tests.** Beyond `target.ships + 1`, what fleet size dominates? `arrival_size` is already in `DEFAULT_MECHANISMS` so the baseline sizing is production-aware; this axis pushes further.

**Variants:**
- `simple_size_min` — control. `target.ships + 1`.
- `simple_size_overshoot` — `target.ships + production*eta + 5` (defensive overshoot; catches +k mid-flight reinforcement).
- `simple_size_garrison_reserve` — strip down to `K` ship reserve at source (G.1 lazy reserve).
- `simple_size_risk_adjusted` — multiply send-size by `1 + α·incoming_enemy_count` (G.8).

**Strategy: build on top of `roi`** (current champion), so each variant is "roi target selection + axis-1 sizing." Panel composition: 4 sizing variants + `roi` (control) + `v1_orbitfix` (reference) + `baseline` (floor) = 7 agents × 32 seeds × 2 sides ≈ 1500 games, ~15 min CPU.

**Hypothesis (pre-registered):** overshoot beats min by ≥3% panel WR; garrison-reserve loses ≤2% (paying for defense it doesn't get without arrival ledger).

### Axis 2: Source selection (≤ ½ day)

**What it tests.** The sister panel launches from every owned planet every turn. Restricting source set may free ships for bundling, or may starve frontiers.

**Variants:**
- `simple_src_all` — control.
- `simple_src_drain_low` — only planets with `production ≤ 2` launch (G.4 — preserve high-prod engines).
- `simple_src_launch_high` — only `production ≥ 4` planets launch (G.5 — bigger garrison ⇒ faster fleet).
- `simple_src_top_garrison` — only top-K-garrison planets launch (K=3).
- `simple_src_safe_only` — skip planets with any incoming enemy fleet (G.2 — don't strip a planet about to be hit).

**Build on:** `roi` target selection. Panel: 5 variants + `roi` + `v1_orbitfix` + `baseline`. ~15 min CPU at 32 seeds.

**Hypothesis (pre-registered):** `safe_only` beats `all` by ≥4% (free defense). `drain_low` and `launch_high` are within ±2% of each other and within ±3% of `all` (the source choice is less load-bearing than target choice in pre-v2 setups, since arrival ledger doesn't yet exist).

### Axis 3: **Coordination** — multi-source simultaneous arrival (≈ 1 day)

**What it tests.** Research note §E.3: combat resolver groups same-owner same-step arrivals at a planet. So if 2–3 source planets time their launches such that all fleets land on **the same step**, we get combat mass for free with **no speed penalty**. Public-top heuristics including Roman 1224 don't seem to do explicit timing (per `external/kernels/` references); this is the largest unexploited lever.

**Variants:**
- `simple_coord_uncoord` — control. Each source independently picks its top-ROI target and launches now.
- `simple_coord_pair_sync` — for top-1 ROI target, find the 2 sources that minimize `max_eta`; delay the closer source so both arrive on `step + max_eta`. Other sources launch uncoordinated.
- `simple_coord_triple_sync` — same logic, 3 sources.
- `simple_coord_top2_sync` — sync on top-1 AND top-2 ROI targets independently (so each owned planet contributes to its locally-best sync mission).

**Implementation sketch (~80 LOC):**
```
def propose_intents(self, world):
    targets = rank_by_roi(world.targets)
    sources = world.my_planets

    intents = []
    for target in targets[:K_TARGETS]:
        contributors = pick_n_closest(sources, target, n=N_SYNC)
        max_eta = max(eta(s, target, ships) for s in contributors)
        for s in contributors:
            delay = max_eta - eta(s, target, ships)
            if delay == 0:
                intents.append(Intent(s.id, target.id, ships))
            else:
                # Defer launch by `delay` steps. This needs a tiny
                # per-strategy state buffer keyed by (source, step+delay).
                ...
    # Remaining sources fall back to roi-greedy.
    return intents
```

**State requirement.** Strategies in the sister panel are **stateless** (`propose_intents(self, world)`). Coordination requires a small per-instance buffer: `pending: dict[(src_id, step) -> Intent]`. The Strategy ABC may need a tiny extension. Confirm during implementation.

**Build on:** `roi` target selection. Panel: 4 coord variants + `roi` (control) + `v1_orbitfix` + `baseline`. **Plus a 32-seed cross-panel** vs the Axis-1 and Axis-2 winners once those land — so the best-of-each-axis can be combined.

**Hypothesis H4 (pre-registered):** `simple_coord_pair_sync` beats `roi` by ≥55% over 24 seeds × both sides. If H4 holds, this becomes the v3 mission class `gang_up` substrate (roadmap mechanism 4).

**Risk.** Stale-target risk: if the target's garrison swells between when we plan and when the slow source arrives, the bundled mass may still be insufficient. Mitigation: re-evaluate on launch; abort the plan if the target's projected arrival-time garrison exceeds bundled mass × 0.9.

### Axis 4: Phase segmentation (≤ ½ day)

**What it tests.** Research note §D.6: different scoring rules per game phase (land-grab → frontier → consolidation → endgame).

**Variants:**
- `simple_phase_const` — control. ROI throughout.
- `simple_phase_landgrab_then_roi` — steps 0–60: production-greedy on neutrals only (skip enemies entirely); steps 60+: `roi` everywhere.
- `simple_phase_endgame_burn` — steps 0–470: `roi`; steps 470+: launch every owned planet's full garrison toward nearest enemy planet (in-flight ships count for our score at game end per env scoring).
- `simple_phase_full_segmented` — landgrab + roi-frontier + endgame-burn combined.

**Build on:** `roi`. Panel: 4 variants + `v1_orbitfix` + `baseline`. ~15 min CPU.

**Hypothesis (pre-registered):** `endgame_burn` adds ≥2% panel WR (ships in flight at step 500 are scored; we currently waste them sitting in garrison). `landgrab_then_roi` adds ≥1% (cleaner early game). `full_segmented` is the ceiling.

### Axis 5: Defense / counterfactual (≤ ½ day; lift uncertain pre-v2)

**What it tests.** Research note §G.1, §G.15. These all need to know "is an incoming enemy fleet about to hit us?" — observable from `obs.fleets`. Real combat forecasting needs the v2 arrival ledger; the simple panel can use a coarse "any enemy fleet inbound within K steps" predicate.

**Variants:**
- `simple_def_lazy` — control. Strip everything to launch; no reserve.
- `simple_def_threat_aware` — for each planet, if `Σ enemy_fleet_ships_inbound_within_K > current_garrison + production·K`, skip launch and reserve garrison.
- `simple_def_counterfactual` — heuristic combat sim per planet: "would I survive doing nothing?" If yes, don't launch defensively (free up the ship for offense).

**Build on:** `roi`. Panel: 3 variants + `v1_orbitfix` + `baseline`.

**Hypothesis (pre-registered):** `threat_aware` adds ≤1% panel WR pre-v2 (without proper arrival forecasting, the K-step heuristic is noisy). The real test happens at v2.

### Axis 6: Compete-relative (parked)

**What it tests.** Research note §F: 4P FFA spoiler-mode. Requires `kaggle_environments.make("orbit_wars", agents=[a, b, c, d])` panel infra. `scripts/strategy_panel.py` is currently 1v1; extension is a separate piece of work.

**Action:** create a `scripts/strategy_panel_4p.py` once Axes 1–4 are done. Until then, this is a v4 design candidate, not a v2/v3 axis.

## 6. Sequencing — priority order

| Order | Item | Effort | Blocked by | Expected lift |
|---|---|---|---|---|
| 1 | 32-seed confirmation of `roi`/`production` | 30 min CPU | sister merge | gating |
| 2 | **Axis 3 — coordination** | 1 day | (1) | high (research's biggest claim) |
| 3 | Axis 1 — sizing | ½ day | (1) | medium |
| 4 | Axis 2 — source selection | ½ day | (1) | medium-low |
| 5 | Axis 4 — phase segmentation | ½ day | (1) | low-medium (endgame-burn likely free %) |
| 6 | Axis 5 — defense (heuristic) | ½ day | (1) | low pre-v2 |
| 7 | Cross-panel of best-of-each-axis | 1 hour CPU | 2–6 | combinatorial lift |
| 8 | Promote winning combo → v1.2 submission candidate | — | (7) + v1.1 μ settled | live |
| 9 | Axis 6 — 4P FFA infra + spoiler-mode | 2 days | (8) or v4 trigger | uncertain |

Axes 1+2+4+5 can run in parallel once the framework is in place — they're file-disjoint and panel-runner-disjoint. Axis 3 is the only one needing a small Strategy-ABC tweak (per-instance state); do it first to surface that change.

## 7. Decision criteria for shipping anything new

Inherit sister's gates verbatim:
- 32-seed Wilson-lo ≥ 0.55 vs current ladder champion (v1.1 currently).
- p95 turn wallclock < 500 ms (sister measured ~0.4 ms — orders of magnitude under, no concern).
- `scripts/pre_submit_diff.py` (TODO per roadmap line 123) → 10-game head-to-head vs previous submit; abort if < 55%.
- Rolling-last-2 economy: never push a speculative variant on the same UTC day as a known-good submit.

## 8. Diagnostic instrumentation to add to every panel run

`lib/fingerprint.py` (sister branch) maps a K-turn replay prefix to a feature vector. A behavioural-collapse diagnostic should be part of any axis we run — if two variants produce indistinguishable fingerprints under a strategy ablation, the axis is not actually doing anything (the mechanism layer is doing all the work, and the strategy is a no-op).

**Add to each axis audit:** "Fingerprint-distance matrix between variants on the seed bag. If average pairwise distance < threshold τ (TBD post-(1)), flag as collapsed and re-examine."

This also feeds Phase-2/3 of the opponent-modelling work the sister branch started.

## 9. Risk register

| Risk | Mitigation |
|---|---|
| Rolling-last-2 evicts known-good v1.1 prematurely | Decision (5) above; don't push axis-winners until v1.1 μ settles |
| Coordination Axis 3 needs Strategy-ABC tweak (per-instance state) | Land tweak before any variant code; back-port to sister panel agents (no-op for them) |
| 32-seed panel result inverts 8-seed `roi` finding | Falls back to `production` (75% panel) as champion; rerun coordination axis on top of that |
| Behavioural collapse across axis variants | Fingerprint diagnostic per (8) flags it; iterate strategy until variants diverge |
| 4P-FFA matchmaker on the live ladder differs from local 1v1 | Axis 6 + sister fingerprint Phase 2 designed around this; out-of-scope for v2/v3 critical path |
| Sister branch re-bases / squashes before merge | Re-derive paths from the merged tree; logical content here doesn't change |

## 10. What this doc does NOT decide

- Choice of axis-3 timing-buffer implementation (instance state on Strategy vs world-model integration). Defer to implementation session.
- Whether axis 4's `endgame_burn` should fire at step 470 or step 480. 32-seed sweep is the answer.
- v3 mission-class API design — that lives in `docs/strategies/roadmap.md` v3 section; this doc only feeds inputs.
- RL kill-switch trigger condition — unchanged from `roadmap.md` line 109.

---

## Appendix A — file-by-file post-merge state expected

After (1) rebase:

```
agents/simple/{nearest,production,roi,weakest,enemy_first}.py    # sister
agents/v1_orbitfix/main.py                                       # main
audit/2026-05-10-simple-strategy-panel.md                        # sister
audit/2026-05-10-merge-prep-next-experiments.md                  # this doc
audit/tournaments/20260510T*.json                                # sister
docs/strategies/README.md                                        # sister edits
docs/strategies/heuristics-research.md                           # this branch
docs/strategies/roadmap.md                                       # main
docs/strategies/simple-*.md                                      # sister
docs/strategies/v1_orbitfix.md                                   # main
knowledge-base/thoughts/2026-05-10-strategy-heuristics-PI-dump.md  # this branch
lib/fingerprint.py                                               # sister
lib/intent.py                                                    # main
lib/mechanism.py                                                 # main
scripts/strategy_panel.py                                        # sister
scripts/manifold_check.py                                        # sister
scripts/tournament.py                                            # sister edits
state/*.md                                                       # sister edits + our promotion
tests/test_fingerprint.py                                        # sister
tests/test_simple_strategies.py                                  # sister
```

Everything our axes need to import is on this list.
