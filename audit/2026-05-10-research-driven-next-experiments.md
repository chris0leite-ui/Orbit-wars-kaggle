# 2026-05-10 — research-driven next experiments (post-Phase-1)

> Branch: `claude/game-strategy-research-8w7EO`.
>
> Supersedes `audit/2026-05-10-merge-prep-next-experiments.md` (deleted
> in same commit; that doc was based on a pre-merge snapshot in which
> the 32-seed confirmation, the v1.2 submission, and the Phase 1
> manifold verdict had not yet happened).
>
> Companion docs (now all on main):
> - `docs/strategies/heuristics-research.md` (this branch's design
>   research; §K.4–K.6 reference this plan).
> - `docs/strategies/roadmap.md` (v2 → v4 stack).
> - `audit/2026-05-10-simple-strategy-panel.md` (8-seed target-selection).
> - `audit/2026-05-10-phase1-manifold-verdict.md` (Phase 1 gate failure +
>   3 paths forward).
> - `audit/2026-05-10-meta-strategy-prior-art.md` (Grover / DRON /
>   AlphaStar / Pluribus / Ganzfried).
> - `state/current.md`, `state/hypothesis-board.md`,
>   `state/mechanism-ledger.md`.

---

## 1. Post-merge state (one-paragraph recap)

`roi` (production / distance) is shipped as v1.2 (#52518060, PENDING),
confirmed at 32 seeds with 100% (64/64) vs `v1_orbitfix` and 97.1%
mean panel winrate. Rolling-last-2 is `[v1.1=μ597.4, v1.2=PENDING]`;
v1 was evicted; 1 daily submission slot remains. Phase 1 fingerprint
infrastructure (`lib/fingerprint.py` v1 + `scripts/manifold_check.py`)
shipped with a **partial-refute** verdict: 5-class gate failed at RF
80.5% (target 90%) but the broad-class structure is informative —
ROI-family collapses to one basin; `weakest`, `enemy_first`,
`baseline` sit in their own basins. PI choice between path A (3-class
relabel), B (richer fingerprint v2), C (learned embedding) is open.
This plan recommends path A (rationale in
`docs/strategies/heuristics-research.md` §K.4) and treats it as
**Axis 0** — the highest-EV next move.

## 2. Six axes in priority order

| # | Axis | Effort | Reuses | Expected lift | Source |
|---|---|---|---|---|---|
| 0 | **3-class meta-router (Phase 1 path-A)** | 1–2 days | `lib/fingerprint.py`, `scripts/manifold_check.py`, replay corpus | ≥3 μ vs `roi` standalone (H9) | research-note §F + manifold-verdict |
| 3 | Multi-source simultaneous-arrival timing | 1 day | `scripts/strategy_panel.py`, requires Strategy-ABC tweak (per-instance state) | ≥55% beat over `roi` (H4) | research-note §E.3 |
| 4 | Phase segmentation (endgame-burn primarily) | ½ day | `scripts/strategy_panel.py` | ≥2% panel WR (H7-related) | research-note §D.6 |
| 1 | Sizing variants (overshoot / reserve / risk-adjusted) | ½ day | `scripts/strategy_panel.py` | medium | research-note §C.1, §G.8 |
| 2 | Source-selection variants (drain-low / launch-high / safe-only) | ½ day | `scripts/strategy_panel.py` | medium-low | research-note §G.4–G.5 |
| 5 | Defense (heuristic, pre-v2) | ½ day | `scripts/strategy_panel.py` | low pre-v2 | research-note §G.1, §G.15 |

Axis 6 (4P-FFA spoiler-mode, research-note §F.3) is **parked** until
4P panel infra is built (`scripts/strategy_panel_4p.py`); v4 candidate.

## 3. Axis 0 — 3-class meta-router (NEW; top priority)

### What it is

A small wrapper agent that:
1. Captures the first K turns of the live game as a behavioural prefix.
2. Runs `lib/fingerprint.py::compute()` to produce a feature vector.
3. Classifies into one of three basins via a pre-trained RF/LR model:
   `production_aware_greedy` / `weakest` / `enemy_first`.
4. Looks up the basin in a best-response table and dispatches to the
   matching strategy.

### Why path A (not B or C)

Per `docs/strategies/heuristics-research.md` §K.4: §F (compete-relative)
is **rank-aware override behaviour at the basin level**, not
fine-grained per-strategy scoring. There is no §F-derived policy that
distinguishes `nearest`-style from `production`-style opponents —
both get the same response (continue running ROI). Path B's
discrimination is unused at the policy level; Path C is heavier and
only justified if A and B both fail.

Path A is also a one-line experiment per the verdict:

```bash
python -m scripts.manifold_check \
  --label-merge nearest=production_aware_greedy \
                production=production_aware_greedy \
                roi=production_aware_greedy \
  --K 100
```

Predicted result: RF ≥ 92% at K ≤ 100 (verdict's prediction). Gate
clears.

### Best-response table (initial; from research-note §K.5)

| Detected basin | Response | Rationale |
|---|---|---|
| `production_aware_greedy` | run `roi` (current v1.2) | symmetric ROI-vs-ROI → RNG-tie-break; panel showed 7/8 draws self-play |
| `weakest` | `roi` + Axis-4 endgame-burn schedule | their 15.6% panel WR leaves high-prod neutrals open; we keep ROI + flush in-flight at step 470+ |
| `enemy_first` | `roi` + Axis-1 sizing-overshoot on home cluster | their attrition loses to economy; overshoot absorbs siege; we continue ROI expansion |

The two non-default rows are extrapolations validatable on the
existing replay corpus (`audit/replays/20260510T132957Z/`) without
new captures.

### Implementation sketch

New files:
- `agents/meta/router.py` — wrapper that owns the classifier + BR table.
- `agents/meta/br_policies/{default_roi,roi_endgame_burn,roi_sizing_overshoot}.py`
  — three concrete policy implementations sharing
  `DEFAULT_MECHANISMS`. The default is just `agents/simple/roi.py`
  re-exported; the other two are minimal extensions.
- `tests/test_meta_router.py` — fingerprint-roundtrip + BR-dispatch
  unit tests.
- `scripts/train_meta_router.py` — loads existing replay corpus,
  fits the 3-class RF/LR with `--label-merge`, persists the model
  under `models/meta_router_v1.pkl` (gitignored).
- `audit/<utc>-meta-router-eval.md` — reports zoo-panel result.

### Decision gate (H9)

H9 (new): the meta-router wrapped around `roi` beats `roi` standalone
by ≥3 μ on a 60-game zoo panel including `[weakest, enemy_first,
baseline, roi, v1_orbitfix]`. If yes → submit candidate (subject to
rolling-last-2; do not push speculatively while v1.2 settles).

### Risks

- **Path A's predicted +12pp gate lift is unconfirmed.** If the relabel
  re-run still doesn't clear 90%, fall back to path B (FEATURE_VERSION=2
  with distribution-shape + temporal-split features). Plan loses a day.
- **The live opponent may be in a basin we haven't trained for.**
  Pluribus's anti-exploitation argument (cited in
  `audit/2026-05-10-meta-strategy-prior-art.md`) applies — keep an
  `ABSTAIN_BELOW` confidence threshold + epsilon-randomised fallback
  to default `roi` when classifier is unsure (Ganzfried's "safe BR").
- **Joint policy correlation (PSRO)** — BR-table overfits the zoo if
  the live opponent doesn't match any zoo class. Mitigation: keep
  `roi` as the default action; never abstain into a non-`roi` policy
  on low confidence.

## 4. Axis 3 — Multi-source simultaneous-arrival (still high priority)

Per research-note §E.3 + H4. Combat resolver groups same-owner
same-step arrivals at a planet; therefore timing N source planets
to land on the same step gives **combat mass without speed loss**.

### Variants

- `coord_uncoord` — control (current `roi` behaviour).
- `coord_pair_sync` — top-1 ROI target, 2 closest sources, sync delay.
- `coord_triple_sync` — same with 3 sources.
- `coord_top2_sync` — sync independently on top-1 and top-2 ROI.

### Strategy-ABC tweak required

Sister-panel strategies are stateless (`propose_intents(self, world)`).
Coordination needs a small per-instance buffer:
`pending: dict[(src_id, step) -> Intent]`. Land this ABC tweak first
so it's available to all coord variants and a no-op for existing
sister strategies.

### Hypothesis (H4)

`coord_pair_sync` beats `roi` (current champion) by ≥55% over 24
seeds × both sides. If H4 holds, this becomes v3 mission class
`gang_up` substrate (roadmap mechanism 4).

## 5. Axes 1, 2, 4, 5 — refresh of pre-merge plan

These are lifted from the superseded prep doc with minimal change —
the post-merge state didn't invalidate them.

### Axis 1 — Sizing
- `simple_size_min` (control: `target.ships + 1`).
- `simple_size_overshoot` (`+ production*eta + 5`).
- `simple_size_garrison_reserve` (leave K reserve at source per G.1).
- `simple_size_risk_adjusted` (multiply by `1 + α·incoming_enemy`).
- Build on `roi` target selection.

### Axis 2 — Source selection
- `simple_src_all` (control).
- `simple_src_drain_low` (G.4: low-prod planets only launch).
- `simple_src_launch_high` (G.5: high-prod planets only launch).
- `simple_src_top_garrison` (top-K-garrison only).
- `simple_src_safe_only` (G.2: skip planets with incoming enemy).

### Axis 4 — Phase segmentation
- `simple_phase_const` (control).
- `simple_phase_landgrab_then_roi` (steps 0–60 production-greedy on
  neutrals only; steps 60+ roi everywhere).
- `simple_phase_endgame_burn` (steps 470+ launch full garrison toward
  nearest enemy; in-flight ships count for our score at step 500).
- `simple_phase_full_segmented`.
- **Highest expected free %** of the four cheap axes (endgame-burn
  alone plausibly ≥2%).

### Axis 5 — Defense (heuristic)
- `simple_def_lazy` (control).
- `simple_def_threat_aware` (reserve garrison if incoming-enemy ≥
  current + production·K).
- `simple_def_counterfactual` (skip defensive launch if planet would
  survive doing-nothing).
- **Lift uncertain pre-v2** — proper combat forecasting needs the
  arrival ledger. Keep this axis last.

## 6. Sequencing

| Order | Item | Effort | Blocked by | Note |
|---|---|---|---|---|
| 1 | Promote H4–H8 + H9 to `state/hypothesis-board.md` | 5 min | — | done in same commit as this doc |
| 2 | **Axis 0 — Phase 1 path-A relabel + 3-class gate re-run** | ½ day | (1) | confirms ≥92% RF |
| 3 | **Axis 0 — meta-router agent + BR table + zoo-panel eval** | 1 day | (2) | H9 decision |
| 4 | **Axis 3 — Strategy-ABC per-instance state tweak** | 2 hours | — | unblocks coord variants |
| 5 | Axis 3 — `coord_pair_sync` + 32-seed panel | ½ day | (4) | H4 decision |
| 6 | Axis 4 — `simple_phase_endgame_burn` + 32-seed panel | ½ day | — | likely free % |
| 7 | Axes 1, 2 in parallel | ½ day each | — | medium-priority |
| 8 | Axis 5 — defense (heuristic) | ½ day | — | low priority pre-v2 |
| 9 | Cross-panel of best-of-each | 1h CPU | (3,5,6,7) | combinatorial lift |
| 10 | Submission decision (v1.3) on best combination | — | (9) + v1.2 μ settled | live |

Axes 4, 1, 2, 5 are file-disjoint and can run in parallel once the
panel infra is in place.

## 7. Decision criteria (inherited from sister + roadmap)

For any submission candidate (v1.3+):
- 32-seed Wilson-lo ≥ 0.55 vs current ladder champion (currently v1.2
  / `roi` once it lands μ).
- p95 turn wallclock < 500 ms.
- `scripts/pre_submit_diff.py` (TODO per roadmap line 123) — 10-game
  head-to-head vs previous submit; abort if < 55%.
- Rolling-last-2 economy: never push speculative variant on the same
  UTC day as a known-good submit.

For Axis 0 specifically:
- Path A re-run gate: RF ≥ 90% at K=100 on the 3-class problem.
- H9 EV gate: +3 μ vs `roi` standalone on a 60-game zoo panel.
- Computational gate: meta-router p95 turn < 500 ms (fingerprint
  compute + classifier predict + BR dispatch).

## 8. Diagnostic instrumentation

Per Phase 1's verdict the existing 15-feature fingerprint is too
coarse for ROI-family discrimination. For our axes (1–5), variants
will all live inside the production-aware-greedy basin. So:

- **Don't use the fingerprint as a strategy-collapse diagnostic for
  Axes 1–5** — it can't separate them by design. Instead use the
  panel winrate matrix directly: if two variants have winrates within
  Wilson-CI of each other, they may be redundant.
- **Do use the fingerprint as the meta-router's classifier** — it
  works at the basin granularity Axis 0 needs.

## 9. Risk register

| Risk | Mitigation |
|---|---|
| Path A relabel doesn't clear 90% gate | Fall back to path B (FEATURE_VERSION=2 with distribution-shape + temporal-split features); +½ day |
| Live opponent is in a basin we don't have | `ABSTAIN_BELOW` confidence threshold; default to `roi` |
| BR-table overfits zoo (joint-policy-correlation) | Default action is always `roi`; never abstain into non-`roi` |
| Rolling-last-2 evicts v1.2 before μ settles | Don't submit anything until v1.2 lands a μ |
| Coord Axis 3 introduces ABC change that breaks sister strategies | Land ABC tweak as no-op for stateless strategies; tested under existing `tests/test_simple_strategies.py` |
| Behavioural collapse across Axes 1–2-5 variants (panel signal indistinguishable) | Detect via panel WR Wilson-CI overlap; iterate or drop variant |

## 10. What this plan does NOT decide

- **The final BR-table contents** for `weakest` and `enemy_first` rows
  — they're extrapolations that need validation on the replay corpus.
- **Whether to bump fingerprint to FEATURE_VERSION=2** — only triggered
  if path A fails or §F.5 demands per-strategy refinement.
- **The Strategy-ABC per-instance-state shape** — defer to
  implementation session; sketch in §4 is illustrative.
- **4P-FFA panel design** — Axis 6 is parked.
- **Search / RL paths (B.3, B.4 in ISSUES)** — unchanged from
  `docs/strategies/roadmap.md`.

---

## Appendix A — Hypotheses now on `state/hypothesis-board.md`

Promoted in same commit as this doc:

- **H4 (research-note §E.3 / Axis 3 here):** Multi-source
  simultaneous-arrival timer beats `roi` standalone by ≥55% over 24
  seeds × both sides.
- **H5 (research-note §B.2):** Production-dominance lock predicate
  fires before step 200 in <10% of self-play games but, when it
  does, switching to defense-only loses no further games (≥95%
  retention).
- **H6 (research-note §F.3):** A spoiler-vs-leader rule in 4P FFA
  improves μ by ≥30 vs always-expand baseline. Parked behind 4P
  infra.
- **H7 (research-note §D.1):** Front-loading neutral capture by
  re-weighting ROI in steps 0–60 by `(500 − step − eta)^1.5` gains
  ≥3% winrate over `roi` baseline.
- **H8 (research-note §G.6):** Replacing greedy per-source-best with
  Hungarian-assignment solver gains ≥2% winrate at <100 µs added
  per-step cost.
- **H9 (NEW, Axis 0 here):** A 3-class meta-router (Phase 1 path-A
  relabel) with §K.5 BR-table beats `roi` standalone by ≥3 μ on a
  60-game zoo panel.

## Appendix B — what's expected to ship in `agents/meta/`

After Axis 0:
```
agents/meta/__init__.py
agents/meta/router.py                    # main entry point
agents/meta/classifier.py                # wraps the pre-trained model
agents/meta/br_policies/__init__.py
agents/meta/br_policies/default_roi.py   # re-exports agents/simple/roi
agents/meta/br_policies/roi_endgame_burn.py
agents/meta/br_policies/roi_sizing_overshoot.py
models/meta_router_v1.pkl                # gitignored; built from replay corpus
scripts/train_meta_router.py
tests/test_meta_router.py
```

After Axes 3–5: `agents/coord/`, `agents/phase/`, `agents/sizing/`,
`agents/source/`, `agents/defense/` — each following the
`agents/simple/` template.
