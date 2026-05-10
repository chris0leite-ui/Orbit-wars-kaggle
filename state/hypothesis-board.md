# state/hypothesis-board.md — open agent-design hypotheses

## Open

### 2026-05-10 — Phase 1 manifold hypothesis: partial refute

> Plan: `/root/.claude/plans/read-the-handover-next-imperative-whisper.md`.
> Audit: `audit/2026-05-10-phase1-manifold-verdict.md`.
> Reports: `audit/manifold/20260510T141114Z/` (7-class),
>          `audit/manifold/20260510T141409Z/` (5-class — gate target).
> Capture: `audit/replays/20260510T132957Z/` (1568 games, gitignored).

The user's hypothesis "competitor strategies live on a small-dim
manifold so a short prefix is informative enough to identify a class"
**partially confirmed** at 32 seeds × 5-strategy zoo with 15
hand-designed features:

- `weakest` (89.7%), `enemy_first` (83.4%), `baseline` (95% in 7-class)
  sit in their own basins — broad-class routing works.
- `nearest`, `production`, `roi` form a single "production-aware-
  greedy" basin with mutual confusion 12-17%; our 15-feature
  fingerprint can't separate them at K ≤ 200.
- Best 5-class score: RF 80.5% / LR 80.6% at K=100. Gate target was
  90%; **gate ❌ NOT cleared.**

**H-coarsen-labels (open, unranked):** merging the ROI-family into
a single class `production_aware_greedy` likely lifts RF to ≥92%
at K=100. Lets a 3-class meta-router proceed (broad-class routing
is what the panel actually needs — there's no submission incentive
to distinguish ROI-family members because ROI dominates them all).

**H-richer-fingerprint (open, queued behind H-coarsen):** adding
target-distance/production distribution-shape features + early-vs-
late split + target-id Shannon entropy plausibly separates the
ROI-family at K ≤ 100. Bumps `FEATURE_VERSION` to 2.

**H-learned-embedding (parked):** Grover et al. ICML 2018 protocol —
last resort if H-coarsen and H-richer-fingerprint both fail.

### 2026-05-10 — simple-strategy panel (target-selection ablations)

Five strategies under `agents/simple/` share v1.1's mechanism stack
(`[validate, arrival_size, lead_aim]`); they differ only in the score
function for picking a target. Run via
`python -m scripts.strategy_panel --seeds 32` for confidence;
`--seeds 8` for quick iter. Plan:
`/root/.claude/plans/read-the-handover-next-imperative-whisper.md`.

**8-seed smoke results (audit/tournaments/20260510T123059Z.json):**

| Strategy      | Hypothesis (one-liner)                                                  | Mean panel winrate | vs v1_orbitfix | Verdict (8-seed) |
| ------------- | ----------------------------------------------------------------------- | ------------------ | -------------- | ---------------- |
| `roi`         | production / distance is the right travel-adjusted ROI signal           | 96.9%              | 100% (16/16)   | ✅ strong        |
| `production`  | highest-production target beats nearest                                 | 75.0%              | 69% (11/16)    | ✅ confirmed     |
| `nearest`     | (control) reproduces v1's distance-greedy under the shared stack        | 56.2%              | 19% (3/16)     | ≈ tied with v1   |
| `enemy_first` | pressure-on-opponent beats economy                                      | 32.3%              | 12% (2/16)     | ❌ refuted       |
| `weakest`     | cheap snipes dominate                                                   | 15.6%              |  0% (0/16)     | ❌ refuted       |

**Open verdicts pending 32-seed confirmation:**
- H-roi-32: confirm `roi`'s 100% beat over v1_orbitfix holds at 32 seeds.
  If Wilson lo ≥ 0.6 over 32 seeds, `roi` is a v1.2 submission candidate
  (subject to roadmap submission economy: rolling-last-2 means do NOT
  push until v1.1's live μ has settled).
- H-production-32: same for `production`'s 69% beat — narrower margin,
  needs the seed bag to confirm or invert.
- H-nearest-vs-v1: nearest using DEFAULT_MECHANISMS is statistically
  the same agent as v1_orbitfix; the 19/81 split observed at 8 seeds
  is within the seat-asymmetry noise floor (sp 2/1/5 in own self-play
  cell). Confirm at 32 seeds — if it does diverge, dig into RNG seed
  ordering or whether `propose_intents` mirrors v1's exactly.

### 2026-05-10 — research-driven extensions (game-strategy-research-8w7EO)

> Source: `docs/strategies/heuristics-research.md` (universal heuristics
> + 15-item brainstorm). Plan: `audit/2026-05-10-research-driven-next-experiments.md`.
> All six hypotheses are pre-registered with explicit decision gates;
> H9 in particular is the immediate-priority Axis 0 (Phase 1 path-A
> meta-router) and operationalises §F (compete-relative) on top of the
> Phase 1 fingerprint infra already on main.

- **H4 (research-note §E.3 — Axis 3 in prep doc):** Multi-source
  simultaneous-arrival timer beats `roi` standalone by ≥55% over 24
  seeds × both sides. Combat resolver groups same-owner same-step
  arrivals; timing N source planets to land on the same step gives
  combat mass without speed loss. Predicted to become v3 mission class
  `gang_up` (roadmap mechanism 4) if H4 holds.
- **H5 (research-note §B.2):** The production-dominance lock predicate
  fires before step 200 in <10% of self-play games but, when it does,
  switching to defense-only retains the win in ≥95% of cases.
  Diagnostic-only at v2; policy-actionable at v3.
- **H6 (research-note §F.3):** A spoiler-vs-leader rule in 4P FFA
  improves μ by ≥30 vs the always-expand baseline over a 60-game
  panel. Parked until 4P-FFA panel infra exists
  (`scripts/strategy_panel_4p.py`).
- **H7 (research-note §D.1 — Axis 4 in prep doc):** Front-loading
  neutral capture by re-weighting ROI in steps 0–60 by
  `(500 − step − eta)^1.5` (vs linear) gains ≥3% winrate over `roi`
  baseline.
- **H8 (research-note §G.6):** Replacing greedy per-source-best with
  Hungarian-assignment solver gains ≥2% winrate at <100 µs added
  per-step cost. Future axis (heavier than current panel agents).
- **H9 (NEW — Axis 0 in prep doc):** A 3-class meta-router (Phase 1
  path-A relabel: nearest+production+roi → `production_aware_greedy`)
  with the BR table from `heuristics-research.md` §K.5 beats `roi`
  standalone by ≥3 μ on a 60-game zoo panel including
  `[weakest, enemy_first, baseline, roi, v1_orbitfix]`. Builds on the
  existing `lib/fingerprint.py` v1 (15 features) and
  `scripts/manifold_check.py` already on main; one-line `--label-merge`
  re-run; predicted RF ≥ 92% at K=100. **Recommended path A on §F's
  rank-aware-override grounds** — see `heuristics-research.md` §K.4
  for the rationale vs paths B and C.

### Pre-existing seeds (carried over from Day 1)

- H-search: A search-based agent (MCTS over short horizons) beats a
  hand-coded heuristic on the baseline-opponent panel.
- H-rl-curriculum: An RL agent trained on self-play overfits to
  symmetric strategies and loses to rule-based opponents — needs
  opponent-curriculum diversity.
- H-replay-mining: Replay statistics from top public-LB agents reveal
  a load-bearing tactic that no public notebook documents.

## Killed

(empty — `weakest` and `enemy_first` are leaning falsified at 8 seeds
but stay in **Open** until 32-seed confirmation. Falsified entries get
moved here with the audit-JSON path attached.)

## Hedge ladder

> Per CLAUDE.md R2: PRIMARY = best-current; HEDGE = next-best agent
> that regressed ≤ defined-bracket on the rank ladder. Populate during
> the final 3-day window.
