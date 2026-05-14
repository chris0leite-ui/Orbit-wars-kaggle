# Geometry-Conditional Strategy EDA — Roll-up

**Date**: 2026-05-14
**Branch**: claude/game-strategy-eda-roatN
**ISSUES leaf**: A.7 `[wip → done after this hand-back]`
**Plan**: `~/.claude/plans/you-are-a-senior-woolly-nest.md`
**Audit JSONs**:
- `audit/2026-05-14-public-notebook-scan.md`
- `audit/2026-05-14-board-taxonomy.json` (Mine 1)
- `audit/2026-05-14-planet-importance.json` (Mine 2)
- `audit/2026-05-14-opening-atlas.json` (Mine 3)
- `audit/2026-05-14-endgame.json` (Mine 4)
- `audit/2026-05-14-sun-shadow.json` (Mine 5)

## TL;DR

Four of five falsification gates passed. The hypothesis "optimal strategy is map-conditional" is supported in the OPENING (Mines 1+3) and in PLANET PRIORITISATION (Mine 2). The hypothesis is REJECTED for the ENDGAME (Mine 4 — winners universally expand, no regime switch) and for SUN-SHADOW (Mine 5 — too rare to matter). One experimental option from the original five is killed outright; two have inverted direction; one new option emerged from the data.

## Mine-by-mine result

| # | Mine | Gate | Result | Verdict |
|---|---|---|---|---|
| 1 | Board taxonomy | Cluster-conditional opening prod or dist spread ≥1.0 | prod 1.25, dist 17.3 | **PASS** — 4 archetypes |
| 2 | Per-planet importance | 5-fold logistic-regression AUC ≥0.70 | 0.7704 | **PASS** — radius + low garrison dominate |
| 3 | Opening atlas | Cluster-conditional opening templates differ ≥1 in prod or dist | yes (see table below) | **PASS** — 4 distinct playbooks |
| 4 | Endgame anatomy | Cluster-conditional 60/40 consolidate-vs-expand split | universally 76% expand | **PASS but inverts the hypothesis** |
| 5 | Sun-shadow exploitation | Spearman ≥0.2 shielded-frac vs survival | mean 0.025 | **FAIL** — drop axis |

## Headline numbers

- **Top-10 winners fire by median step 3.5** (range 1-22). 84.5% of first-30-turn targets are neutrals; **0% are enemy planets in the median game**. This is a land-grab, not a raid.
- **76% (45/59) of top-10 winners expand ship share by >2pp in the final 100 turns.** Only 1/59 contract. Median delta: +16.7 pp. *Late-game throttling is wrong.*
- **Planet importance is driven by radius and garrison size**, NOT by sun-shadow, denial, perp-bisector position, or centrality. Standardised LR coefficients: radius +0.78 (production proxy), starting_ships -0.71, min_home_dist +0.21.
- **Sun-shadow signal is essentially zero** — planets are shielded only 6.1% of owned-turns; mean Spearman correlation with survival is 0.025.

## Four opening-book templates

Pulled directly from the top-10 corpus, broken down by board-taxonomy cluster. These are the per-game medians (so they round to integer or half-integer):

| Cluster | n | First launch | Target prod (med) | Target dist (med) | Launches in first 30 |
|---|---|---|---|---|---|
| **C0 — wide-and-sparse** (home pair ≈108, prod 2.5, 35% orbital) | 10 | t=4 | 3.0 | 37.4 | 7 |
| **C1 — close-quarters-orbital** (home pair ≈66, 44% orbital) | 10 | t=3.5 | 3.0 | 29.6 | 6 |
| **C2 — wide-and-rich** (home pair ≈109, prod 3.0, dense) | 13 | t=3 | **4.0** | **20.1** | **10** |
| **C3 — sparse-static-rich** (21 planets, 23% orbital, prod 3.0) | 4 | t=4 | **4.2** | 37.0 | 6 |

The "wide-and-rich" board (C2) is the OPENING-BLITZ regime — earliest first launch, most launches, highest-production nearby targets. The "sparse-static-rich" board (C3) is the patient-long-arm regime — fewer launches at long distance to the few high-prod planets that exist. Different boards genuinely demand different openings.

## Ranked experiment options (replaces the original list of five)

### Tier 1 — ship this cycle

**Option 1 — "Don't throttle late" audit.** *Cheapest, lowest risk, highest information density per slot.*
Action: instrument v7_pv to log per-turn value-head outputs across a 32-game self-play set. Check whether the value of late-game expansion drops near turn 400+. If so, scale up; if not, the agent is already aligned with the data. Falsification: <1pp difference in winrate vs current v7_pv → null finding, but the diagnostic is valuable for free.
- Compute: ~30 min.
- Eviction risk: zero unless we submit a tweaked v7.
- Calibration: 3-agent local panel before any submission (v7_pv, v7_0, v3.5.1).

**Option 2 — Map-type-conditional opening book.** *Strongest empirical support of any single change in the EDA.*
Action: at t=0, compute the 10-feature board fingerprint, classify into one of 4 clusters, then for the first 30 turns override the proposer's first-launch with a cluster-specific template (target distance + target production + first-launch turn). Beyond turn 30, full v7_pv kicks in.
- Compute: ~1 hr build + 1 hr local panel.
- Risk: medium — only the opening is touched; existing v7 search runs on every subsequent turn.
- Pre-submit calibration ladder (Rule 27): Spearman vs v7_pv ≤0.999, panel Wilson-lo ≥0.55 on all 3 opponents.

### Tier 2 — queue for next cycle

**Option 3 — Planet-value head with EDA coefficients.** Replace v7's heuristic per-planet score with the Mine 2 LR coefficients. Bigger change than Option 2 (touches search ordering at every turn), so worth doing AFTER Option 2 stabilises on the LB.
- Compute: ~2 hr build + 2 hr ablation panel.
- Risk: higher; Rule 37 friendly axis (value-function not chooser).

**Option 4 — Meta-router refresh.** Feed the 4-cluster fingerprint into the existing D.6 manifold diagnostic (currently RF 80.5% at K=100). Higher signal-to-noise fingerprint may finally clear the 90% gate and unlock the meta-router track. Worth doing alongside Option 2 once that ships.

### Killed by EDA findings

- ~~Option 5 — Sun-shadow valuation bonus.~~ Mine 5 fails the gate. No further work.
- ~~Option 4 (original) — Endgame mode switch.~~ Mine 4 shows the opposite of what the option assumed; instead, see Tier 1 Option 1 ("Don't throttle late").

## What we don't know

- **Survivor bias on Mine 4**: every game in the corpus is a top-10 WIN. The "76% expand" finding measures what WINNERS do, not what LOSERS should do. The loss-mode-correct strategy may differ. Confirming would require a top-10-LOSS corpus we do not have.
- **Cluster softness**: silhouette ≈0.17 means clusters are real but not sharply separated. A hard k=4 classifier will misroute marginal boards. The map-conditional opening book needs to handle uncertainty — e.g. fall back to a global default when the classifier's confidence is below threshold.
- **Top-10 ≠ optimal**: every opening template is top-10 behaviour against top-10 opponents. Against a fundamentally different opponent class (e.g. a hyper-aggressive bot), the template may misfire. Pre-submit calibration must include diverse opponents (v7_0, v3.5.1, plus a few simple-strategy panel members).

## Public-notebook context (Rule 22)

Four notebooks scanned (`audit/2026-05-14-public-notebook-scan.md`). Headline observations from the scan that the EDA confirmed or contradicted:

- **Confirmed**: sigmaborov's Voronoi-style frontline detection — Mine 1 cluster C1 (close-quarters-orbital) is precisely the regime where this matters most.
- **Confirmed**: top-10 (and sigmaborov) fire early; melccoro's "early-off until turn 50" is correct against weak baselines, wrong against strong opponents (Mine 3).
- **Contradicted**: rahul's "neutral denial" term shows ~zero coefficient (-0.035) in Mine 2. May still work as a TACTICAL gain for the leading player, but does not predict capture priority in the top-10 corpus.
- **No notebook covers**: per-cluster opening templates, garrison-priority weighting calibrated against winner data, "don't throttle late" diagnostic. These are ours.

## Compute spent

- Public-notebook pull + scan: ~1 hr
- Top-10 corpus re-pull: 4.2 s (60 episodes, 337 MB; 8-thread)
- Self-play generation: ~25 min so far (81/500 at audit write-time; running in background)
- Mine 1-5 feature extraction + analysis: ~3 min total
- Total wall: ~1.5 hr, mostly notebook reading
- Total CPU: <5 min

Next session can pick this up from `audit/2026-05-14-eda-rollup.md` + the five mine JSONs. Recommendation: PI approves Tier 1 (Options 1 + 2) for the next 3-day cycle.
