# 2026-05-18 — Geometry-stratified 128-seed eval panel

## Why

Local strategy evaluation has been running against `SEEDS_20 =
[42, 1, 7, 13, 31, ...]` (`scripts/eval_v1.py`). That list was
chosen for early correctness gates, not for geometry coverage —
nothing guarantees those 20 seeds cover the archetype range the
Kaggle ladder actually samples.

The PI flagged (eyes-on inspection of replays) that initial states
vary along several distinct flavours: sparse-tiny vs dense-large,
mostly-static vs mostly-rotating, big-static-with-small-rotating vs
the reverse, slow vs fast games. A strategy that wins on average
across random seeds can still lose hard on one specific flavour;
without coverage we can't see the regression.

## What

`scripts/build_seed_panel.py` enumerates seeds `[0, 10_000)`, extracts
~15 geometry features per seed, and stratified-samples 128 seeds
into a 4 × 4 × 2 = 32-cell grid:

- **`total_production`** (4 bins, percentile-based) — captures
  "fast game / high prod" ↔ "few tiny planets". Originally drafted
  on `n_planets` but swapped because `n_planets × rotating_share`
  has unreachable cells (the simulator enforces
  `MIN_STATIC_GROUPS = 3`, so 20-planet boards can have at most 2
  rotating groups). `total_production` has a wider range and is
  less coupled with `rotating_share`.
- **`rotating_share`** (4 bins) — direct PI ask, core gameplay axis.
- **`size_split`** (2 bins, sign of
  `radius_rotating_mean − radius_static_mean`) — the
  big-static-vs-big-rotating flavour the PI named most specifically.

4 seeds per cell, picked by greedy farthest-point sampling on
secondary features (`angular_velocity`, `home_orbital_radius`,
`nearest_neighbor_mean`) so the 4 in-cell seeds also spread.

## Files

- `lib/geometry_features.py` — `extract_geometry(seed)` returns a
  flat feature dict from `env.reset()`. Reuses `lib/orbit.is_orbiting`
  and `lib/geometry.CENTER`.
- `scripts/build_seed_panel.py` — full pipeline; writes
  `audit/seed-panel/features.json` (cache) and
  `data/seed_panel_128.json` (final).
- `lib/seed_panel.py` — importable: `SEED_PANEL_128`,
  `SEED_PANEL_BY_ARCHETYPE`, `ARCHETYPE_OF_SEED`.
- `scripts/render_seed_panel.py` — 32-archetype preview PNG to
  `audit/seed-panel/preview.png`.
- `scripts/validate_seed_panel.py` — baseline-vs-baseline tournament
  on the panel; reports per-archetype winrate variance.
- `scripts/eval_v1.py` — adds lazy `SEEDS_128` import next to the
  existing `SEEDS_20`.

## How to use

```python
from lib.seed_panel import SEED_PANEL_128, SEED_PANEL_BY_ARCHETYPE
from scripts import tournament  # or importlib pattern from eval_v1.py

# Full panel run
result = tournament.run_tournament(
    agents={"my_agent": "agents/v1_orbitfix/main.py", "baseline": "data/main.py"},
    seeds=SEED_PANEL_128,
    include_self_play=False,
)

# Per-archetype regression report
from lib.seed_panel import ARCHETYPE_OF_SEED
for game in result.matrix["my_agent"]["baseline"].games:
    archetype = ARCHETYPE_OF_SEED[game.seed]
    # bucket / aggregate as needed
```

## Real-replay coverage check (submission 52710995, 100 games)

`scripts/validate_panel_vs_replays.py` extracts the same geometry
features from 100 real Kaggle-ladder games (55 × 2-player, 45 × 4-player)
and bins them into the 32 archetypes.

**Headline:** every real game falls inside a panel bin (no out-of-range
features), so the panel is structurally valid as a coverage set. The
panel uses **uniform 4 seeds per cell**, but live-play frequency is
not uniform:

- **Over-covered (good for tail-regression detection):**
  - 2P: 5 archetypes never appear in 55 live games (mostly
    `high_prod__*__big_rotating`). Panel still keeps 4 seeds each
    so we'd catch a regression there before it landed on the ladder.
  - 4P: 11 archetypes missing from 45 live games.
- **Under-covered (slots we should consider doubling):**
  - 2P top live cells run 2-3× the panel's uniform 3.1% allocation.
  - `med_high_prod__mixed_static__big_static`: 9.1% live vs 3.1% panel.
  - `med_high_prod__mostly_rotating__big_static`: 7.3% live.
  - `low_prod__mostly_static__big_static`: 7.3% live.
  - `low_prod__mixed_static__big_rotating`: 7.3% live.

Full output: `audit/seed-panel/replay-coverage-52710995.txt`.

**Interpretation.** Uniform stratification trades sample-efficiency
for tail-regression detection. Since the goal of this panel is "catch
geometry-conditional failures BEFORE they hit the ladder," uniform is
the right choice — a frequency-matched panel would have ≤1 seed per
rare archetype and miss those regressions. A future
`SEEDS_DISTRIBUTION_MATCHED_64` panel mirroring live frequency would
complement, not replace, the uniform `SEEDS_128`.

**4P caveat.** The panel is 2P-only (seat-0 vs seat-1 home assignment).
4P games sample the same initial-planet RNG path, so the geometry
features are comparable, but 4P-specific regressions need their own
panel.

## Self-play validation (baseline vs baseline, 128 panel games)

`python scripts/validate_seed_panel.py` ran the comp-shipped baseline
against itself across all 128 panel seeds (~9 min CPU on this container).

- 128/128 games completed cleanly.
- Aggregate: 31 P0 wins / 29 P1 wins / 68 draws — P0/P1 split well
  within the ±15 % gate from ISSUES.md::A.6.
- **Geometry-conditional signal is strong**: 27/32 archetypes show
  P0 winrate outside [0.40, 0.60]; stdev across archetypes = 0.22.
- This confirms the panel actually exposes flavour-dependent
  differences. The draw rate (53 %) is the well-known
  baseline-self-play step-500 attrition pattern — irrelevant here
  since we only need the panel to surface variance, not produce
  decisive games.

Outputs:
- `audit/seed-panel/selfplay-validation.txt` — per-archetype readout.
- `audit/tournaments/20260518T111505Z.json` — full tournament record
  (per-game seed, ship-delta, statuses).

## Open follow-ups (out of scope here)

- Synthetic / handcrafted geometries (PI flagged as later). Now that
  we know which archetype cells actually exist, handcrafting
  representative boards becomes well-posed.
- Seed → archetype prediction (PI's "fun" side question). The
  features.json this pipeline produces is the dataset for it.
- Wire the per-archetype regression report into every tournament
  callsite (currently only `validate_seed_panel.py` reports it).
