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

## Open follow-ups (out of scope here)

- Synthetic / handcrafted geometries (PI flagged as later). Now that
  we know which archetype cells actually exist, handcrafting
  representative boards becomes well-posed.
- Seed → archetype prediction (PI's "fun" side question). The
  features.json this pipeline produces is the dataset for it.
- Wire the per-archetype regression report into every tournament
  callsite (currently only `validate_seed_panel.py` reports it).
