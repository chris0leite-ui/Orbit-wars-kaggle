# Per-archetype winrate gap: top-10 vs our submission 52710995

Corpora: top-10 = 30 2P + 20 4P replays (from 2026-05-11 curation); ours = 55 2P + 45 4P from submission 52710995 ladder games.

All top-10 corpus entries are WINS by construction (5 wins per team curated). Their per-archetype winrate is therefore 100% wherever they have samples; the analytical signal is **which archetypes top-10 plays + wins on** vs **which our submission wins**. A high gap means top-10 wins in cells we lose — IL target.

## Ranked gap table (2P, sorted by gap)

| archetype | top-10 W/n | ours W/n | gap |
|---|---|---|---|
| `low_prod__mixed_static__big_static` | — | — | — |
| `low_prod__mostly_rotating__big_rotating` | — | 1/1 | — |
| `med_low_prod__mostly_static__big_static` | — | 1/1 | — |
| `med_low_prod__mixed_static__big_static` ⚠ | — | 1/1 | — |
| `med_low_prod__mixed_rotating__big_rotating` | — | 0/1 | — |
| `med_low_prod__mostly_rotating__big_rotating` | — | 0/2 | — |
| `med_high_prod__mostly_static__big_static` ⚠ | — | 1/1 | — |
| `med_high_prod__mostly_static__big_rotating` | — | 0/1 | — |
| `med_high_prod__mixed_static__big_rotating` | — | 0/1 | — |
| `med_high_prod__mixed_rotating__big_static` | — | 0/2 | — |
| `med_high_prod__mixed_rotating__big_rotating` | — | 2/2 | — |
| `high_prod__mostly_static__big_static` | — | — | — |
| `high_prod__mostly_static__big_rotating` | 1/1 | — | — |
| `high_prod__mixed_static__big_rotating` | 5/5 | — | — |
| `high_prod__mixed_rotating__big_static` | — | 0/1 | — |
| `high_prod__mostly_rotating__big_rotating` | — | — | — |
| `low_prod__mostly_static__big_rotating` | 1/1 | 0/2 | +100% |
| `low_prod__mixed_rotating__big_rotating` ⚠ | 1/1 | 0/1 | +100% |
| `low_prod__mostly_rotating__big_static` | 2/2 | 0/1 | +100% |
| `med_low_prod__mixed_static__big_rotating` ⚠ | 1/1 | 0/2 | +100% |
| `med_high_prod__mostly_rotating__big_rotating` | 1/1 | 0/1 | +100% |
| `high_prod__mixed_static__big_static` | 2/2 | 0/2 | +100% |
| `high_prod__mostly_rotating__big_static` | 1/1 | 0/2 | +100% |
| `med_high_prod__mixed_static__big_static` | 6/6 | 1/5 | +80% |
| `med_low_prod__mixed_rotating__big_static` ⚠ | 2/2 | 1/2 | +50% |
| `high_prod__mixed_rotating__big_rotating` | 1/1 | 1/2 | +50% |
| `low_prod__mixed_rotating__big_static` | 1/1 | 2/3 | +33% |
| `med_high_prod__mostly_rotating__big_static` | 1/1 | 3/4 | +25% |
| `low_prod__mostly_static__big_static` | 1/1 | 4/4 | +0% |
| `low_prod__mixed_static__big_rotating` | 1/1 | 4/4 | +0% |
| `med_low_prod__mostly_static__big_rotating` | 1/1 | 3/3 | +0% |
| `med_low_prod__mostly_rotating__big_static` | 1/1 | 3/3 | +0% |

⚠ = baseline known-regression archetype from the 2026-05-18 A/B vs v7_0.
