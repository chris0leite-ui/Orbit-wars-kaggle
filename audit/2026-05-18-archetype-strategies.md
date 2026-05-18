# Per-archetype strategy spec — 32 cells

Source of truth for *what good play looks like* on each of the 32
panel archetypes (`total_production × rotating_share × size_split`).
Machine-readable form lives at `lib/archetype_strategy.py`. Tests in
`tests/test_archetype_strategies.py` and the divergence report in
`scripts/archetype_report.py` reference both.

## Game-shaping numbers (turn-budget arithmetic)

- 500 steps total. Sun at (50, 50), radius 10. Board diagonal 141.
- Fleet speed: 1.0 (1 ship) → 6.0 (≥1000 ships), log-scale.
  Sweet spot is **200-300 ships at ~4 u/turn** — speed without
  single-point-of-failure risk.
- Combat: `survivor = max_force − second_max_force`; against
  garrison G, you need **G + 1 attackers** to flip a defended
  planet. Equal forces both die (tie kills all).
- Production: P ships/turn while owned. A P=5 planet held all
  game = 2 500 ships. Typical total board production = 20-150,
  giving endgame ship totals of ~10k-75k.
- Comets at steps 50/150/250/350/450 (low-prod, short-lived
  hot-potatoes).
- Sun-clipping: 10-15 unit detour for cross-board routes.
- Orbital aim lead ≈ `omega × eta × orbital_radius` (e.g. at
  omega=0.04 and eta=6 turns, a planet at r=30 has moved
  7.2 board units).

## Dimensional rules (one per axis value)

### Production tier — sets the **tempo / economy budget**

**`low_prod`** (≤64 total production; endgame ≈10-25k ships).
*Every capture is high-value*; payback period is long (50+ turns
to recoup a 30-ship investment from a P=1 planet). Conservative
sizing, sparse launches, prioritise defense. Crucial points:
adjacent neutrals, never the far prizes. Counterplay risk: a
single bad trade loses 5 % of your endgame economy.

**`med_low_prod`** (64-76; ≈25-40k endgame). The known regression
cluster. Tighter resource competition — deeper lookahead and
opp-response modelling pay (v7_0's K=10 search wins here vs
baseline's K=5 reactive). Crucial points: contest the central
production density before opp does. Counterplay risk: opp's
counter-launch can turn a successful capture into a bounce → trade.

**`med_high_prod`** (76-88; ≈40-55k endgame). Baseline's strong
region — generic ROI-weighted play with reactive rollouts works.
Crucial points: standard play; capture-defense balance.

**`high_prod`** (>88; ≈55-75k endgame). Tempo dominates. *The
H11 finding* from the public-kernel teardown: top agents fire
from **90%+ of planets by step 5**; baseline fires from ~40%.
A parallel-launch opening compresses the neutral-grab window
and locks the production density before opp can react. Crucial
points: ALL high-prod planets within reach, simultaneously.
Counterplay risk: opp also goes H11; the race is decided by
who has the smaller home-to-mid-planet travel time.

### Rotating share — sets the **prediction cost**

**`mostly_static`** (rot ≤ 33 %). Classical RTS. Targets are
stable, orbit prediction unnecessary. Launch angles cluster
around fixed bearings. Aim-at-current-position is fine.

**`mixed_static`** (33-40 %). Most static, a few rotating.
Light prediction needed for the rotating subset; static targets
dominate the priority list.

**`mixed_rotating`** (40-44 %). Balanced mix — highest
cognitive load. You must reason about BOTH stable bearings AND
moving targets, and decide which class to prioritise per game.
Three of the five baseline regression archetypes live in this
band, which suggests baseline's mission-portfolio doesn't
gracefully handle the mode-switching.

**`mostly_rotating`** (>44 %). Most targets move. Orbital aim
lead (`omega × eta × r`) is mandatory; "fire at current x,y"
misses by 5-15 units on inner orbits. Top public agents use 5-iter
aim refinement (`lib/aim.py`).

### Size split — sets the **prize locus**

**`big_static`** (`radius_static_mean > radius_rotating_mean`).
The high-prod planets are stationary — easy to LOCATE and lock
down (no aim math) but **expensive** because they typically
have larger garrisons too (production scales radius, so P=5
planet has 30+ ship garrison). Bigger fleets needed.

**`big_rotating`** (`radius_rotating_mean > radius_static_mean`).
The high-prod planets ORBIT. You need precise lead-aim, but
once captured the inner orbit positions are easy to defend
because they're closer to your home. Worth the aim investment.

## Archetype playbook (32 cells)

Composite of the dimensional rules above. Read as
`prod × rot × split → strategic priority`.

| Archetype | Key objective | Crucial points to conquer | Counterplay risk |
|---|---|---|---|
| `low_prod__mostly_static__big_static` | Defend, contest only adjacent | 1-2 adjacent high-prod static | A bad early bounce = 10 % economy loss |
| `low_prod__mostly_static__big_rotating` | Defend; lead-aim the few prize orbits | 1 rotating prize within 25-unit reach | Wasted lead-aim on far targets |
| `low_prod__mixed_static__big_static` | Sparse opening, big garrisons | The high-prod static cluster nearest home | Opp captures the same cluster first |
| `low_prod__mixed_static__big_rotating` | Lead-aim 1-2 rotating prizes early | The largest rotating planet in your near hemisphere | Bad lead-aim → fleet death |
| `low_prod__mixed_rotating__big_static` | Static-first; rotating is decoy | Static prize density in Q1 | Opp uses rotating to outflank |
| `low_prod__mixed_rotating__big_rotating` ⚠ | Lead-aim a single rotating prize | Closest big rotating planet | Aim miss compounds — KNOWN REGRESSION |
| `low_prod__mostly_rotating__big_static` | Tiny static base + protect; orbit the rest | The 1-2 static high-prod | Opp dominates the orbiting market |
| `low_prod__mostly_rotating__big_rotating` | Pure orbital play, conservative | One big rotating prize, captured & defended | Aim errors cumulative |
| `med_low_prod__mostly_static__big_static` | Standard small-board play | 2-3 static prizes near home | Opp deeper search outscores |
| `med_low_prod__mostly_static__big_rotating` | Mixed; lead-aim where worth it | The big rotating planet AND a static anchor | Mode confusion |
| `med_low_prod__mixed_static__big_static` ⚠ | Density race + careful sizing | Central static cluster | KNOWN REGRESSION — baseline mis-sizes |
| `med_low_prod__mixed_static__big_rotating` ⚠ | Mid-pace + lead-aim prizes | Big rotating in inner band | KNOWN REGRESSION |
| `med_low_prod__mixed_rotating__big_static` ⚠ | Static prize + parry rotation | Central static, the big one | KNOWN REGRESSION |
| `med_low_prod__mixed_rotating__big_rotating` | Both classes — pick fast | Whichever class is denser near home | Mode-switching cost |
| `med_low_prod__mostly_rotating__big_static` | Static anchor + scrap rotating | The 1-2 static high-prod | Bad aim on rotating |
| `med_low_prod__mostly_rotating__big_rotating` | Pure orbital, deeper search | Inner-orbit big rotating | Opp counter-prediction |
| `med_high_prod__mostly_static__big_static` ⚠ | Race for central density | Static prize cluster | KNOWN REGRESSION — opening tempo gap |
| `med_high_prod__mostly_static__big_rotating` | Mix; lead-aim selectively | Both anchors and the big rotating | Diluted focus |
| `med_high_prod__mixed_static__big_static` | Standard density play | Central static + Q1 anchor | Opp's settle_plan |
| `med_high_prod__mixed_static__big_rotating` | Aggressive aim play | Inner-orbit big rotating | Aim drift |
| `med_high_prod__mixed_rotating__big_static` | Standard | Central static density | Opp parallel opening |
| `med_high_prod__mixed_rotating__big_rotating` | Standard rotating | Inner band big rotating | Mode-switch cost |
| `med_high_prod__mostly_rotating__big_static` | Static anchor + chase rotating | The 1-2 big static | Mis-sized fleets |
| `med_high_prod__mostly_rotating__big_rotating` | Pure orbital | Inner big rotating | Aim drift |
| `high_prod__mostly_static__big_static` | **H11 OPENING**, lock the prize cluster | EVERY big-prod static within step-10 reach | Opp also H11s |
| `high_prod__mostly_static__big_rotating` | H11 + lead-aim the prize | Big rotating + the static cluster | Opp wins the static race |
| `high_prod__mixed_static__big_static` | H11 + density attack | Central static prize density | Opp gets there first |
| `high_prod__mixed_static__big_rotating` | H11 + aim-lead big | Big rotating + Q1 anchor | Lead-aim cost slows tempo |
| `high_prod__mixed_rotating__big_static` | H11 + static prize | Static prize cluster | Mode confusion |
| `high_prod__mixed_rotating__big_rotating` | H11 + orbital aim | Inner big rotating + close anchors | Hardest archetype — high cost, high reward |
| `high_prod__mostly_rotating__big_static` | H11 + lock the static | Static prizes (few but huge) | Opp captures the static gold |
| `high_prod__mostly_rotating__big_rotating` | H11 + pure orbital | Inner big rotating | Hardest aim demand of all 32 |

(⚠ = known regression in baseline as of 2026-05-18; cell listed in
`lib.archetype_strategy.KNOWN_REGRESSIONS`. Tests mark these xfail
until archetype-aware logic ships.)

## What we measure (mapping spec → fingerprint metrics)

The spec is enforced through these measurable behavioural metrics:

| Strategic claim | Metric (in `lib/fingerprint.py` / `scripts/extended_features.py`) |
|---|---|
| "H11 opening — fire fast & wide" | `first_launch_step ≤ 5`, `early_launches ≥ 6`, `multi_launch_turn_rate ≥ 0.2` |
| "Conservative low-prod sizing" | `mean_fleet_size ≥ 12`, `launches_per_turn ≤ 1.5` |
| "Standard mid-pace tempo" | `launches_per_turn ∈ [0.8, 2.5]` |
| "Static-mostly games — tight angles" | `launch_angle_var ≤ 2.0` |
| "Rotating-rich games — diverse angles" | `launch_angle_var ≥ 0.5` |
| "Target the high-prod prizes" | `mean_target_production ≥ 1.5-1.8` |
| "Economy first (low prod)" | `targets_neutral_fraction ≥ 0.4` |

Test thresholds are deliberately loose (~factor-2 tolerance) so
per-seed RNG variation doesn't trip them. What we want tests to
catch is *systematic* mismatch (e.g. H11 missing in high_prod
games — first_launch_step consistently > 5).

## What this enables next

1. **`pytest tests/test_archetype_strategies.py`** — verify any
   agent against the spec; XFAIL flips to PASS = closed regression.
2. **`python scripts/archetype_report.py <agent>`** — table of
   divergence scores per archetype. The high-divergence rows tell
   you which archetype to tune NEXT.
3. **Follow-up — archetype-aware agent**: branch on the live
   `obs.archetype` (computed via `lib.geometry_features` and binned
   via `lib.seed_panel.PANEL_BIN_EDGES`) and apply the dimensional
   rules. The largest expected gain is the H11 opening tier for
   `high_prod__*` cells — that alone explains the gap to public
   top-10.
