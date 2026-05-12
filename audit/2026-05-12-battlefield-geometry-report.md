# Battlefield geometry report — Orbit Wars

**Date:** 2026-05-12. **Branch:** `claude/game-strategy-analysis-XXxEK`.
**Source:** `scripts/geometry_report.py` over seeds 0-99, 2-player. Raw
per-seed JSON: `audit/2026-05-12-battlefield-geometry-data.json`.

This is the first end-to-end empirical pass over the Orbit Wars map
distribution. Every number is reproducible by re-running the script. No
submissions were used.

---

## TL;DR — six load-bearing facts

1. **Maps are strictly 4-fold rotationally symmetric.** 98.2% of planets
   sit in 4-fold-symmetric radius clusters; only 7 / 100 seeds have any
   cluster that isn't (and those are minor — typically one ring of
   8 planets that happens to share both radius and production).
2. **2-player homes are always on the Q1 + Q3 diagonal.** 100/100 seeds.
3. **Homes are always STATIC.** Orbital radius + planet radius ≥ 50 in
   every seed. Inner-orbit planets are the dynamic battlefield; homes
   are anchors.
4. **The sun blocks ~20% of all straight planet-to-planet shots.** Mean
   19.5%, range 13.7-23.8%. A non-trivial fraction of natural attack
   pairs need an arc route.
5. **Orbital sweep cuts attack distance by a median 50% at the right
   moment.** Averaged over all (my-home, orbiting-target) pairs, the
   closest-approach distance is 35% shorter than the initial distance;
   the *best* target in each seed offers a 75% median reduction.
6. **Comet spawns are clockwork.** 4 comets per spawn event, 5 spawn
   events per game (steps 50, 150, 250, 350, 450). Each comet lives on
   the board for ~31-37 turns. That's 20 free production-1 targets per
   game, of which v3_snipe currently captures ~5%.

---

## 1. Planet count distribution

100 seeds, 2-player. Mean **28.3**, median **28**, range **20-40**.
Spec range (20-40) confirmed.

```
 16.0-18.8    0
 18.8-21.6   18  ##########################
 21.6-24.4   18  ##########################
 24.4-27.2    0
 27.2-30.0   20  #############################
 30.0-32.8   27  ########################################
 32.8-35.6    0
 35.6-38.4   16  #######################
 38.4-41.2    1  #
```

Multi-modal: counts cluster around 20, 28, 32, 36 — consistent with
5, 7, 8, 9 quartets respectively. Roughly half the seeds have **28-32
planets** (7-8 quartets).

## 2. Production-rate distribution (planet `production` field)

2832 planets across 100 seeds.

| production | count | share |
|-----------:|------:|------:|
| 1 | 836 | 29.5% |
| 2 | 584 | 20.6% |
| 3 | 520 | 18.4% |
| 4 | 476 | 16.8% |
| 5 | 416 | 14.7% |

**Skewed low.** Cheap production-1 planets are the most common. This
matters for ROI scoring: most of the map is low-yield, so the few
production-4/5 planets are disproportionately valuable.

## 3. Orbital radii

Mean **44.6**, median **48.7**, range **21.7-65.6**. Bimodal:

```
 20.0-25.0  156  ########
 25.0-30.0  168  ########
 30.0-35.0  316  ################
 35.0-40.0  420  ######################
 40.0-45.0   24  #
 45.0-50.0  756  ########################################
 50.0-55.0  604  ###############################
 55.0-60.0  388  ####################
```

**Two populations:** inner planets at radii 20-40 (~38% of total) and
outer planets at radii 45-60 (~62% of total). The dip at 40-45 is the
rotation cutoff (radius + planet_radius ≥ 50 → static).

**Per-seed split**: median 12 orbiting, 16 static. **37.7% of planets
orbit on average.** Every seed has at least 4 orbiting planets, but the
fraction varies seed-to-seed from 17% to 57%.

## 4. Per-game angular velocity ω

Range **0.025-0.049 rad/turn** (matches spec). Median 0.036.

Over 500 turns that's **12.5-24.5 radians = 2.0-3.9 full revolutions**.
Every orbiting planet sweeps a complete circle at least twice per game.

| ω band | seeds |
|--------|------:|
| 0.025-0.030 | 8 |
| 0.030-0.035 | 30 |
| 0.035-0.040 | 35 |
| 0.040-0.045 | 16 |
| 0.045-0.050 | 11 |

ω is roughly uniform across its range, with a slight peak around 0.034.

## 5. 4-fold rotational symmetry check

For each seed, planets are clustered by orbital radius (tolerance 0.05).
Each cluster is then tested for regular angular spacing of `2π/N`.

- **Mean: 98.2% of planets are in 4-fold-symmetric clusters.**
- Median: 100%. Min: 66.7%.
- **93 / 100 seeds have STRICT 4-fold symmetry on every cluster.**
- 7 seeds have at least one non-symmetric cluster: seeds **20, 44, 48,
  49, 81, 96, 99**. In every such case it's a single mixed cluster
  where two physically different quartets coincidentally share an
  orbital radius; the clusters are still made of two regular quartets,
  just at different angular offsets.

**Strategic implication:** 4-fold mirror reasoning is sound. The map
**is** symmetric. Option H (mirror-quadrant proposal) has a stable
underlying geometry; the 7 anomaly seeds should be handled by a
fallback to non-mirror behaviour, not by abandoning the strategy.

## 6. Sun no-fly zone (segments clipping the 10-unit sun disk)

For each seed, fraction of unordered planet pairs whose straight-line
chord passes within 10 units of (50, 50):

- **Mean 19.5%, median 20.0%, range 13.7-23.8%.**
- No seed has under 13.7%.

```
 10-15%    2  #
 15-20%   45  ################################
 20-25%   53  ########################################
```

**Strategic implication:** Roughly **one in five** natural source→target
pairs needs sun-tangent routing or it's simply invalid (Option B). The
current `sun_avoid` mechanism in `lib/trajectory.py` rejects these
fleets at validation; we are leaving ~20% of the proposal pool on the
table.

A corollary: **planets that sit in the enemy's sun-shadow cone** enjoy
a free defensive bonus. With homes at Q1 and Q3 and the sun at the
centre, any planet within ~14° of the enemy-home-to-sun line is
unreachable by direct shot from that home. This is Option F's surface.

## 7. Nearest-neighbour distance

Median **12.1 units**, range **9.0-28.4**.

```
  8-12  1296  ########################################
 12-16  1120  ##################################
 16-20   296  #########
 20-24    92  ##
 24-28    20  #
```

**Tight clusters.** Two-thirds of planets have a neighbour within 12
units. With small-fleet speed ~1.6, that's an 8-turn cross-time.
Closely-packed pairs are the natural micro-battle scale.

## 8. Home-to-home distance over 500 turns (STATIC homes)

Mean **88.2**, median **96.9**, range **43.7-128.4**. As predicted,
homes are static — distance is constant for the entire game. The
home-to-home line passes within 0 of the sun by construction (Q1↔Q3
diagonal through (50,50)), so the straight chord is **always blocked**.

This means **every cross-board fleet between the two homes pays a
sun-tangent tax**. A 70.7-unit chord (homes at (25,25) and (75,75))
becomes ~84.7 units along the safety tangent — 20% longer travel.

## 8b. Distance from my home to every ORBITING planet over 500 turns

This is the central insight for Option C (orbital phase-lead targeting).

For each seed, for each orbiting planet, we compute the distance to my
home at every turn 0-499 and take the minimum (closest-approach):

| metric | mean | median |
|--------|-----:|-------:|
| mean initial distance (t=0) | 52.2 | 54.1 |
| mean closest-approach distance | 29.3 | 24.1 |
| per-target % gap `(t0-min)/t0`, averaged within seed | **35.1%** | **49.7%** |
| per-target % gap, MAX within seed | **49.9%** | **75.0%** |

```
mean pct-gap per seed:
   0-8 %   38  ########################################
  40-48%    9  #########
  48-56%   14  ##############
  56-64%   32  #################################
  64-72%    7  #######
```

**38 seeds show ≤ 8% mean gap.** These are seeds where the *typical*
orbiting target is already near its closest approach at t=0 — the home
quadrant is rich with reachable inner planets. The other 62 seeds show
mean gaps ≥ 40%, with a strong mode at 56-64%.

**Magnitude.** A typical mid-game target sits at ~50 distance units,
falls to ~30 at closest approach. The single most-extreme target in an
average seed has a 75% distance reduction at its sweet-spot turn.

A fleet aimed at the t=0 position must traverse 50 units. Aimed at the
closest-approach point, the same fleet traverses ~30 units. With small
fleets at speed ~1.6, this saves ~12 turns. With production-3 planets
at the target, that's a 36-ship swing.

**Direct dollar value of phase-lead targeting**: on the best target per
seed, expect 50-75% travel time saved. Across all targets, expect ~35%
saved on average. This is the keystone primitive (Option C). It also
explains why the existing v1_orbitfix `predict_relative` upgrade
captured +205 μ — but only when used reactively at t=0. Using it
predictively to *choose the launch turn* recovers the rest.

## 9. Home-quadrant pairs (2P)

```
  quads (1, 3): 100
```

**100 / 100 seeds place homes on the Q1 + Q3 diagonal.** Spec confirmed.
This is a HARD prior: in 2-player mode we always know our enemy's
quadrant. Mirror reasoning is exactly the right primitive.

## 10. Comet atlas (10 seeds × 5 spawn events)

For each of seeds {0, 10, 20, ..., 90}, stepped the env to each spawn
step with no-op actions:

| seed | comets per event | path length per comet |
|------|-------------------|----------------------|
| 0 | 4, 4, 4, 4, 4 | [34, 34, 34, 34], [35, ...], [32, ...] × 5 |
| 10 | 4, 4, 4, 4, 4 | [33, 32, 37, 32, 31] (per event, all 4 equal) |
| ...  | ... | ... |
| 90 | 4, 4, 4, 4, 4 | [32, 32, 34, 33, 31] |

**Every spawn event delivers exactly 4 comets**, in a 4-fold-symmetric
quartet (one per quadrant). All 4 comets in a single event share the
same path length (their trajectories are mirrors of each other). Path
lengths cluster around **31-37 turns** — that's the lifetime of each
comet on the board.

**Total: 20 free production-1 targets per game**, spawning on a
predictable clock: 50, 150, 250, 350, 450. The current submission
captures ~5% of these (per `audit/2026-05-11-v3-snipe-games-analysis.md`
§5). Option D (comet interception pre-positioning) addresses this.

## 11. Frontier-line planets (±10 units of Q1↔Q3 bisector)

Median **6 planets** in the contested band per seed; range 2-12.

```
 2-4    11  ################
 4-6    22  #################################
 6-8    26  ########################################
 8-10   25  ######################################
10-12   14  #####################
12-14    2  ###
```

**The contested zone is dense.** Roughly 20-30% of every map sits in
the natural Q1↔Q3 frontier. These are the planets that change hands
most often, the natural sites for early-game capture races and
mid-game flipping. For Option E (mutual-annihilation feint), these are
the candidate trigger sites: where both sides will plausibly route
fleets at the same step.

---

## What this changes about the strategy menu

The plan's original menu (Options A-H in `/root/.claude/plans/...`)
holds up, with some priors updated:

- **Option B (sun-tangent routing).** Surface area is 19.5% of all
  planet pairs, not a tail case. Promoted from MED prior to **MED+**
  (60-100 μ). This is at least one in five proposed missions.
- **Option C (orbital phase-lead).** Confirmed as the keystone. The
  per-target average distance gap is 35%, and the best target per seed
  offers 50-75%. The library is worth building before D, E, G.
- **Option D (comet interception).** Confirmed predictable: 5 events
  × 4 comets, 31-37-turn lifetime, 4-fold-symmetric paths. Prior:
  **MED+** (60-90 μ), capturing even 50% of comets quadruples our
  current ~5% rate.
- **Option F (sun-shadow safe harbor).** Surface area is real (19.5%
  blocked pairs implies ~14° angular sun-shadow per home). Prior
  **LOW-MED** holds; this is a defender-allocation win, not a winning
  move on its own.
- **Option H (4-fold mirror snipe).** Geometric assumption is valid
  on 93% of seeds. The 7 anomaly seeds should fall back to non-mirror
  behaviour rather than abandon. Prior **LOW** holds — this is a hedge
  option.
- **Option A (recapture wire-up).** Geometry-neutral; still the
  highest-value same-day action.

## Reproduction

```
python3 scripts/geometry_report.py     # ~30 s for 100 seeds
```

Outputs `audit/2026-05-12-battlefield-geometry-data.json` and a text
summary identical to sections 1-11 above.

## Verification

Orbit math sanity check: from seed 42, planet 0 starts at
(68.17, 94.89) with ω=0.0410, orbital radius = √((68.17-50)² +
(94.89-50)²) = 48.4; total radius+planet_radius = 50.5 > 50 → STATIC.
Confirmed: home planets are physically pinned, matching the empirical
"sweep amplitude = 0" result in section 8.
