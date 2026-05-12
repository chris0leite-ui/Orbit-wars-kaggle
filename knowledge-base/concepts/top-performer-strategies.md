# Top-performer strategies — Orbit Wars Kaggle competition

> Permanent reference. Built 2026-05-11 on branch
> `claude/analyze-leaderboard-strategies-sdZlE`. Sources: a single
> leaderboard CSV pull (top-30 snapshot), 50 replay JSONs sampled
> from current top-10 ladder leaders + 10 midpack control replays
> from our v2's live ladder, 14 pulled public kernels, 13
> discussion-forum threads, and the existing public-kernel teardown.
>
> All quantitative claims trace to:
>   - `audit/2026-05-11-lb-top30-snapshot.csv` (leaderboard)
>   - `audit/2026-05-11-top-performer-fingerprints.json` (60 rows × 15 features)
>   - `audit/2026-05-11-top-performer-extended.json` (60 rows × 10 extended features)
>   - `audit/2026-05-11-top-performer-profiles.json` (per-team aggregates)
>   - `audit/2026-05-11-top-performer-corpus.md` (source manifest)
>
> External corpus (not committed; gitignored): replays in
> `audit/external/replays/` (~305 MB), kernel JSONs in
> `audit/external/kernels-pulled/`, episode lists in
> `audit/external/episodes/`, discussion threads in
> `audit/external/discussions/`.

## 1. Why this exists

`state/current.md` puts our v2 at μ=974.3 with the top-10 prize cliff at
μ=1439.5 (3Comets) and #1 at μ=1682.9 (bowwowforeach). We had one prior
public-kernel teardown (`audit/2026-05-10-public-kernel-teardown.md`)
for Roman Tamrazov's μ=1224 published agent — Roman now sits ~250 μ
below the cliff, so adopting his architecture wholesale is **necessary
but not sufficient**. This document characterises what top-10 players
empirically do (replay-derived) plus what the strongest public agents
(those at the bottom of the top-30 band who have published code)
architect for, so the gap between Roman's μ=1224 and the cliff's μ=1440
is not opaque.

The 50-agent local round-robin tournament organised by @marcodg in the
official discussion (`audit/external/discussions/topic-698614.json`)
ranks public agents at the konbu17-hybrid → Roman → Pilkwang tier
(72-85% panel winrate); the unpublished top-10 lives above that. The
two corpora together let us triangulate.

## 2. Leaderboard snapshot — 2026-05-11

Source: `audit/2026-05-11-lb-top30-snapshot.csv`. Top-10:

| Rank | Team                 | μ      | User           | Last submit (UTC)   |
| ---- | -------------------- | ------ | -------------- | ------------------- |
| 1    | bowwowforeach        | 1682.9 | bowwowforeach  | 2026-05-11 12:51:55 |
| 2    | flg                  | 1598.6 | ferdinandlimburg| 2026-05-07 15:28:44 |
| 3    | Vadasz               | 1556.3 | vadasz         | 2026-05-11 15:26:30 |
| 4    | Isaiah @ Tufa Labs   | 1553.3 | pressman1      | 2026-05-07 22:18:34 |
| 5    | Ebi                  | 1548.8 | ebinan92       | 2026-05-11 10:01:26 |
| 6    | Shun_PI              | 1525.3 | shunrcn        | 2026-05-11 15:52:06 |
| 7    | Erfan Eshratifar     | 1485.0 | erfaneshrati   | 2026-05-11 18:26:43 |
| 8    | kovi                 | 1468.2 | kovi22         | 2026-05-11 08:15:16 |
| 9    | sash                 | 1440.9 | sash2104       | 2026-05-09 07:10:52 |
| 10   | 3Comets              | 1439.5 | javelin1991,lightmk | 2026-05-11 02:18:21 |

None of the top-10 have published their code. The strongest published
kernel author is @woosungyoon (Roche Overflow, rank #26) — a tutorial,
not their submitted agent. So **top-10 strategies are inferable only
from replays + cross-checked against public-agent ceilings**.

## 3. Per-player profiles (top-5)

Built from 5 replays per player (3 × 2P-wins + 2 × 4P-wins), K=100 turn
prefix. Numbers below are means across the 5 replays; "midpack" is the
10-replay v2-vs-mid-tier comparison set. Bold = top-5 differs from
midpack by ≥1.5σ.

> Plain-English label is the strategy archetype each player's
> fingerprint argues for — derived from replay statistics, not author
> claims (since none of them have published).

### #1 bowwowforeach (μ=1682.9) — "concentrated artillery"

- 0.88 launches/turn (midpack 0.63); **fleet mean 47, p95 135**
  (midpack 29 / 82). Largest fleets in the entire top-10.
- Mean garrison-at-launch **7.7** (midpack 22) — nearly empties the
  source on every launch.
- Sun-clip rate **4.4%** (midpack 9.8%) — best sun discipline of any
  top-10 player.
- Targets **42% enemy, 27% neutral** (midpack 14% enemy, 62% neutral)
  — overwhelmingly enemy-focused.
- First launch at step 4 (midpack 11) — opens immediately.
- 125 launches per game total. Comet capture rate 2.1% — does NOT
  chase comets.

The signature is **few, very large, very accurate fleets**. The 4.4%
sun-clip is genuinely low — possibly indicates a fixed-point intercept
solver (Roman-style 5-iter + safe-intercept) running on every shot. The
combination of high mean fleet (47) and low mean garrison (8) tells you
each source planet typically launches its FULL stockpile when it acts,
not a 1+ETA·prod minimum.

### #2 flg / ferdinandlimburg (μ=1598.6) — "saturation skirmisher"

- **1.21 launches/turn** (top-10 max launches/turn). 3330 launches
  across 5 games — ~666/game; mostly 2P wins.
- **Targets 61% enemy, 21% neutral** — most enemy-focused player in
  top-10. flg points at the opponent, not the map.
- Recapture rate **52%** — highest. flg routinely re-takes lost ground.
- First launch at step **1.6** — fastest opener of any top-10.
- Mean fleet 41, p95 116 — medium-large fleets like #1, but at twice
  the rate.
- Sun-clip 8.4% — surprisingly high for top-10, but compensated by
  volume; "fire many, lose some, still net positive".

The signature is **constant pressure**. flg launches roughly twice as
often as #1, picks enemy targets harder than anyone else, and recaptures
half of lost planets. Reads like a strategy that values *denying* the
opponent over *building* economy. The early-step-1 opener says flg
does NOT wait to scout — fixed opener.

### #3 Vadasz (μ=1556.3) — "balanced positional"

- 0.88 launches/turn (same rate as #1) with smaller fleets (mean 33,
  p95 82).
- Mean garrison-at-launch **12.8** — keeps a small reserve at home,
  unlike #1 and #4.
- Mean target distance **38.8** — longest-range targeting in top-10
  (midpack ~35).
- Targets 38% enemy, 25% neutral — closer to "mixed" than flg's pure
  enemy focus.
- Episode steps median 402 — Vadasz games go to near-completion.

Reads as **the careful positional player**. Smaller fleets but
launched at the same rate as #1 means total ship investment per turn is
similar; the difference is Vadasz reaches farther on each shot. The
longer mean target distance hints at a global-target-pool solver
(Hungarian-assignment-style) rather than nearest-greedy. The remaining
12-ship garrison-reserve discipline matches what @ykhnkf's published
notebook calls `frontier_keep`.

### #4 Isaiah @ Tufa Labs (μ=1553.3) — "siege artillery"

- **Lowest launches/turn in top-10**: 0.54. But **biggest fleets**:
  mean 69, p95 130.
- Mean garrison-at-launch 7.2 — empties sources like #1.
- Targets 45% enemy, 21% neutral — enemy-leaning like flg, but at
  one-quarter the launch rate.

Isaiah's profile is **one big punch every 2-3 turns**. Each fleet is
double the size of the median top-10 fleet. This is what you'd build
if the cost of small, lost fleets is high (Tufa Labs is a known
robotics/agents lab; possibly a search/RL agent that batches its
launches to amortise compute). Combined with the low launch rate, the
production-per-turn investment is comparable to others; the difference
is **lumpiness**.

### #5 Ebi / ebinan92 (μ=1548.8) — "volume + multi-launch"

- 1.10 launches/turn, **multi-launch turn rate 51%** (top-10 highest).
  Multi-launch turn = a turn where ≥2 fleets are dispatched.
- Targets 43% enemy, 27% neutral.
- Sun-clip rate **11.7%** (top-10 highest, even worse than midpack
  9.8%). Yet wins at μ=1548. Suggests Ebi's pathfinding **accepts**
  some sun crossings as long as the expected-value is positive — i.e.,
  sun_clip detection is OFF or has a low priority.

Ebi reads as **massively parallel** — coordinates many simultaneous
launches per turn from many sources. The very high multi-launch rate
(half of all turns ≥2 launches) is consistent with a settle_plan-style
global allocator running every turn. The sun-clip tolerance is the
oddity — either confident in their physics (rare misses get absorbed
by volume) or genuinely missing the sun-check that everyone else has.

## 4. Common qualities across all top-10

These features differentiate top-10 (any rank 1-10) from midpack at
≥0.5σ, in 2P games only (cleaner signal). All numbers are means.

| Feature                       | top-10 | midpack | gap (×) | What it means                                                          |
| ----------------------------- | -----: | ------: | ------: | ---------------------------------------------------------------------- |
| **launches_per_turn**         |  1.20  |    0.63 |  **1.9×** | Top-10 acts roughly twice as often.                                  |
| **mean_fleet_size**           |  38.2  |    28.7 |  +33%   | Bigger ship investments per launch.                                    |
| **p95_fleet_size**            |  98.9  |    82.0 |  +21%   | Top-10 occasionally swings very large fleets.                          |
| **mean_garrison_at_launch**   |  10.6  |    22.0 | **0.5×** | Top-10 empties sources; midpack hoards.                              |
| **targets_neutral_fraction**  |  0.30  |    0.62 | **0.5×** | Top-10 spends only 30% on neutrals; midpack 62%.                     |
| **targets_enemy_fraction**    |  0.32  |    0.14 | **2.3×** | Top-10 hits enemy planets 2.3× more often.                           |
| **sun_clip_launch_rate**      |  0.062 |   0.098 |  0.6×   | Top-10 has ~36% lower sun-disaster risk.                               |
| **multi_launch_turn_rate**    |  0.48  |    0.38 |  +27%   | Top-10 dispatches ≥2 fleets per turn 48% of turns.                     |
| **first_launch_step**         |  4.1   |   10.5  |  0.4×   | Top-10 opens by step 4; midpack waits 10+ steps.                       |
| **n_launches_total_per_game** |  751   |    213  |  3.5×   | Top-10 makes 3.5× more launches per game.                              |

The picture is **active, aggressive, enemy-focused, ships flowing**.
Midpack is **passive, neutral-grabbing, ships hoarded**.

Counterintuitive finding: midpack actually accumulates **more total
ships** per turn (12.9 vs top-10's 8.5) and ends games with more total
ships, yet **loses anyway**. Ships sitting in garrisons are score
points (per the comp's `final_score` rule) but they don't win planets,
and the margin-agnostic TrueSkill (`comp-context.md:153`) means the
*win/loss* is what moves μ — accumulated ship counts that don't flip a
planet are wasted potential.

## 5. What top-10 systematically AVOID

Inverse of the above, plus extended features:

- **Comet chasing.** Top-10 comet-capture rate = 3.4% of launches.
  Midpack = 13.4%. Midpack's intuition is "moving high-prod target =
  high value." Top-10's correction is that comet speed 4.0 + 4-turn
  trajectory window + planet collision-on-arrival make the
  break-even hostile in most spawn geometries. @emanuellcs's spoofing
  agent codifies this: any comet target needing `eta + recoup_time >
  remaining_path` is filtered with negative score.
- **Idle garrisons.** Mean garrison-at-launch 11 (top-10) vs 22
  (midpack). A ship sitting at home only generates production-equiv
  passive value; a ship in motion captures something.
- **Slow openings.** First launch step 4 vs 11. Six wasted turns of
  not-launching ≈ 6× one-fleet's-worth of value forfeited at the start.
- **Targeting just-neutral planets.** Top-10 picks enemy targets 2.3×
  more often than midpack does, even though neutrals have lower
  capture-cost. The math: capturing a neutral grants its (often-low)
  production and denies it to no one. Capturing an enemy grants
  production AND removes it from an opponent. In a margin-agnostic
  rating system, denial > acquisition.
- **Single-source greedy launches.** Multi-launch turn rate 48% (top)
  vs 38% (mid). Top-10 dispatches from multiple sources per turn
  (gang-up/swarm patterns), which is what @yijue1's published kernel
  calls "Swarm" and @yuriygreben's `WorldModel.simulate_planet_timeline`
  optimises for.

## 6. Within-top-10 archetypes

Two visibly distinct playstyles emerge:

**Concentrated artillery** — #1 bowwowforeach, #4 Isaiah @ Tufa Labs,
#6 Shun_PI. Few launches per turn (0.5-0.9), very large fleets (mean
45-69, p95 121-135), low garrison-at-launch. **Quality over quantity.**

**Saturation pressure** — #2 flg, #5 Ebi, #7 Erfan Eshratifar, #10
3Comets. High launches per turn (1.1-2.2), medium fleets (mean 32-41,
p95 86-116), high multi-launch rate (≥40%). **Quantity, with each shot
still carefully aimed.**

A third axis is enemy-focus: #2 flg and #6 Shun_PI lean hardest on
attacking opponents (enemy-target fraction 0.46-0.61); #7 Erfan and
#9 sash lean toward neutrals (0.37-0.45).

The fact that both concentrated-artillery and saturation-pressure
finish in top-10 says **the game has at least two stable optima at
the current top of the ladder**. There is not a single dominant
recipe; the game is sufficiently nuanced that different solutions to
the "how to allocate the per-turn fleet budget" question can converge
on similar μ.

## 7. First-principles cross-check

Why do the empirical features above match what the game *should*
reward?

### TrueSkill margin-agnostic (comp-context.md:142-158)

Only win/loss matters. Implications:
- **No incentive for ship hoarding past survival.** Midpack's 22-ship
  mean garrison and +50% larger total-ships count is value that didn't
  convert into a win. Top-10 spends.
- **Risk profile: close wins as good as routs.** Top-10's higher
  sun-clip rate at #5 Ebi (12%) is still profitable as long as it
  flips the win/loss bit; the lost ships don't penalise μ as long as
  the game outcome is favourable.
- **Denying enemies is mechanically equivalent to capturing
  neutrals** for win/loss purposes, but on top of that, denied
  enemy production no longer feeds their counter-attack. The 2.3×
  enemy-target ratio top-10 ↔ midpack is the in-game read on this.

### Margin-agnostic + 4P FFA + rolling-last-2

- Top-10 plays a roughly 55%/45% 2P/4P mix (74-651 episodes per
  recent submission, sizes={2:38-482, 4:36-241}). In FFA, **kingmaker
  / spoiler** behaviour matters: see @emanuellcs's
  `ffa-mode-aware-strategist` (Crash Exploitation Amplification ×1.5,
  Safe-Neutral Proximity Bias ×1.15).
- Rolling-last-2 means top-10 cannot risk speculative regressions.
  Most top-10 last-submit dates cluster within ~7 days of 2026-05-11;
  the few stable ones (#9 sash 2026-05-09, #4 Isaiah 2026-05-07) have
  paused while they probably re-tune locally.

### Sun risk math

`comp-context.md:47` puts the sun radius at 10 with **zero collision
margin**. Path-clears-sun is binary. A 1-unit-off mis-aim ≈ a fleet
death. Top-10's 6.2% sun-clip rate (and the 4.4% from #1 bowwowforeach)
is **roughly the irreducible floor** given that:
- Orbiting targets move; aim must lead.
- Multi-step intercept requires fixed-point iteration to converge
  (Roman uses 5 iter + safe-intercept fallback, public-kernel
  teardown lines 99-106).
- Edge geometries (target across the sun from source) are sometimes
  unreachable; you must skip those launches rather than push through.

The midpack 9.8% sun-clip indicates the typical agent does *not* have
arrival-aware sun-check (they check `path_clears_sun(src,
target.current)` not `path_clears_sun(src, target.arrival)`). This is
the exact bug v2 had and v3_snipe's `lib/trajectory.predict_fleet_fate`
fixes (audit/2026-05-11-block-e-snipe-mvp.md, capture probe
77.2% → 97.2%).

### Comet ROI math

Comets at steps 50,150,250,350,450, speed 4.0, one per quadrant,
production 1.0. A comet with N ships at radius R from path-end has a
remaining lifetime ≈ R/4. Break-even: `fleet_cost / 1.0 + eta_to_comet
≤ remaining_lifetime`. Most comets fail this test; only fresh
mid-trajectory ones are positive-EV. Top-10's 3.4% comet rate matches
this constraint; midpack's 13.4% is over-fitting on the high
production-per-radius and missing the lifetime constraint.

## 8. Public-agent ceiling (the gap between Roman and the cliff)

@marcodg's 50-agent round-robin (`audit/external/discussions/topic-698614.json`)
ranks the strongest public agents:

| # | Agent                                    | Win% | Author        | Architectural note                                     |
| - | ---------------------------------------- | ---: | ------------- | ------------------------------------------------------ |
| 1 | rule-base × ML shot validator hybrid     | 85.4% | @konbu17      | NumPy MLP filters bad shots from a Roman-style rule-base |
| 2 | two-bot-combine-v3                       | 84.4% | @nina2025     | Ensemble of two heuristics                             |
| 3 | meta-optimized-spoofing-agent            | 83.3% | @emanuellcs   | Spoofing single-ship fleets every 18 turns to poison opponent's defensive sim |
| 4 | orbit-star-wars-lb-max-1224              | 79.2% | @romantamrazov | Reference rule-base (our v3_snipe converges to this archetype) |
| 5 | physics-aware-architect                  | 79.2% | @yuriygreben  | 5-layer architecture: Config / Physics / WorldModel / Strategy / Execution |
| 6 | marcodg-v3.3                             | 79.2% | @marcodg      | Mid-tier published reference                           |
| 7 | ender's-fleet-score-1000-heuristic       | 77.1% | @zacharymaronek | Score-1000 heuristic                                 |
| 8 | pilkwang-structured                      | 76.0% | @pilkwang     | Ancestor of Roman; shared skeleton                     |

The public-agent ceiling is at 85% panel winrate (konbu17 hybrid),
which translates to roughly μ=1200-1300 on the live ladder. The
top-10 cliff at μ=1440 implies **+150-250 μ over the best public
ceiling**. Three of the public agents (#1, #2, #3) sit at 83-85%
panel — these are the candidates for "best architectural ideas that
made it into the open."

### What the best published agents add over Roman

1. **konbu17 — ML shot validator (+19pp over rule-base).** A 24-input
   numpy MLP (no torch) predicts "10 turns after this shot, will I
   own the target?" Inputs: src/target ships, prod, radius, ships
   sent, ship fraction, distance, ETA, fleet speed, in-flight friendly
   & enemy fleet stats, turn, total ships diff, planet count diff.
   Conservative: only rejects shots, never proposes new ones. **This
   is the most credible v4 path for us** given we already have a
   functioning rule-base (v3_snipe).
2. **emanuellcs — spoofing fleets.** Every 18 turns, send a 1-ship
   fleet at the highest-scoring opponent's highest-value planet. Cost
   negligible; effect is to make opponents' simulation-based defensive
   reservation models over-react. Pure adversarial metagaming. Adds
   ~2-4% winrate in local panels (tournament position #3).
3. **emanuellcs FFA-aware (separate kernel).** 4P-mode triggers:
   crash-exploit value ×1.5, safe-neutral proximity ×1.15, terminal
   mode at turn 60 (not 100), attack margin +0.05. Confirms top-10
   tune 4P separately from 2P.
4. **yuriygreben — WorldModel.simulate_planet_timeline.** Event-loop
   simulator that, per planet, tracks every arrival to compute "who
   owns it on turn X with how many ships." Lets the strategy query
   `min_ships_to_own_by(target, turn)`. Roman has the same shape; this
   is the v2 → v3 ledger substrate we're already building.

### What the unpublished top-10 are likely doing on top

Inferred from replay statistics + the absence of equivalents in
public kernels:

- **Better opening (step 1-10).** Top-10 first-launch at step 1-5;
  best public agents wait. Bowwowforeach's step-4 with full-source
  emptying suggests a pre-computed opening recipe based on the
  randomised map geometry, not a runtime-evaluated launch.
- **Higher per-turn launch density.** Top-10 launches/turn 1.2;
  midpack 0.63. Public Roman maxes at ~0.8 launches/turn (estimated
  from his structure; he allocates per-source greedy). To hit 1.2+,
  you need a planner that considers **multi-source same-turn
  arrivals** (yijue1's "Swarm") as first-class mission types, plus
  cheap-to-evaluate Hungarian-style assignment.
- **Empty-the-source discipline.** Top-10 mean garrison-at-launch
  ~11; Roman's `frontier_keep` and similar published patterns leave
  ~30-50 ships reserved. The top-10 reads as "the in-flight ledger
  is reliable enough that we trust an empty home planet to be
  defended by what's coming back". This requires accurate
  defensive prediction (the WorldModel substrate Roman ships) PLUS
  conviction in it.
- **Tighter sun discipline at the very top.** #1 bowwowforeach's
  4.4% sun-clip is below the 6-8% range most top-10 sit at and the
  9-10% midpack baseline. The remaining few percent gap might be
  achievable by a higher-iteration intercept solver, a richer
  path-clears-other-planets check, or selective abstain on
  geometries where the intercept can't converge.

## 9. Implications for our v3 → v4 roadmap

Promoted to `state/hypothesis-board.md` in the same commit as this
doc; numbered to extend the existing H4-H9 series.

- **H10 (high EV, ~3 days):** **Add a "kill enemy" target multiplier
  to v3_snipe's ROI scoring.** Top-10 picks enemy targets 2.3× more
  than midpack at otherwise-comparable rolls. Operationalisation:
  multiply target.value by 1.3-1.5 when `target.owner ≠ ourselves
  AND target.owner ≠ -1`. Decision gate: 32-seed 2P vs v3_snipe
  ≥55% Wilson-lo; 4P FFA parity-or-better. Falsifiable.
- **H11 (medium EV, ~5 days):** **Opening-only first-fleet rule —
  launch from EVERY home/near-home planet at step 1 toward a
  pre-scored neutral cluster.** Top-10 launches at step 4; midpack
  at step 11. The 7-step gap = ~7×production×N_planets worth of
  forfeit value (≈30 ships across opening). Implement as a
  Mission class `opening_landgrab` that fires once if `step <= 5
  AND ours.ships > 8` and then disengages. Decision gate: 32-seed
  2P winrate vs v3_snipe ≥55%; first-launch-step measurable ≤3.
- **H12 (medium EV, ~5 days):** **Source-emptying mission class.**
  Right now v3_snipe leaves ~production×ETA reserved at the source.
  Top-10 mean garrison-at-launch is 11; we should target 12-15. New
  mission `drain` triggered when our planet has ships > 30 AND no
  incoming enemy fleet predicted within ETA+5. Decision gate:
  mean_garrison_at_launch reduces from ~25 to ≤15 without
  fleets_lost_to_enemy_recapture increasing by >2 per game.
- **H13 (high EV, larger build, ~10 days):** **Multi-source
  same-turn arrival as a Mission class (`swarm` / `gang_up`).**
  yijue1 and yuriygreben both have this; our v3_snipe planner does
  per-source greedy. Implementation: in settle_plan, after each
  per-source candidate is scored, run a second pass that considers
  pairs of sources whose ETAs to the same target are within ±2
  turns; bonus the pair if their combined ships > target's
  predicted-at-arrival-defenders. Decision gate: gang_up_rate
  rises from current ~35% to ≥50%; 32-seed 2P winrate ≥55%.
- **H14 (high EV, ML, ~15 days):** **konbu17-style shot validator.**
  After v3_snipe proposes its action set, a small numpy MLP
  (24-dim input, 32-16-8-1 hidden, sigmoid) drops shots predicted
  to fail. Training: from our captured replay corpus
  (`audit/replays/`) label each launched shot by whether the target
  was ours 10 turns later. Conservative: only reject, never propose.
  Decision gate: +5pp panel winrate over v3_snipe baseline, no
  regression vs Roman-1224. Equates to a non-trivial RL workstream
  but no PPO sample-efficiency wall (konbu17 explicitly notes pure
  PPO failed five times against the same opponents).
- **H15 (cheap probe, ~½ day):** **Drop comet chasing entirely.**
  Currently `agents/v3_snipe` chases comets when ROI score wins. Top-10
  has comet-capture rate 3.4%, midpack 13.4%, and the public
  emanuellcs kernel codifies break-even filtering. Hypothesis: a hard
  filter `comet_target → require (eta + cost/prod) < remaining_path`
  drops the rate to ~3% without losing wins. Decision gate: panel
  winrate parity-or-better with v3_snipe.

## 10. Limitations and caveats

- **Sample size**: 5 replays per top-10 player. Confidence intervals
  on each feature are wide (Wilson on a 5-of-5-win cluster spans
  56-100%). The cross-team aggregates (50 top-10 replays vs 10
  midpack) are more reliable, but per-team profiles should be read
  as "where the player likely sits" not "where the player
  definitely sits."
- **Survivorship in the win-selection**: we biased toward wins (3/5
  2P-wins per top-10 player) to characterise the strategies that
  work for them. Their losses might reveal more failure modes; not
  studied here.
- **K=100 prefix**: features computed over the first 100 turns. Late-
  game shifts (steps 200-500) are NOT in this analysis. The
  `mean_total_ships` and `ships_growth_per_turn` already hint that
  midpack accumulates more total ships — what happens late-game is
  worth a follow-on.
- **Replay action format**: action items in the replay can have ≥3
  fields; we used only `[from_pid, angle, ships]`. If the env supports
  a 4th `target_pid` field, we treated it as inferred-by-ray-cast
  (which can mis-attribute targets when two planets are colinear with
  the source). Bias estimated <5% on the target-distribution features.
- **Public-agent ceiling is a proxy.** The 50-agent tournament was
  run locally with one configuration; live-ladder μ depends on
  matchmaking and opponent density which differ from the panel.
  Treat the konbu17 "85% / μ≈1250" mapping as a rough anchor, not a
  guarantee.

## 11. References (verbatim sources)

- **Leaderboard pull**: `audit/2026-05-11-lb-top30-snapshot.csv` (raw)
- **Replay corpus**: `audit/external/replays/r*.json` (50 top-10 +
  10 midpack), pulled via `https://www.kaggle.com/api/v1/competitions/episodes/{episode_id}/replay`
- **Episode crawl frontier** (~1860 unique submissions BFS'd):
  `audit/external/episodes/crawl-frontier.json`,
  `audit/external/episodes/crawl-hop{2,3}.json`
- **Public-kernel teardowns** (extends the prior `audit/2026-05-10-public-kernel-teardown.md`):
  konbu17 ML hybrid, nina2025 two-bot-combine, emanuellcs spoofing,
  emanuellcs FFA-aware, yuriygreben physics-architect, yijue1
  1103-peaker, djenkivanov OW-Proto, ykhnkf distance-prioritized,
  woosungyoon baseline-tutorial, rahulchauhan016 target-2000,
  marcodg-v3.3, thisisn0mad RL-pipeline-public. All in
  `audit/external/kernels-pulled/{user}_{slug}.json`.
- **Discussion threads** (sources for "Aggressive Expansion Wins",
  RL lessons, daily replay datasets, fleet-tunneling bug, 50-agent
  tournament): `audit/external/discussions/topic-{697413, 697725,
  696214, 696219, 697397, 698614, 692800, 693755, 696043, 698395}.json`
- **Comp spec**: `comp-context.md:42-141` (board, planets, fleets,
  comets, combat, scoring, action IO)
- **Our internal state**: `state/current.md` (rank), `state/hypothesis-board.md`
  (where H10-H15 are appended), `state/mechanism-ledger.md`,
  `audit/2026-05-11-block-e-snipe-mvp.md` (current v3_snipe baseline),
  `audit/2026-05-11-lookahead-phase2-forward-sim.md` (Sim<K> scorer)

## 12. PI takeaway (one paragraph)

The top-10 of the Orbit Wars leaderboard plays an active, aggressive,
enemy-focused style with 1.2 launches per turn, fleet sizes around 38,
sources emptied to 11 ships, sun-clip rate 6%, and targets that are
2.3× more often enemy planets than midpack chooses. They open
immediately (step 1-5, not step 11), they ignore comets (3% capture
rate, not 13%), and they multi-launch from many sources at once (48% of
turns). The strongest published agent (konbu17 at 85% panel winrate)
adds a small NumPy MLP to a Roman-style rule-base — a "shot validator"
that rejects bad launches. The top-10 gap above the best public agent
(+150-250 μ) is most plausibly closed by **better target valuation
(enemy-bias multiplier), faster opening, source-emptying discipline,
multi-source coordinated arrivals, and (longer-term) a learned shot
validator**. These map to hypotheses H10-H15 on the board.
