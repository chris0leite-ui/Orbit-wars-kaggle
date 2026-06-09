# 2026-06-01 — Loss-mode diagnosis: it is NOT ship-hoarding

Replay study of the champion's own live games (sub **53182323**, μ=1183.7):
121 episodes, 50 of them 2P, 70 4P. Goal: locate the dominant loss mode
before committing a build slot. **PI-steered; the corrections below are
load-bearing.**

## Headline

The prior "ship-hoarding / under-expansion" framing
(`audit/2026-06-01-live-replay-diagnosis.md`) does **not** survive scrutiny:

- **We launch plenty.** In losses we carry a *higher* in-flight ship
  fraction than in wins (2P midgame 38% vs 26%; late 36% vs 8%). The loser
  is *more* active, not hoarding. The large on-planet pile in wins is a
  *consequence* of winning (more planets → more ships), not a cause.
- **The opening is roughly even.** Planet-gap at steps 0-50 ≈ 0 in both
  wins and losses; net captures +8.5 (win) vs +7.3 (loss). We are not
  visibly behind early on planet *count*.
- **The split is midgame capture *rate*.** Captures/game by phase (2P):

  | phase | WINS cap/lost (net) | LOSSES cap/lost (net) |
  |---|---|---|
  | open 0-50 | 8.6 / 0.2 (+8.5) | 8.2 / 0.9 (+7.3) |
  | mid 50-150 | 25.1 / 13.8 (**+11.3**) | 12.0 / 17.7 (**−5.6**) |
  | late 150+ | 17.1 / 7.7 (+9.4) | 1.3 / 3.3 (−2.0) |

  Captures *halve* in the midgame of losses (25→12); planets *lost* are
  nearly identical (defense is not the differentiator). 4P shows the same
  shape, stronger (total captures wins 61.5 vs losses 21.3).

So the visible loss signature is **"launch a lot, capture little" — a
conversion gap, not a volume gap.** This also explains the flat-expand-
credit regression (−124μ): it added launch *volume* to a problem that was
never about volume.

## PI corrections (these bound every conclusion above)

1. **Selection bias is severe.** Win-vs-loss is confounded by opponent
   strength: a stronger opponent both captures more *and* reinforces
   faster, suppressing our conversion. So the midgame conversion gap is
   **not established as a fixable mechanism** — it may be largely "we met a
   better agent." Treat the win/loss table as descriptive, not causal.
2. **Fleets do NOT die in flight.** Fleet-vs-fleet collision does not exist
   in this game; a fleet only dies to sun / out-of-bounds (already filtered
   by the trajectory pre-pass). **The H44 "65% destroyed in-flight"
   framing is dropped as a lever.** "Arrive but don't capture" can only
   mean: arrived under-strength, arrived after the target was reinforced,
   or arrived at a planet already taken → the lever is **sizing + timing +
   winning the race**, never survival.
3. **Opening tempo is real** ("we sometimes open too slowly"). Planet
   *count* being even early hides tempo/position differences; a slow open
   lets the opponent take the contested middle and box us in, which surfaces
   as the midgame conversion gap. Opening and midgame are one chain.
4. **Fleet inefficiency / "not moving ships to the front" is a known,
   long-standing issue**, not a new finding.

## Selection-bias-free check: front positioning (paired)

To test "we fail to move ships to the front" *without* the win/loss
confound, compare US vs the OPPONENT **within the same 2P game** (opponent
strength cancels), midgame steps 50-150, n=50:

| | US | OPP |
|---|---|---|
| ship-weighted mean distance to nearest enemy | 39.2 | 39.2 |
| fraction of ships in the rear (>35 from any enemy) | 56% | 55% |
| paired: we are "more rear" than the opponent | 27/50 (coin-flip) | — |

**Null** for the simple hypothesis: on the field average we are *not* more
rear than our opponents. This does not refute the PI's intuition — it
narrows it to forms a static field-average washes out: **opponent-specific**
(more rear only vs the strong agents we lose to) or **tempo/rate** (we move
ships forward, just too slowly). A static snapshot cannot see either.

## Where this leaves the build candidates

- **Contest-urgency / win the race (untested half).** Conversion = sizing +
  timing + race. Sizing alone already failed (size-balance regressed); the
  **timing/race** half — prioritise captures we *narrowly win* the race for,
  defer bankable ones — is the untested, most-direct lever. Genuinely novel
  vs the closed-tracks list.
- **Opening tempo** — still in play (PI: "open too slowly"); feeds the
  midgame. Best pursued via the horizon work below, not a flat credit.
- **Adaptive horizon K — state-driven (see the K investigation doc).** K is
  a *predictability horizon*, so it should equal how far ahead the board is
  actually predictable (few fleets in motion / uncontested target
  neighbourhood → large; churning → small) plus compute headroom — **not a
  fixed step schedule.** Possibly per-target. This naturally raises the
  horizon in midgame lulls, which a step-schedule cannot.

## Dropped / de-prioritised

- **"Ship-hoarding" as the loss mode** — refuted (we out-launch in losses).
- **Fleet-survival / H44 in-flight-death lever** — dropped (no air
  collisions).
- **Static "ships too rear" positioning fix** — null on the paired metric.

## Method notes (reproducibility)

All from `audit/live-episodes/53182323/*.json`. Our seat detected via the
constant `info.TeamNames` entry (`ChrisLeiteScha`); win = our final reward
is the unique max. Phase split open<50 / mid<150 / late. Capture/loss =
planet-owner transitions in `steps[i][0].observation.planets`
(`[id,owner,x,y,radius,ships,prod]`). Positioning = ship-weighted distance
to nearest enemy planet, paired US-vs-OPP per game.
