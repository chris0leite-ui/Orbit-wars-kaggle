# Between-band stratification check — partial answer

Date: 2026-05-27. Follow-up to `audit/2026-05-27-hold-time-empirical.md`.

## What we tried to do

Answer the open question from `knowledge-base/concepts/evaluation-metrics.md`
§"Within-band vs between-band": does production-share-of-integral
separate us from top-10, or does it only separate winners-from-losers
*within* our μ band?

## What we actually did (and why)

The original plan was "pull ~20 top-10 replays via `kaggle competitions
replay`." That requires a top-10 submission ID. The Kaggle CLI exposes
the leaderboard with team IDs and team names but **not submission IDs
for other teams**. `kaggle competitions episodes <sub_id>` requires the
sub_id as input. So a clean top-10 pull is blocked by the CLI surface.

Alternative used: the 92 replays already on disk each contain full
state for every seat. We re-ran the share-of-integral extraction
treating *every* seat as the focal seat, joined to the downloaded
leaderboard CSV to get each opponent's current μ, then bucketed by μ.

This is **not** a true top-10 self-play sample. It is "opponents we
matched, stratified by their current rating." That answers a related
question: does the share signal separate the opponents we faced into
strength tiers consistent with their leaderboard μ? The trend the
bucketing produces is suggestive, but with one big caveat (see §"What
this does NOT tell us" below).

## Result

Per-opponent median share-of-integral, in our games, bucketed by
opponent's current leaderboard μ:

| μ bucket | n seat-games | Median share | Mean share |
|---|---:|---:|---:|
| μ ≥ 1400         |   2 | 0.034 | 0.034 |
| μ 1200-1400      |  28 | **0.463** | 0.389 |
| μ 1000-1200      |  58 | **0.260** | 0.341 |
| μ < 1000         |   5 | 0.195 | 0.315 |
| us (μ = 1119)    |  92 | 0.276 | 0.357 |

**Headline:** the trend in the meaningful cells (μ 1000-1200 and
μ 1200-1400, n=86 of 93 seat-games) is monotonic and substantial:
opponents in the 1200-1400 band take 0.46 share in our games, opponents
in the 1000-1200 band take 0.26 share. Our own across-the-board median
(0.276) sits right at the 1000-1200-band line, exactly where the
leaderboard places us.

The μ ≥ 1400 cell is broken: only one team (skalermo, μ=1425) appears,
and they happened to play two 4P matchups against us where they were
crushed. Two games is no signal at all; this row should be ignored.

## What this tells us

The within-band signal extends through the μ=1000-1400 range. Higher-
rated opponents take meaningfully more share-of-integral in mixed
matchups, by roughly +0.20 median per +200 μ. Extrapolating linearly
(which is not justified, but a working guess) puts the μ=1500 band at
share ≈ 0.66 in mixed matchups — close to what 2P-winners already
achieve in pure 2P self-play.

For the chooser build this means:
- Production-share is on the right axis through the band our chooser
  realistically targets first (μ=1100 → 1300-ish).
- A chooser that gets us to median share 0.46 across our mix would
  put us roughly in the 1200-1400 band — a 80-200 μ lift, fully
  worth the build cost.
- Whether the same chooser breaks through to μ=1500+ (true top-10) is
  STILL unanswered. The data we can extract from our matches goes dark
  above 1400 because we so rarely match teams there.

## What this does NOT tell us

Three caveats worth being honest about:

1. **Matchmaking bias.** We mostly match teams within ±100 μ of our
   own. Opponents in our games at μ > 1300 are either (a) on a
   downward swing in rating, (b) in a sparse-opponent regime where
   TrueSkill matched them down, or (c) playing a 4P FFA where seat
   composition is more random. These are not representative samples
   of those teams' typical play. A team's share in a *mismatched*
   game is not their share in a *typical* game.
2. **True top-10 missing.** μ ≥ 1500 is the top-10 band; we have zero
   seat-games against any team at that μ. The extrapolation above is
   a guess, not data.
3. **Sample sizes are thin per opponent.** No opponent has more than
   6 games with us. Per-opponent medians are noisy; only the bucketed
   aggregates are interpretable.

## What would close the gap

To get a real "top-10 typical-play share" measurement we need true
top-10 replays, which means top-10 submission IDs. Three plausible
paths, ranked by effort:

1. **Manual sub-ID discovery (~10 min, low effort).** Browse top-10
   teams' Kaggle profile pages, copy submission IDs from their public
   submissions tab into a CSV, feed to our pull script. Cleanest
   path; needs a human action.
2. **`kaggle competitions submissions` for our team-of-interest** —
   doesn't work; that command is for *your own* submissions only.
3. **Pull all our episodes' opponents recursively** — for each of our
   replays we know which teams are in it and could in principle find
   their sub_ids by cross-referencing on Kaggle's episodes page (web
   UI shows opponent sub_id per episode). Not automatable with the
   CLI as it stands.

Path (1) is what to do next if we want a clean top-10 sample. It's
~30-min of PI time to copy-paste a handful of sub_ids, and we already
have the measurement script ready.

## Status

Within-band-vs-between-band question: **partially answered, directional
positive.** The chooser build is justified by the within-band signal
through μ=1400; the break-through-to-top-10 question remains open and
should be answered by the manual sub-ID-discovery path before any
chooser is submitted (so we know whether to expect μ ≈ 1300 or μ ≈
1500 as the chooser ceiling).
