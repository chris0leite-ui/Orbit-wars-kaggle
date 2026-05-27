# Evaluation Metrics — what to measure when scoring production-integral agents

Authored 2026-05-27 from `audit/2026-05-27-hold-time-empirical.md`.

This is a short operating doc: which numbers to look at (and not look at)
when evaluating any agent that targets the production-time integral
`S_i ≈ S_i(0) + Σ p̃·τ_p^i` — including the queued reach-frontier
chooser, but also any future variant that descends from the doctrine
framework.

The 92-game empirical study taught us five things about how to measure
this class of agent. They are listed in priority order.

## 1. Production-share is the primary, hold_fraction is secondary

The original empirical-verification plan used per-capture median
hold_fraction as the primary metric and the production-weighted share
of integral `Σp̃·τ_me / (Σp̃·τ_me + Σp̃·τ_opp)` as the confirmatory
secondary. **This was the wrong way round.**

Production-share separates wins from losses far more cleanly than
hold_fraction does, and — critically — it does NOT collapse in 4P the
way the per-capture median does.

| Cohort | Median share_focal | Median hold_fraction |
|---|---:|---:|
| 2P wins   | 0.711 | 1.000 |
| 2P losses | 0.315 | 0.235 |
| 4P wins   | 0.660 | 0.332 |
| 4P losses | 0.113 | 0.120 |

The 4P winner takes 66% of total production-time integral despite a
median per-capture hold_fraction of 0.33 (below the doctrine §9
"strong" threshold). The hold_fraction underreports because 4P winners
attempt 3× more captures than 4P losers, and many of the extra
attempts are short contested-band trades that dilute the per-capture
median. Production-share absorbs the dilution because the short trades
contribute little p̃·τ; the long holds dominate.

**Operational consequence:** any future A/B panel that scores a
production-integral agent reports the production-share-of-integral
*first* and hold_fraction *second*. Reverse this and you will
mis-evaluate 4P-correct agents.

## 2. Hold_fraction is bimodal — report quartiles and tail fractions

The per-capture hold_fraction distribution is two-population, not a
hump. A typical 2P winning game has ~53% of captures stick to game-end
(hf ≥ 0.9), ~21% fail within the first tenth of remaining game
(hf ≤ 0.1), and the middle is sparse. Losers show the inverse: ~3%
sticky, ~28% fail-fast.

Reporting only mean or median erases the bimodality. Future eval
output should include:

* `frac_sticky` = fraction of focal segments with hf ≥ 0.9
* `frac_trade`  = fraction of focal segments with hf ≤ 0.1
* full quartiles Q1, Q2, Q3 (not just the median)

The chooser's success signal is "many sticky + few trades" — a shift
in *which side* of the bimodality the agent operates on, not a shift
in mean.

## 3. Never report aggregate without a game-size cut

The aggregate STRONG verdict on 92 games masked a 2P/4P asymmetry that
matters operationally. The 2P-only sub-sample cleanly clears the
strong gate; the 4P-only sub-sample looks weak-positive *at best* by
the per-capture metric (the share metric rescues it). Any agent
evaluation that pools 2P and 4P games into one median is reporting a
composition-confounded number, in violation of Rule 41.

**Default cut:** every evaluation table emits separate rows for 2P and
4P (and 3P if any of those slipped into the sample), even if both
rows show the same direction. This forces the reader to see the
composition.

## 4. Per-cell n_games thresholds

Per-capture medians are usefully tight at n ≥ 500 segments per cell
(achieved in our 92-game sample for 2P-wins, 2P-losses, 4P-losses).
Per-game medians on share need more *games*, and the 4P-wins cell at
n=8 games is thin enough that the 0.660 share estimate has wide CIs.

**Cell minima for an agent-evaluation panel:**

| Metric | Minimum cell n | Notes |
|---|---:|---|
| Per-capture hold_fraction medians | n ≥ 200 segments | Q1/Q3 noise-floor; medians stabilise below this |
| Per-game share_focal medians      | n ≥ 20 games    | For a tight per-cell estimate |
| 4P-specific cells                 | n ≥ 30 games    | 4P winners are rarer; need more total games |

For a chooser-evaluation run: target 60+ games (20+ per cell for 2P
wins, 2P losses, 4P combined) at minimum, 100+ if the chooser is
meaningfully different in 4P. This guides the `--max-pulls-per-sub`
sizing for any future replay pull.

## 5. 4P winners capture later, not earlier — flag this as a behaviour gate

Median `t_capture` for 4P winners is 137; for 4P losers it is 72.
4P losers attack early and the planets do not stick; 4P winners wait
longer and the planets do stick. The doctrine's "(T − t_capture)
compounding" principle still holds in 2P (medians 102 wins / 86 losses
— a wash) but in 4P the constraint that flips first is "can you defend
what you just took with your current production cushion."

**Operational consequence:** a 4P-aware chooser should be expected to
launch its first capture later than its 2P sibling, and the panel
should report the median first-launch step. A chooser that launches at
step 4 in both formats is not yet 4P-correct.

This is a *prediction* the eval harness should check, not a free
parameter to tune.

## Within-band vs between-band — partial answer, directional positive

The 92-game study was within-band: winners-vs-losers among our own
ladder play in the μ=900-1150 band. The follow-up
`audit/2026-05-27-between-band-stratification.md` re-extracted share-
of-integral for every seat in those 92 replays, joined to the public
leaderboard CSV, and bucketed by opponent μ:

| Bucket | n seat-games | Median share |
|---|---:|---:|
| μ 1200-1400 | 28 | 0.463 |
| μ 1000-1200 | 58 | 0.260 |
| us (μ=1119) | 92 | 0.276 |

The trend is monotonic and substantial through μ=1400. **The chooser
build is justified by this signal: a chooser that targets share-of-
integral should plausibly lift us into the 1200-1400 band.**

The break-through-to-top-10 question (μ ≥ 1500) is still open — we
have zero seat-games against any team at that μ, because we so
rarely match them. The Kaggle CLI does not expose other teams'
submission IDs, so a clean top-10 self-play pull requires manual sub-
ID discovery via the Kaggle web UI. ~30 min of PI time, then ~10
min of wallclock to pull and measure. That answer reshapes the
**chooser ceiling expectation** but does not invalidate the chooser
build itself.

## References

* `audit/2026-05-27-hold-time-empirical.md` — the raw study this doc
  generalises.
* `knowledge-base/concepts/reach-frontier-doctrine.md` §9 — the
  pre-registered gates this study used (now superseded by §1 above
  for any future production-integral agent eval).
* `knowledge-base/concepts/top-performer-strategies.md` §4 — the
  behavioural fingerprint of top-10 (the implicit comparator for the
  between-band question in §6).
* CLAUDE.md Rule 41 — confound-sweep before correlational conclusion;
  the 2P/4P split discipline (§3 above) is a direct application.
* CLAUDE.md Rule 48 — binds these metrics as the default for any
  production-integral-class agent evaluation.
