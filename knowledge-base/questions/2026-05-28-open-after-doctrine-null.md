# 2026-05-28 — Open questions after the doctrine null

The doctrine null doesn't answer these — surfaced for PI consideration
before the next strategy choice.

1. **Is share-of-integral useful as a pre-submit gate even when no
   chooser ships?** Rule 48 substrate is built and works
   (`scripts/measure_hold_times.py --replay-dir`). The n=92 study
   showed winners have 0.488 more share than losers. We never wired
   the metric into the existing baseline submit flow. Would catching
   share-of-integral regressions BEFORE a Kaggle submit have prevented
   any of: sub 53083109 (μ=920, post-fix regression), sub 53099001
   (μ=600, ship-turn-kappa regression)? Worth a retroactive pass on
   the saved local replays from those submits.

2. **Top-10 self-play gap (μ ≥ 1500).** `audit/2026-05-27-between-band-
   stratification.md` showed the share signal monotonic through
   μ=1400 but we have zero seat-games against top-10 (μ ≥ 1500).
   The manual sub-ID discovery path (Kaggle web UI → copy sub_ids
   → feed to our pull script) is ~30 min PI time. Still open. If
   top-10 share IS substantially higher than μ=1400's 0.46, the
   ceiling expectation for any future agent shifts up. If it's not,
   we're near the ladder's structural ceiling and submit-strategy
   matters more than agent-strategy.

3. **What's the right NEXT axis?** With the chooser line closed,
   plausible directions in order of expected leverage:
   - (a) baseline knob calibration via the Rule 48 substrate
     (BASELINE_GAMMA, BASELINE_VALUE_HEAD, BASELINE_REINFORCE_*,
     BASELINE_STAGNANT_DRAIN — many are default-OFF).
   - (b) Defensive modeling (the 4P 0% win-rate in our peak's
     n=2 sample is a gap; cushion-as-pure-silence failed but
     "defensive carve-out within cushion" wasn't tested).
   - (c) ML value head training against share-of-integral as the
     target.
   No data yet on which has highest EV.

4. **Does the parity-gate bundler bug affect any other modular
   agent?** `scripts/bundle_agent.py` CLI parity failed on
   `agents/reach_frontier/` due to kaggle_environments sys.path
   mutation putting `kaggle_environments/envs/lux_ai_s3/` ahead of
   our local `agents/`. Likely affects any new agent whose import
   chain is sensitive to module resolution order. Untested on the
   existing modular agents (baseline). Worth a one-line check on
   the next iteration.
