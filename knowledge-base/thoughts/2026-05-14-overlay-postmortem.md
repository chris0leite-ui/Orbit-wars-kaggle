# 2026-05-14 — opening-overlay attempt: what we keep, what we don't

Session: `claude/game-strategy-eda-roatN` (game-strategy EDA → cluster-
conditional opening overlay → falsified).

## What's load-bearing (port to main)

1. **The opening-launch-rate gap is the real lever**, not late-game
   throttle. v7_pv launches at 0.44 / turn in its own wins and
   0.29 / turn in its losses; the top-10 corpus medians 0.70 / turn.
   By turn 100 of our matches, our wins lead our losses by +30
   percentage points of ship share — the W/L split is decided
   before the late-game phase Mine 4 was framed against. Median
   episode length in our ladder is 180 turns, so the "endgame
   throttle" framing was an artifact of the top-10 corpus, where
   games run far longer because both sides survive.
   Doc: `audit/2026-05-14-loss-mode-mine.md`.

2. **The hardened-panel calibration is a permanent gate.** 32-seed
   `v7_0_drop_one × v3.5.1 × roi × baseline` matrix: v7_0 mean
   78.6%, worst-Wilson-lo 53.4% (vs v3.5.1). Any future v7-family
   candidate measures against this row. The panel preset
   (`scripts/strategy_panel.py --panel hardened`) and the resolver
   change (`scripts/_agent_paths.py` finds nested
   `agents/v7_ablations/<name>/main.py`) are infrastructure I'll
   reach for every variant from here. JSON:
   `audit/tournaments/20260514T194550Z.json`.

## What's falsified (stays on the branch, doesn't ship)

3. **Cluster-conditional opening overlay underperforms pure v7.**
   v3 sweep (post-bugfix): 17W/15L = 53% vs v7_0; Wilson-lo ~36%;
   overlay-active games 12W/14L = 46% vs pure-v7 fallback 4W/1L =
   80%. The mechanism — classify board into one of 4 archetypes,
   force template-specific launch cadence in turns 0-30 — does
   not survive the gate. The classifier picks the wrong template
   on enough boards to nullify the cadence benefit on the rest.

4. **The "encouraging" v2 result was a bug.** v2 sweep ran at 67%
   vs v7_0 with a broken `_board_fingerprint` that reported
   `orbital_frac=1.00` on every self-play board (vs training
   corpus 0.27-0.44). Every board got force-classified into
   cluster 3 — the high-cadence "blitz" template. So v2 was
   actually an unconditional aggressive-cadence agent, not a
   classifier-driven one. After fixing the proxy, cluster
   assignments spread sensibly across C0-C3 and the winrate
   collapsed 14 percentage points.

## Three lessons (friction-promoted; see audit/friction.md 2026-05-14)

- **Inlining a library helper for bundling-friendliness needs the
  source line verbatim, not a paraphrase.** I rewrote `is_orbiting`
  from scratch and got the threshold wrong. The bug was silent
  because both proxies return numbers in [0, 1]; only the
  *distribution* differed (training median 0.4, runtime 1.0).
- **A positive sweep result is not signal until you verify the
  mechanism is functioning.** A 30-second cluster-distribution
  print would have caught the orbital_frac bug before the v2
  result was reported. The encouragement was the bug.
- **Soft clusters (silhouette < 0.20) need a confidence threshold
  on day 1.** Mine 1 had already flagged the silhouette ≈0.17 risk
  in its rollup; I shipped the hard nearest-centroid classifier
  anyway. Threshold landed in v3 — should have been in v1.

## What the next session shouldn't repeat

The real Tier-1 lever (opening launch rate, finding 1) is still
unopened. The cluster-conditional overlay path is dead, but
that's one specific *implementation*. Two paths remain unexplored:

- **Push v7's value head**, not an overlay. Raise the per-launch
  reward inside `lib/v7_search.score_candidate` for steps 0-30
  instead of bolting an overlay on the outside. Same mechanism,
  no separate proposer, no second classifier to misfire.
- **Question whether opening launch rate is causal at all.** Top-10
  launches more in the opening because they *can* (more ships
  from earlier captures), not because launching more *causes*
  wins. The 0.44 → 0.70 correlation in v7_pv wins vs top-10 wins
  may be downstream of "we win more captures," not a behavioural
  prescription. Worth a one-pass mine on the top-10 corpus —
  does early launch rate predict later ship-share, controlling
  for win/loss outcome?
