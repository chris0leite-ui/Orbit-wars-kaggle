# Flag — trajectory chooser is now default in production but 4P untested locally

`agents/baseline/main.py:f192cf4` flipped `BASELINE_CHOOSER` default
to `trajectory`. Live submission `52754310` ships this default.

All local A/B was 2-player (vs v15). 36% of ladder games are 4-player.

What we know:
- Value head is unchanged (favor_hybrid: composite in 2P, A2-favor
  in 4P). Composite_a2 ran with same head in 4P and shipped fine.
- The chooser change is orthogonal in principle to the value head's
  4P branch. But trajectory's predict_fleet_fate filter behaves
  differently in 4P obs (more planets, more comets, more potential
  path-blockers).

What we don't know:
- Whether the `predict_opp_responses` heuristic (1-turn lookahead,
  4 nearest non-opp targets, 80% ship-fraction) gives meaningfully
  worse 4P projections than 2P.
- Whether the wallclock budget enough room for 4P's larger candidate
  count.

Risk window: live μ settles over ~6h / 50 games per
`early-trueskill-mu-unreliable` (Rule 12 caveat). If 4P regresses,
we won't see it cleanly until μ stabilises.

Mitigation if μ tanks: revert default to composite via
`os.environ.setdefault("BASELINE_CHOOSER", "composite")` (any non-
"trajectory" value falls through), re-bundle, re-submit. Rollback
diff is ~3 lines.

Re-bundle path: composite_a2_hybrid baseline.py is at submission
52744856 (μ=1158.6 settling) — the rolling pair partner. If
52754310 settles lower, we have the floor.
