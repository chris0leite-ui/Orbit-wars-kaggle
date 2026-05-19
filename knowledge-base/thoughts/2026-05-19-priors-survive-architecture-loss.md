# Priors mechanism survives even when host architecture loses

2026-05-19, branch `claude/reverse-engineer-seat-geometry-BPJKs`.

The simple-baseline + priors + ROI A/B vs the latest submission
(`82df5b8`, μ=1121.2) failed decisively (5/32, Wilson [0.069,
0.318]). The denom_floor sweep over {0.5, 1.0, 2.0, 5.0} also
failed at every floor (best 5/32). All 32-seed-panel runs.

**The signal worth keeping:** the two archetypes where the simple
agent does win, it wins 2/2 cleanly — `low_prod__mixed_static__
big_static` and `low_prod__mostly_static__big_rotating`. Both
are low-production, static-dominant boards. These are the exact
classes the priority prior boosts (`low_prod_static_inner` α=+0.017,
`low_prod_static_outer` α=+0.003, `low_prod_rotating_inner` α=+0.10).
The pattern is consistent across all four sweep variants — when
priors land on a board whose dominant geometry class is positive-α,
the simple agent matches or beats the trajectory chooser.

**What this might mean:**
- The priors mechanism (per-class multiplier on cheap_marginal_value)
  carries real strategic information even when the surrounding
  agent is mechanically weaker. The win is target-selection, not
  search depth.
- On high-production or rotating-dominant boards, the trajectory
  chooser's deep search + composite_capture_value head dominate.
  The priors can't overcome that gap because there's no static
  alpha that says "be smarter at search" — it only says "prefer
  these classes."
- The right experiment next is to port the priors multiplier ONTO
  the trajectory chooser's value path (`agents/baseline/value.py`
  on `origin/main`'s state). Small surgical change; might lift
  the trajectory agent's low-prod-static performance without
  touching anything it already does well. The math is settled
  (multiplicative scalar on the value head).

**What this rules out:**
- The simple-baseline architecture is not a viable substrate to
  build on. Even with priors + ROI both active at design defaults,
  it loses 13 of 16 archetypes. The 3000-line divergence from
  origin/main is doing real work that hand-coded heuristics can't
  replace.
- ROI denominator tuning is a non-axis — the floor sweep clustered
  6.2-15.6% with no monotone trend. Whatever's wrong, it isn't
  "cost denominator too sharp / too soft."

**Open thread for next session:** does the trajectory chooser
already have an implicit class-weighting via composite_capture_value?
If yes, the priors might be redundant rather than additive on the
trajectory base. The replay-diagnose-the-two-wins option (offered
to PI, not taken this session) would surface this empirically.
