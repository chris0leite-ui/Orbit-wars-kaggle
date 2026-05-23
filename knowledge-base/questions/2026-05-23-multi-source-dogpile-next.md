# Question: should multi-source dogpile be lagrange_simple v2?

**Filed**: 2026-05-23 (claude/session-EqJuT)

`agents/lagrange_simple` v1 ships with a known structural ceiling:
**single-source-per-target** capture. The per-target argmax in
`dual.py:_inner_solve` picks at most ONE candidate per target. When
a target's predicted garrison-at-arrival exceeds any single source's
ship budget, the agent has no path to capture.

Concretely: vs `agents/baseline` at n=8 = 0/8 LOSS, root-caused to
this ceiling (mid-game the cohort-attack pattern baseline produces
via its dedup + chooser pipeline outpaces our solo-min-per-target).
Vs `random` at n=16 = 16/16 ELIM after the two bug fixes — `random`
never produces dense-enough defenses to trigger the ceiling.

**Open question**: is the ~50 LOC multi-source dogpile addition
(per `(target, arrival_step)` bucket, greedy-add candidates by
reduced-cost-per-ship until ships > defense; commit if subset
reduced cost positive) the right next iteration?

**For** dogpile:
- Closes the single biggest known structural gap.
- Stays inside the Lagrangian framework.
- Adds ≤50 LOC; remains "simple and maintainable" per PI directive.

**Against** dogpile:
- PI explicitly said "simplest" this session and rejected the
  multi-source variant. The current v1 may BE the deliverable PI
  wanted as a clean reference, with vs-baseline gap acknowledged
  but not actionable.
- Multi-source requires per-(src, ships, launch_tick) ship-count
  variants in enumeration (otherwise solo-min combinations are
  pure waste). That's the part that pushes complexity up.

**Decision deferred** to next session with PI input. The agent
currently passes its stated gate; whether the next move is
dogpile, port to live ladder, or a completely different track
(physics modeling sweep / Konbu17 ML / etc.) needs PI direction.

**PI direction (2026-05-23 session-end)**: climb rungs 2-3:

1. **Rung 2** — run `scripts/random_elim_gate.py
   agents/lagrange_simple/main.py` BUT with `starter` substituted for
   `random` in `_play_one`. Expected: likely 16/16 already (starter
   is only marginally stronger than random). If it fails, fix the
   surfaced bug class FIRST before proceeding.
2. **Rung 3** — add multi-source dogpile (~50 LOC in `dual.py`):
   per `(target, arrival_step)` bucket, greedy-add candidates by
   reduced cost per ship until cumulative ships > defense_at_arrival;
   commit subset if subset reduced cost > 0. Then run the gate vs
   `agents/baseline`. Target: 100% ELIM at n=16. Iterate until pass.

Goal for the next session: pass rung 3 (100% ELIM vs baseline). Both
rungs preserve the Lagrangian structure. The dogpile addition requires
per-source PARTIAL candidates (ships < solo-min), which is the only
non-trivial enumeration change.

Rungs 4-5 (analytical_phase_c, composite_a2_hybrid) are explicitly
NOT this session's targets; queued for the session after.

**Related**:
- `knowledge-base/thoughts/2026-05-23-simplest-lagrangian-shadow-
  prices.md` (the full design walkthrough)
- `audit/2026-05-23-postmortem-session-EqJuT.md` (the session
  postmortem, with carry-forward summary)
