# 2026-05-17 — substrate reframe (PI voice)

Branch `claude/space-fleet-physics-engine-lrLE6`, mid-session
after the JAX value head was diagnosed broken and a pivot to
fast_sim landed.

The session started as a kill-or-keep test for v8_analytic. Plan-
mode WRAPUP encoded a strict gate: Wilson 95% lower bound on
win-rate < 40% versus nearest → kill. After the pivot to fast_sim
at K=15, the bench came back at 4/8 wins versus nearest. Wilson
LB at n=8, 4 wins is 21.5% — below 40% — strict reading says kill.

PI overrode the strict reading with two messages:

> "we do not need to win, we just need to know if we can use the
> architecture as a strong baseline to leverage our ambitious ideas"

and earlier in the session:

> "it even loses against nearest. we need signs that this can
> lead to a competitive strategy"

The reframe: the kill-or-keep verdict is **not** outcome-based
(absolute win rate against a baseline). It's **substrate-quality-
based** — is this architecture buildable-on? Specifically, three
properties matter:

1. Does a single tunable knob produce monotone improvement in a
   measurable micro-metric?
2. Did a predicted-from-microtrace outcome match an actual bench
   outcome on at least one seed?
3. Is timing healthy enough that there's headroom for ambitious
   extensions?

If all three pass, the architecture is "alive" and worth carrying
forward even if absolute quality is mid. If any fail, kill.

This is consistent with decision-quality-vs-outcome-quality
(see `knowledge-base/concepts/decision-quality-vs-outcome-quality.md`
if exists). Outcome-based kills at small n throw away substrates
whose uncertainty intervals are just wide; substrate-quality kills
target genuinely structural dead-ends.

The micro-trace evidence at K-sweep (1 → 5 → 8 → 9 candidates
beating no-op as K rises from 8 to 40) and the seed-1 win-flip
predicted-from-microtrace both cleared the substrate-quality
bar. The 4/8 versus nearest is informational about absolute
quality but not the kill signal.

**Promotion candidate added to today's postmortem:** encode this
dual-gate (Wilson LB AND substrate-viability check) in the kaggle-
comp improvements file. Pending PI ratify next session.

Related: the same reframe explains why we shouldn't have spent
5 sessions tuning before running the value-head micro-trace.
"Does the agent ever pick action X" is a substrate-quality
question and would have been asked earlier under this lens.

PI quote that sums it up:

> "we will need this architecture to build upon"

— so the verdict is KEEP, and the next-session priority order
(wallclock-guarded longer K, beam-over-fastsim, wait-then-fire,
stronger opp model, learned head) is captured in
`knowledge-base/concepts/v8-analytic-architecture-state.md`.
