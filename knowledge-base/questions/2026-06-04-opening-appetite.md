# Open questions — 2026-06-04 (opening launch appetite)

- **Does our opening waiting get punished by an aggressive early-expander, or only
  matter in self-play?** We wait 27/31 opening turns by value-function choice (not
  horizon — that's settled, see audit/2026-06-04-opening-wait-diagnostic.md). In
  self-play that's symmetric/even. The decisive test: vs a deliberately aggressive
  expander, measure opening launch-rate gap + territory/production gap by step 30.
  This is THE next experiment.

- **Can a higher early-launch appetite win vs aggressive expanders while staying
  neutral in self-play?** "Launch more early" regressed before — but the regression
  cohort was all self-play (Rule 41 confound). Cut the A/B by opponent class:
  self-play (expect ~neutral/slightly down) vs aggressive-expander (expect up). If
  the split holds, the fix is a *conditional* appetite (strategic mode), not a global
  threshold.

- **Where exactly does the value function decline the opening launches?** Trace one
  WAIT turn (e.g. step 5: 8 candidates, 0 launched) through the chooser scoring —
  is every candidate scored negative (hoarding is "correct" by the current value
  head), or is there a launch-threshold gate? Determines whether the lever is the
  value head's expansion-credit term or a separate launch gate.

- **(Carried) Does refine sub 53336920 settle ≥ 1170 live?** As of 2026-06-03 18:13
  it read 860 but was only ~20 min old (siblings took hours to settle). Re-check
  first; if it plateaus low, resubmit champ_adaptiveK_on (μ1188, recoverable). See
  knowledge-base/flags/2026-06-03-rolling-pair-uncalibrated.md.
