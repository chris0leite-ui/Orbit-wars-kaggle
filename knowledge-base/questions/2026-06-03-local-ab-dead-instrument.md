# Open questions — 2026-06-03

- **Is local A/B a dead measuring instrument at the champion's level?** Every
  local opponent is either saturated (v7_0 16/16) or too heavy to batch
  (v7_minimax, champion-vs-champion ~2min/game). If so, the only real validation
  is submit-and-measure on the live ladder. Should we formalise that — i.e.
  budget submissions as the primary experiment channel (within rolling-last-2
  discipline) rather than chasing local lift signals that the instrument can't
  resolve?

- **Would a decided-lead early-call actually help throughput enough?** The grind
  in champion-vs-champion is heavy per-turn cost, not games-running-past-decided.
  Early-call trims only the lopsided tail. Quantify: what fraction of a
  champion-vs-champion game is post-decided? If small, the real lever is a
  lighter opponent or cheaper agent config for triage, not early-call.

- **Does ME-defends help vs weak-expander opponents (istinetz-type) even though
  it loses the mirror?** It targets hoarding, which costs vs spread-fast
  opponents, not vs a mirror. We have no local istinetz-like opponent to test
  this — could one be built/frozen from a public notebook archetype?

- **Is the hoarding loss-mode actually fixable in the leaf at all?** ~8 value-leaf
  mechanisms now falsified. Either the leaf is near-optimal (champion is just
  good) or the right fix is structural (reach-frontier chooser, designed but
  unbuilt) rather than a leaf tweak.
