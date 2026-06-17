# Flags — 2026-06-17 (take-and-hold session)

- **[MONITOR] sub 53768768 (take-and-hold) was PENDING ~2 h** at session end with
  no score and no Error. The exact bundle passed a local real-loader self-play
  (2P + 4P-to-250, max 469 ms, 0 timeouts, entry=`agent`) and uses the same
  packaging as the prior lr bundle that scored — so the delay is almost certainly
  the Kaggle eval queue, not us. **First fresh-session action: re-check the status.**
  COMPLETE = passed (compare μ to the 1115 backstop); Error = diagnose (unexpected).

- **[TIMING] latent slow turn:** a **1441 ms** 4P midgame turn appeared earlier when
  `least_resistance` played a 4-way game vs *other* public bots (self-play is clean,
  ≤570 ms). The per-turn budget bails the greedy + 2-ply loops separately, so their
  sum + unbounded candidate-gen can exceed 1 s in heavy states. On the ladder that
  eats the 60 s overage bank / risks a slow-turn loss. **Bound the total turn.**

- **[CLEANUP] dead-weight flags in `agents/least_resistance/main.py`:** the five
  refuted levers (`LR_LEADER_RELATIVE_4P`, `LR_VALUE_COMMIT`, `LR_ENEMY_BOOST`,
  `LR_ANYTIME`, `LR_ROLLOUT_DEPTH`) are gated default-OFF but are noise — strip them.

- **[METHOD] A/B independence is now a rule** (improvements.md): one game per fresh
  seed, rotate seat across seeds, never within a seed. Use `scripts/verify_confirm.py`.
