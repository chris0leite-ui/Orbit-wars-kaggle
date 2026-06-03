# Open questions — 2026-06-03 (refine teamwork submit)

- **Does refine settle ≥ 1170 live?** (THE gate question.) Sub `53336920`
  scored 70% h2h vs the adaptive-K champion locally, but local→live μ is noisy
  (our own precedent: sync 88–94% local → 1150 live). If it settles below the
  evicted adaptiveK μ~1170.4, resubmit adaptiveK (recoverable, just behind the
  rolling window). Check first thing next session.

- **Why does refine break the 4 long contested seeds** (2P1, 9P0, 13P0, 14P0)?
  The generator is fine; the oracle *selects* a coalition that backfires in long
  games. Is it horizon-too-short (can't see the downside), or a recapture the
  marginal-gain scorer misses, or a defense it strips? Regenerate replays and
  trace. Answering this is the +9→+13 improvement.

- **Does compute-by-ships cannibalize coalition opportunities?** It raises solo
  reach for high-ship planets (more targets solo-takeable ⇒ fewer "needs two
  sources" situations). Measure raw coalition count with `BASELINE_COMPUTE_BY_SHIPS`
  on vs off. Determines whether the two levers compete or compose.

- **Why did the h2h winrate drift 78% (n=32) → 70% (n=57)?** Normal regression to
  the mean, or a seat/seed-mix artifact of the larger sample? Not urgent, but the
  true value matters for sizing the next candidate (calibration ladder).
