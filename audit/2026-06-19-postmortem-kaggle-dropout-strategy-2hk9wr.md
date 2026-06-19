# Postmortem — 2026-06-19 kaggle-dropout-strategy-2hk9wr

## What went wrong

- **Over-generalized a single 2-point comparison into a directional principle.**
  After the V2-as-opponent A/B regressed (13/32, −802), I framed "the *strength*
  of the modelled adversary, not its accuracy, is what disciplines plan selection"
  as a *principle* and recommended testing a *stronger* adversary. The only
  evidence was a single comparison (simultaneous-producer 20/32 vs V2-model
  13/32), which was equally consistent with a **calibration sweet-spot**
  explanation. The best-response A/B then confirmed the sweet-spot reading: a
  *too-tough* adversary regressed identically (13/32, −959) to the *too-weak* one.
  Decision-quality miss: at decision-time I had enough to see the two hypotheses
  were indistinguishable on the data in hand; I should have presented them as
  competing and tested the discriminating case, not promoted one and steered a
  ~25-min compute spend + a recommendation the PI acted on. (Given the priors,
  proposing *a* test was fine; promoting it to a principle was not.)

- **Re-ran the deterministic OFF baseline three times.** The OFF config
  (`LR_HOLD_SEARCH=0` etc.) is byte-identical and reproduces 20/32 +319 every run,
  yet I included it in three separate A/Bs (hold-search untuned, hold-search
  tuned, win-leaf) before the PI stopped me ("stopping running the OFF comparison
  again and again"). ~20 min × 3 of compute spent re-deriving a known constant.
  Fix adopted mid-session: measure ON-only and compare against the recorded
  baseline.

- **Minor:** ran heavy HTML replay renders concurrently with timing-sensitive
  A/Bs, injecting CPU-contention spikes into the `max_turn_ms` column
  (1422 → 3122 → 4054 ms) that then needed repeated caveating. Renders and timing
  A/Bs should not share the box.

## Frictions logged this session

(No `audit/friction.md` block was written by WRAPUP step 4 before this postmortem;
the three items above are the session's frictions and are cross-listed in the
2026-06-19 block appended to `audit/friction.md` as part of this wrap-up.)

## Promotion candidates (PI ratified: NO — keep in postmortem only)

- **[ ] Don't re-measure a deterministic baseline config — compare new variants
  against the recorded constant.** Bar met (≥1 h compute waste AND required PI
  override). **PI decision: do not promote to improvements.md.**
- **[ ] When one comparison admits multiple explanations, frame competing
  hypotheses and test the discriminating case before promoting to a principle or
  recommending action.** Bar met (caused a wrong recommendation the PI acted on).
  **PI decision: do not promote to improvements.md.**

## PI additions (from step 4)

- Nothing to add.

## Framework version at session-end

- Commit SHA: `e4ad8d58e2d957ff3525a882a0ec36aceeb2fd62`
- Active operating rules: CLAUDE.md 1, 12, 32, 35, 36, 38, 39, 40, 42, 45, 46.
- Loaded skills this session: `postmortem` (this artifact); `kaggle-comp` context.
