# Postmortem — 2026-05-12 analyze-leaderboard-strategies-sdZlE

## What went wrong

- **v3.5 stack regressed 39.1% at 32 seeds (Wilson lo 28.1%).** All
  four new Mission classes (opening, drain, gang_up, recapture) failed
  the 55% Wilson-lo gate individually in 16-seed ablations: 40.6%,
  46.9%, 50.0%, 53.1% respectively. Decision quality given priors:
  *defensible* — the top-performer fingerprint analysis was real,
  the PI directive was "be ambitious," and each Mission's conditional
  gate (step-window, garrison threshold, eta-window, lost-planet
  window) was reasonable in isolation. Decision quality given
  *prior-art recall*: weaker — v3.3 (blanket eta+1 → 42.2%) and v3.4
  (NEUTRAL_BONUS=1.5 → 28.1%) had already established that broad-shape
  proposal-mix changes regress on first try. I knew this pattern;
  I should have run a one-Mission-at-a-time ablation BEFORE stacking
  the full v3.5 portfolio. The full-stack A/B was the right gate but
  not the right FIRST gate.

- **Three new A/B driver scripts where one would have sufficed.**
  `scripts/run_v35_ab.py`, `scripts/run_ablation_panel.py`,
  `scripts/run_phys_ab.py` are nearly-identical thin wrappers around
  `tournament.run_tournament`. Should have factored the common
  argparse + Wilson-lo + dump-result scaffolding into one parametric
  script the first time I needed it. Decision quality: poor — visible
  by the second copy.

- **No PI overrides this session.** Calibration count: 0/M. Per Rule
  5b: 2 consecutive 0/M sessions would flag `pi-stamp-risk`. This is
  the first; not yet a flag.

## Frictions logged this session

Appended under `## 2026-05-12 (analyze-leaderboard-strategies-sdZlE)`
in `audit/friction.md`:

- `tag: multi-mission-stack-regresses-even-with-conditional-gates`
  (v3.5 ambitious stack at 39.1%, third consecutive session with the
  same regression family)
- `tag: ab-harness-not-reusable-for-arbitrary-pairs`
  (three thin wrappers around `tournament.run_tournament`)
- `tag: kaggle-env-var-case-confusion`
  (15 min wasted on `$KAGGLE_key` lowercase vs `$KAGGLE_KEY` uppercase)

## Promotion candidates (PI ratified: NO promotions this session)

PI was asked to ratify two candidates:

- **`multi-mission-stack-regresses-even-with-conditional-gates` →
  `.claude/skills/kaggle-comp/improvements.md`** (would have added a
  "new Mission classes must clear a 16-seed Wilson lo ≥ 0.55 ablation
  gate before stacking" rule). **PI verdict: NO** — "Only Orbit Wars
  has the per-source-greedy + Mission-class setup; not yet a
  cross-comp lesson." Kept as in-comp friction only.

- **`ab-harness-not-reusable-for-arbitrary-pairs` →
  `.claude/skills/kaggle-comp/improvements.md`** (would have added a
  candidate to subsume the three drivers into one
  `scripts/run_ab.py`). **PI verdict: NO** — "Defer to next session
  that touches tournament.py." Kept as in-comp friction only.

## PI additions (from step 4)

PI: *"Nothing to add."*

No additional frictions surfaced. No PI override events to flag.

## Framework version at session-end

- Commit SHA: `493418ed151500596b1da2151e20fecb9315e959`
- Branch: `claude/analyze-leaderboard-strategies-sdZlE`
- Active rules: CLAUDE.md `Operating rules` 1-36 (no changes this session)
- Loaded skills this session:
  - `kaggle-comp` (per CLAUDE.md / current-comp metadata)
  - `postmortem` (this skill, invoked by `wrap up`)

## Calibration snapshot (Rule 5b)

No live submission this session → no live μ to calibrate against
local-A/B predictions. The local A/B itself was the calibration:
predicted "ambitious stack lifts μ +30-60" → empirical "stack
regresses 11pp." That's a strong negative datapoint and feeds the
"new Mission classes regress on first try" prior — already encoded
in the friction entry.
