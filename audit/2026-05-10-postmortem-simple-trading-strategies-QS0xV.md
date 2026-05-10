# 2026-05-10 — postmortem (simple-trading-strategies-QS0xV)

> Branch: `claude/simple-trading-strategies-QS0xV`.
> Pre-commit at `dd603be`; this postmortem is part of the wrap-up
> commit that follows.
>
> **PI ratification of friction-promotion candidates: deferred.**
> Per CLAUDE.md Rule 0 / user instruction this session, I am not
> using `AskUserQuestion`; promotion candidates are listed below
> and the PI may ratify in the next session.

## What happened (one-paragraph TL;DR)

Built two batches in this branch. **Batch 1** shipped a five-strategy
target-selection panel under `agents/simple/` plus a round-robin
runner; the 8-seed run surfaced `roi` (production / distance) as the
clear winner at 96.9% mean panel WR / 100% (16/16) vs `v1_orbitfix`.
**Batch 2** rewrote the plan around the user's much bigger ambition
(strategy zoo → behavioural fingerprint → opponent classifier →
best-response routing) and shipped the Phase-1 infrastructure: replay
capture, `lib/fingerprint.py` (15 features), `scripts/manifold_check.py`,
and a prior-art survey. The 32-seed capture confirmed `roi` (97.1% mean
WR, 100% (64/64) vs `v1_orbitfix`) but the manifold gate did NOT clear
(RF 80.5% at K=100, target 90%) — `nearest` / `production` / `roi`
collapsed into a single "production-aware-greedy" basin. Wrapped up
with a bundled `submissions/roi.py` staged for PI submission approval;
v1.1 settled at μ=597.4 in the meantime.

## Calibration table — predicted vs actual

(see also `state/calibration-ladder.md` — single source of truth)

| Item                      | Predicted          | Actual         | Δ                    |
| ------------------------- | ------------------ | -------------- | -------------------- |
| v1.1 vs v1 live μ delta   | +50 to +200        | +89 (508→597)  | within range ✅      |
| roi vs v1 8-seed → 32-seed| ≈ same             | 100% → 100%    | extremely consistent |
| roi mean panel WR 8 → 32  | minor regression   | 96.9 → 97.1    | actually held ✅     |
| production mean WR 8 → 32 | hold near 75%      | 75 → 67.7      | regressed ~7pp       |
| Phase 1 manifold gate     | ≥90% RF at K≤100   | 80.5% RF       | ❌ not cleared       |

PI overrides this session: 0 (the Phase-0 plan was approved with
clarifying answers; the Phase-1+ plan was approved as written).

## What worked

- **The "5 small modules sharing one mechanism stack" architecture**
  paid off twice: once when the 8-seed result surfaced ROI, again
  when 32 seeds confirmed it without any rebuild. The Strategy /
  Intent / realize abstraction held under three different uses
  (per-strategy panel, round-robin runner, downstream replay corpus).
- **Replay capture as opt-in flag** — the additive `capture_replays`
  kwarg means existing callers pay zero cost; the manifold-check
  pipeline re-runs cheaply once the corpus is captured. Gzipped
  ~250 KB per game keeps a 1568-game panel at 404 MB.
- **GroupKFold-by-seed in `manifold_check.py`** — caught what would
  otherwise have been a 10-20pp accuracy inflation from same-seed
  leakage. Per-seed CV is the right discipline for paired-game data.
- **Prior-art note as a committed artifact** rather than only-in-chat
  research grounds the design in published references (Grover 2018,
  DRON 2016, AlphaStar's "discrete basins" framing, Pluribus's
  exploitation-counter-exploitation warning) and sets expectations
  for which path the Phase-1-fail branch should take.
- **Honest negative result for the manifold hypothesis.** The user
  asked me to push back on the framing — I did, and the data
  confirmed the pushback (basins, not a smooth low-dim manifold).

## What didn't / what to fix

(promotion candidates for `.claude/skills/kaggle-comp/improvements.md`,
flagged for PI ratification)

1. **Bootstrap doesn't actually fetch comp data on this container.**
   `bash bootstrap.sh` runs to "bootstrap done" but never executes
   `kaggle competitions download`. This is the second session in a
   row where this caused a 5-minute friction. Promotion: either fix
   `bootstrap.sh` to always check `data/main.py` exists post-run, or
   add a session-start hook (per the `session-start-hook` skill) that
   refuses to proceed without `data/main.py`.

2. **Bundler hardcoded directory layout.** Ten lines of code that
   blocked submission staging by 5 minutes; should have been caught
   the moment the simple-strategy panel folder layout diverged from
   v1's. Promotion: add a smoke test that bundles every `agents/**`
   path on every commit (extends `tests/test_bundle.py`).

3. **`AskUserQuestion` UX friction.** User flagged twice that they
   dislike the question tool. The "Other" option came back without
   the user's free-text answer (lost in transit / not surfaced
   to me). Promotion: switch to inline text-options in user-facing
   prose for clarification questions; ExitPlanMode for plan approval.
   Already applied in this session after the second flag; promotion
   is to make this the default in the kaggle-comp skill.

4. **`pip install --break-system-packages` recipe should be in
   `bootstrap.sh`.** Friction `pip-blinker-system-conflict` reappears
   on every fresh container. The bootstrap.sh `pip install` line is
   `pip install -q -r requirements.txt` which fails the same way
   every time. Promotion: bootstrap.sh should use the
   `--break-system-packages --ignore-installed blinker` flags.

5. **Sun-clip detection in fingerprint may double-count.** The
   probe-ray method projects 200 board-units; if a strategy fires
   *toward* the sun area but the planet sits beyond, we count it
   even though the env's collision check is path-based and the fleet
   would die. The feature is consistent across same-strategy agents
   (so the classifier still discriminates), but the absolute number
   isn't a clean "fleets actually destroyed" rate. Promotion (low
   priority): replace probe-ray with same-step fleet-disappearance
   tracking once D.2 replay-logging is fully consumed by Phase 2.

## Open at end of session

- **PI submission approval for `submissions/roi.py`** — staged but
  not pushed. Predicted live μ 700-1000; rolling-last-2 means pushing
  it evicts v1 (μ=508) but keeps v1.1 (μ=597.4). 2 submission slots
  left today.
- **Phase 2 path choice** — H-coarsen-labels (cheap, 3-class router)
  vs H-richer-fingerprint (medium, full 5-class) vs learned embedding
  (last resort). Recommended A then B in the verdict audit.
- **v1.1 PR / merge to main** — not in scope this session; the merge
  cadence is the comp-team's call.

## Files added / changed this branch

(load-bearing only; full list in commits)

- `agents/simple/{nearest,production,roi,weakest,enemy_first}.py`
  (Phase 0)
- `scripts/strategy_panel.py` (Phase 0; later extended for Phase 1
  capture)
- `scripts/tournament.py` (Phase 1: replay capture kwargs + helper)
- `scripts/manifold_check.py` (Phase 1)
- `scripts/bundle_agent.py` (wrap-up: file-or-dir agent input)
- `lib/fingerprint.py` (Phase 1)
- `tests/test_simple_strategies.py` + `tests/test_fingerprint.py`
  (40 new tests; full suite 151/151)
- `docs/strategies/simple-{nearest,production,roi,weakest,enemy_first}.md`
- `audit/2026-05-10-simple-strategy-panel.md`,
  `audit/2026-05-10-meta-strategy-prior-art.md`,
  `audit/2026-05-10-phase1-manifold-verdict.md`,
  `audit/2026-05-10-postmortem-simple-trading-strategies-QS0xV.md`
- `submissions/roi.py` (staged — PI approval pending)
- State updates across `state/{current,hypothesis-board,
  mechanism-ledger,calibration-ladder}.md` + `ISSUES.md` +
  `audit/friction.md` + plan file
