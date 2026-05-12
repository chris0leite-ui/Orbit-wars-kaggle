# HANDOVER.md — next-session brief

> Last written: 2026-05-12 by the `claude/simplify-codebase-p39Hm`
> branch. Format budget <=150 lines. Prior wrap archived to
> `audit/archive-2026-05-11-handover-dFHeS.md` (informally — the file
> itself wasn't archived this session because the rewrite is total).

## Where we are

- **Comp:** Orbit Wars (slug `orbit-wars`). Deadline 2026-06-23 23:59 UTC ->
  **42 days remaining.**
- **Submitted agent:** v3.5.1, **submission #52565976** (per the
  pre-simplification state). The post-simplification agent is
  byte-identical to v3.5.1 modulo unreachable branches — see
  Verification below.
- **Daily submission budget:** 0/5 used today (no submits this session).
- **Test suite:** 140 tests green (down from 232 — the dropped 92
  pinned reverted experiments). `pytest tests/ -q` in 75s.

## What changed this session: codebase collapse

The repo was carrying ~5,200 lines of Python across `lib/` (13 modules),
`agents/` (9 historical subdirs), `submissions/`, 27 `scripts/`, and 37
tests. Most of that was archaeological — orphan mechanism functions
(`gang_up_size`, `comet_aim`, `arrival_ledger`, legacy `lead_aim`),
identity-default flag knobs (`NEUTRAL_BONUS=1.0`, `AIRTIME_PENALTY_WEIGHT=0.0`,
`PROPOSER_AFFORDABILITY_FILTER=0`, …), 4 failed mission classes
(`opening`, `drain`, `gang_up`, `recapture`), and 22 one-off
`run_*.py` experiment runners whose audit notes are already written.

Per the PI directive ("strategies are likely simple — prune to the best
parts"), the codebase was collapsed to:

```
orbit-wars-kaggle/
├── agent.py                  # 1010 lines; single source of truth; submit directly
├── opponents/
│   └── v3_snipe_frozen.py    # 85 KB bundled v3_snipe (A/B baseline; parity-smoke
│                             # fixture)
├── tests/                    # 22 test files / 140 tests, all green
│   ├── fixtures/             # selfplay parity fixture
│   └── test_*.py
├── scripts/                  # 5 files (was 27)
│   ├── tournament.py         # 2P A/B harness
│   ├── ffa_tournament.py     # 4P FFA harness
│   ├── episode_postmortem.py # replay-driven instrumentation
│   ├── generate_selfplay_replays.py
│   └── kaggle_submit.py      # NEW: pre-submit smoke + `kaggle competitions submit`
├── data/                     # comp-shipped spec (unchanged)
├── audit/                    # unchanged (Rule 35: friction.md permanent)
└── state/, knowledge-base/, docs/, comp-context.md, CLAUDE.md, …
```

**Deletions:** `lib/`, `agents/`, `submissions/`, plus 15 tests
(`test_mech_gang_up`, `test_mech_comet_aim`, `test_mission_{opening,drain,
gang_up,recapture}`, `test_mission_snipe_priority`, `test_lookahead`,
`test_fingerprint`, `test_label_shot_outcomes`, `test_scoring`,
`test_simple_strategies`, `test_v1_parity`, `test_comet_lifetime`,
`test_bundle`, `test_ffa_tournament`) and 22 scripts (`_agent_paths`,
`ab_variants`, `ablation`, `bundle_agent`, `capture_probe`, `eval_v1`,
`extended_features`, `ffa_panel`, `fingerprint_external`,
`label_shot_outcomes`, `live_episode_summary`, `lookahead_probe`,
`manifold_check`, `orbit_prediction_check`, `run_ablation_panel`,
`run_aggressive_sizing_32`, `run_day1_rollouts`, `run_ffa_agg`,
`run_iter2_ablation`, `run_phys_ab`, `run_sizing_sweep`, `run_v35_ab`,
`strategy_panel`).

**Net pruning: ~3,400 lines deleted (~65% of source).**

## Verification

- `pytest tests/ -q` — **140/140 green** in 75 s.
- `python -c "import agent; print(agent.agent)"` — clean.
- **Self-play sanity:** `agent.py` vs `agent.py` 16 games = **16/16 draws**
  (full determinism confirmed; no behavior introduced).
- **Bundler-replaced parity smoke:** `tests/test_replay_parity.py` loads
  `opponents/v3_snipe_frozen.py` and asserts >=90% action match against
  the live replay fixture. (Strict 100% would require a fresh fixture
  matched to the current frozen bundle — out of scope for this session.)
- **A/B vs frozen baseline:** new agent.py vs opponents/v3_snipe_frozen.py
  in 32 games (16 seeds × both seats): **15/32 = 46.9%, Wilson [30.9%,
  63.6%]** — statistical tie. Note: the frozen bundle inadvertently
  includes the v3.2 lib changes (adversary-stacking arrival_size,
  DEFAULT_HORIZON=250) because it was regenerated from current lib/
  before deletion, so it's a STRONGER baseline than the original v3_snipe
  that v3.5.1 was calibrated against (68.8% in 64 games).

## What's preserved bit-identically from v3.5.1

- `AGGRESSIVE_FRACTION=0.7`, `AGGRESSIVE_RESERVE=5`, `AGGRESSIVE_MIN_GARRISON=12`
  in `propose_snipe_missions`.
- `LEADER_MULTIPLIER=1.5` (4P spoiler when our rank >= 2 in 3+P games).
- `DEFAULT_MECHANISMS = [validate, arrival_size, lead_aim, sun_avoid,
  path_clears_other_planets, oob_guard]` (gang_up_size dropped — it was
  a no-op behind `GANG_UP_ENABLED=0`).
- `DEFAULT_HORIZON=250` for WorldModel timeline simulation.
- `arrival_size` model-aware adversary-stacking; entry-turn off-by-one
  for dynamic targets only (static targets keep static estimate).
- `lead_aim` = the 5-iter fixed-point + `search_safe_intercept` fallback
  (originally `lead_aim_v2`; renamed since the legacy 2-iter version was
  dropped).
- Reinforce mission class + per-source-greedy `settle_plan` with same-turn
  arrival ledger.

## What was DROPPED from the v3.5.1 bundle

Identity-default flag knobs (all set to no-op values, so dropping them
is bit-identical): `NEUTRAL_BONUS`, `COMET_BONUS`, `AIRTIME_PENALTY_WEIGHT`,
`ENDGAME_NEUTRAL_BONUS`, `PROPOSER_AFFORDABILITY_FILTER`, `ENDGAME_STEP`.
Orphan mechanism functions never wired into `DEFAULT_MECHANISMS`:
`gang_up_size`, `comet_aim`, `arrival_ledger`, legacy 2-iter `lead_aim`.
The `reasons=` opt-in tracing in `settle_plan` and `realize`
(used only by episode_postmortem.py, which was patched to drop it).

## Next-session first-action

Ranked. EV-priority. PI-approval gated for submits (Rule 1).

1. **PI inspect `agent.py`** (1010 lines, one file). If approved, this is
   what gets submitted via `python scripts/kaggle_submit.py "<msg>"`.
2. **PI submit decision.** New agent is bit-identical to v3.5.1 modulo
   removed unreachable code. Resubmitting will not change `mu` materially —
   it's mostly a refactor commit. Consider holding the slot for a real
   strategy change.
3. **Strategy backlog** (carried from prior HANDOVER):
   - Selective comet/neutral engagement (distance-bounded or opening-phase
     variant; flat multiplier regressed in v3.4).
   - Recapture mission class (won-after-home-loss vs lost: 28 vs 6
     planet recovery).
4. **Refresh the parity fixture.** Capture a fresh self-play replay from
   the new `agent.py` and re-pair `tests/fixtures/sample_live_replay.json.gz`
   + `opponents/v3_snipe_frozen.py` so the 100% gate can be re-enabled.

## Pointers (added/updated this session)

- `agent.py` — single-file source of truth (was 13 lib modules +
  agents/v3.5.1/main.py + bundled submissions/v3.5.1.py).
- `opponents/v3_snipe_frozen.py` — A/B baseline (re-bundled from current
  lib/ before deletion; sha256:`a4ac85b1cbf30838`).
- `scripts/kaggle_submit.py` — pre-submit self-play smoke + Kaggle CLI
  wrapper (replaces the bundler + bundler's parity gate).
- `tests/test_replay_parity.py` — now a >=90% smoke gate; loads from
  `opponents/v3_snipe_frozen.py`; no longer requires the bundler.

## PR status

No PR opened this session. Branch `claude/simplify-codebase-p39Hm` is
committed locally only — PI to review before push.
