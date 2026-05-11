# HANDOVER.md — next-session brief

> Last written: 2026-05-11 PM (Day 2 wrap PM) by the
> `claude/analyze-submission-logs-dFHeS` branch. Format budget ≤150 lines.
> Prior `bootstrap-agentic-systems-lqnm6` wrap archived to
> `audit/archive-2026-05-11-handover-lqnm6.md`.

## Where we are

- **Comp:** Orbit Wars (slug `orbit-wars`). Deadline 2026-06-23 23:59 UTC →
  **43 days remaining.**
- **Submitted agent:** v3_snipe, **submission #52544634**, status COMPLETE
  with **publicScore μ=1055.5** (+90.2 over v2). Rolling-last-2:
  `[v2 (μ=965.3), v3_snipe (μ=1055.5)]`. v1.2/roi (μ=1006.9) was evicted.
- **Gap to top-10 prize:** v3_snipe at 1055.5 → **+392 μ** to cliff at
  1447.6 (ShunkiKyoya).
- **Daily submission budget:** 0/5 used today. 5 slots remain.
- **Live winrate** (34 games): 14W/20L (41.2%); 2P 47.1% / 4P 35.3%.
  TrueSkill places us against stronger opponents than the absolute
  winrate suggests.
- **PR #10 open** to main: 100%-parity gates + v3.2 (arrival_size adversary
  stacking + DEFAULT_HORIZON 250) + v3.4 partial (4P spoiler retained,
  neutral/comet bonus disabled after regression). Awaiting merge —
  direct push blocked by branch protection (403).
- **Test suite:** 232/232 non-slow tests green.

## Today's PM progress

Branch `claude/analyze-submission-logs-dFHeS`, ~14 commits this session.
Load-bearing only:

1. **Local↔live parity 100%.** Two postmortem bugs fixed: off-by-one
   on `steps[t].action` indexing (kaggle_environments stores action at
   `steps[t+1]`) and missing `obs.step` backfill for non-seat-0 obs.
   Self-play match rate jumped from 33.9% to **100%**; live replays
   from 53% to 100%. Permanent gates: `tests/test_replay_parity.py`,
   post-bundle self-play parity check in `scripts/bundle_agent.py`,
   sha256 bundle hash printed at bundle time.
2. **v3.2 lib changes** (`lib/mechanism.py::arrival_size` adversary-
   stacking via WorldModel + `lib/world_model.DEFAULT_HORIZON` 110→250).
   2P 32-seed A/B vs frozen v3_snipe: **57.8% Wilson [45.6%, 69.1%]**
   (matches v3_snipe's pre-live calibration). 4P FFA 16-seed: **93.8%
   vs frozen 90.6%** (+3.2pp directional lift).
3. **v3.3 blanket off-by-one fix REVERTED.** 32-seed A/B: 27/64=42.2%
   Wilson [30.9%, 54.4%]. Static targets get over-sized because the
   env's center-to-center distance over-estimates eta by
   `(r_src + r_target)/v`.
4. **v3.4 neutral/comet bonus REVERTED.** Tried `NEUTRAL_BONUS=1.5,
   COMET_BONUS=1.3` based on live finding "78.6% of comets sit
   neutral, only 4.9% to us." 32-seed A/B regressed to 28.1% Wilson
   [18.6%, 40.1%] — flat multiplier tips target selection toward easy
   neutrals when contested enemy planets are the binding constraint.
5. **v3.4 4P SPOILER** (`LEADER_MULTIPLIER=1.5` when our rank≥2 in
   ≥3P games): kept; 2P A/B confirmed no-op in 2P (54/64 draws).
   **Pointed 4P FFA test** (v3.4 focal vs 3×v3.2 background) was
   running at session wrap — read its result before deciding whether
   to retain or revert.
6. **Postmortem Fleet-schema fix** (scripts/episode_postmortem.py:211):
   `init_entry[5]` → `init_entry[6]` (Fleet schema is `[id, owner, x,
   y, angle, from_planet_id, ships]`; we were reading from_planet_id).
   Per-episode `fleet['ships']` field is now meaningful; aggregate
   outcome categorization (captured/bounced/etc.) was unaffected.
7. **Games analysis write-up** (`audit/2026-05-11-v3-snipe-games-
   analysis.md`): five distinct weakness patterns + ranked improvement
   backlog. Plus the original critical review (`audit/2026-05-11-v3-
   snipe-critical-review.md`).

## Falsified-or-dead this session

- **Blanket `+1` production tick in `arrival_size`** (v3.3): regressed
  in 32-seed A/B because static targets are already over-sized by the
  center-to-center distance estimate.
- **Flat `NEUTRAL_BONUS=1.5` + `COMET_BONUS=1.3`** in snipe scoring
  (v3.4 first pass): regressed in 32-seed 2P A/B.
- **The "one ship too little" near-miss claim from the critical
  review §4.6** is real (106 of 518 enemy bounces within ±5 of
  threshold) but doesn't admit a flat-formula fix. Needs a
  selective approach.

## Next-session first-action

Ranked. EV-priority, cost on local CPU.

1. **Read the pointed 4P FFA result** (cost: <1 min, EV: decides
   v3.4 fate). `cat /tmp/claude-0/.../tasks/b84x27box.output` or
   `ls -la audit/tournaments/ffa-panel-*.json`. If v3.4-spoiler
   wins >30% (chance vs 3 identical = 25%), bundle as v3.4 and
   stage for submit. If parity/regression, revert spoiler (one-line
   constant change in `lib/missions/snipe.py`).
2. **Merge PR #10.** Direct push to main is 403-blocked. PR is
   the only path. Branch:
   `claude/analyze-submission-logs-dFHeS`.
3. **PI submit decision** for v3.2 (or v3.4). Rule 1 requires
   PI-authorized push. Bundle paths:
   - v3.2: `submissions/v3_2.py` sha256:`ce304fff67c5f879`
   - v3.4-spoiler-only: `submissions/v3_4.py` sha256:`410b3c2ee370f943`
   Rolling-last-2 eviction record per the §6.3 critique discipline.
4. **Selective comet/neutral engagement** (v3.5). Flat multiplier
   regressed; try distance-bounded (`d < 30`) or opening-phase
   (`step < 50`) variants. The 78.6% neutral-comet finding is real
   but the shape matters.
5. **Recapture mission class** (P4 from games analysis). Most
   relevant improvement for the comeback-gap finding: in wins after
   home loss, we recover to 28 planets; in losses, 6. Roman has
   recapture; we don't. ~4-6h.
6. **(DEFER)** RL training; gang_up mission class (H4); 4P score-
   function rebalance for bigger fleets (P3).

## Pointers (added/updated this session)

- `audit/2026-05-11-v3-snipe-critical-review.md` — critical review of
  submission 52544634 (live μ + error breakdown + project critique).
- `audit/2026-05-11-v3-snipe-games-analysis.md` — five distinct
  weakness patterns + ranked improvement backlog.
- `audit/archive-2026-05-11-handover-lqnm6.md` — prior session's
  HANDOVER.
- `audit/live-episodes/52544634/` + `52532938/` — pulled + postmortem-
  processed live replays (100% action match).
- `audit/live-episodes/SELFPLAY/` — gold-standard parity test fixture.
- `audit/tournaments/20260511T1[5-9]*Z.json` + `ffa-panel-*` —
  A/B + FFA artifacts for v3.2, v3.3 (regressed), v3.4.
- `scripts/episode_postmortem.py` — new replay-driven instrumentation
  diagnostic (100% action match on live replays).
- `tests/test_replay_parity.py` — permanent parity gate against
  `submissions/v3_snipe_frozen.py`.
- `tests/test_mission_snipe_priority.py` — 7 tests for the neutral/
  comet bonus + 4P spoiler (bonuses disabled, spoiler active).
- `submissions/v3_2.py` — v3.2 bundle (validated, ready to submit).
- `submissions/v3_4.py` — v3.4 spoiler-only bundle (pending 4P
  pointed-test validation).
- `submissions/v3_snipe_frozen.py` — frozen v3_snipe baseline (parity
  test pin).

## PR status

**PR #10** (https://github.com/chris0leite-ui/Orbit-wars-kaggle/pull/10):
`claude/analyze-submission-logs-dFHeS` → `main`. Contains all of:
parity infrastructure, v3.2 (validated), v3.3 (reverted), v3.4
(spoiler retained, bonuses disabled). Awaiting human merge.
