# Postmortem — 2026-05-28 PM3 (kaggle-submission-review-gZsCu)

Third postmortem of the day. Companion to:
- `audit/2026-05-28-postmortem-pv-eta-pm-pv-eta-and-silent-turns.md` (PM1, PV_ETA ship)
- `audit/2026-05-28-postmortem-pm2-leaf-pv-2p-compute-variance.md` (PM2, leaf_pv_2p ship + compute-variance investigation)

This PM3 covers the macro-layer build + null-result A/B + opp-model pivot.

## What went wrong

- **Wallclock-confound on the first A/B.** Set `BASELINE_WALLCLOCK_MS=2000`
  in the harness env, not realising kaggle env's `actTimeout: 1.0` is
  a separate hard cap. 38/64 games hit double-TIMEOUT. The fact was in
  `comp-context.md` and the env source; I conflated soft internal
  chooser budget with the env's hard turn limit. Rule 45b (ratified
  PM2 same day, same agent) covers the spirit but its literal text
  scopes to "sub-gate-strength submit," so I didn't apply it.
  Decision-quality verdict: a known fact existed and was not
  consulted; the fix re-ran the A/B at `WALLCLOCK_MS=800` with 0
  timeouts, confirming the diagnosis.
- **Launched a 2.5h background sweep at session-end.** Container was
  reclaimed between sessions; sweep killed mid-baseline-rerun (46/64
  games complete in variant 1, 0/64 in variants 2-4). Rule 32
  (session-start fetch) implies but does not codify "don't launch
  >30 min background compute near session-end." Decision-quality
  verdict: should have foreground-run, chunked, or deferred — Rule
  32's existence should have generalised to "container reclaim costs
  unfinished computes too, not just stale state."

## Frictions logged this session

- `tag: wallclock-ms-exceeds-env-acttimeout` — see `audit/friction.md`
  PM3 block.
- `tag: background-compute-killed-by-container-reclaim` — see same.
- `tag: macro-layer-functionally-redundant-with-chooser` — see same.

## Promotion candidates (PI ratified: NO — nothing to promote this session)

Two candidates were drafted, both DECLINED by PI:

- **Candidate A** — Codify the env actTimeout vs WALLCLOCK_MS
  distinction in `comp-context.md` as a loud block similar to
  `SCORES DO NOT SETTLE`. Proposed text included the safe value
  (≤800ms) and the origin (PM3 macro A/B). PI: "Nothing to promote."
- **Candidate B** — Add a rule (or amend Rule 32) about long-running
  background compute near session-end. PI: "Nothing to promote."

Both stay as friction one-liners only. Future agents will see them via
`audit/friction.md` but not via the CLAUDE.md rules block.

## PI additions (from step 4)

PI answered: "Nothing to add or to promote."

## What landed this session

- **`lib/missions/macro.py`** (~300 LOC) — 4-state machine: EXPAND,
  STOCKPILE, STRIKE, DEFEND. Closed-form geometry: forward lateral
  via `home_angle + π/2` (`omega > 0` always). Contiguous-id home
  group (env uses 90° rotation, NOT the README's claimed mirror).
- **`agents/baseline/main.py`** — pre-chooser hook behind
  `BASELINE_MACRO=1` env var. `macro_reserved` blocks chooser from
  draining stockpile; `macro_moves` prepended to chooser output.
  Layered on top of PV_ETA + LEAF_PV_2P; byte-parity OFF preserved.
- **`scripts/bundle_agent.py`** — `lib/mirror.py` and
  `lib/missions/macro.py` added to `DEFAULT_LIB_ORDER`.
- **`tests/test_macro.py`** — 10 unit tests; bundle parity green.
- **Commits:** `1640792` (feat), `faa8401` (bug fixes from seed-7 trace).

## A/B results

| Run | Config | Result | Notes |
|---|---|---:|---|
| First A/B | `WALLCLOCK_MS=2000`, n=64 | **13/26 = 50.0%**, Wilson [0.32, 0.68] | 38/64 TIMEOUT (wallclock confound) |
| Calibration baseline_rerun | `WALLCLOCK_MS=800`, n=64 partial | **24/46 = 52.2%**, Wilson [0.38, 0.66] | 0 timeouts; killed by container reclaim before STRIKE_MARGIN variants ran |

**Verdict:** macro adds no measurable lift at default knobs over the
PV_ETA + LEAF_PV_2P stack. State machine fires correctly per seed-7
trace, but the chooser was already producing equivalent forward-bundled
launches via its existing scoring.

## What I'm carrying out of session

- **The macro is opt-in dead-weight at default.** Code ships but
  `BASELINE_MACRO=0` is the default — no behavioural change on the
  live rolling pair. Either revert or leave dormant.
- **The opp-model is the next frontier (PI directive).** `lite_greedy_policy`
  inside the rollout currently models opp counter-attacks from "any
  opp planet" — PI's "opponents from everywhere" diagnosis. Restricting
  to planets where `eta(opp_planet → our_target) < safe_horizon` is a
  modeling-side fix (Rule 40 priority over restriction-tuning).
  Expected mechanism: each candidate is currently scored against a
  pessimistic opp counter-pool → fleet sizes undercounted, captures
  under-emitted → "silent turns" + "fleet too small" pathology that
  PM1 documented. Fixing the opp pool spatial restriction propagates
  uniformly through the chooser's value computation, unlike the macro
  which layered a new constraint.
- **Diagnostic next step (cheap, ~1-2 hr):** instrument the chooser
  to log per-candidate `lite_greedy_policy` predicted opp ships vs
  actual post-game realised opp ships. If the predictor over-estimates
  by >30% on average, opp-model spatial restriction has a strong prior.
  Trace seed=7 with the logging on to see whether the candidate
  launches I observed had inflated counter-attack predictions.

## What I would NOT do again

- Use `BASELINE_WALLCLOCK_MS > 1000` for any evidence-generating A/B.
  The env's `actTimeout: 1.0` is the load-bearing constraint.
- Launch >30 min background compute near a likely session boundary.
  Container reclaim is a known systemic cost; should foreground,
  chunk, or defer.

## Framework version at session-end

- Branch: `claude/kaggle-submission-review-gZsCu`
- Commit SHA at write-time: see `git rev-parse HEAD` in the commit
  that ships this artifact
- Active CLAUDE.md rules: 1..48 (48 = same-day-reads-are-climb-
  snapshots, added PM2 today; 45b = confound-check, also added today)
- Skills loaded this session: postmortem (this), kaggle-comp (via
  /improvements.md reads earlier)
- Newly added artifacts:
  - `lib/missions/macro.py` (commit 1640792, fixed 4357fe4)
  - `agents/baseline/main.py` macro hook
  - `scripts/bundle_agent.py` DEFAULT_LIB_ORDER additions
  - `tests/test_macro.py` (10/10 green)
  - `audit/friction.md` PM3 block
  - `knowledge-base/thoughts/2026-05-28-pm3-macro-null-opp-model-pivot.md`
  - This postmortem
