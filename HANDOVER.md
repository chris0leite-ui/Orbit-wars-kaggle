# HANDOVER.md — next-session brief

> Last written: 2026-05-18 (late) by
> `claude/audit-workflow-performance-btjeK`. Spatial leaf AND
> post-chooser idle drain (H1) BOTH falsified this session. The
> chooser's reserves are correctly calibrated; single-step
> heuristics targeting "idle ships" all fail. Direction B (joint
> candidates / multi-step) is the only known sound next direction.

## Where we are

- **Comp:** Orbit Wars. Deadline 2026-06-23 23:59 UTC. ~36 days.
- **Live production:** `52754310 baseline.py` — trajectory chooser
  v4 + wait_N + wallclock budget + hybrid value head. Settled at
  **μ=1271.8** (well above prior session's prediction of ~1158).
  Sets `BASELINE_CHOOSER=trajectory` and `BASELINE_VALUE_HEAD=hybrid`
  via setdefault in `agents/baseline/main.py`.
- **Rolling-last-2** (auto-eval pair):
  - `52754310` trajectory v4 (5/17 22:06 UTC) — COMPLETE, μ=1271.8
  - `52744856` composite_a2_hybrid (5/17 14:17 UTC) — COMPLETE,
    μ=1152.7 (the floor)
- **Daily submission budget:** 5/day; 5/17 used 3 (52744234 ERROR,
  52744856 OK, 52754310 OK); 5/18 used 0.
- **NO new submission this session.** Spatial leaf experiment was
  net-negative (see audit/2026-05-18-spatial-leaf-negative-result.md).
  Floor stays at 1152.7; champion at 1271.8.
- **Calibration UPDATE**: the trajectory chooser's local A/B
  predicted ~1140-1180 mu vs v15-as-reference; live μ landed at
  1271.8 — **+90 above prediction**. Local A/B systematically
  UNDERPREDICTS for the trajectory chooser architecture. Future
  candidates that show clear local A/B lift may settle even higher.
  Conversely, A/B losses (like spatial leaf this session) are real
  and should be trusted.

## What this session shipped (no submission)

6 commits this session:
- `b5f5296` — spatial leaf head (opt-in, env-gated) + idle-trajectory
  audit infrastructure
- `cc38e11` — summary.json for 52754310 live episodes
- `558bd61` — spatial leaf 2P-only short-circuit (4P regression fix)
- `70fcc28` — spatial leaf experiment: negative result, no submission
- `1b3f920` — H1 post-chooser idle drain implementation (initial)
- `90c6adb` — H1 A/B FAIL: 11/32 vs hybrid, default flipped OFF

### A/B receipts (clean bundle-based, NOT env-based)

| variant | n | wins/rate | Wlo | max-ms | verdict |
|---|---:|---:|---:|---:|---|
| spatial+trajectory vs hybrid+trajectory (2P) | 64 | 26/40.6% | 0.295 | 2541 | **FAIL** |
| spatial+trajectory in 4P vs 3x hybrid | 32 | 3 first-place/9.4% | 0.032 | 1503 | **FAIL** |
| H1-idle-drain+trajectory vs hybrid+trajectory (2P) | 32 | 11/34.4% | 0.204 | 1528 | **FAIL** |

Both attempts to "drain idle rear ships" hurt winrate. Spatial perturbed
chooser Δ globally. H1 force-emitted launches the chooser correctly
rejected.

**Generalizable conclusion**: the 43.8% isolated ship-turns measured
on the trajectory champion is NOT a leak. It's the natural distribution
of correctly-held defensive reserve. The chooser at μ=1271.8 already
optimizes this. Any single-step heuristic that "drains" or "pulls
forward" hurts.

### What's reusable

- `scripts/idle_trajectory_audit.py` — re-runnable measurement
  (ship-turn density by distance bucket, launch ETA distribution,
  staging-opportunity rate). See `audit/replays/idle-trajectory-
  2026-05-17.md`.
- `agents/baseline/value.favor_hybrid_spatial` — opt-in spatial
  leaf via `BASELINE_VALUE_HEAD=hybrid_spatial`. Default OFF.
- New env vars: `BASELINE_SPATIAL_WEIGHT` (default 0.5),
  `BASELINE_SPATIAL_DECAY` (default 30).
- Friction tag `env-var-shared-process-breaks-ab-isolation` —
  documents the within-process A/B isolation issue. **Use hard-coded
  bundle patches for clean A/B between configurations.**

## Confirmed: idle-fleet leak IS real but spatial leaf does NOT fix it

`scripts/idle_trajectory_audit.py` measured on submission 52754310:
- **43.8% of ship-turns** are on planets >50 units from any
  non-our planet ("isolated")
- 22.6% mid, 10.7% rear, 22.9% frontier
- Long launches (>20 ETA): 11.6% of all launches
- Staging-opportunity rate: 46.2%

The leak is real. The spatial leaf is just not the right fix.

## Next-session first-action (ranked by EV / cost)

**1. Direction B — joint candidate evaluation (PI directive from
prior session).** Multi-step planning is more likely to drain
isolated ship-turns than single-step leaf tweaks. The joint
chooser scores "A→B stage then B→T capture" as a unit, so forward-
deploy emerges naturally if the joint score is positive. Concrete
plan: `knowledge-base/thoughts/2026-05-17-direction-b-joint-action-
scoping.md`. Open question on joint baseline still pending PI:
`knowledge-base/questions/2026-05-17-joint-scoring-baseline.md`.

**2. Mine top-5 public notebooks (Rule 22).** We're at μ=1271.8.
Romantamrazov LB-MAX was 1224 (47 below). Top-of-LB might be
1300+. Pull and compare structural choices. Cheap; informs every
later direction.

**3. Alternative positional formulation.** Rule 37 budget: 2 more
variants available on positional axis.
  - Weight by production (target capture EV, not just distance).
  - Restrict to "contested neutrals only" (exclude opp planets
    that have full garrisons and aren't realistic captures).
  - Different spatial signal: distance-to-opp-FLEETS-only (not
    planets).
  - Risk: still double-counting concern. Probably better to wait
    for Direction B which uses different framing.

**4. Staging proposer.** Per audit, only 12% of launches are long
and 46% of those have staging options ≈ 5.5% of launches could
benefit. Low leverage compared to other directions.

## Pointers

- `agents/baseline/main.py` — entry, sets BASELINE_CHOOSER=trajectory
  + BASELINE_VALUE_HEAD=hybrid via setdefault.
- `agents/baseline/value.py` — favor_hybrid (validated production),
  favor_hybrid_spatial (opt-in, 2P-only). New env vars
  BASELINE_SPATIAL_WEIGHT, BASELINE_SPATIAL_DECAY.
- `agents/baseline/chooser_trajectory.py` — trajectory chooser v4
  (extend here for Direction B).
- `agents/baseline/proposer.py` — multi-wait grid + banded dedup.
- `scripts/idle_trajectory_audit.py` — NEW; ship-turn density
  measurement.
- `audit/2026-05-18-spatial-leaf-negative-result.md` — NEW;
  this session's negative-result postmortem (full A/B receipts +
  failure-mode hypotheses).
- `audit/replays/idle-trajectory-2026-05-17.md` — measurement
  output for 52754310 / 52744856 / 52710995.
- `knowledge-base/thoughts/2026-05-17-direction-b-joint-action-
  scoping.md` — Direction B plan.
- `knowledge-base/questions/2026-05-17-joint-scoring-baseline.md` —
  open question for PI.
- `state/current.md` — submitted-agent state (no μ values).

## Rule reminders

- Rule 1: submissions are single-shot, PI-approved. No retry loops.
- Rule 12: rolling-last-2 — third push evicts oldest. Current pair
  is [52754310 (1271.8), 52744856 (1152.7)]. Push order matters.
- Rule 22: at every plateau, mine top-5 public notebooks. **Fires
  next session — current μ=1271.8 stable, exploration warranted.**
- Rule 32: session-start `kaggle competitions submissions
  orbit-wars` is the source of truth for μ. State files do NOT
  record μ.
- Rule 37: 3-variant axis cap. Spatial-positional axis: 1/3 used
  (this session's negative result). 2 more available.
- Rule 38: fix-verification reproduces failure. The 2P-only
  short-circuit was verified via test_favor_hybrid_spatial_skips_
  spatial_in_4p with SPATIAL_WEIGHT=5.0 but not by re-running 4P
  A/B with the new bundle. **Next session: re-bundle and verify
  4P matches hybrid baseline before any spatial-leaf submission.**
- Rule 40: prefer modeling-correctness over restriction-tuning.
  Applied this session (spatial leaf IS modeling). Rule 40 doesn't
  guarantee the modeling fix is RIGHT — it guarantees you TRY
  modeling. Spatial leaf was wrong direction.
