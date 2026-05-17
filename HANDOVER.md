# HANDOVER.md — next-session brief

> Last written: 2026-05-17 evening by
> `claude/audit-workflow-performance-btjeK`. Next session opens
> with Direction B (joint candidate evaluation).

## Where we are

- **Comp:** Orbit Wars. Deadline 2026-06-23 23:59 UTC. ~37 days.
- **Just submitted:** `52754310 baseline.py` PENDING — trajectory
  chooser v4 + wait_N + wallclock budget. Sets BASELINE_CHOOSER=
  trajectory and BASELINE_VALUE_HEAD=hybrid via setdefault in
  `agents/baseline/main.py`. **Do NOT hardcode μ here.** Query
  Kaggle at session start: `kaggle competitions submissions
  orbit-wars`.
- **Rolling-last-2** (auto-eval pair):
  - `52754310` trajectory v4 + wait_N (5/17 22:06 UTC) — PENDING
  - `52744856` composite_a2_hybrid (5/17 14:17 UTC) — COMPLETE,
    μ ≈ 1158.6 settling
  - v20 (`52721807`, μ=1082.4) evicted by 52754310 push.
- **Daily submission budget:** 5/day; 5/17 used 3 (`52744234`
  ERROR, `52744856` OK, `52754310` PENDING).
- **Calibration WARNING** (still active per prior sessions): local
  Wilson lower bound consistently over-predicts live μ. Wait for
  ~6h / 50 games per `early-trueskill-mu-unreliable` before
  reading the new submission's settled μ.

## What just landed (2026-05-17 evening)

The trajectory-first reframe (4 iterations across this and the
prior session) finished as a production swap. Live agent now uses
the trajectory chooser with deterministic admissibility filter
(predict_fleet_fate rejects sun/oob/expired-comet/comet-collision
candidates before scoring).

3 commits this session:
- `da2473f` — wait_N>0 candidates threaded into v4 (was previously
  filtered with `if int(wait_N) != 0: continue` at chooser entry).
- `10c9601` — wallclock budgeting via `affordable_validate_cap` +
  `safe_deadline` pre-bail. Mirrors composite chooser pattern.
- `f192cf4` — `BASELINE_CHOOSER=trajectory` setdefault in main.py.

### A/B receipts (n=64 vs v15, BASELINE_VALUE_HEAD=hybrid)

| variant | wins | rate | Wlo | max-ms |
|---|---:|---:|---:|---:|
| v4 fire-now only (no wait_N) | 31/64 | 48.4% | 0.36 | — |
| **v4 + wait_N (shipped)** | **42/64** | **65.6%** | **0.534** | **1077** |
| composite_a2 (ref / rolling partner) | 40/64 | 62.5% | 0.503 | 1292 |

+3pp point estimate over composite at better max-turn-ms. Both
INCONCL — within statistical noise.

### Bench post-wallclock-fix

p50=138, p95=412, p99=566, max=623, over_1000ms=0. PASS.

## Next-session first-action (ranked by EV / cost)

**1. Direction B — joint candidate evaluation (~1-2 weeks, +30-50μ
expected if it works).** PI directive at session end. Concrete
shape filed at `knowledge-base/thoughts/2026-05-17-direction-b-
joint-action-scoping.md`. Summary:

- Enumerator: top-K=5 independent candidates form seed pool. Joint
  candidate set = K singles + C(K,2)=10 pairs = 15 candidates.
- Scorer: fast_sim with ALL constituent launches injected at their
  respective wait_N steps; favor-Δ vs idle baseline.
- Emit: highest-Δ joint; emit fire-now constituents; reserve src/tgt
  for wait>0 constituents.
- Wallclock estimate: 15 × ~30ms = 450ms. Fits in 600ms budget.

Open question (PI to weigh in pre-implementation): which baseline
for joint candidates? Same idle baseline (singles vs joints
non-comparable), best-single baseline (marginal lift), or sum-of-
singles baseline (interaction term only). See
`knowledge-base/questions/2026-05-17-joint-scoring-baseline.md`.

**2. Mine the leaderboard (~1-3 days, cheap).** Rule 22 fires at
plateau. We haven't pulled top-5 public notebooks since the
romantamrazov (LB μ=1224) reference. Rolling μ at ~1158 suggests
we're in the 5-15% percentile band, not top-1%. Pull replays from
top-LB submissions, run `attribute_fleets`, find the structural
gap. Cheap; informs every later direction.

**3. Watch 52754310 settle (~6h / 50 games).** If μ lands ≥1158
(matches/beats composite_a2), Direction B builds on a solid base.
If μ tanks <1100, revert trajectory→composite via 3-line edit and
re-submit composite_a2 as the rolling-pair floor. Rollback flag:
`knowledge-base/flags/2026-05-17-trajectory-default-4p-untested.md`.

## Falsified or dead this session

- N_VALIDATE=60 cap as the wallclock fix: cost 8pp winrate (37/64
  vs 42/64). Friction tag
  `validate-cap-too-tight-cost-winrate-not-just-wallclock`. Rule 40
  applies — restriction-tuning lost to letting safe_deadline bind.
- Replacing composite entirely with trajectory chooser (v1/v2/v3
  binary leaf): 0/32 × 3 = Rule 37 axis-saturation. Closed.

## What this session deliberately did NOT do

- **Did not implement Direction B or C.** Rule 37 already constrained
  this session's diff. Direction B is next session's first task.
- **Did not test trajectory chooser in 4P.** All A/B was 2P. Flag
  filed at `knowledge-base/flags/2026-05-17-trajectory-default-4p-
  untested.md`.
- **Did not modify proposer, value head, or emit logic.** The
  trajectory chooser slots in via two env-var setdefaults; the
  upstream/downstream pipeline is untouched.

## Pointers

- `agents/baseline/main.py` — entry, sets both env vars.
- `agents/baseline/chooser_trajectory.py` — trajectory chooser
  (extend here for Direction B).
- `agents/baseline/chooser.py` — composite chooser (REFERENCE for
  wait_N pattern at score_action:60-73 and validate budgeting at
  affordable_validate_cap:78).
- `agents/baseline/proposer.py` — multi-wait grid + banded dedup.
- `agents/baseline/value.py` — favor_hybrid dispatcher.
- `lib/trajectory.py` — predict_fleet_fate (SUN_SAFETY=0 post-fix).
- `lib/fast_sim.py`, `lib/world_model.py` — primitives. Do NOT
  rewrite.
- `audit/2026-05-17-trajectory-chooser-shipped.md` — what shipped
  this session (full receipt).
- `audit/2026-05-17-sun-safety-cushion-fix.md` — earlier bug fix
  for predict_fleet_fate's false-rejection cushion.
- `knowledge-base/concepts/probability-of-winning-framework.md` —
  Direction A/B/C framing.
- `knowledge-base/concepts/trajectory-first-architecture.md` —
  the architectural reframe doc.
- `knowledge-base/thoughts/2026-05-17-direction-b-joint-action-
  scoping.md` — concrete plan for next session.
- `state/current.md` — submitted-agent state (no μ values).

## Rule reminders

- Rule 1: submissions are single-shot, PI-approved. No retry loops.
- Rule 12: rolling-last-2 — third push evicts oldest. v20 just
  evicted; pair is now [composite_a2 52744856, trajectory 52754310].
- Rule 22: at every plateau, mine top-5 public notebooks.
- Rule 32: session-start `kaggle competitions submissions
  orbit-wars` is the source of truth for μ. State files do NOT
  record μ.
- Rule 37: 3-variant axis cap. v1/v2/v3 trajectory-as-replacement
  hit it; v4 succeeded by making trajectory chooser LOOK LIKE
  composite + smarter filter. Direction B is a DIFFERENT axis
  (joint vs single scoring).
- Rule 40: prefer modeling-correctness over restriction-tuning.
  Hit again this session — N_VALIDATE=60 cap was a band-aid; the
  modeling fix was "let safe_deadline bind."
