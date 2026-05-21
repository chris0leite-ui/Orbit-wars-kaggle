# 2026-05-22 — session summary

Branch: `claude/review-skills-improvements-moKOR`
Duration: ~2 hours
Result: **no significant A/B lift produced**, but two real findings.

## Live ladder (settled 15:50 UTC)

| Sub ID | Agent | μ | Position |
|---|---|---:|---|
| 52894340 | _phase4_step1_FND (sibling) | 940.8 | Rolling (recent) |
| 52893236 | baseline_full (this branch) | 1079.2 | Rolling (older) |
| 52882014 | consolidated (this branch) | 1124.0 | EVICTED — best on branch |
| 52744856 | composite_a2_hybrid (team peak) | 1149.2 | EVICTED — peak |

baseline_full settled ~45 μ below consolidated. The four stacked
features in baseline_full (orbital_safety + stagnant_drain +
combat_stack + sniper) are collectively net-harmful.

The sibling-branch `_phase4_step1_FND` push (analytical LP endgame
predicate; described as 8/0 vs consolidated locally) settled at 940.8
on the live ladder — **almost certainly a victim of the same env-var
leak diagnosed below** when run in fast.py's in-process A/B.

## Finding 1 — test-infra env-var leak (BLOCKING)

Documented at `knowledge-base/thoughts/2026-05-22-test-infra-env-var-leak.md`.

**Mechanism:** `agents/baseline_*/main.py` shims and most
`submissions/*.py` bundles use `os.environ.setdefault(KEY, VALUE)` at
module load. `agents/baseline/main.py` constants
(`STAGNANT_DRAIN_ENABLED`, `OPENING_MILP_ENABLED`,
`JOINT_TOP_K_PER_TARGET`, etc.) are evaluated at import time and
cached as module-level globals.

When fast.py loads two agents in the same Python worker:
- First-loaded agent's `setdefault` sets env keys.
- Second-loaded agent's `setdefault` is a no-op for already-set keys.
- Both agents' bundles read from the SAME polluted env.

**Verified empirically:**
```python
m1 = load('variant_topk8_locked', ...)  # would set TOP_K=8 via env
m2 = load('consolidated', ...)           # setdefault TOP_K=5 → no-op
# m2 reads env, gets TOP_K=8 (WRONG — should be 5)
```

**Consequence:** Variant-vs-variant local A/B tests in fast.py
that share env-gated constants produce INVALID results — both
agents end up with the same effective config. This likely explains
why the previous session's small-n ablations of drain / orbital /
sniper produced numbers inconsistent with the live ladder, AND why
the sibling branch's "8/0 vs consolidated" claim crashed to μ=940
on the live ladder.

**Workaround prepared (not testable in budget this session):**
Six hardcoded variant bundles in `submissions/variant_*.py` with
inline constants instead of env reads:
- `variant_topk5_locked.py` / `variant_topk8_locked.py`
- `variant_milp_on.py` / `variant_milp_off.py`
- `variant_wallclock800.py` / `variant_wallclock600.py`

These are immune to env pollution because constants live in each
bundle's unique `sys.modules` namespace.

## Finding 2 — topk8 (MAX_PAIRS=100) is too slow for fast.py

n=8 (8 games at 4 workers) ran 17 min wallclock without completing.
Per-worker CPU profile showed 5-7 min CPU per game for the topk8
variant — 4x slower than the consolidated bundle. Root cause is
the chooser scoring more JOINT pairs (TOP_K=8, MAX_PAIRS=100 vs
5/60).

**Implication:** the JOINT axis can't be cheaply A/B'd at n=8 in
fast.py given the current 1000ms-actTimeout wallclock_budget=600ms
config. A test would need either:
- (a) Smaller TOP_K (6 or 7) to keep pair count manageable
- (b) Raise wallclock_budget AND use shorter rollouts
- (c) Use the geometry-panel adaptive harness with timeout-resistant
  early-stop

## Finding 3 — chooser family saturation (carryover)

Per `knowledge-base/thoughts/2026-05-16-chooser-family-saturation.md`,
the v9-trajectory chooser family hits μ ≈ 1120 ceiling.
Consolidated (μ=1124) sits ON this ceiling. Team peak μ=1149 was
+25 μ above ceiling and within ladder noise (σ~25-50).

**Implication:** TOP_K expansion within the same chooser framework
is unlikely to break through the ceiling. To produce a real lift
requires a structural change — different chooser entirely (MCTS,
beam), different value head architecture, or proper opponent
modeling (Tier 2 placeholder is unimplemented).

## What was prepared this session

- `scripts/clean_ab.py` — subprocess-per-game harness (helps for
  fresh shell env, but the within-game env leak remains).
- `submissions/variant_{topk5,topk8}_locked.py` — hardcoded JOINT
  TOP_K=5/8 and MAX_PAIRS=60/100.
- `submissions/variant_milp_{on,off}.py` — hardcoded
  OPENING_MILP_ENABLED=True/False.
- `submissions/variant_wallclock{600,800}.py` — hardcoded
  WALLCLOCK_BUDGET_MS=600.0/800.0.
- `submissions/variant_milp_opening.py` (deprecated; superseded by
  variant_milp_on.py).
- `agents/baseline_joint_aggr_consolidated_topk8/main.py` — shim
  (supplanted by the locked bundle).

## What I'd do next session

1. **Bundle parity verify.** Run the parity-gate on each
   `variant_*_locked.py` to confirm bundled behavior matches source.
2. **Run topk8 A/B with smaller wallclock-budget knob.** Build
   `variant_topk8_locked_wb400.py` (TOP_K=8, MAX_PAIRS=80,
   WALLCLOCK_BUDGET_MS=400.0). 80 pairs at 400ms is still > 5 per
   pair compute budget; tighter timing risk but probes the axis.
3. **MILP opening A/B is feasible.** `variant_milp_on` vs
   `variant_milp_off` at n=8 should run within fast.py budget
   (MILP only fires at step<30, doesn't touch chooser hot path).
4. **Konbu17 shot-validator MLP** (H14 from
   `knowledge-base/concepts/top-performer-strategies.md`).
   ~1-week build but +5pp panel lift in prior eval, only ML attack
   with empirical precedent. Best EV per build-week.

## Pre-submit gate (Rule 42 / 43 / 45 — NOT cleared)

No submissions attempted this session. Live `baseline_full` push
from the previous session is settling and will inform the next
submit decision.
