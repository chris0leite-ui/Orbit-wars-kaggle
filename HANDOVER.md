# HANDOVER.md — next-session brief

> Last written: 2026-05-24 22:50 UTC by `claude/agent-design-exploration-Q0q9T`.
> Two submits today + four-layer refactor shipped (default-off) + full
> ablation matrix learned. Older sections archived to
> `audit/archive-2026-05-24-handover.md`.

## Read order (Rule 44 — mandatory)

1. **`state/MULTI_BRANCH.md`** — live rolling pair, push claim board.
2. **`state/TOOLS.md`** — A/B harnesses, diagnostics, bundle/validation.
3. **`CLAUDE.md`** — rules 1-48 (Rule 48 added 2026-05-24: Kaggle scores
   adapt over hours; never panic-react to an early μ poll).
4. **This file.**
5. `audit/friction.md` if you're about to touch a fragile path.

## Where we are (2026-05-24 22:50 UTC)

- **Comp:** Orbit Wars. Deadline 2026-06-23 23:59 UTC. **~30 days remain.**
- **Rolling pair (auto-kept by Kaggle):**
  - **53000996** (Phi-1 leaf only, 2026-05-24 22:38) — just submitted.
    `BASELINE_VALUE_HEAD=phi` hard-set in `agents/buildup_planner/main.py`
    so it wins over baseline's bundle-inlined setdefault to "hybrid".
    Local n=8 vs sub 52968889 lineage: **4/8=50% parity**, Wilson [0.22, 0.79].
    Adapting from μ=600 — **do NOT re-poll until ≥4-12h elapsed (Rule 48).**
  - **52993021** (concentration A+B, 2026-05-24 16:10) — at last poll
    (22:30) **μ=1116.2 and still climbing**. Confirms Rule 48 (early poll
    at 600 was the start value, not regression).
- **Just evicted by sub 53000996:** sub 52968889 μ=1144.5 (the strong half).
- **Team peak ever:** μ=1149.2 (sub 52744856, `composite_a2_hybrid`,
  evicted long ago). Architecture preserved in `agents/baseline/value.py`
  (favor_composite + 4P A2). Treat as **break-glass defensive reserve**.
- **Daily submits:** 2/5 used today. 3 remaining.

## Day-N PM agent-design-exploration-Q0q9T (2026-05-24)

**Session arc:** plan the holistic Φ refactor → ship four layers (default-off)
→ A/B all-on (catastrophic 0/16) → ablate → ship Phi-1 alone (parity).

**What landed (4 layers, each default-off, each independently A/B-able):**

- **Layer Z** (commit `109c01a`) — effective-landing prune in proposer +
  opening_planner. `BASELINE_EFFECTIVE_LANDING_PRUNE=1`. Filters
  candidates with `ships - prod·eta < SAFETY_MARGIN`.
- **Layer R** (commit `9117474`) — reliability multiplier in
  `lib/reliability.py`. `BASELINE_RELIABILITY_PRICING=1`. `risk_adjusted
  = nominal × eta_rel × wait_rel × landing_rel`.
- **Layer D** (commit `ac80723`) — plan-level drop-one validator in
  `lib/drop_one.py`. `BASELINE_DROP_ONE_VALIDATE=1`. Leave-one-out over
  the chooser's emit list; closed-form plan-value via `predict_fleet_fate`.
- **Phi-1** (commit `1f020e1`) — `favor_phi` leaf in
  `agents/baseline/value.py`. `BASELINE_VALUE_HEAD=phi`. Adds 2P
  elimination bonus + 250-tick pv_horizon.
- **Submit chore** (commit `3a3fa10`) — bundler-aware hard-set of
  `BASELINE_VALUE_HEAD=phi` in `agents/buildup_planner/main.py`.
  See "Bundler trap" below.

**Full ablation matrix at n=8 vs sub 52968889 lineage (250-step cap):**

| Variant | Wins/n | Winrate | Notes |
|---|---|---|---|
| All 4 ON | 0/16 | **0%** | catastrophic compound interaction |
| Z+R+Φ (no D) | 2/8 | 25% | D alone contributes -25 pp |
| Z only | 3/8 | 37.5% | mild regression alone |
| R only | 3/8 | 37.5% | mild regression alone |
| **Phi-1 only** | **4/8** | **50%** | **the only parity result — shipped** |

**Falsified / dead-ends:**

- **Drop-one as currently implemented.** Closed-form plan-value via
  `predict_fleet_fate(src, src, ...)` misclassifies many captures →
  plan_value=0 → marginal=0 → everything pruned → agent goes idle.
  Trace confirmed: at turn 5 of seed 1, phi agent emitted 1 candidate
  vs baseline's continuous expansion.
- **The 4-layer compound regression.** Strong negative interaction
  not predicted by individual-layer math. Operational consequence:
  **strict one-layer-at-a-time discipline going forward.** Per Rule 37
  + today's evidence, never stack independent-looking changes without
  per-layer verification.

**Bundler trap (worth documenting):** `agents/buildup_planner/main.py`
runs `os.environ.setdefault("X", ...)` BEFORE the baseline import in
source order, so my setdefault would "win" in plain Python. But the
bundler inlines `agents/baseline/main.py` FIRST (dependency order),
so baseline's own setdefault to "hybrid" runs first in the bundle's
top-down execution and any later setdefault is a no-op. Fix: use
HARD assignment `os.environ["BASELINE_VALUE_HEAD"] = "phi"` so it
overrides regardless of bundle order. Same pattern applies to any
future buildup_planner-specific env var that conflicts with a
baseline setdefault.

## Next-session first actions (strategist's order)

### Priority 0 — Reconnaissance (one afternoon, zero submits, highest leverage)

We've been *building* without *scouting*. Pull 5-10 replays of top-50
agents on the live ladder and read their move sequences:
- Wave-strike or trickle?
- Concentration vs spread?
- Fleet size distribution? eta distribution?
- Heuristics or RL?

Until we know what beats us, every layer redesign is a guess.

### Priority 1 — Settle assessment, then defensive triage

Re-poll subs 53000996 and 52993021 **≥4h after submit time** (Rule 48).
Categorize:
- **Both ≥1100:** floor holding; proceed to Priority 3.
- **Either <1000:** floor breached; rebundle the `composite_a2_hybrid`
  lineage (sub 52744856 μ=1149) as a recovery push. Code is already
  on this branch in `agents/baseline/value.py` (favor_composite +
  4P-A2). Treat this as the break-glass reserve, not for experiments.

### Priority 2 — Layer Z v2 (proposer-only, fleet-speed-aware formula)

The current Z formula `n - prod·eta < MARGIN` is too crude. The
fleet-speed-aware version is `n - prod·d/√n < MARGIN` (since
eta = d/√n for `n>4`). Apply this only in the proposer; drop the
opening_planner site (where `needed` already encodes regrowth via
`gar_at_arr`). ~30 LOC; smallest, safest, most-principled.

### Priority 3 — Layer R v2 (reliability with floor)

`reliability = max(0.5, eta_rel × wait_rel × landing_rel)`. The floor
prevents value-collapse (the v1 failure mode). Same env var
`BASELINE_RELIABILITY_PRICING=1`, just clamp the output.

### Priority 4 — Layer D v2 (replace predict_fleet_fate plan-value)

The current `plan_production_advantage` uses
`predict_fleet_fate(src, src, ...)` which has edge cases. Replace with
`aim_and_eta`-based target lookup (the proposer's own primitive).
Bigger redesign than Z/R v2; queue after those land.

### Priority 5 — Φ refactor Stages 2-5

MILP value migration, proposer migration, mission migration, retire
constants. Big diff, only after Phi-1 confirmed at ladder. Plan file:
`/root/.claude/plans/go-also-checknfor-similar-purring-flute.md`.

## Submission discipline (hard lesson from this session)

**Don't burn daily submits on speculative probes.** Today's two submits
were uncertain (concentration bimodal n=8, Phi-1 parity n=8). Future
submits require either:
- **n=32 serial Wilson-lo ≥ 0.55** (lift claim, per Rule 45), OR
- **A clear defensive purpose** (rebundle a known-good).

That gates submits to ~3-5 per week, not per day.
**Slow is smooth; smooth is fast.**

## Pointers

- `state/MULTI_BRANCH.md` — live state, push claim board.
- `state/TOOLS.md` — A/B harnesses, bundle/validation.
- `/root/.claude/plans/go-also-checknfor-similar-purring-flute.md` —
  full Φ refactor specification, 4-layer architecture.
- `audit/archive-2026-05-24-handover.md` — older Day-N PM sections.

## Rule reminders (most relevant this session)

- **Rule 12:** rolling-last-2; today we evicted the strong half twice.
- **Rule 37:** consecutive-falsification cap. The 4-layer combo +
  3 individual ablations all failed — that's 4 axis falsifications
  in one session. Next session must pivot the axis, not iterate it.
- **Rule 40:** prefer modeling-correctness over restriction-tuning.
- **Rule 45:** n=16 with bimodal seed pattern is INCONCLUSIVE, not
  parity. n=8 has Wilson CI ±20-25 pp — never sole submit gate.
- **Rule 48:** Kaggle μ is adaptive; sub 52993021 confirmed (600 → 1116
  in 6h). Never interpret early poll as settled.
