# Plan — marco-v3-3 opponent model + adversarial re-rank in the opening

> **Drafted:** 2026-06-02, branch `claude/game-theory-winning-strategy-SEU7P`.
> Source of truth for the next session. The plan is staged so a fresh
> session can start without needing the (ephemeral) `/tmp/kernels/`
> downloads from the previous session.

## What we are building, in one paragraph

The Kaggle public field contains a deterministic 5-move opening planner
that ~7-9 of the top-30 public agents share (marco-v3-3 lineage:
romantamrazov, ykhnkf, yuriygreben, agentzz, pascalledesma, debugendless,
and others). The planner is depth-5 beam-width-8, optimises capture-time,
and assumes the enemy plays straight-line single-fleet captures with no
coordination. We exploit this by porting the marco planner as our
**opponent model** for the opening window (step < 50, opponent owns ≤ 6
planets — the exact gate marco's own EAM uses), then running a one-ply
**adversarial re-rank** of our chooser's top-3 candidates against the
predicted opponent reply. The change is **default OFF behind env vars**,
adds no new physics primitives, and gates itself to the opening only —
worst case is no-op identical to the live champion.

## Live state we are extending

| Sub ID | Date (UTC) | Agent | μ | Role |
|---|---|---|---:|---|
| 53280733 | 2026-06-02 07:05 | `baseline_state_driven_k.py` | **1172.9** | Rolling pair (newest) — new live peak |
| 53277693 | 2026-06-02 05:18 | `baseline_launch_rules_universal.py` | 1076.7 | Rolling pair (older, evicted by our next submit) |

The next submit evicts the 1076.7 backstop. Rule 42 threshold:
**predicted-μ must clearly exceed 1076.7**. Local Wilson-lo ≥ 0.50 vs
the 1172.9 anchor approximates that.

Code path we extend (everything else stays byte-identical):
- `agents/baseline/main.py` (entry)
- `agents/baseline/chooser_trajectory.py` (decision)
- `agents/baseline/proposer.py` (candidate gen — read-only, no changes)
- `lib/opp_model.py` (Tier 0/1/2 exists; we add Tier 3)

What we do NOT touch: `lib/kinematic_table.py`, `agents/baseline/launch_rules.py`,
late-game logic, the proposer.

## Reference material (committed to this directory)

- `kernels/marco-dg-v3-3.py` — the lineage ancestor. Key sections to port:
  - `class Mission` line 209
  - `class ShotOption` line 197
  - `def target_value` line 1123 (FYI only — we don't port this; we have our own value)
  - `def _plan_best_launch` line 2213 — earliest-affordable launch + intercept (port)
  - `def _enemy_earliest_capture` line 2263 — FYI only, we don't port this (it's marco's model OF US)
  - `def _plan_evaluate` line 2296 — simulate a 5-deep plan, return V + moves (port)
  - Beam search body line 2346-2484 (port: depth=5, width=8, EAM constants at 2199-2208)
  - `EAM_OPENING_LIMIT=50`, `EAM_MAX_MY_PLANETS=6` line 2207-2208
- `kernels/marco-dg-v56.py` — the v56 upgrade. Useful for understanding what we are
  NOT replicating (the C-extension + true 2-ply IBR). Reference only.
- `kernels/romantamrazov-lb-max-1224.ipynb` — a confirmed marco-v3-3 fork.
  Use for phase-4 panel test (the marco-fork win-rate gate).
- `kernels/suntzu-launch-safety.ipynb` — reference for a separate idea
  (continuous launch-safety multiplier) we discussed; NOT in scope for this plan.

## Build phases

### Phase 1 — port the marco planner (2-3h)

**Deliverable:** `lib/opp_marco.py` exposing:

```python
def predict_marco_plan(world, opp_seat, time_budget_ms=30) -> list[Commit] | None:
    """Return the first-5 captures a marco-v3-3 fork would commit to from
    this observation, or None if planner inapplicable (step >= 50, opp owns
    > 6 planets, time budget blown, or planner found no feasible plan).

    Commit = (src_id, tgt_id, t_launch_relative, fleet_size, eta_relative).
    """
```

Port from `kernels/marco-dg-v3-3.py`:
- `_plan_best_launch` (intercept loop, fleet-size growth, ETA fixed point) — keep faithful
- `_plan_evaluate` (beam-step simulator) — keep faithful
- The beam loop itself — depth 5, width 8, sort by V

Substitute our primitives where the math is identical:
- `lib/orbit.predict_planet_position` for marco's orbital prediction (parity-tested already)
- `lib/geometry.dist` / `fleet_speed` / `is_static_planet` — same formulas

**Parity test (REQUIRED before continuing):**
1. Capture an observation from `fast.py play kernels/romantamrazov-lb-max-1224 --emit-trace`
2. Run `predict_marco_plan` on the romantamrazov-seat observation
3. Diff predicted launches vs actual launches in the trace for steps 0-15
4. **Gate: ≥ 80% match on the first 3 launches.** If < 60%, abort the plan
   and file the negative result.

### Phase 2 — wire Tier 3 into opp_model (1h)

`lib/opp_model.py` gains:

```python
def make_opp_policy(my_id, tier=3, fork="marco_v33"):
    # Tier 3: returns a policy that, given (snap, opp_seat), emits the
    # marco-predicted launches at their predicted t_launch, otherwise empty.
    # Cached per (episode_id, opp_seat). Falls through to Tier 1 if the
    # opening gate fails or the marco predict returned None.
```

Cache key prevents re-planning every rollout step. The marco port runs
**once per real game turn per opponent**, not per rollout step.

### Phase 3 — adversarial re-rank in the chooser (2h)

Add to `agents/baseline/chooser_trajectory.py`:

```python
def _adversarial_rerank_opening(snap_base, top3_candidates, opp_marco_plans,
                                me, num_seats, gamma, world, model):
    # For each of top-3 my-candidates:
    #   simulate forward H=25 ticks with:
    #     - my move = this candidate
    #     - opp moves = marco-predicted launches arriving in window
    #   score using existing leaf scorer
    # Return the candidate with the best adversarial-leaf-score.
```

Wire into `choose_trajectory`:
1. Build top-K shortlist via existing code (no change to ranking)
2. If `BASELINE_ADVERSARIAL_RERANK=1` AND step < 50 AND any opp owns ≤ 6
   planets AND `predict_marco_plan` returned non-None for at least one opp:
   run `_adversarial_rerank_opening` on top-3 and return its pick
3. Else: original behaviour (top-1 by leaf score)

The re-rank can only pick from the top-3 already generated; it cannot
introduce new candidates. Worst case: returns the same top-1, no-op.

### Phase 4 — local validation (Rule 43, 45, 46 — 2-3h)

All gates must pass before submit. None are optional.

**(a) Byte-parity with default OFF — Rule 46:**
```
python scripts/bundle_agent.py baseline
pytest tests/test_bundle.py
python fast.py play <bundled_submission>
```
All GREEN, byte-identical to live champion when env vars are 0.

**(b) n=32 A/B vs current champion — Rule 45:**
```
python fast.py eval baseline_with_marco \
    --vs agents/baseline (state_driven_k bundled) \
    --seeds 32 --crn --env BASELINE_OPP_MARCO=1 BASELINE_ADVERSARIAL_RERANK=1
```
Required: **Wilson-lo ≥ 0.50.**

**(c) n=32 vs a marco-fork — exploit verification:**
```
python fast.py eval baseline_with_marco \
    --vs audit/2026-06-02-marco-lineage-reference/kernels/romantamrazov-lb-max-1224 \
    --seeds 32 --crn
```
Required: **win-rate ≥ 0.55** (Rule 43 sub-gate — confirms the exploit fires).

**(d) Geometry panel — Rule 43:**
```
python fast.py eval baseline_with_marco \
    --vs baseline_state_driven_k --geometry-panel --by-archetype
```
Required: **no archetype regression > 10 pp.**

**(e) Wallclock — Rule 2:**
```
python fast.py play baseline_with_marco --vs v7_0 --seeds 5
```
Required: **max turn ≤ 950 ms** (env actTimeout=1000, 50 ms safety).
If we blow past 950 ms in the opening, abort and fall back to the
single-pre-game-prediction caching fallback (phase-1 risk #1).

### Phase 5 — submission (Rule 42 — 15 min)

If and only if all of (a)-(e) GREEN:
1. Bundle:
   `python scripts/bundle_agent.py baseline --env BASELINE_OPP_MARCO=1 BASELINE_ADVERSARIAL_RERANK=1`
2. Append claim row to `state/MULTI_BRANCH.md` push claim board:
   - branch: claude/game-theory-winning-strategy-SEU7P
   - candidate μ: predicted from local A/B (anchor: vs-champion Wilson-mu)
   - evicted: sub 53277693 baseline_launch_rules_universal, μ 1076.7
   - block-condition: predicted-μ < 1076.7 — STOP for PI sign-off
3. **ASK PI before pressing submit.** Show the predicted-μ, the
   evicted-μ, the panel/vs-champion numbers, and wait for explicit
   approval (Rule 12, Rule 42).
4. Submit only after PI says yes.
5. Update `state/MULTI_BRANCH.md` with new sub_id and pending μ.

### Phase 6 — wrap (30 min)

Rule 36: log thought in `knowledge-base/thoughts/`, questions in
`knowledge-base/questions/`, frictions in `audit/friction.md`.

## Risks and abort triggers

1. **Marco port too slow.** Phase-4-(e) any turn > 950 ms → fallback to
   pre-game one-shot prediction caching (predict once at game start, never
   re-run). If still > 950 ms, abort.
2. **Marco port not faithful.** Phase-1 parity test < 60% match → don't ship.
   File negative result, move on.
3. **Re-rank picks worse candidate vs non-marco opponents.** Phase-4-(b)
   Wilson-lo < 0.50 → tighten the activation gate to opp-detected-as-marco
   only (a few-shot fingerprint check on the opp's first 3 launches).
4. **Few marco-forks on the live ladder.** Tournament results don't
   guarantee ladder population. If exploit fires too rarely to move μ,
   the re-rank infrastructure still pays forward for Tier-2 logistic opp-model.
5. **Rule 37 (axis cap).** This is axis attempt #1 for opp-model +
   adversarial re-rank. If v1 fails, v2 must be a different opening-exploit
   axis (swarm tactic, deception layer) — not another tuning of this one.

## Pre-flight (Rule 16)

- Q1 already explored? No — opp-model + adversarial re-rank is a new axis
  per `state/MULTI_BRANCH.md` closed-tracks list.
- Q2 rank-lock-vulnerable? Mildly. Gating (env-var + default OFF + opening-only)
  keeps the live champion byte-identical when flags off.
- Q3-Q5 prediction: +30-60 μ if exploit fires for ~25% of ladder games;
  precedent is marco-v56's own adversarial 2-ply re-rank gained ~20 pp wr
  over v54 in the public 78-agent round-robin.
- Q6 metric alignment: yes — we're optimising chooser pick-quality against
  a specific opponent distribution, which is what TrueSkill measures.

## Effort estimate

~8-10 hours, fits in one focused session.

| Phase | Time |
|---|---|
| 1 — port + parity | 2-3 h |
| 2 — Tier 3 wiring | 1 h |
| 3 — rerank + gating | 2 h |
| 4 — local validation | 2-3 h |
| 5 — bundle + submit | 15 min |
| 6 — wrap | 30 min |
