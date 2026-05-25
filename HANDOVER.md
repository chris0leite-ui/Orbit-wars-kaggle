# HANDOVER.md — next-session brief

> Last written: 2026-05-25 12:00 UTC by `claude/agent-design-exploration-Q0q9T`.
> K1 (kinematic_table) + Layer Z v2 shipped as sub 53018599 (PENDING).
> Three proposer-tightening axis falsifications → Rule 37 cap reached;
> next axis is chooser-side. Older sections archived to
> `audit/archive-2026-05-24-handover.md`.

## Read order (Rule 44 — mandatory)

1. **`state/MULTI_BRANCH.md`** — live rolling pair, push claim board, closed tracks (includes today's proposer-tightening closure).
2. **`state/TOOLS.md`** — A/B harnesses, diagnostics, bundle/validation.
3. **`CLAUDE.md`** — rules 1-48.
4. **This file.**
5. `audit/2026-05-25-postmortem-k1-zv2-axis-exhaustion.md` if you're about to touch the proposer pre-filter chain.

## Where we are (2026-05-25 12:00 UTC)

- **Comp:** Orbit Wars. Deadline 2026-06-23 23:59 UTC. **~29 days remain.**
- **Rolling pair (auto-kept by Kaggle):**
  - **53018599** (K1+Z v2 bundle, commit 603f45f, 2026-05-25 11:54). PENDING; predicted μ-band 1100-1180. **Do NOT re-poll <4h after submit (Rule 48).**
  - **53013786** (`baseline_joint_aggr_consolidated_orbitfix` RESUBMIT, sibling ESwSv, 2026-05-25 08:40) — μ=**1120.1** (adapting). Strong half.
- **Just evicted by sub 53018599:** sub 53001857 (baseline_wave v3.1, μ=1126.8).
- **Team peak ever:** μ=1149.2 (sub 52744856, `composite_a2_hybrid`). Treat as break-glass reserve.
- **Daily submits today (2026-05-25 UTC):** 1/5 used. 4 remaining. **HOLD until 53018599 settles.**

## Today's progress (2026-05-25)

**Session arc:** diagnose seed-1622482326 BUILDUP misprice → ship surgical fix → discover wallclock breach → wire pre-built kinematic_table cache → A/B parity-band but budget-safe → bug-fix Layer Z (`pred_ships` subtraction) → A/B +11pp lift → submit → chase live-game cheap-recapture diagnosis (seed-2020490432) → over-restrict and revert.

**What shipped (sub 53018599):**

- **K1** (commit `c0035ff`) — wires `lib/kinematic_table.py` (already extracted to origin/main; never wired). Two file edits: `agents/baseline/main.py` + `agents/buildup_planner/main.py` add `begin_turn(world)` priming + `KINEMATIC_TABLE_ENABLED=1` setdefault. Bit-parity by construction (21/21 parity tests + 564 byte-identical FleetFate assertions from sibling-branch `c48e143`). **Production p95 1101ms → 893ms; predict_relative cumtime -90%.** Max 651ms → 718ms still under the 1000ms Kaggle budget.
- **Z v2** (commit `603f45f`) — fixes `effective_landing` formula in `agents/baseline/proposer.py:516`. V1 was `ships - prod·eta`; v2 is `(ships - pred_ships) - prod·eta`. Dropped the redundant opening_planner site (`needed` already encodes regrowth via `gar_at_arr`). **n=64 A/B vs phi1_only: 56.2% (+10.9pp over K1-alone)**, Wilson [0.441, 0.677]. PI override on Rule 45's Wlo<0.50 accepted given the directional signal + bit-parity K1 below.

**What landed and got reverted (failed proposer-tightening axis):**

- **Cache-the-MILP** (commit `9870575` → reverted `52e771c`). Hypothesis: opening_plan was 1.7s/call. Reality: it was ~5ms/call. Caching regressed behavior 34% → 9% because the re-derive was a strategic feature (mid-opening adaptation), not a wallclock bug. Profile FIRST next time.
- **Fix A + Fix B** (commit `03cb25b` → Fix B reverted `e277c53` → Fix A reverted `6627420`). Triggered by user's seed-2020490432 game showing cheap recapture. Fix A: opp-model floor 10→5. Fix B: holdability floor 20→5. New 5×250×no-swap A/B: 80% vs phi1_only (replicates root-cause diagnosis), **0-20% vs joint_aggr** (the live strong half). Three consecutive proposer-tightening falsifications = Rule 37 cap.

**New tooling:**

- `scripts/ab_quick.py` (commit `0f2a23d`) — **new A/B standard per PI: 5 games × 250-step cap × no seat swap × 3-opp panel (phi1_only + joint_aggr + v7_0)**. ~30 min single-worker. Triage signal at n=5 (point estimate + direction); not falsification (Wilson CI ±35pp).

## Falsified or dead today

- **Proposer pre-filter tightening axis (vs joint_aggr).** Rule 37 cap. Z v2 parity at n=64; Fix A+B 20% at n=5; Fix A alone 0% at n=5. Pattern: wins vs quiet opp, loses vs aggressive opp. Future filter changes must be A/B'd vs joint_aggr BEFORE shipping. Closed in `state/MULTI_BRANCH.md`.
- **Commit-and-execute MILP caching.** Re-derive each turn is a feature, not a bug. Confirmed by cProfile (`predict_relative` was the real hot path, not opening_plan).
- **HANDOVER P2's literal `d/√n` recipe.** Env `lib/fleet.py:speed` uses `(log n / log 1000)^1.5`, not √n. `aim_and_eta` already passed real `fleet_speed(ships)` eta. The real Z v2 bug was the `pred_ships` omission, not the eta formula.

## Next-session first actions (ranked)

### Priority 0 — Read sub 53018599's settled μ

Wait ≥4h after 11:54 UTC submit. Then `kaggle competitions submissions orbit-wars`:
- **μ ≥ 1130:** K1+Z v2 confirmed; pivot to P1.
- **μ 1080-1130:** parity-band on the ladder; un-stack Z v2 and A/B in isolation at n=32 vs joint_aggr (NOT phi1_only — the n=64 vs phi1_only was a misleading panel).
- **μ < 1080:** Z v2 was the regression source. Revert Z v2 too; re-submit K1-only bundle. K1 is bit-parity safe.

### Priority 1 — Replay-scout (deferred from HANDOVER 5/24 P0)

Pull 5-10 top-50 ladder replays. Catalog opening + midgame patterns. Specifically check the asymmetry the seed-2020490432 screenshot showed: how do top agents handle "opp closer to the midline neutral than we are"? Do they avoid the capture, contest at fewer-ships, or always counter-launch? Until this is answered, every chooser-side fix is a guess.

### Priority 2 — Better rollout opp model (chooser-side, not proposer-side)

The cheap-recapture diagnosis was REAL (80% vs phi1_only confirms). But the proposer-pre-filter axis is closed. The right fix is structurally better opp model inside `score_candidate_v4`'s rollout. Today's evidence: `lib.opp_model.lite_greedy_policy` skips 5-9 ship launches (10-ship floor); lowering to 5 over-launches and regresses vs joint_aggr. The fix isn't a threshold — it's a different model. Candidates:
- Priority-based projection (opp targets the highest-prod planet in their nearest-K, regardless of ship count).
- Behavior-cloned policy from Priority 1's scouted replays.
- Cap rollout opp launches per turn at K=2 (avoid the over-launch failure mode of Fix A).

### Priority 3 — Z v2 isolation A/B vs joint_aggr (n=32)

Whether sub 53018599 settles strong or weak, the n=64 vs phi1_only result is suspect (Wilson CI [0.441, 0.677] includes parity widely). Run n=32 A/B with Z v2 ON vs OFF, both vs joint_aggr. If lift survives, Z v2 stays. If not, revert and only K1 ships in next iteration.

### Priority 4 — Φ refactor Stages 2-5

Same priority as before (HANDOVER 5/24 P5). Plan: `/root/.claude/plans/go-also-checknfor-similar-purring-flute.md`. Don't start until P0-P3 resolved.

## Submission discipline

Per HANDOVER 5/24: submits require **n=32 Wilson-lo ≥ 0.55** OR clear defensive purpose. Today's sub 53018599 was n=64 Wilson-lo 0.441 + PI override — borderline. **Slow is smooth.**

For the new 5×250×no-swap standard: **n=5 is triage only**. Single-opp wins of 4/5 or 0/5 have Wilson CIs ±35pp; never sole submit gate.

## Pointers

- `state/MULTI_BRANCH.md` — live state, push claim board, closed tracks (today's proposer-tightening closure added).
- `state/TOOLS.md` — A/B harnesses, bundle/validation.
- `scripts/ab_quick.py` — new 5-game/250-step/no-swap A/B standard.
- `/root/.claude/plans/go-also-checknfor-similar-purring-flute.md` — full Φ refactor spec.
- `audit/2026-05-25-consolidation-profile.md` — pre-K1 cProfile (predict_relative 84s cumtime / 219 turns).
- `audit/2026-05-25-consolidation-review.md` — K1 finding + Rule-44 cross-reference.
- `audit/2026-05-25-consolidation-profile-post-K1.md` — post-K1 cProfile (-52% p95, -90% predict_relative cumtime).
- `audit/2026-05-25-postmortem-k1-zv2-axis-exhaustion.md` — full session postmortem + promotion candidates.
- `knowledge-base/thoughts/2026-05-25-k1-zv2-ship-and-axis-exhaust.md` — narrative arc + open questions.

## Rule reminders (most relevant)

- **Rule 37:** consecutive-falsification cap. Today closed the proposer pre-filter axis after 3 attempts. Next move pivots the axis (chooser-side opp model), not iterates it.
- **Rule 40:** prefer modeling-correctness over restriction-tuning. The two reverts today (Fix A+B) were tuning constants; the right next move is a model change.
- **Rule 44:** state-of-truth read. Today found a parity-gated optimization (kinematic_table) wired on sibling branches but never our agent. Always cross-reference siblings before assuming a feature is missing.
- **Rule 45:** n=5 panel = triage; n=16 = falsification; n=32 = submit gate. Today's revert chain was driven by n=5 column results — informative but not dispositive on their own.
- **Rule 48:** Kaggle μ adapts over 4-12h. Do not panic-poll sub 53018599 before then.
