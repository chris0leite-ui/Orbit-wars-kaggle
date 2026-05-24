# HANDOVER.md — next-session brief

> Last written: 2026-05-24 17:00 UTC by `claude/agent-design-exploration-Q0q9T`.
> Wrap-up after submitting the concentration A+B variant as sub
> 52993021. Older sections (2026-05-20, 2026-05-22) archived to
> `audit/archive-2026-05-24-handover.md`.

## Read order (Rule 44 — mandatory)

1. **`state/MULTI_BRANCH.md`** — live rolling pair, push claim board.
2. **`state/TOOLS.md`** — A/B harnesses, diagnostics, bundle/validation.
3. **`CLAUDE.md`** — rules 1-48 (Rule 48 added 2026-05-24: Kaggle
   scores adapt over hours; never interpret early μ as settled).
4. **This file.**
5. `audit/friction.md` if you're about to touch a fragile path.

## Where we are (2026-05-24 17:00 UTC)

- **Comp:** Orbit Wars. Deadline 2026-06-23 23:59 UTC. **~30 days remain.**
- **Rolling pair (auto-kept by Kaggle):**
  - **52993021** (just submitted, 2026-05-24 16:10) — concentration
    A+B (α=1.5, C_open=1.0, C_prop=0.05; commit `2878bfd`). Local
    A/B at 250-step n=16 was bimodal 8/16 parity vs sub 52968889;
    100% winrate / 81% elim-rate vs simpler v3.5.1. **μ is adapting
    (Rule 48). First poll at +15 min showed 600.0 — that's the start
    value, not a settled score. Re-poll at session start.**
  - **52968889** (2026-05-23 23:59) — buildup_planner pre-fix
    (commit `eb1653a` bundler-trailer fix, otherwise current
    production lineage). μ=1144.5 stable.
- **Team peak (evicted long ago):** μ=1149.2 (sub 52744856,
  `composite_a2_hybrid`).
- **Today's submit budget:** 5/day. Used: 1 (sub 52993021). 4 remaining.

## Day-N PM agent-design-exploration-Q0q9T (2026-05-24)

**Session arc:** bundler-ERROR diagnosis → opening-MILP rotation-aware
ETA fix → Tier-1 concentration patch → submit → wrap-up.

**Commits (latest first):**

- `6ae9958` — chore: submit + handover (sub 52993021 push claim + Φ
  refactor queue).
- `2878bfd` — Tier-1 A+B concentration. All four env vars default no-op;
  active variant ran at α=1.5, C_open=1.0, C_prop=0.05. Plus
  `fast.py --episode-steps` plumbing.
- `564fbc9` — two-call rotation-aware ETA in `_expected_hold_duration`.
  Stage 2A race uses `arrival_eta=0`; Stage 2B hold-scaling gated on
  `BASELINE_ORBITAL_SAFETY=1` uses `arrival_eta=arrival`.
- `f590c22` — revert of broken `24111ac` (universal `arrival_eta=arrival`
  no-op'd the race-to-planet gate).
- `eb1653a` — bundler trailing-entrypoint wrapper (sub 52968305 ERROR
  root cause; permanent fix).

**Falsified / dead-ends this session:**

- `24111ac` arrival_eta=arrival as the universal fix: silently no-op'd
  the race-to-planet gate. Documented in commit `f590c22` body.
- Local n=16 with bimodal seed pattern: per Rule 45 a bimodal split
  (focal wins all 4 P0/P1 seats on seeds 1,2,3,5,6; opp wins all 4 on
  0,4,7) is seed-dependent noise with Wilson CI ±200 μ trivially.
  Next session should treat bimodal A/B as INCONCLUSIVE, not parity.

**New Rule 48 (added this session):** Kaggle submission scores are NOT
settled when first observed. Every submission starts at μ=600.0 and
adapts upward over hours-to-days. Do not panic-react to an early poll.

## Next-session first actions (ranked by EV / cost)

### Priority 0 — Re-poll sub 52993021 (≤2 min)

```
KAGGLE_API_TOKEN="$KAGGLE_KEY" kaggle competitions submissions orbit-wars | head -3
```

Read the actual settled μ. Then choose the next priority based on it:
- μ ≥ 1100: concentration broadly parity-or-lift; proceed to Priority 1
  (Φ refactor). The submit was a success.
- 900 ≤ μ < 1100: concentration is parity-with-noise; queue α=1.2
  ablation as a tiebreaker before committing to Φ.
- μ < 900: concentration regressed live (despite local A/B looking ok).
  Write `audit/2026-05-24-postmortem-concentration-AB.md`; queue a
  rebundle of sub 52968889 as a recovery push only if floor is at risk.

### Priority 1 — Effective-landing prune (small-fleet long-haul waste)

PI observation during wrap-up (2026-05-24 17:30): the agent still
emits **size-2 fleets traveling long distances**. That's pure waste —
2-ship speed is floored at 2 (fleet_speed = `max(2, √n)`), so 2-ship
fleets at distance 50 take 25 turns, during which opp regrows ~75
ships. Effective landing is negative; the launch contributes nothing.

**Closed-form prune (1-line per candidate):**
- `eta = d / max(2, sqrt(n))`
- `effective_landing(n, d) = n - prod_target · eta`
- Reject candidate iff `effective_landing < SAFETY_MARGIN` (e.g. 1).

For neutral targets (prod=0), tiny short-haul launches still pass
(eta·0 = 0 bleed). For high-prod or long-haul, the prune kicks in
naturally.

**Files to touch:**
- `agents/baseline/proposer.py` near `enumerate_ship_counts` (line
  236 onward) and inside `cheap_marginal_value` — drop candidates with
  `effective_landing < 1`.
- `lib/joint_solver/opening_planner.py` — pre-filter in
  `_build_candidates` before MILP setup.

**Env var:** `BASELINE_EFFECTIVE_LANDING_PRUNE=1` (default on once
A/B confirms — this is a Rule 40 modeling-correctness fix, not
restriction-tuning).

**Why ship this BEFORE the Φ refactor:** small change (~30 LOC), pure
prune (no new value math), directly addresses an observed waste pattern.
Likely net-positive lift even if Φ refactor is delayed.

### Priority 2 — Holistic Φ refactor, Stage 1 (leaf `favor_phi` + 2P elim bonus)

**Plan file:** `/root/.claude/plans/go-also-checknfor-similar-purring-flute.md`
(full 5-stage spec).

**Goal:** replace the four disjoint approximations (opening MILP value,
proposer cheap-delta, `favor`/`composite`, finisher special case) with
one unified function `delta_phi(action)` derived from the discounted
production-advantage integral

  `Φ(s,t) = Σ_{τ≥t} γ^(τ-t)·(P_my − P_opp) + B·𝟙{opp eliminated}`.

**Key leverage point:** `chooser_trajectory.score_candidate_v4` is
already in Δ-form (`favor(leaf) − favor(baseline)`), so swapping `favor`
→ `favor_phi` propagates ΔΦ through the entire rollout chooser
automatically — no chooser-side edits needed.

**Stage 1 files (only):**
- NEW: `lib/value_heads/phi.py` (~200 lines).
- NEW: `tests/test_value_head_phi.py` (~120 lines, 5 oracle cases).
- EDIT: `agents/baseline/value.py` (+~40 lines: `favor_phi` +
  `select_favor_fn` route via `BASELINE_VALUE_HEAD=phi`).

**Env vars (all default no-op):** `PHI_HORIZON=250`, `PHI_GAMMA=0.99`,
`PHI_ELIM_BONUS=300`.

**Key gap Stage 1 closes:** the current 2P leaf returns ZERO elimination
bonus (4P has `ELIMINATION_BONUS=55` at value.py:99 but 2P branch
doesn't reach it). Team peak μ=1149 (sub 52744856) ran with the
composite head's 2P-aware capture mechanic; Φ closes that gap without
the composite head's PV-augmentation fragility.

### Priority 3 — Φ refactor Stages 2-5

Once Stage 1 ships and clears A/B (n=16 at 250-step + 500-step both
≥9/16): MILP value (Stage 2), proposer cheap-delta (Stage 3), mission
formulas (Stage 4), retire `OPP_BONUS=1.10` / `OPENING_VALUE_GAMMA=0.95`
/ A+B env vars / dedicated finisher / `drain_*` mechanisms (Stage 5).

### Priority 4 — Composite head + A2 restoration (fallback)

If Φ Stage 1 fails or regresses, rebundle sub 52744856's lineage
(`composite_a2_hybrid`, μ=1149). Code already imported per the
2026-05-20 audit notes.

### Priority 5 — `used_tgts` lock removal + JOINT cap expansion

Demoted from prior Priority 1. Φ subsumes the multi-source coordination
question; skip unless Φ Stage 1 fails.

## Pointers

- `state/MULTI_BRANCH.md` — live state, push claim board.
- `state/TOOLS.md` — A/B harnesses, bundle/validation.
- `/root/.claude/plans/go-also-checknfor-similar-purring-flute.md` —
  full Φ refactor specification.
- `audit/archive-2026-05-24-handover.md` — older Day-N PM sections
  (extract-physics-trajectory-Vjaz9 + review-skills-improvements-moKOR +
  2026-05-20 What-just-landed block).

## Rule reminders (most relevant this session)

- **Rule 12:** rolling-last-2; never push speculative variants after a
  known-good submit unless willing to lose its ladder spot.
- **Rule 40:** prefer modeling-correctness over restriction-tuning. The
  concentration patch is restriction-tuning; the Φ refactor is the
  modeling-correctness alternative.
- **Rule 45:** n=16 with bimodal seed pattern is NOT a parity claim —
  treat as INCONCLUSIVE.
- **Rule 48 (new):** Kaggle μ is adaptive; never interpret early poll
  as settled.
