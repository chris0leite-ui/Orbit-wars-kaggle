# 2026-05-23 — coord Day 13: feature pile-up, prune-needed

## What shipped today

Two submissions in quick succession:

- **sub 52936894 (coord v2)** — deadline-bounded enumerate + smooth-ΔW
  endgame bonus (λ_W=0.002) + code-review fixes (per-kind gates,
  EPISODE_STEPS assert, env truthy, threaded `model`, leaf-floor with
  default 0.0) + COORD_REDUCED_FLOOR knob (default 0 = current
  behavior). 4P attribution via `_strongest_opp` + `_largest_threat_owner`.
- **sub (coord v3)** — same stack + **demand-spread mixing (Option 3
  LITE)**: per-opp defensive capacity × per-bundle attention demand →
  per-bundle mixing_weight ∈ [0, 1] interpolating composite between
  tier2 (defended) and cheap_score (undefended). Plus floors raised:
  LEAF_FLOOR_DEFAULT 0 → **2.0**, REDUCED_FLOOR_DEFAULT 0 → **2.0**
  (PI observation in v2: too many small/far wasted fleets).

Both self-evicted the predecessor before any μ data landed. v3 evicted
v2 ~10 min after v2 was submitted.

## What I learned

1. **Per-bundle scoring isn't enough for ensemble emission.** The
   leaf head via `score_candidate_v4_joint` rolls out `lite_greedy_policy`
   for the opponent, assuming opp's full attention is on this single
   bundle. With 19 own planets and 480 ships dominating, individual
   ATTACK bundles got leaf = −485 to −679 (opp piles defenders);
   minimal emits 8 moves, coord emits 1. Diagnosed via the new
   `scripts/check_coord_turn0_diagnostic.py` (one-turn dump of all
   scored bundles + Lagrangian selection).

2. **The Lagrangian was solving the right problem with the wrong scores.**
   The fix isn't to replace the Lagrangian — it's to feed it scores
   that already factor in the ensemble effect on the opponent.
   Demand-spread mixing does this in closed form (no new dual
   variable, no convergence concerns).

3. **Multiple null A/Bs in a row are diagnostic.** At λ_W=0.002,
   λ_W=0.01, λ_W=0.03, LEAF_FLOOR=-1e9, REDUCED_FLOOR=-1e9 — every
   n=4 swapped A/B vs orbitfix landed at 1W/3L same outcomes.
   That's the signature of "wrong axis." It pointed at the
   isolation-scoring substrate, not the chooser axis we kept
   tuning. The deeper fix (ensemble-aware scoring) was what was
   needed.

4. **Deadline-bounded enumerate matters BEFORE anything else.** The
   timing probe revealed 84% idle turns because enumerate (p50 607ms)
   was eating the 600ms budget alone. Tier-2 then pre-bailed empty.
   No amount of objective tuning helps when the agent has no scored
   bundles to choose from. ~50 LOC fix; took the idle rate from 84%
   to 77% and put total wallclock under budget.

## The prune-needed concern

This session added FIVE knobs to coord in rapid succession:

- `COORD_DELTA_W` + `COORD_LAMBDA_W` (smooth-ΔW endgame bonus, Day 12)
- `COORD_ATTACK_BONUS` + `COORD_DEFEND_BONUS` (per-kind gates,
  code review)
- `COORD_LEAF_FLOOR` (tactical viability floor; default 2.0)
- `COORD_REDUCED_FLOOR` (Lagrangian break threshold; default 2.0)
- `COORD_DEMAND_SPREAD` + `COORD_OPP_CAPACITY_FACTOR` (Option 3
  demand-spread mixing)

The Day 11 coord was simpler and had μ=905.6. We've changed five
things at once. If v3's μ is similar or worse, we won't easily
identify which feature is doing harm vs help. **Next session must
prune.** Recommended order:
- Toggle each feature off in isolation (single env var change) and
  A/B vs v3 (n=4 swapped). If win rate within ±25pp, the feature is
  null or harmful — disable in production default.
- The features most likely to hurt: COORD_DEMAND_SPREAD (changes
  scoring fundamentally), COORD_LEAF_FLOOR=2.0 (drops more bundles
  than 0.0), COORD_REDUCED_FLOOR=2.0 (Lagrangian admits fewer).
- The features most likely to help: deadline-bounded enumerate
  (measured: 84%→77% idle), code-review fixes (correctness only,
  no behavior change).

## Where the design could compound (if v3 lifts μ)

If demand-spread mixing produces a real lift on Kaggle:
- Tune `COORD_OPP_CAPACITY_FACTOR` empirically (sweep 0.5, 1.0, 2.0)
- Tune `DEMAND_REACH_WINDOW` (sweep 8, 12, 20)
- Upgrade to per-opp shadow price (Option 3 canonical) — the LITE
  mixing uses uniform-mean over responders, the canonical version
  gives finer per-opp-source pricing via dual iteration
- Multi-turn portfolio planning (Mission Renaissance's intended fix,
  now on a substrate where ensembles work)

## What I'm watching

- **sub coord v3 settled μ** — single fact that picks the next
  major direction. Expected ~12-24h.
- Whether v3's small-fleet behavior changes vs v2 (the immediate
  symptom that prompted the floor raise).
- The "isolated leaf head pessimism" question: is mixing toward
  cheap_score the right "less pessimistic" anchor, or does it
  overshoot in the opposite direction?
