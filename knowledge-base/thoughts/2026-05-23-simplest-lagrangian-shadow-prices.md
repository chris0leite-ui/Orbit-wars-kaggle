# 2026-05-23 — what the "simplest Lagrangian shadow-prices" agent
# revealed

PI directive this session: build the simplest possible agent that
uses our high-precision physics + a Lagrangian shadow-price coupling
across selfish per-planet decisions. Maintainable, ~250 LOC target,
"don't reinvent the wheel."

## What the structure is

Per turn:

1. **Enumerate** candidates `(src, tgt, launch_tick)` with the precision
   physics substrate already on the branch:
   - `aim_and_eta(src, tgt, ships, ω, wait_N=launch_tick)`
   - `predict_fleet_fate(src, tgt, angle, ships, world, wait_N=)`
     → keep only `outcome == "target"`
   - `predict_garrison_at(tgt, launch+eta, base_arrivals)` →
     ships > predicted defense
   - `_target_holdable_after_capture(...)` (B1, gated off in
     dominant-endgame)
2. **Score** each candidate `V(c) = production(tgt) ·
   (EPISODE_STEPS − arrival_step)`.
3. **Solve** with 3-sweep Lagrangian on per-source ship budgets:
   per-target argmax of `V(c) − λ_{src(c)} · ships(c)`; subgradient on
   λ_s using mean-launch_tick effective budget; final feasibility
   fix-up enforces the exact per-time cumulative constraint.

That's it. No MILP, no topology features, no smooth-ΔW, no opening
planner, no maximin search.

## What the random-elim gate revealed

Two structural bugs that the smoke test (vs random, 1 game) didn't
catch but n=16 with random seat assignment did:

- **Planet id 0 truthiness drop.** `int(x or -1)` evaluates to -1 when
  `x == 0`. Silently dropped every shot at planet 0 → game ran 500
  steps with opp owning planet 0 the whole time.
- **Midgame filter bleed into endgame.** B1 hold filter, calibrated
  for ladder midgame play (where it gives +63 μ), over-rejects in
  dominant endgame. Game ran 500 steps with opp owning a 3-planet
  pocket because every approach was filtered as "unholdable."

Pattern: **single-game smoke tests are blind to seed-conditional
late-game corner cases.** The n=16 random-elim gate (every game runs
to natural termination by elimination) is the cheapest screen that
forces every late-game edge case to surface.

PI promoted "100% win-by-elim vs random" as a pre-submit hard
requirement; queued as Rule-48 candidate in improvements.md.

## What this agent isn't

The simple agent is **single-source-per-target only**. Each target is
captured by exactly one fleet from one source (the per-target argmax
picks one candidate). When a target's predicted garrison-at-arrival
exceeds any single source's ship budget, the agent has no fallback.

vs `agents/baseline` (2000 LOC, multi-source dogpile via its dedup +
chooser pipeline), lagrange_simple loses 0/8 at n=8. The gap is
structural, not a bug.

## Where the design space opens next

The PI explicitly directed against multi-source dogpile this session
("simplest, maintainable, do not reinvent"). That makes the
single-source-per-target ceiling **deliberate** — a known limit traded
for code clarity. Next-cycle option-space:

A. **Multi-source dogpile** — ~50 LOC addition. Per (target, arrival
   step) bucket, greedy-add candidates by reduced cost per ship until
   sum-ships > defense; if subset's reduced cost > 0, commit. Captures
   the cohort-attack pattern that baseline does implicitly. Maintains
   the Lagrangian structure.
B. **Reinforce / migration moves** — currently we `continue` on
   own-target arrivals. Adding reinforce columns (own-target, score =
   ship-saving-defense-value) would let the dual coordinate offensive
   AND defensive launches in one solve.
C. **Source-time-cell dual** — current λ_s is a single scalar per
   source; the proper dual decomposition per `dual_decomp.py` uses
   λ_{s,u} per-source-per-time-cell. Faithful to the math, more
   coordination, ~30 extra LOC.

None of these are necessary for the **PI gate as stated** (vs random).
They become necessary the moment the target shifts to vs-baseline or
vs-ladder.

## Method-of-discovery observation

The two bugs were both surfaced by **running the gate, not by reading
the code**. The truthiness gotcha is technically catchable by static
review, but I missed it; the midgame-filter calibration is invisible
to any review because it requires endgame state to reveal. Lesson:
the n=16 gate is the cheap fuzzer; trust its output over `git diff`
self-review for behaviour-sensitive code.

## Reference

- `agents/lagrange_simple/{score,dual,main}.py`
- `scripts/random_elim_gate.py`
- `audit/2026-05-23-postmortem-session-EqJuT.md` (this session's
  postmortem)
- `.claude/skills/kaggle-comp/improvements.md` § random-elim-gate-
  mandatory (pending Rule-48 candidate)
