# 2026-05-22 — coord (Lagrangian bundle market) shipped

12 days of work. Built `agents/coord/` from scratch on the
`claude/consolidate-codebase-refactor-dQAWA` branch. Core idea: instead
of orbitfix's per-source-greedy-then-pair-then-defend brain, formulate
the per-turn decision as a **Lagrangian-priced combinatorial auction**.

## The substrate

- Each candidate bundle = (sources, target, per-source ships,
  arrival_step, kind ∈ {ATTACK, DEFEND, RECAPTURE}).
- All bundles enumerated up-front (per target × arrival window ×
  ≤3-source subset, capped at MAX_BUNDLE_SIZE=2 after Gate 3 falsified
  3-source at 1.8% wins).
- Cheap-filter ranks all (~1100) bundles by closed-form Δ-favor +
  production-PV, keeps top-75.
- Tier-2 simulation scores top-75 via the existing N-leg-generic
  `score_candidate_v4_joint`.
- Lagrangian dual ascent with greedy primal: per-planet ship-budget
  shadow prices iterate until best-feasible primal found.
- Defense bundles in the SAME market as attack — they compete for
  source ships via shadow prices. PI override of my Day-2
  conservative recommendation; the unified-market design landed
  cleanly.

## What this gives us

The chooser-axis ceiling at μ≈1175 has held across multiple branches.
The orbitfix brain shape — per-source greedy → post-hoc joint pair →
post-hoc reinforce — seems to be the binding constraint. Coord's
Lagrangian market is the first **structurally different brain** on
this branch line. Whether it breaks the ceiling is the empirical
question.

Local A/B vs orbitfix (sub 52912707, μ=1174.2): **4W/2L over n=6
unswapped seeds**. Wilson 95% CI [0.30, 0.90] — directionally
positive but n=6 is below Rule 45's n=32 threshold.

## Submitted

sub 52927313 (PENDING), evicts sub 52894340 (phase4_step1_FND,
μ=1093.0). The evicted-μ < predicted-μ low end of 1100-1250 → Rule
42 auto-GREEN. PI explicit "Submit now" gave Rule 1 sign-off.

## Tonight's H44 patch

Cherry-picked the wait_N>0 trajectory admissibility fix from
`claude/extract-physics-trajectory-Vjaz9` (commit c6a0c80). Two
sites in coord. Correctness only — source branch's three A/Bs
landed at parity vs orbitfix even with this fix included. The
fix removes a 65%-of-live-failures class of in-flight death
that may explain some of coord's 2 losses in n=6.

Not resubmitted (sub 52927313 still pending; resubmitting would
evict the team champion orbitfix). Decision on resubmit waits for
the live μ.

## What I learned

1. **The bundler's regex strips lines individually** — multi-line
   imports leak continuation lines as orphan indented text. Solved
   on other branches (commit 4094aa1, 2026-05-17) but the
   cross-agent inlining gap still requires the modular-agent pattern
   (helpers in the agent's own directory). Documented as friction
   `bundler-multiline-cross-agent-import-orphans`.

2. **Discipline of gate-by-gate validation pays.** Each gate had
   acceptance criteria documented before running. When Gate 1 and
   Gate 3 failed strict thresholds, we diagnosed structural causes
   and proceeded with documented caveats rather than tuning until
   the gates passed. The result: a cleaner agent shipped with
   honest documentation of its limitations.

3. **PI override of "conservative defense" was correct.** My Day-2
   recommendation kept defense post-hoc; PI pushed for unified
   market. The Lagrangian's shadow-price mechanic makes defense
   competitive against attack without recreating the Mission
   Renaissance failure (forced commits + drop-one can't undo).

4. **Cross-branch synergy hunting is high EV.** The H44 fix took
   1.5h to identify and land. The 65%-of-live-failures impact is
   massive relative to that cost. Reading other branches' wrap
   notes + recent commits should be standard before any major
   feature work.

## What I'm watching

- **sub 52927313 settled μ** — single fact that picks the next
  major direction (compound on Lagrangian / portfolio planning /
  structurally different).
- **Whether coord's extra captures (Gate 1 showed coord launches
  when minimal doesn't) translate to live wins.** If yes, the
  no-hard-cheap-reject design is validated. If no, we add a
  threshold filter to match minimal's economy.

## Where the design could compound

Coord's bundle market is the substrate that finally makes
**multi-turn portfolio planning** possible. Mission Renaissance
failed because the per-source greedy chooser couldn't undo forced
commits. The Lagrangian can. The Mission Renaissance audit (H30,
2026-05-14) named the fix — "portfolio search: multiple incumbent
plans, drop-one within each, score across all" — and the design
has been waiting for a substrate that supports it. Coord is now
that substrate. ~5 days build.
