# Plan — next-session directions (synthesised from 2026-05-18 PM)

## Context — where we are

- **Submission 52784853** snapshot at **μ=1083.1** — UNDER-performed.
  Built from commit `82df5b8`: PV term off in production, bug #3 /
  #4 / #11 / #12 clean math fixes shipped. Local A/B vs prior bundle
  was 26/32 = 81.2% Wlo=0.647 Whi=0.911 PASS, predicted snapshot μ
  1130-1160 → actual 1083.1. Repeats the recurring local-vs-live
  calibration miss; new friction tag
  `local-ab-vs-ladder-calibration-miss-30mu`.
- **Rolling pair now**: [52754310 (1143.7 snapshot, trajectory
  champion), 52784853 (1083.1, NEW FLOOR — net cost ~30μ vs the
  1113.4 floor it replaced)]. 52766596 (1113.4) evicted.
- This session falsified the bug #15 PV-term-cures-everything hypothesis
  AND the bug #14 option-5 rollout-defense-cures-PV hypothesis. The
  chooser was calibrated WITHOUT PV; the natural next direction is
  structural improvements that **stop bleeding ships** rather than
  inflate the leaf-value head.

## Prediction for 52784853 (for calibration)

| Where it should perform well | Where it likely won't help |
|---|---|
| Multi-wave defense (bug #12 wider window) | 4P games — no 4P sub-panel was run pre-submit |
| Sources under known inbound threat (bug #4 drain-frontier pre-cut) | Endgame cleanup — `test_oracle_cleanup` still xfail (no PV → captures of last opp planet score Δ ≈ 0) |
| Mid-game reinforce sizing (bug #3 symmetric math) | Coordinated multi-source captures — `test_oracle_coordinated_capture` still xfail |
| Orbital-target attribution (bug #11, already shipped previously) | Long-runway captures where opp counter-attacks past rollout horizon (the wasted-ships pattern — Tier 2 below) |

## PI direction from this session — four lever sets

1. **Bundling tax** is a friction multiplier on every other iteration.
   Make submission iteration cheaper.
2. **The chooser models opp as ourselves** (`lite_greedy_policy` in
   `lib/opp_model.py:155`) and learns to beat itself. Real LB
   opponents play differently — we need an **asymmetric opp model**.
3. **Backline planets sit idle** while frontline planets carry every
   action. Need an **active-planets / coalition proposer** where
   every planet pitches; far planets contribute ships to the highest-EV
   proposal.
4. **We waste ships capturing planets we can't hold.** Long-distance
   capture of a neutral planet adjacent to a strong opp planet → opp
   counter-attacks cheaply from short range → we lose the planet AND
   the ships. **Hold-feasibility** is encodable as synthetic oracles
   and addressed by a proposer-side pre-cut sibling to drain-frontier.

These are independent levers. Tier 1 (bundling) is mechanical;
Tier 2 (hold-feasibility) is the biggest ship-savings lever and most
directly encodable as oracles; Tier 3 (asymmetric opp) and Tier 4
(active planets) compound and should land together.

---

## Tier 1 — Bundling-tax cleanup

### Problem
- `scripts/bundle_agent.py agents/baseline --force` regenerates a 392kB
  single-file bundle and runs a 626-turn self-play parity gate.
  Each cycle costs ~30-60s on top of the actual edit+test loop.
- Multi-line parenthesised imports break the line-by-line strip regex
  (friction `bundler-modular-agent-namespace-access-breaks-bundle` —
  already documented in `agents/baseline/main.py`). Every new module
  added forces single-line import hygiene.
- The bundle is gitignored so submission artifacts aren't versioned;
  re-bundle on every push.

### Fix sketch
- Add `--skip-parity-gate` to the iteration loop; run parity only
  pre-submit. Already exists at `scripts/bundle_agent.py:503`; just
  document and adopt.
- Cache the parity result keyed by bundle sha256 so repeat parity
  runs are skipped.
- Optional: track `submissions/baseline.py` so PR diffs show the
  bundled change (current `.gitignore: submissions/*` keeps it
  out — flip if PI wants the audit trail).

### Critical files
- `scripts/bundle_agent.py` — the bundler.
- `.gitignore` — currently excludes `submissions/*`.
- `agents/baseline/main.py:1-30` — bundler-safe import comments.

---

## Tier 2 — Hold-feasibility filter (THE wasted-ships lever)

### The pattern PI observed in live games

```
Our planet M  ──── long flight ────►  Neutral P  ◄── short flight ── Opp O (strong)
       (far)          T_MP                 T_OP
```

- We launch F ≥ P.defense + 1 from M.
- We capture P at tick T_MP with residue ≈ F − P.defense.
- Opp O is close to P; opp's counter-launch costs ≤ residue + 1 from
  short range. Opp recaptures by tick T_MP + T_OP.
- **Net**: we spent F ships, opp spent F − P.defense + 1, opp owns P.
  We've wasted P.defense ships AND given opp a position.

### Why the rollout misses it

- Rollout horizon is 25 ticks. Long-runway captures (T_MP ≈ 15+) leave
  little simulation budget for opp's counter to land.
- `lite_greedy_policy` (the rollout's opp model) picks targets by
  production / distance ratio — it doesn't *specifically* counter
  the planet we just captured. A real opp at μ1200+ does.
- With PV off (current production), the leaf base ship-delta for a
  marginal capture is `−P.defense + production·t` ≈ 0 — borderline
  candidates slip through the Δ > 0 emit gate when the post-rollout
  recapture isn't in the leaf state.

### Fix mechanism — proposer-side pre-cut, sibling to drain-frontier

Add `_target_holdable_after_capture(src, tgt, ships, eta_arrival, world, model, me)`
in `agents/baseline/proposer.py` alongside `_source_survives_launch`
(`agents/baseline/proposer.py:355`). Wire as a post-dedup filter in
`propose()` (line 407) behind a new env var
`PROPOSER_HOLD_FEASIBILITY` (default on for production).

Logic per candidate:
1. For each OPP planet O (or each NEUTRAL planet that opp could plausibly
   leverage), compute `T_OP = ceil(dist(O, tgt) / fleet_speed(O.ships))`.
2. Compute `our_garrison_at_eta_arrival + T_OP` =
   `max(0, ships − tgt.defense_at_arrival) + tgt.production × T_OP`.
3. Compute `counter_force_O = O.ships + O.production × (eta_arrival + T_OP)`
   (worst case: opp commits everything).
4. Identify the nearest opp planet that BOTH (a) has > some threshold
   ships and (b) `counter_force_O ≥ our_garrison_at_eta_arrival + T_OP + 1`
   — i.e. would beat us. If exists → unhold-able → drop candidate.

Reuse from already-existing code:
- `lib.fleet.speed` — fleet-speed formula.
- `lib.world_model.WAVE_LOOKAHEAD = 12` — same window semantics.
- `agents/baseline/proposer.py:355 _source_survives_launch` — code
  pattern to mirror.
- `kaggle_environments.envs.orbit_wars.orbit_wars.Planet` — namedtuple.

### Synthetic oracles to pin the contract (write FIRST, before the filter)

Add to `tests/test_planner_oracles.py`. Each test uses the existing
`_planet` and `_obs` helpers (line 31-53). All three should FAIL on
current production and PASS post-filter.

1. **`test_oracle_hold_feasibility_neutral_near_strong_opp`** —
   M at (10, 10) with 50 ships, P neutral at (50, 50) with 5
   defense, O opp at (60, 60) with 80 ships. Chooser MUST NOT emit
   M → P at the "barely capture" size; either skip the launch
   entirely OR emit a size that survives O's counter (verify via
   post-launch residue + production accrual ≥ 81).

2. **`test_oracle_hold_feasibility_size_threshold`** — same geometry,
   vary M's ship count from 6 (minimum capture) to 200. Find the
   threshold where the chooser starts emitting. Pin it. Documents
   the filter's exact cut-off so regressions are visible.

3. **`test_oracle_hold_feasibility_with_nearby_ally`** — same M + P + O
   geometry, plus M' near P with 100 ships. Chooser SHOULD emit a
   **coordinated launch** (M attacks, M' supports). Verifies hold-
   feasibility filter doesn't kill the legitimate joint case; the
   M' contribution makes residue + production large enough to hold.

### Verification gates

- Unit tests above must FAIL pre-filter, PASS post-filter.
- Existing oracles must still pass: `test_oracle_sanity_trivial_capture`
  (xfail expected with PV off), `test_oracle_defense_*` (must pass),
  `test_drain_frontier_filter_*` (must pass — independent layer).
- Bench: `python fast.py bench agents/baseline/main.py --vs v7_0
  --games 4` must hit verdict PASS (filter is O(planets²) per
  candidate; negligible vs rollout cost but verify).
- A/B vs 52784853 bundle: `python fast.py eval --max-seeds 32`
  Wlo ≥ 0.55 PASS. This A/B specifically measures whether stopping
  wasteful captures translates into ladder lift.
- 4P sub-panel via `scripts/play4p.py`: first-place rate ≥ 25% (no
  4P regression).

### Critical files
- `agents/baseline/proposer.py:355` — `_source_survives_launch`
  (the pattern to mirror).
- `agents/baseline/proposer.py:407` — `propose()` filter chain.
- `agents/baseline/proposer.py:511-528` — drain-frontier filter
  wiring (the closest existing precedent).
- `tests/test_planner_oracles.py` — oracle home.
- `lib/world_model.py:42-132` — `fleet_target_planet` (for any
  ray-cast logic we might need to attribute opp's counter).

---

## Tier 3 — Asymmetric opp model in the rollout

### Problem

`agents/baseline/chooser_trajectory.py:score_candidate_v4` drives opp
seats via `opp_actions_for_snap` (`agents/baseline/chooser.py:31`),
which calls `lib.opp_model.lite_greedy_policy`
(`lib/opp_model.py:155`). lite_greedy is a ROI-greedy launcher that
ignores ME's specific actions — it picks targets by global
production / distance. So:

- Our chooser learns to beat a dumb mirror of itself.
- lite_greedy doesn't **counter** us — it doesn't re-target the
  planet we just captured, doesn't switch focus to threaten our
  sources. Real LB opponents do (especially top-tier μ1200+).
- This is the root cause of the wasted-ships pattern (Tier 2 is the
  symptom-level pre-cut; this is the structural fix).

### Approach options (ranked simplest → most ambitious)

**A. Switch to `top_tier_mirror_policy`** (`lib/opp_model.py:92`,
   already implemented). Per-call cost ~10ms (vs 1-2ms for
   lite_greedy) — must verify wallclock budget. Represents Roman 1224
   / romantamrazov LB-MAX-1224 patterns. Already on the rollout-tier
   shelf.

**B. Targeted counter-policy**: build a NEW opp policy that
   specifically targets ME — at each tick, pick the move that
   maximises **damage to me** (combinable with lite_greedy's ROI as
   a weighted blend). Forces our chooser to find ME-robust strategies.

**C. Archetype panel**: at each candidate scoring, sample opp policy
   from a panel (lite_greedy + top_tier + hoarder + aggressive) and
   average the leaf value across opp variants. Robust scoring but ~2-4x
   wallclock cost.

### Critical files
- `lib/opp_model.py:92-153` — `top_tier_mirror_policy` already
  exists; just verify per-call cost.
- `lib/opp_model.py:155-233` — `lite_greedy_policy` (current default).
- `agents/baseline/chooser.py:31` — `opp_actions_for_snap` (the call
  site to switch).
- `agents/baseline/chooser_trajectory.py:42` — `lite_greedy_policy`
  import alias `_me_policy` (used in the parked option-1 cheap-mirror
  experiment; rename if confusing).

### Verification
- Bench gate first (top_tier_mirror is ~5-10x slower per call).
  `BASELINE_OPP_TIER=1` env var to toggle between tiers.
- Re-run the existing defense oracles (must still pass).
- A/B vs 52784853 to measure ladder effect.
- Pair with Tier 2 hold-feasibility filter — they should compound
  (filter pre-cuts wasted captures; asymmetric opp lets the rollout
  see remaining wasted-capture cases for non-pre-cut geometries).

---

## Tier 4 — Active-planets / coalition proposer

### The structural critique (PI's framing)

> "Every planet should suggest its improvement, the thing it would do.
> The estimator: how much it would improve our position / ROI / state
> of world. Far planets that can send their ships SUPPORT the
> highest-EV mission."

Translation: today the proposer is **planet-centric** — for each
source, enumerate candidates. The chooser then scores them
independently. The chooser's joint path
(`score_candidate_v4_joint` at `chooser_trajectory.py:502`)
generalises this to **pair joints** (top-K solos per target +
pair-enumerate). It's bounded at `JOINT_TOP_K_PER_TARGET=3,
JOINT_MAX_PAIRS=20` (chooser_trajectory.py:154-155).

PI wants **N-way joints**: every planet pitches; rank by EV; planets
that didn't pitch the winner contribute ships to it. Conceptually:
a market / coalition where each mission attracts supporters by EV.

### Implementation sketch

Two-phase per turn:

**Phase 1 — bid generation.** For each MY planet P:
- Generate P's best candidate (same as current proposer logic) — best
  capture target, best reinforce target, OR best "support" target
  (planets nearby that have an active mission needing ships).
- Score it with the existing `cheap_marginal_value` (proposer.py:373).
- Tag P's bid with `(target_id, P, ships, eta, ev)`.

**Phase 2 — coalition formation.** Group bids by `target_id`. For
each target group:
- Sort planets in the group by distance to target.
- Take the closest planet as "the primary" (the breaker).
- Iterate remaining planets: would adding their ships improve
  hold-feasibility (Tier 2) without violating drain-frontier (bug #4)?
  If yes, add them to the coalition.
- Coalition gets one joint candidate with `inject_at[wait_N_per_planet]`
  using the existing joint rollout (`score_candidate_v4_joint`).

The cap is the existing `JOINT_MAX_PAIRS` generalised to
`JOINT_MAX_COALITIONS` with `COALITION_MAX_PLANETS_PER_TARGET`.

### Why this fixes the structural critique

- **Active planets**: every planet contributes to the per-turn move
  set, not just the closest one to a target.
- **Asymmetry preserved**: opp still plays via `opp_actions_for_snap`
  (whatever tier is active per Tier 3). Our coalition formation is on
  ME's side only — the rollout's leaf state evaluates the full
  coalition's impact.
- **Synergy with Tier 2**: a coalition that adds enough supporters
  satisfies hold-feasibility for captures the solo couldn't safely
  make. Tier 2 prunes unhold-able solos; Tier 4 promotes hold-able
  coalitions.

### Critical files
- `agents/baseline/proposer.py:407` — `propose()` (Phase 1 expansion).
- `agents/baseline/chooser_trajectory.py:502` — `score_candidate_v4
  _joint` (Phase 2 evaluation; existing infrastructure).
- `agents/baseline/chooser_trajectory.py:154-155` — joint caps.

### Verification
- New oracle: `test_oracle_three_source_coalition` — three of our
  planets at varying distances from an opp planet; coordinated
  capture is the only winnable mode. Chooser should emit a 3-way
  coalition.
- Bench (coalition formation adds enumeration cost — must verify).
- A/B vs 52784853 (with Tier 2 + Tier 3 already landed).
- 4P sub-panel.

---

## Tier 5 (parked) — PV recalibration

Open question filed at
`knowledge-base/questions/2026-05-18-can-chooser-be-recalibrated-for-PV.md`.

Investigation sketch:
1. Profile Δ magnitudes per candidate over 20 games with PV on vs off.
   Histogram them.
2. Identify the chooser-gate threshold that reproduces pre-fix 50%
   winrate with PV on (call it `Δ_emit_floor + shift_PV`).
3. A/B with the new gate.

If recalibration is structurally possible, re-enable PV (sanity oracle
returns to PASS, captures register correctly at leaf, no over-
emission). If not, leave PV disabled permanently.

This work is independent of Tiers 1-4 and can run in parallel.

---

## Recommended priority for next session

1. **First**: query `kaggle competitions submissions orbit-wars` to
   see 52784853's snapshot μ. Use that to calibrate which tier is
   worth investing in.
2. **Tier 1 (bundling)** — cheap mechanical improvement; unblocks
   faster iteration on everything else. Spend ≤ 30 min.
3. **Tier 2 (hold-feasibility filter)** — biggest expected ship-
   savings lever. Three oracles + one filter; ~1-2 hours total.
   A/B should be conclusive in one tier (n=32). **This is the
   primary work item.**
4. **Tier 3 (asymmetric opp model)** — pair with Tier 2 (they fix
   the same root cause from two angles). Try `top_tier_mirror_policy`
   first; if too slow, design the targeted counter-policy.
5. **Tier 4 (active-planets coalition)** — bigger redesign; commit
   only after Tiers 2 + 3 are landed and ladder-validated.
6. **Tier 5 (PV recalibration)** — parallel investigation; doesn't
   block anything else.

## Verification across the whole stack

Each tier ends with the same gate stack:
- **Oracle suite**: `python -m pytest tests/test_planner_oracles.py
  tests/test_me_defensive_policy.py -v` — all targeted oracles pass.
- **Bench**: `python fast.py bench agents/baseline/main.py --vs v7_0
  --games 4` — max < 1000ms, zero >= 1000ms.
- **A/B**: `python fast.py eval agents/baseline/main.py --vs <prev
  bundle> --max-seeds 32` then 64 if positive. Wlo ≥ 0.55 PASS.
- **4P sub-panel**: `scripts/play4p.py` with the new toggle on.
  First-place rate ≥ 25% (parity or better with current 4P agent).

Only after ALL FOUR gates pass for a tier: bundle, submit (single
submission per session, PI-approved per Rule 1), update
`state/current.md` rolling-pair record.

## Key references

- `audit/2026-05-18-bug-catalog.md` — 15-bug catalog (yesterday's
  artifact, still current).
- `audit/2026-05-18-postmortem-bug-15-v2-and-bug-14-option-5.md`
  — this session's postmortem of the 3 failed A/Bs (PV + option 5).
- `knowledge-base/thoughts/2026-05-18-PV-term-recalibration-debt.md`
  — value-head/chooser-gate calibration mismatch lesson.
- `knowledge-base/flags/2026-05-18-pv-term-regression-shipped-as-default-on.md`
  — historical flag; PV is now disabled by default (commit 82df5b8).
- `HANDOVER.md` — session 2026-05-18 PM brief (must read first
  next session).
- `state/current.md` — rolling-pair record with 52784853.
- `tests/test_planner_oracles.py` — oracle suite (8 tests +
  3 xfails; conditional xfails keyed on `_value_heads._COMPOSITE_PV_
  ENABLED`).
- `tests/test_me_defensive_policy.py` — option-5 unit tests
  (dormant feature, idempotency contract pinned).
