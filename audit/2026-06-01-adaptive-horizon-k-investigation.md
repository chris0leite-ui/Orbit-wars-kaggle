# Adaptive horizon K — investigation (2026-06-01, champion-strategy-rules-00JzI)

> **REVISED 2026-06-01 PM** after the empirical loss-mode diagnosis
> (`audit/2026-06-01-loss-mode-diagnosis.md`) and PI corrections. Two
> changes bind everything below: (1) the H44 "fleets die in flight" risk in
> §6 is **dropped** — fleets do not collide in air; (2) the step-decay
> schedule in §5 is **superseded** by a **state-driven** horizon (PI
> direction) — see §8. The §3 reachability data stands; the §4 "opening is
> where we lose" thesis is **weakened** (the opening is roughly even — the
> visible divergence is midgame). A v1 step-schedule was built and is in
> A/B (§8); the redesign is what ships if it pays.

PI directive: *"explore and think hard about an adaptive horizon K. In the
beginning it should be feasible and allow faster expansion. With the
increased horizon, also planets farther away should be feasible. It needs
to be adopted consistently across agent functionality."* Later refined:
*"it should scale according to our capabilities and the current game-state
complexity, not to a fixed schedule necessarily."*

This is the "opening / dynamic-lookahead" axis from the 2026-06-01 AM
handover, reframed precisely: the lever is the **universal launch ceiling
K** (`launch_rules.DEFAULT_CAPTURE_HORIZON_K = 10`), made a function of
game phase instead of a constant.

## 1. What K=10 is and why it exists

K is the "predictability ceiling": `enforce_launch_rules` drops EVERY
launch whose fleet arrives after turn K (opponent captures, neutral
captures, own reinforcements, comet-sourced). Rationale (launch_rules.py
docstring): *"Beyond K the board state is unpredictable, so a far launch
routinely lands at a flipped/contested planet and loses its fleet."*

K=10 is **load-bearing for the champion.** The universal-K=10 ceiling is
the difference between `baseline_launch_rules_k10` (sub 53175658, μ=1102)
and `baseline_launch_rules_universal` (sub 53182323, μ=**1183.7** — our
best agent ever). So we relax K at our peril; the discipline is what made
the champion good.

## 2. The single-lever finding (consistency is structurally easy)

`capture_horizon_k()` is the **single source of truth for K**, read in
exactly three subsystems — so "adopt consistently across agent
functionality" reduces to making *that one function* phase-aware:

1. **`launch_rules.enforce_launch_rules`** (the gate; 4 call sites in
   `main.py` — opening-MILP path + 3 normal-return paths). `step > k` →
   drop. This is the hard chokepoint every launch funnels through.
2. **`proposer.py:1086-1088`** — efficiency pre-prune: `if eta > _k:
   continue`, so far candidates aren't even enumerated/rolled-out.
3. **`chooser_trajectory.py:1276`** — sync-coalition arrival cap =
   `min(max_horizon, capture_horizon_k())`.

The **value function already looks past K**: `value_heads.INFLIGHT_EXTRA_
HORIZON = 30` gives effective leaf horizon ~40, and the trajectory
chooser's leaf scores `eta + SETTLE_TURNS=3` (scales with arrival). So a
far capture, *once K admits it*, is scored on its real production — no
value-side change needed. The proposer's `MAX_HORIZON=40` already
enumerates eta up to ~38, so it generates far candidates; only the K
pre-prune (point 2, same function) currently kills them.

**Conclusion:** thread an adaptive K out of `capture_horizon_k(step)` and
all three subsystems move together automatically. `world.step` is in scope
at every call site. No other horizon constant needs to change for v1.

## 3. Empirical reachability (Rule 47 physics grounding)

Probe: 10 fresh 2P openings (step 0), min-ETA per neutral over my sources
(`aim_and_eta`, modest fleet). N=288 neutral min-ETAs.

| metric | value |
|---|---|
| opening neutral min-ETA: min / median / p75 / p90 / max | 3 / **22** / 29 / 38 / 50 |
| nearest neutral per game | always ETA 3-6 (grabbed at any K≥10) |

**Avg # neutrals reachable per game at each ceiling:**

| K | reachable | vs K=10 |
|---|---|---|
| 10 | 5.4 | 1.0× |
| 15 | 8.5 | 1.6× |
| 20 | 12.7 | 2.4× |
| 25 | 17.5 | 3.2× |
| 30 | 22.8 | 4.2× |

**The static K=10 hides ~75% of the opening map.** Only the nearest ring
(ETA≤10, ~5 planets) is considered; the median target (ETA 22) is
forbidden. K=10 is geometrically very restrictive in the opening.

> **CAVEAT (revised):** this was originally pitched as "the structural
> mechanism behind ship-hoarding." That loss-mode framing is now refuted
> (`audit/2026-06-01-loss-mode-diagnosis.md`): we are not behind on opening
> planet *count*. Reachability being restricted is a real fact; that it is
> *the* loss lever is not established. The opening still matters via tempo
> ("we open too slowly") and the snowball, but as a chain into the midgame,
> not as an opening planet-count deficit.

## 4. Why "adaptive" and not just "bigger" (the thesis)

K=10's unpredictability rationale **does not bind in the opening**: at step
0-30 few/no fleets are in flight, planets sit at known positions, opponents
have not committed → a far launch is *safe*. As the game develops, fleets
fill the board, combat scatters trajectories → far launches start landing
at flipped planets → K=10 discipline becomes correct again. So:

> **K should be large early (unlock the map for fast expansion) and decay
> back to the champion's disciplined K=10 by midgame.** This keeps the
> exact champion behavior where the discipline matters and only relaxes it
> in the provably-predictable opening.

This is a modeling fix (Rule 40), not a constant bump: the right behavior
(expand far when safe, stay disciplined when not) emerges from making K
track board predictability instead of being fixed.

## 5. Proposed design

`capture_horizon_k(step=None)` returns a phase schedule. Default-OFF env
gate `BASELINE_ADAPTIVE_K` so the champion is byte-identical when off.

- **Floor** `K_FLOOR = 10` (the champion's disciplined ceiling).
- **Opening ceiling** `K_OPEN` (candidate 20-25 from §3 — unlocks the 2nd
  ring without over-committing to the ETA 30+ fringe).
- **Decay** linear from `K_OPEN` at step 0 to `K_FLOOR` at `T_SETTLE`
  (candidate 30-40, aligned with the existing `OPENING_HORIZON=30`).

  `K(step) = max(K_FLOOR, round(K_OPEN - (K_OPEN-K_FLOOR)*step/T_SETTLE))`

Driver = **step** (turn number) for v1: deterministic, opponent-independent
→ clean common-random-number A/B. A v2 variant can drive off in-flight
fleet count (a more direct "how much is going on" proxy) if step-based
shows promise — that gives the ≥3 variants for Rule 21 falsification along
the schedule axis (K_OPEN magnitude, T_SETTLE length, driver step-vs-entity).

## 6. Risks / open questions

- **Wallclock:** bigger K → proposer enumerates more candidates (2-4×
  targets) → more rollouts. *Verified OK so far:* single-game smoke (ON vs
  v7_0) p95=283ms, max=337ms — far under the 1000ms actTimeout, even under
  concurrent-A/B contention. Re-check under the state-driven version, which
  can raise K in busier midgame states.
- **Wrong phase (the big one).** The step-decay version raises K only in
  steps 0-30. The loss-mode diagnosis shows the opening is roughly even and
  the divergence is **midgame (50-150)** — where this schedule has already
  decayed K back to 10. So the v1 schedule may target the wrong phase. This
  is the §8 redesign motivation; the running A/B (§8) measures it directly.
- **Does it convert to wins?** §3 is geometry only — it shows what becomes
  *reachable*, not what becomes *held*. The A/B is the real test. Gate:
  champion h2h n≥32 Wilson-lo ≥ 0.50, then vs-panel Wilson-lo ≥ 0.55 per
  opponent (Rules 43/45). (No "far launches die" risk — fleets don't
  collide in air; the only flight deaths are sun/OOB, already filtered.)

## 7. What was built (v1, committed `9985e98`)

`capture_horizon_k(step)` step-decay behind `BASELINE_ADAPTIVE_K` (default
OFF); `world.step` threaded to the 3 call sites (gate, proposer prune, sync
cap). `K(step)=max(10, round(K_OPEN - (K_OPEN-10)·step/T_SETTLE))`,
defaults `K_OPEN=20`, `T_SETTLE=30`. Verified: OFF → flat 10 = byte-identical
champion; 19 launch-rules tests green; fires in the champion's default
composite path; wallclock smoke clean (§6).

## 8. Redesign: state-driven horizon (PI direction — the version that ships)

The step-schedule is the wrong shape. K is a *predictability horizon*, so it
should equal **how far ahead the board is actually predictable**, driven by
state + capability, not the clock:

- **Board complexity** — few in-flight fleets / uncontested target
  neighbourhood → predictable → large K; churning board → small K.
- **Per-target** — `K_target ≈ time until an enemy fleet could interfere at
  this target`, clamped `[K_floor, K_ceil]`. A far but uncontested planet
  is safe to commit to; a near but contested one is not.
- **Compute headroom** — fewer entities → cheaper rollouts → we can *afford*
  a deeper horizon (ties "our capabilities").

This naturally raises the horizon early **and** in midgame lulls — exactly
the phase the step-schedule misses. Build only if the v1 A/B is non-negative
(evidence the lever has signal) **or** if it is neutral *and* the diagnosis
holds that the schedule shape is the reason.

### Status

- v1 step-schedule A/B (ON vs immune champion, CRN, n=32): **in flight**.
  Interim n=18 = 12W/6L (67%) — leaning positive, not yet conclusive.
- Earlier ON-vs-OFF A/B was **discarded** (contamination: the OFF bundle has
  the live env-read code, so the ON bundle's baked `ADAPTIVE_K=1` leaked into
  the shared process and turned OFF adaptive too). Valid A/Bs use the
  pre-edit `baseline_champion_nokt.py` as the immune opponent.

