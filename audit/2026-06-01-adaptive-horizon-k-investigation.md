# Adaptive horizon K — investigation (2026-06-01, champion-strategy-rules-00JzI)

PI directive: *"explore and think hard about an adaptive horizon K. In the
beginning it should be feasible and allow faster expansion. With the
increased horizon, also planets farther away should be feasible. It needs
to be adopted consistently across agent functionality."*

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
forbidden. This is the structural mechanism behind the diagnosed loss mode
("we had ≥ ships but FEWER planets; opponents spread + snowballed" —
`audit/2026-06-01-live-replay-diagnosis.md`). If opponents grab the ETA
15-30 ring in the opening and we are capped at 10, we lose the planet race
before midgame.

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

- **Wallclock:** bigger K early → proposer enumerates more candidates (2-4×
  targets) → more rollouts. Opening has time headroom (fewer entities), but
  must verify p95 stays < the 1000ms actTimeout (Rule 2 two-tier smoke).
- **Far-capture survival (H44):** even in the opening, 65% of *failed*
  captures historically die in-flight. Need to confirm opening far-launches
  actually land (single-game trajectory trace, Rule 47) — if they die, the
  reachability gain is illusory and this collapses to the flat-expand-credit
  regression.
- **Does reachability convert to wins?** §3 is geometry only. The A/B
  (Rules 43/45) is the real test. Gate: vs-panel Wilson-lo ≥ 0.55 per
  opponent AND vs current champion n≥32 Wilson-lo ≥ 0.50, evaluated
  against **aggressive expanders** (not the champion mirror — the
  flat-credit lesson: a hoarder-vs-hoarder mirror can't see expansion
  value).

## 7. Next-step plan (not yet built)

1. Implement `capture_horizon_k(step)` adaptive schedule behind
   `BASELINE_ADAPTIVE_K` (default OFF); thread `world.step` to the 3 call
   sites. ~40 LOC, single function + 3 one-line call-site edits.
2. Parity test: ADAPTIVE_K off → byte-identical to champion (move-parity).
3. Two-tier wallclock smoke (Rule 2): opening p95 < 1000ms.
4. Single-game opening trajectory trace (Rule 47): far opening launches
   land (sun/oob/flip waste < 2%).
5. A/B vs aggressive-expander panel + champion h2h, n≥32 (Rules 43/45),
   sweeping K_OPEN ∈ {15,20,25} and T_SETTLE ∈ {30,40} (Rule 21).
</content>
