# 2026-05-12 — v4_planner "shoots later than necessary" diagnosis

> PI observation from live ladder watch (16:00 UTC):
> *"v4_planner is taking some really absurd actions. It shoots at
> planets and it could hit them easily early, but it waits. Shoots
> further in the orbit. So there, it loses like 10 time steps or so.
> It could hit them earlier — maybe that's that comes from backwards
> iteration. So somehow it's converting to the wrong conclusion."*

## Initial hypothesis (ruled out)

First suspect: `lib/aim.py::aim_orbiting` or `search_safe_intercept`
returns a multi-orbit-later intercept instead of the soonest one.

**Tested:** ran the lead-aim solver on two configurations:
- Fast fleet (50 ships, speed ≈ 4), inner orbit (orbit_radius=30,
  omega=0.05). eta ≈ 20 turns vs orbital period 126. Converges to
  cand_t=20 (delta=0.16) — earliest self-consistent intercept; no
  wraparound. ✓ correct.
- Slow fleet (1 ship, speed=1.0), same orbit. Direct eta=71 turns;
  iteration oscillates (orbital period 126, eta 36–75 in successive
  iters) → falls back to `search_safe_intercept`. Picks cand_t=39
  (delta=0.78). No earlier delta≤1 candidate exists in the search
  horizon. ✓ also correct.

In both cases the lead-aim returns the soonest valid intercept. The
"shoots further" behavior isn't from the aim code.

## Actual hypothesis: receding-horizon pathology in v4_planner

v4_planner uses 1-ply receding-horizon control:
- 5 portfolios per turn (incumbent / conservative / per_source_swap /
  drop_weakest_source / **noop**).
- Each portfolio simulated K turns ahead (K adaptive 8–30, truncated
  by comet-spawn-boundary).
- Leaf scored by `evaluate_value(observation, my_id)` —
  `prod_share + 0.4·prod_denied + 0.05·ships_share + 5·sole_survivor`.
- Pick argmax.

### The bug

If the incumbent's chosen fleet has **eta > K** (target arrives AFTER
the rollout horizon), the leaf doesn't see the capture. In the
terminal state:
- "fire" portfolio: −N ships in flight (not yet arrived).
- "noop" portfolio: +0 (kept N ships at home).

`evaluate_value` rewards production-share at terminal. The "fire"
candidate has REDUCED our ship-share without (yet) increasing our
production-share. So **noop scores higher**.

The lookahead picks noop. Source idles. Next turn, same logic:
target is now eta−1 turns away; if eta−1 still > K, lookahead picks
noop again. This continues until eta drops below K (typically when
the planet rotates close enough for K=8–30 to cover it).

PI's observation matches exactly: the agent "waits 10 time steps"
because at the launch turn, the LATER-orbit-position intercept eta
fits within K, while the EARLIER intercept's eta doesn't.

### Why drop-one (v7_0) doesn't have this bug

v7_0's candidate set is `[incumbent, incumbent − each launch]`. Every
candidate that REMOVES a launch is a STRICT SUBSET of the incumbent.
There is no "noop" or "wait" option. Worst-case is the incumbent
(v3.5.1's action) — the parity floor.

If a fleet's eta exceeds K, v7_0 still fires (the incumbent fires).
The K-step rollout might not credit the capture, but it also doesn't
penalize the launch beyond the ships-in-flight cost. argmax-over-
drop-one picks whichever variant has best terminal V — typically
the incumbent itself when ship-delta is the head.

This is why v7_0 beats v4_planner 75% locally despite v4_planner
having a "richer" search structure.

### Why K=30 isn't enough

K=30 catches most fleet etas (max board diagonal at 1-ship speed is
≈ 142 turns; typical fleet 30+ ships travels ≈ 50–80 turns). But
adaptive K starts at 8 (low entropy) and is comet-truncated. Near
spawn boundaries it can drop to single digits. Long-distance launches
to far targets are systematically excluded from the rollout's value
calculation.

### Why v4_planner still scored 84% locally

v4_planner's local A/B was vs v7_minimax. v7_minimax has the SAME
pathology in its 2×2 maximin (K=3 is way too short to see captures).
Both agents are blind to long-flight captures, so the pathology
cancels out in their head-to-head. v7_0 with its drop-one structure
sidesteps it.

## Confirmation in the v0-v4 A/B

The v7_0 vs v4_planner A/B (just completed, Wilson lo 55.1% for v7_0)
shows v7_0 winning. Per-game ship-deltas show consistent v7_0 lead
even when v4_planner's portfolio search would have plausibly picked
a "better" mission set in absence of the K-horizon bug.

## Fix candidates (next-session work)

1. **Longer adaptive K_min** (cheap): raise K_MIN from 6 to e.g. 15.
   Costs more rollout time but eliminates the short-horizon blind
   spot at low entropy. May cap candidate count at 3-4 per turn.

2. **In-flight value credit** (smarter): modify `evaluate_value` to
   add a partial bonus for our in-flight fleets weighted by
   `1 - eta/K` (the closer to arrival, the more credit). Avoids the
   "fire is strictly worse than noop" failure mode.

3. **Reward terminal SHIP TOTAL plus PRODUCTION-CAPTURED**: change
   the head to credit both "we own this planet at terminal" AND "we
   have N ships in flight aimed at planet X". Bootstrapped: simulate
   the in-flight fleets' arrivals OUTSIDE the K-step rollout via
   `WorldModel.simulate_planet_timeline`, add that to V.

4. **Remove noop from the portfolio set** (band-aid): force the
   chooser to always fire something. Risks regressing in legit
   "ships are scarce, don't waste" situations. Worst-case-best is
   the incumbent.

5. **Use v7_0's drop-one with `evaluate_value`** (= v8_minimal, A/B
   running). If σ-equiv's regression in v7.6 was actually FROM the
   value-head-stacking on K=10 ship-delta rather than σ-equiv itself,
   v8_minimal might fix it. Result pending.

## Implication for the current submission decision

The current rolling-last-2 is `[v7_minimax (1063), v4_planner (PENDING)]`.
v4_planner has the receding-horizon pathology documented here. Its
live μ will tell us how badly it manifests on the ladder. Possibilities:

- v4_planner converges at μ ≥ 1063 → pathology is a minor inefficiency
  the value function partially compensates for. Submit v7_0 if it
  beats it locally (already established 75%).
- v4_planner converges at μ < 1063 → pathology costs ~30+ μ. Submit
  v7_0 immediately to evict v7_minimax → [v4_planner (poor), v7_0
  (better)]. We lose the v7_minimax floor but v7_0 is the new best.

Decision tree per Rule 1 (PI authorises every submit). Recommendation:
submit v7_0 as soon as the v8 A/B finalises a target.
