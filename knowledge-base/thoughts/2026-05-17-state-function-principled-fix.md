# 2026-05-17 — PI principle: model the state, don't tweak the restrictions

## The PI's call

> "max wait should not be a restriction. not waiting should emerge from
> a proper modeling not from a restriction. similar for other tweaks
> or restrictions"

Mid-session this turn I had bumped `MAX_WAIT` from 10 to 20 (and
`MAX_HORIZON` from 30 to 40) after Felipe-seed forensic showed
`agents/v8_scavenge/main.py` couldn't enumerate a wait-then-fire
capture of path-blocking planet 12 (wait_N=17 needed, capped at 10).
The PI caught it: those caps are band-aids on a state function that
misvalues actions. Reverted both back to session-start values.

## What's actually broken in the state function

The chooser pipeline computes:

```
Δ = leaf_favor(my_action + opp_idle_K) − baseline_favor(everyone_idle_K)
```

with `_favor = F1 + F2 × pv_horizon(γ=0.99)`. Multiple defects compound:

1. **Strict-idle opp model.** Both terms assume opp does nothing for K
   turns. Real opp captures 4 planets in those K=30 turns on Felipe.
   So the chooser overvalues "I do nothing" (baseline = my natural
   growth) and undervalues "I act fast" (candidate Δ shrinks because
   the only thing that changed was MY ship count).
2. **Truncated rollout, full-game F2.** Leaf state is at turn K+arrival
   (say 34), but `pv_horizon(34, 0, γ=0.99)` integrates production for
   the REMAINING 466 turns ≈ 99. So a 4-prod neutral captured at eta=30
   scores `4 × 99 = +396` in F2 alone — overwhelming everything else.
   In reality, the captured planet might only produce for 90 turns
   before we get eliminated.
3. **Artificial caps as workarounds.** MAX_WAIT=10, MAX_HORIZON=30,
   MIN_FLEET_SIZE=2, `cheap > -10` filter, adaptive N_VALIDATE,
   wait-then-fire only when capture-now infeasible — each is a tweak
   to make defects (1)+(2) tractable. None addresses the root cause.

PI's principle: stop adding tweaks; fix the modeling.

## The principled fix (next iteration)

**One pre-computed opp trajectory per turn, replayed deterministically
in baseline and every candidate.**

```python
# Once per turn (cost ~30 × 10ms = 300ms with top_tier_mirror_policy,
# or ~30 × 1ms = 30ms with lite_greedy if we fix its sizing bug):
opp_traj = []
snap = fs_clone(snap_base)
for step_i in range(MAX_HORIZON):
    if snap.fake_env.done:
        opp_traj.append([[] for _ in range(num_seats)])
        continue
    actions = [[] for _ in range(num_seats)]
    for opp_id in range(num_seats):
        if opp_id == me: continue
        actions[opp_id] = opp_policy(snap.state[opp_id].observation)
    opp_traj.append(actions)
    snap = fs_step(snap, actions, in_place=True)

# Baseline: me idle throughout + opp_traj
baseline_favors = []
snap = fs_clone(snap_base)
baseline_favors.append(_favor(snap.state[me].observation, me, num_seats))
for step_i, actions in enumerate(opp_traj):
    snap = fs_step(snap, actions, in_place=True)
    baseline_favors.append(_favor(snap.state[me].observation, me, num_seats))

# Candidate: replay opp_traj, splice in me's action at wait_N
def _score_action(snap_base, ..., wait_N, src_id, angle, ships):
    snap = fs_clone(snap_base)
    for step_i in range(horizon):
        if snap.fake_env.done: break
        actions = [list(a) for a in opp_traj[step_i]]
        if step_i == wait_N:
            actions[me] = [[src_id, angle, ships]]
        snap = fs_step(snap, actions, in_place=True)
    return _favor(snap.state[me].observation, me, num_seats) - baseline_favors[horizon]
```

Why this is principled:
- **No more strict-idle blindness.** Opp's expansion is in the baseline,
  so "I idle" correctly costs me planets.
- **No more "wait 30 turns to capture distant planet" overscoring.**
  By the time the leaf is computed, opp has captured 4 planets and my
  ship balance is wrecked — Δ goes appropriately negative.
- **Wait-then-fire emerges from value.** If waiting 17 turns lets opp
  capture 3 planets, the wait candidate's Δ is negative. If waiting
  buys me a CRUCIAL capture (e.g., planet 12, blocking my path to
  better territory), Δ is positive. MAX_WAIT can be 50 or unlimited;
  the value computation chooses.
- **Common random numbers.** Same opp trajectory across all candidates
  → opp's stochasticity cancels in Δ. This was the Iter 2 regression
  cause; CRN fixes it.

## Cost / wallclock

Per-turn budget after overhead = 550ms. Allocation:
- opp_traj build: 30-300ms depending on policy choice
- baseline rollout: ~90ms (30 × ~3ms per fs_step)
- N candidates × ~90ms each

For top_tier_mirror (10ms/step) trajectory + adaptive N_VALIDATE:
- 300 + 90 + N × 90 = 390 + N×90
- Budget left after traj+baseline: 160ms → N=1
- Way too few candidates.

For lite_greedy (1ms/step) with sizing-bug fix:
- 30 + 90 + N × 90 = 120 + N×90
- Budget left: 430ms → N=4-5
- Workable.

So Step 0 of the principled fix: make `lite_greedy_policy` correctly
size captures (cap = defenders + 1, not 0.7 × source). Then it returns
only feasible-and-winning actions, not 7-ship bounces.

Alternative: top_tier_mirror only for first K=10 steps (the strategically
critical opening), idle thereafter. Cost = 100 + 90 + N×90 ≈ 270+N*90 →
N=3.

## Why I can't just ship this in one commit

- It's a refactor of `_build_idle_baseline` + `_score_action` + a new
  opp-trajectory builder. ~80-120 LoC delta.
- Needs the lite_greedy sizing fix (separate change to lib/opp_model.py
  — that touches every agent's downstream behavior).
- Wallclock impact requires careful re-tuning of N_VALIDATE, adaptive
  cap, and possibly MAX_HORIZON.
- Panel verification per-opponent (Wlo ≥ 0.55 gate) is overnight-scale
  for n=64 on 3 opponents.

This-session decision: land the orthogonal Layer-1 (orbital revert) +
Layer-2 (mirror-opp at step 0) increment because they:
- Fix a real bug (Layer 1)
- Move ONE STEP toward the principled model (Layer 2 — strict-idle is
  gone at step 0, which is when opp's first capture decision happens)
- Do not regress the panel (75% Wlo 0.579 holds)

Next-session work: implement the full opp trajectory as described
above. The Felipe-seed loss is the gate — if the principled fix doesn't
flip Felipe to ≥1/2, the model has a deeper structural issue (e.g.,
path-blocking obstacle detection).

## Other restrictions to revisit in the principled refactor

- `MIN_FLEET_SIZE = 2` — should the agent be able to send 1-ship probes?
  Currently no, by fiat. Principled: small fleets have small upside but
  even smaller cost; the value computation decides.
- `NUM_TARGETS_PER_SOURCE = 8` — caps candidate breadth per source. Why
  8? It's there because validation is expensive. Principled: cheap-rank
  every target, validate as many as wallclock allows.
- `_cheap_marginal_value` "cheap > -10.0" filter — arbitrary threshold.
  Principled: let validation reject; don't pre-filter on a noisy proxy.
- The `if delta > 0` emit filter — biases toward action. Could mask
  negative-Δ "least-bad" emissions OR conversely block defensive moves
  that have small positive Δ. Principled: emit the top-K by absolute
  Δ if Δ > some_threshold_derived_from_idle_value, not zero-pivot.

## TL;DR

PI is right. The fix is rollout-aware opp modeling with common random
numbers, not bumping the caps. Landing Layer 1+2 as the bug-fix and
half-measure increment; planning the principled refactor for next
session.

## Update 2026-05-18 — diagnostic cross-game evidence

Pulled 66 v8 (#52684059) + 60 v9 (#52687411) live replays and ran the
phase classifier. Findings (full notes
`audit/2026-05-18-loss-mode-v8-v9.md`):

- Felipe is NOT representative. v8 loss distribution is **84 %
  mid_economy_lost**, only 16 % opening_lost. v9 is 57 % / 43 %.
- Per-game emission-rate tables across 4 sample losses (Felipe 2P,
  Naoism 2P, dekaineko-led 4P, Troups-led 4P) all show the same
  pattern: **top opponents ramp launches 3–5/window → 19–25/window
  as their planet count grows; we stay flat at 8–13/window.** The
  defect manifests as mid-game launch-rate ceiling, not opening
  cadence.
- All three downstream symptoms (opening misses, mid-game holds,
  reinforce starvation) route to the same upstream defect: **the
  strict-idle baseline lies about opp's behavior, so most candidates
  score Δ ≤ 0 against the lying baseline.**

This sharpens the principled-fix design but doesn't change it: the
opp-trajectory + CRN refactor is the single highest-leverage
intervention. The `pv_horizon` strategic-horizon fix (t_total
500 → 200) becomes additionally justified by the empirical median
game length (v8 171, v9 206) — the integration window is genuinely
mismatched, not a theoretical concern.

Verification gate adjustments:
- Felipe primary gate stays (0/2 → ≥1/2 vs v7_0).
- Add an emission-rate diagnostic: post-fix, our launch rate in the
  step 50–125 decision window should match ≥ 80 % of opp's rate on
  panel games. This is a direct measurable of "did the chooser see
  the pressure."
