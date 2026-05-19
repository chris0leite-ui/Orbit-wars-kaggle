# 2026-05-19 PM — trajectory_roi saturation + analytics-verification pivot

## What I learned about analytical agents in this game

After five trajectory_roi iterations (v1, v1.1, v2, v3, v3.1) all
losing to baseline 0-1/32, three uncomfortable truths surfaced:

**1. Value-maximization via forward projection is structurally
limited.** Our `project()` simulates K=30 turns with `lite_greedy`
as the opp model. The marginal "value" of any candidate action is
computed against that lite_greedy opp. The REAL opp (baseline,
v15-class) is materially stronger than lite_greedy. So we size
captures for a weaker opp → captures arrive under-margined →
bounce in real games. The benchmark proved that running v2 (our
own analytical agent) as the projection's opp is computationally
infeasible (130-216 seconds per turn against the 1000ms env cap).
This means: **for pure forward-projection to beat baseline, we
need a CLOSED-FORM opp model that's at least as strong as
baseline.** lite_greedy is not it.

**2. We've been building on unverified analytical primitives.**
Five iterations of architecture (mirror, multi-source, defense,
forward-projection, joint 2-opt → forward-greedy) on top of:
- `_aim_and_eta` — closed-form fleet ETA. Never compared to env.
- `_net_defenders` — closed-form defender growth + in-flight
  fleet accounting. Never compared to env.
- `project()` — bit-exact env via fast_sim. Trusted but never
  validated end-to-end.
- `delta_us_minus_them` terminal — trusted from lib.

PI's question this turn was load-bearing: **"How can we know
we're doing everything right analytically?"** The honest answer is
we don't know. The next session's first work-item is to find out.

**3. Goal-directed beats value-directed in this game.** PI
articulated the right architecture: define a closed-form
**winning state** (our production advantage × remaining turns >
opp recovery pool), identify the **smallest sufficient portfolio
of planets** to reach it, plan **backwards** to acquire them.
- The objective is structural ("own these planets"), not a noisy
  scalar from a projection.
- Every action has a measurable benefit ("brings us 1 step
  closer to portfolio completion").
- Defense emerges naturally — portfolio members we already own
  MUST be preserved.
- Compute is O(planets² × portfolio_size) — closed-form, no
  rollouts, easily fits in 50ms/turn.

## Anti-pattern I logged about myself

Iterating on the same architecture (forward-projection joint
solve) through 5 versions without verifying the primitives is the
**"add features before checking your math" anti-pattern.** Each
v_(n+1) added a knob, parameter, or sub-feature. None addressed
the systemic risk of "what if my ETA formula is wrong by 1 turn"
or "what if my defender count silently drops in-flight fleets in
one branch but not another."

The Rule 41 candidate from today's friction log captures this:
"no v_(n+1) on a v_(n) that didn't beat random+10%." Random is
the absolute floor; if we don't beat it convincingly, we have
analytical bugs, not strategic ones.

## What changed in PI's framing this session

Three reframings in chronological order:

1. **Mid-session** ("simple stuff first"): replace v1's per-source
   greedy with real joint optimization. Done in v2 + v3 (joint
   2-opt → forward-greedy).
2. **Later** ("backwards analytical"): solve from the future
   backwards. We benchmarked and built v3 + v3.1 with K=30-50
   forward-projection. Both 0/32.
3. **End-of-session** ("are we doing analytics right? identify
   the winning state and solve backwards from THAT"): PI gave up
   on value-maximization; pivoted to goal-state-first
   architecture. v4 plan = verification + portfolio planner.

The third reframing is the load-bearing one. It changes the
*objective* of the agent from "maximize expected ship-diff at
horizon" to "reach a state from which we provably win." Both are
analytical; the second is much more verifiable.

## What's in next-session's pocket

- Five concrete analytics tests (projection-vs-reality is the
  killer one).
- v4 architecture sketch (winning-state → portfolio →
  acquisition planner → defense).
- Copy-paste prompt at the bottom of
  `/root/.claude/plans/read-the-handover-do-abundant-quokka.md`.
- The DI1 + G1 scenario substrate as a regression gate.
- v3.1 in git history as the "best version of the bad
  architecture" — useful as a benchmark against v4.

The work for next session is well-scoped (~400 LOC total,
broken into verification 150 + agent 250). If verification
surfaces bugs, those must be fixed before agent code. That
discipline matters more than getting v4 to compile on day 1.
