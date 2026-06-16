# 2026-06-16 — A simulation-driven "least resistance" agent

PI ask (paraphrased, plain English): build an agent that uses our accurate +
fast physics to streamline ships along the *path of least resistance* to gain
production, minimizing the travel-time (ETA) of its ships to the closest
opponent at all times, in a coordinated way. Follow-up: "It needs to be smart.
Choose the next step that is optimal over a simulation period, use how the
strongest agents work, avoid tuning knobs."

New agent lives at `agents/least_resistance/` (committed on
`claude/dreamy-fermi-8unqi5`). NOT submitted — submission stays gated on PI
sign-off (Rules 1/12/42).

## The one lesson worth keeping

**A weighted-heuristic scorer bled ships and lost to the comp's `nearest`
sniper (6%). Replacing the weighted score with forward simulation —
"only commit a launch if it improves the simulated K-step ship-delta" — lifted
it to 94% vs `nearest` with the SAME candidate moves.** The selection rule,
not the move menu, was the bottleneck. Reserves, gang-up-vs-solo,
attack-vs-expand, "don't bleed ships," and "accumulate when nothing pays" all
emerge from the simulation; none of them needed a knob.

This is a concrete instance of CLAUDE.md Rule 40 (model-correctness over
restriction-tuning): the right behaviour came from a correct evaluator, not
from tuned thresholds.

## How it works (final design)

Each turn:
1. **Enumerate coordinated candidate moves** with the accurate physics
   (`lib/aim` for orbit/comet lead, `lib/fleet` ETA): per non-owned planet, the
   capture launch from the cheapest source, or a multi-source gang-up when one
   source can't afford it; plus forward-staging moves (stream a rear planet's
   idle excess toward the friendly planet nearest the front).
2. **Order** by path-of-least-resistance-to-production (production ÷ ETA),
   tie-broken toward the nearest opponent (the only explicit nod to "minimize
   ETA to opponent"). Ordering decides what we *try first*, not what we keep.
3. **Select by simulation.** Greedily add a candidate to the plan only if it
   raises the simulated horizon ship-delta. Value = roll forward
   `SIM_HORIZON` turns with both sides on `lite_greedy_policy`
   (`lib/fast_sim` + `lib/opp_model`), score with `delta_us_minus_them`. Stop
   when nothing improves.

The leaf value (ships at the horizon) is the self-calibrating "more production"
objective — no strategy weights. The only parameters are compute bounds
(horizon, candidate cap, wallclock), set conservatively.

## Calibration cost surprise (banked)

`opp_model.lite_greedy_policy` is **0.023 ms/call**, not the ~1 ms its docstring
implies; a K=12 fast_sim rollout is ~4 ms. So a full per-candidate rollout
search is cheap: ~100+ rollouts fit in 500 ms. This makes "simulate every
candidate move every turn" affordable for a heuristic-class agent. Worth
remembering for any future search agent — the per-candidate rollout budget is
much larger than the old 0.6-CPU-era intuition.

## Where it stands

- Smoke: `random` 32/32 (100%), `nearest` 30/32 (94%) — PASS.
- Timing (single process, honest): p50=74 ms, p95=400 ms, **max=682 ms, zero
  turns ≥1000 ms**. (Smoke's 908 ms p95 was 8-worker CPU contention; Kaggle
  gives one agent 1.6 cores.)
- vs `v7_0` (our tuned champion, ~μ1115): behind (bench 0/3; fuller A/B noted
  in the session log). Expected: the agent's strategic ceiling is roughly
  "lite_greedy + one ply of exhaustive tactical search," so it crushes the comp
  baselines but trails an agent tuned over weeks.

## The ceiling, and the lever to raise it (for next session / PI call)

The rollout policy (`lite_greedy` for both seats) sets the strategic level. To
make it competitive with `v7_0`/producer, the principled (non-knob) lever is a
**stronger rollout policy** — e.g. the producer's `orbit_lite` evaluator or a
production-aware leaf — so the simulation reflects strong future play and
punishes over-extension the way a strong opponent would. That's a larger,
compute-sensitive change; flagged for PI decision rather than done blind.
Longer horizon (a compute parameter) is the cheap first thing to test.

## Infra notes (cost me time today — avoid the repeat)

- **Only one background bash task runs at a time here**: launching a second
  background task SIGTERMs (143) the running one. Don't launch a "waiter"
  background task while a heavy run is going — read its output file from the
  foreground instead. Foreground commands are safe alongside a background task.
- `kaggle_environments` floods stdout/stderr with the OpenSpiel registry on
  every worker import; filter with `grep -vi open_spiel`. Don't `2>/dev/null`
  blindly — it also hides real tracebacks (bit me twice).
- `scripts/bundle_agent.py`'s DEFAULT_LIB_ORDER is missing `kinematic_table`
  (lazily imported by `lib/trajectory.py`), so the default order can't bundle
  any agent that pulls in `trajectory`. Pass an explicit `--lib` list with
  `kinematic_table` after `orbit`. (Left the shared default untouched to avoid
  perturbing champion bundles.)
- Time-budgeted agents need `ORBIT_WARS_PARITY_WALLCLOCK_MS` honoured at CALL
  time, or the bundle parity gate fails on timing nondeterminism. Verified the
  bundle 34/34 against source with the override on.
