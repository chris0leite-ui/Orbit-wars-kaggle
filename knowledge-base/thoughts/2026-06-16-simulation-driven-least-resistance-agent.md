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
- **Only one background bash task runs at a time** — starting a new background
  task SIGTERMs the running one (cost me three killed A/B runs). Run heavy
  A/Bs as the *sole* background task and read their progress from a file the
  script flushes itself; never launch a "waiter" task alongside.

## UPDATE (same session) — the orbit_lite evaluator made it competitive

PI: "Build a stronger evaluator" + "install torch if you need it." Findings:

- **torch wasn't installed** (orbit_lite/producer couldn't load). Installed CPU
  torch (`--index-url https://download.pytorch.org/whl/cpu`).
- **Diagnosis confirmed by the horizon sweep:** the weak `lite_greedy` rollout
  was the ceiling. K=14 → 12% vs v7_0; K=26 → **0%** (longer = worse, because
  more steps of a weak policy compound the error). Foresight wasn't the
  problem; the evaluator was.
- **The fix:** swap the leaf from a fast_sim rollout to the producer's
  `orbit_lite.score_candidates` — a garrison-flow projection (~18 turns) that
  returns competitive net-ship-delta. Production-aware, policy-free, ~1-2 ms.
  Keep `least_resistance`'s candidate generation; only the evaluator changed.
- **Result: vs v7_0 12% → 62%** (n=16), faster than the rollout version
  (max ~59 ms). vs the full producer: 0/10 (it also has reactive defense +
  idle-ship regroup we lack — the gap to close next). So the agent now sits
  between v7_0 (~μ1115) and the producer (~μ1280).

Lesson restated: for a search/eval agent, the **leaf evaluator is the
strength ceiling.** A cheap-but-strong projector (orbit_lite's garrison flow)
beats a cheap-but-weak rollout policy decisively, and is cheaper to boot.

Verified APIs that work (torch): `single_obs_to_tensor(obs, player_id)` →
`ensure_planet_movement(obs_tensors, expected_cfg=MovementConfig(...),
cached_movement=None)` → `movement.garrison_status(max_horizon=H)`,
`movement.planet_prod`, `movement.alive_by_step[:H+1]` → `make_launch_set(
source_slots, target_slots, ships, eta, valid, player_id)` (slots are planet
ROW INDICES, not ids) → `score_candidates(status, prod=, alive_by_step=,
player_count=, launches=, player_id=)` → `[C]` net-ship-delta. Smoke checks:
empty plan = 0; capture of a gar-5/prod-5 planet at eta 8 = 45 = prod·(H−eta)−gar.

Open: producer is STATEFUL (`_RUNTIME` movement cache, resets on step 0) — do
NOT call `producer.agent` on hypothetical obs as a rollout policy. The scorer
path (build a fresh movement per turn) is safe. Bundling now needs torch +
the orbit_lite package → tar.gz submission, not the single-file bundler.

## UPDATE 2 — chasing producer-parity with defense+regroup FAILED (reverted)

PI: "close the producer gap." The producer has two things least_resistance
lacked: reactive defense (reinforce planets about to flip) and idle-ship
regroup. Added both (defense as scorer-gated reinforce candidates off the
orbit_lite flip projection; regroup as an unscored forward-staging overlay).

**Result (n=16/12/8):** vs producer **0/16** (unchanged — didn't close the
gap), vs v7_0 **2/12 = 17%** (REGRESSED from 62%), vs nearest 8/8. Net-harmful
→ **reverted** to the clean orbit-eval version (commit 490f68b / revert
b27a609).

Why it hurt: (a) the **unscored** regroup over-extends — it streams rear excess
forward every turn with only a reserve/threat guard, so the rear thins and a
strong opponent (v7_0) punishes it; (b) the **static-opp** flip projection
marks many of my planets as "flipping" (it assumes the opponent's in-flight
fleets all land), so defensive reinforcements over-fire and drain the offense
the orbit scorer would otherwise spend on captures.

Lesson: the producer's edge over "my-candidates + producer-evaluator" is its
**whole tuned planner** (multi-size + multi-wave candidates, refined
offensive/defensive shortlists, per-player-count configs), not two bolt-on
behaviours. Bolting features onto a different candidate generator, validated
only against a weak/biased projection signal, regressed the strong baseline.
Producer-parity = reimplement the producer's candidate generation ≈ rebuild
the producer (which already exists) — not worth it. **Keep least_resistance as
the orbit-eval agent that beats v7_0 ~62%.**
