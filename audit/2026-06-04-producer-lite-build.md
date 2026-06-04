# Producer-lite opponent model — build + three bugs (2026-06-04)

_Branch JzIAr. Built the design-locked Producer-lite opponent model
(`knowledge-base/concepts/producer-lite-opponent-model-plan.md`). This records
the build, the bugs found, and the verified gate results. Transfer gate (vs our
champion) was still running at write time — see HANDOVER for its outcome._

## What was built

- `lib/producer_lite.py` — pure-python port of the public Producer agent's
  attack policy: do-nothing forward projection (resolving in-flight fleets) →
  `safe_drain` hold-reserve → capture-floor → production-integral proxy for
  Producer's exact competitive flow score → greedy top-6 waves (one per target,
  role-mutex) → exact aim on fired waves only.
- `agents/producer_lite/main.py`, `agents/lite_greedy/main.py` — standalone
  wrappers for the as-agent A/Bs.
- `scripts/bench_producer_lite.py` — speed gate.
- `tests/test_producer_lite.py` — unit tests (contract, safe_drain, max-waves,
  projection). 5/5 green.
- Wired `BASELINE_OPP_TIER=2` into `agents/baseline/chooser._select_opp_policy`
  (lazy import; default-OFF → champion bundle byte-identical; bundle parity test
  green).

## Three bugs (all fixed)

1. **40× speed regression — masked by a bad bench.** First speed run read
   0.22ms but had collected only 16 *early* boards (the board source crashed —
   see bug 2). On real dense boards (44 fleets) it was **17ms mean / 158ms max**
   — fatal on the rollout hot path. Cause: `world_model.fleet_target_planet`
   does a per-tick orbital scan to DEFAULT_HORIZON=250 (~2M `predict_relative`
   calls/board). Fix: closed-form straight-line ray-cast for the threat
   projection (the plan's sanctioned fallback) + bucket arrivals by planet.
   → **0.42ms mean / 1.1ms max.**

2. **Wrappers crashed under the env loader (`__file__`).** kaggle_environments
   execs an agent as raw source with `exec(code, {})` — no `__file__`. Both new
   wrappers referenced `__file__` at module top-level → NameError on load →
   games ended at step 1 (hence bug 1's 16-board bench). Fix: `try/except
   NameError` → cwd fallback.

3. **Fleets fired into walls — static-target over-lead.** producer_lite never
   expanded (stuck at 1 planet, lost). Cause: aim used `aim_orbiting` whenever
   `omega != 0`, but `predict_relative` rotates a *static* planet (orb_r+radius
   ≥ rotation limit) to a bogus future position → the fleet aimed at empty space
   and flew into a wall planet. Fix: gate orbital lead on `is_orbiting` (mirrors
   `me_defensive_action`); static targets use straight `atan2`. After the fix
   producer_lite's opening matches full Producer byte-for-byte and it
   expands+wins.

## The harness-loader bug in the ORACLE (Rule 38 lesson)

The vendored `agents/producer/producer_agent.py` had the *same* `__file__` bug
(bug 2) — so **full Producer crashed on load and idled every game** under
`env.run` / `scripts/clean_ab`. This produced a bogus "Producer loses 0/16 to
lite_greedy," which led to a wrong "non-transitive ranking" conclusion. The PI
flagged it ("this sounds like a bug"). It was.

The trap: an earlier spot-check loaded Producer via `importlib` (which DOES
define `__file__`), so it looked fine — masking the env-loader failure. **Rule
38: verify the *oracle* runs in the *exact* harness before interpreting its
result. Load-by-path (`env.run`, no `__file__`) ≠ load-by-importlib.** Fix:
recover the dir from `sys.path` (the loader appends the agent's own dir before
exec). With the fix, full Producer plays and beats lite_greedy 32/32.

## Verified gate results (post-fixes)

- **Speed:** 0.42ms mean / 1.14ms max on 3824 real boards. PASS (<3ms).
- **Primary fidelity (producer_lite vs lite_greedy, n=32):** 64/64 = 100%,
  Wilson[0.943, 1.0]. PASS (≫0.65).
- **Premise (full Producer vs lite_greedy, n=16):** 32/32 = 100%,
  Wilson[0.893, 1.0]. Confirmed — Producer trounces lite_greedy.
- **Move-agreement vs full Producer:** ~33% (source/target choice). Ship *sizes*
  match 98% when sources agree. → The scoring proxy is faithful in *strength*
  but diverges on *target choice* (the known compaction risk).
- **Integration smoke (champion + refine + OPP_TIER=2 vs baseline, seed 7):**
  runs clean, 220 steps; max turn 803ms < 1000ms **even under 4-core
  contention** (a clean re-run will be lower).
- **Winrate-transfer (vs our champion):** RUNNING at write time. This is the
  decisive fidelity test (does producer_lite beat our champion at Producer's
  rate?). See HANDOVER.

## Open items

- Transfer-gate outcome (above).
- If transfer lags the oracle: lift move-agreement (~33% → ~70%) by tightening
  the competitive-score proxy toward Producer's exact flow delta.
- A clean (un-contended) Rule 46 timing re-run for the official budget number.
