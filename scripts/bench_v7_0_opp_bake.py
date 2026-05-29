"""Quick wallclock/invocation-count bench for the v7_0-as-opp bake.

Same logic as scripts/verify_mirror_bake.py but truncated to a short
game and instruments per-turn opp-policy invocation count.

Goal: decide if the v7_0 bake is too compute-starved to be worth A/B.
"""

from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def _load(path: str, mod_name: str):
    spec = importlib.util.spec_from_file_location(mod_name, str(REPO / path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    print("=== Loading bundles ===", flush=True)
    p0_mod = _load("submissions/baseline_pv_eta_anchor_1163.py", "_p0_anchor_b")
    p1_mod = _load("submissions/baseline_pv_eta_v7_0_opp.py", "_p1_v7_0_opp_b")
    print(f"  p0._select_opp_policy() → {p0_mod._select_opp_policy().__name__}", flush=True)
    print(f"  p1._select_opp_policy() → {p1_mod._select_opp_policy().__name__}", flush=True)

    # Instrument the opp-policy invocations on each side.
    counts = {"p0": 0, "p1": 0}
    times = {"p0_total": 0.0, "p1_total": 0.0}

    orig_lite = p0_mod.lite_greedy_policy
    def wrap_lite(obs):
        counts["p0"] += 1
        t0 = time.perf_counter()
        r = orig_lite(obs)
        times["p0_total"] += time.perf_counter() - t0
        return r
    p0_mod.lite_greedy_policy = wrap_lite

    orig_v7 = p1_mod.v7_0_opp_policy
    def wrap_v7(obs):
        counts["p1"] += 1
        t0 = time.perf_counter()
        r = orig_v7(obs)
        times["p1_total"] += time.perf_counter() - t0
        return r
    p1_mod.v7_0_opp_policy = wrap_v7

    # Re-route the bake to the wrapped function. We monkey-patch
    # the module-level `v7_0_opp_policy` reference so that the
    # bundle's _select_opp_policy() picks up the wrapper.
    p1_mod._select_opp_policy = lambda: wrap_v7

    print(f"\n=== Running 60-step game (anchor vs v7_0-opp-baked, seed=2083) ===", flush=True)
    from kaggle_environments import make
    env = make("orbit_wars", configuration={"seed": 2083, "episodeSteps": 60}, debug=False)
    t0 = time.perf_counter()
    env.run([p0_mod.agent, p1_mod.agent])
    elapsed = time.perf_counter() - t0
    steps = len(env.steps)

    print(f"  steps: {steps}  elapsed: {elapsed:.1f}s", flush=True)
    print(f"\n=== Opp-policy invocation counts (P0=lite_greedy, P1=v7_0) ===", flush=True)
    print(f"  P0 lite_greedy calls: {counts['p0']:7d}  total {times['p0_total']:.2f}s  "
          f"avg {1000*times['p0_total']/max(1,counts['p0']):.3f} ms/call", flush=True)
    print(f"  P1 v7_0       calls: {counts['p1']:7d}  total {times['p1_total']:.2f}s  "
          f"avg {1000*times['p1_total']/max(1,counts['p1']):.3f} ms/call", flush=True)
    print(f"  per-turn P0: {counts['p0']/max(1,steps):.0f}", flush=True)
    print(f"  per-turn P1: {counts['p1']/max(1,steps):.0f}", flush=True)
    ratio = counts['p0'] / max(1, counts['p1'])
    print(f"  P0/P1 ratio: {ratio:.1f}x  (mirror ratio was 22x)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
