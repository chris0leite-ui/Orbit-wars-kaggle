"""Re-run seed=2083 (the first P0 win in the mirror n=5 sweep) with
per-call instrumentation of `_select_opp_policy` and direct sampling
of `top_tier_mirror_policy` / `lite_greedy_policy` invocations.

Goal: confirm the mirror bake is actually firing — i.e. that
submissions/baseline_pv_eta_mirror_opp.py's chooser is calling
top_tier_mirror_policy during rollouts, NOT silently falling back
to lite_greedy because of module-namespace contamination.

If counts of mirror-side `_select_opp_policy → top_tier_mirror_policy`
returns are 0 in this run, the bake is broken and the 5-0 result
is a wiring bug, not a real falsification.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def _load(path: str, mod_name: str):
    spec = importlib.util.spec_from_file_location(mod_name, str(REPO / path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    print("=== Loading bundles ===", flush=True)
    p0_mod = _load("submissions/baseline_pv_eta_anchor_1163.py", "_p0_anchor")
    p1_mod = _load("submissions/baseline_pv_eta_mirror_opp.py", "_p1_mirror")

    # Sanity: each bundle's _select_opp_policy should be DISTINCT objects
    # and each should return the expected policy.
    print(f"\n=== Bundle-level dispatch sanity ===", flush=True)
    print(f"  p0 (anchor)  _select_opp_policy id: {id(p0_mod._select_opp_policy)}", flush=True)
    print(f"  p1 (mirror)  _select_opp_policy id: {id(p1_mod._select_opp_policy)}", flush=True)
    assert id(p0_mod._select_opp_policy) != id(p1_mod._select_opp_policy), \
        "BUG: both bundles share a dispatch function (module namespace collapse)"
    p0_sel = p0_mod._select_opp_policy()
    p1_sel = p1_mod._select_opp_policy()
    print(f"  p0._select_opp_policy() → {p0_sel.__module__}.{p0_sel.__name__}", flush=True)
    print(f"  p1._select_opp_policy() → {p1_sel.__module__}.{p1_sel.__name__}", flush=True)
    assert p0_sel.__name__ == "lite_greedy_policy", \
        f"BUG: P0 dispatch should be lite_greedy, got {p0_sel.__name__}"
    assert p1_sel.__name__ == "top_tier_mirror_policy", \
        f"BUG: P1 dispatch should be top_tier_mirror, got {p1_sel.__name__}"

    # Verify the policy functions are DIFFERENT objects in the two bundles
    # (each bundle has its own inlined definition).
    print(f"  p0 lite_greedy id:  {id(p0_mod.lite_greedy_policy)}", flush=True)
    print(f"  p1 lite_greedy id:  {id(p1_mod.lite_greedy_policy)}", flush=True)
    print(f"  p0 top_tier id:     {id(p0_mod.top_tier_mirror_policy)}", flush=True)
    print(f"  p1 top_tier id:     {id(p1_mod.top_tier_mirror_policy)}", flush=True)

    # Instrument: wrap each bundle's policy functions to count invocations.
    counts = {
        "p0_lite_greedy": 0,
        "p0_top_tier_mirror": 0,
        "p0_select_calls": 0,
        "p0_select_returns_lite_greedy": 0,
        "p0_select_returns_top_tier_mirror": 0,
        "p1_lite_greedy": 0,
        "p1_top_tier_mirror": 0,
        "p1_select_calls": 0,
        "p1_select_returns_lite_greedy": 0,
        "p1_select_returns_top_tier_mirror": 0,
    }

    def make_wrapper(label_calls, label_lite, label_mirror, mod):
        orig_select = mod._select_opp_policy
        orig_lite = mod.lite_greedy_policy
        orig_mirror = mod.top_tier_mirror_policy

        def lite_wrapped(obs):
            counts[label_lite] += 1
            return orig_lite(obs)

        def mirror_wrapped(obs):
            counts[label_mirror] += 1
            return orig_mirror(obs)

        def select_wrapped():
            counts[label_calls] += 1
            r = orig_select()
            # original returned the unwrapped reference; rewrite to the
            # wrapped one so subsequent calls are counted too.
            if r is orig_lite:
                counts[f"{label_calls}_returns_lite_greedy"] = (
                    counts.get(f"{label_calls}_returns_lite_greedy", 0) + 1
                )
                return lite_wrapped
            if r is orig_mirror:
                counts[f"{label_calls}_returns_top_tier_mirror"] = (
                    counts.get(f"{label_calls}_returns_top_tier_mirror", 0) + 1
                )
                return mirror_wrapped
            return r

        mod.lite_greedy_policy = lite_wrapped
        mod.top_tier_mirror_policy = mirror_wrapped
        mod._select_opp_policy = select_wrapped

    make_wrapper("p0_select_calls", "p0_lite_greedy", "p0_top_tier_mirror", p0_mod)
    make_wrapper("p1_select_calls", "p1_lite_greedy", "p1_top_tier_mirror", p1_mod)

    # Run the game.
    print(f"\n=== Running seed=2083 ===", flush=True)
    from kaggle_environments import make
    env = make("orbit_wars", configuration={"seed": 2083}, debug=False)
    env.run([p0_mod.agent, p1_mod.agent])
    final = env.steps[-1]
    print(f"  rewards: P0={final[0].reward}  P1={final[1].reward}", flush=True)
    print(f"  steps: {len(env.steps)}", flush=True)

    print(f"\n=== Invocation counts ===", flush=True)
    for k, v in counts.items():
        print(f"  {k:42s}: {v}", flush=True)

    # Verdict.
    print(f"\n=== Verdict ===", flush=True)
    bake_ok = (
        counts["p1_top_tier_mirror"] > 0
        and counts["p1_lite_greedy"] == 0
        and counts["p0_lite_greedy"] > 0
        and counts["p0_top_tier_mirror"] == 0
    )
    if bake_ok:
        print("  BAKE OK: P1's rollouts called top_tier_mirror exclusively;", flush=True)
        print("           P0's rollouts called lite_greedy exclusively.", flush=True)
        print("           The 5-0 result is a real falsification, not a bake bug.", flush=True)
        return 0
    else:
        print("  BAKE BROKEN:", flush=True)
        if counts["p1_top_tier_mirror"] == 0:
            print("    P1 NEVER called top_tier_mirror — bake didn't take effect.", flush=True)
        if counts["p1_lite_greedy"] > 0:
            print(f"    P1 called lite_greedy {counts['p1_lite_greedy']}× — leak.", flush=True)
        if counts["p0_top_tier_mirror"] > 0:
            print(f"    P0 called top_tier_mirror {counts['p0_top_tier_mirror']}× — leak.", flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
