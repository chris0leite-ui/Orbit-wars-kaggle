"""Count how often the rerank PROMOTES a non-top-1 candidate to top-1.

Patches `_adversarial_rerank_opening` to log every call with the
result (None = no promotion, else the promoted index). Then runs a
single game and reports counts.

This tells us whether the rerank is actually changing decisions, or
silently picking the same top-1 every turn (no-op overhead).
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("KAGGLE_ENV_QUIET", "1")
os.environ["BASELINE_OPP_MARCO"] = "1"
os.environ["BASELINE_ADVERSARIAL_RERANK"] = "1"
os.environ["BASELINE_ADV_RERANK_MARCO_BUDGET_MS"] = "150.0"


def _load(path: str, alias: str):
    spec = importlib.util.spec_from_file_location(alias, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--start-seed", type=int, default=0)
    ap.add_argument("--opp", required=True)
    ap.add_argument("--max-step", type=int, default=80)
    args = ap.parse_args()

    from kaggle_environments import make
    from agents.baseline import chooser_trajectory as ct

    promote_log: list[tuple[int, int, int]] = []  # (game_seed, step, promoted_idx_or_-1)
    original_rerank = ct._adversarial_rerank_opening

    def _wrapped_rerank(snap_base, scored_top_k, *args_, **kwargs_):
        idx = original_rerank(snap_base, scored_top_k, *args_, **kwargs_)
        promote_log.append((idx if idx is not None else -1, len(scored_top_k)))
        return idx

    ct._adversarial_rerank_opening = _wrapped_rerank

    from agents.baseline.main import agent as focal_agent
    opp_mod = _load(args.opp, "_opp_probe")

    per_seed: list[dict] = []
    for off in range(args.seeds):
        seed = args.start_seed + off
        promote_log.clear()
        env = make("orbit_wars", configuration={"seed": seed,
                                                "episodeSteps": args.max_step + 5,
                                                "actTimeout": 1.0},
                   debug=False)
        env.reset()
        state = env.state
        for step in range(args.max_step + 1):
            obs0 = state[0]["observation"]
            obs1 = state[1]["observation"]
            cfg = env.configuration
            try:
                act0 = focal_agent(obs0, cfg)
            except Exception as e:
                print(f"seed={seed} step={step}: focal crashed: {e!r}")
                break
            try:
                act1 = opp_mod.agent(obs1, cfg)
            except Exception:
                act1 = []
            env.step([act0, act1])
            state = env.state
            if all(s.get("status") in ("DONE","INACTIVE","ERROR") for s in state):
                break

        rerank_calls = len(promote_log)
        promotes = sum(1 for r in promote_log if r[0] >= 1)
        keeps = sum(1 for r in promote_log if r[0] == -1)
        avg_top_k = sum(r[1] for r in promote_log) / max(1, rerank_calls)
        per_seed.append({"seed": seed, "calls": rerank_calls,
                         "promotes": promotes, "keeps": keeps,
                         "avg_top_k": avg_top_k})
        print(f"  seed={seed}: rerank_calls={rerank_calls}, "
              f"promotes={promotes} ({100*promotes/max(1,rerank_calls):.0f}%), "
              f"keeps={keeps}, avg_top_k={avg_top_k:.2f}")

    print()
    total_calls = sum(r["calls"] for r in per_seed)
    total_promotes = sum(r["promotes"] for r in per_seed)
    print(f"TOTAL: {total_promotes}/{total_calls} promotes "
          f"({100*total_promotes/max(1,total_calls):.0f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
