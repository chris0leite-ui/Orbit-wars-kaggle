"""Generate v3_snipe self-play replays for bounce-margin postmortem.

Drops episode-<id>-replay.json + summary.json into audit/live-episodes/
SELFPLAY_MARGIN/ so the (extended) `episode_postmortem.py` can compute
per-fleet margin distributions on a clean v3_snipe corpus.

Usage:
    python -m scripts.gen_selfplay_replays_for_margin --n 10
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from kaggle_environments import make  # noqa: E402


def _load(path: str):
    spec = importlib.util.spec_from_file_location("_a", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.agent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10, help="number of episodes")
    parser.add_argument("--agent", default="v3_snipe",
                        help="agent dir name under agents/")
    parser.add_argument("--out", default="SELFPLAY_MARGIN",
                        help="subdir under audit/live-episodes/")
    args = parser.parse_args()

    agent_path = REPO / "agents" / args.agent / "main.py"
    agent_fn = _load(str(agent_path))
    out_dir = REPO / "audit" / "live-episodes" / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for i in range(args.n):
        seed = 1000 + i
        env = make("orbit_wars", configuration={"seed": seed}, debug=False)
        env.reset(num_agents=2)
        state = env.run([agent_fn, agent_fn])
        # Build a replay dict compatible with episode_postmortem.py's expectations.
        replay = {
            "info": {"TeamNames": [args.agent, args.agent]},
            "rewards": [state[-1][0]["reward"], state[-1][1]["reward"]],
            "steps": state,
            "configuration": dict(env.configuration),
        }
        eid = f"sp{seed}"
        with open(out_dir / f"episode-{eid}-replay.json", "w") as f:
            json.dump(replay, f)
        result = ("p0_win" if replay["rewards"][0] > replay["rewards"][1]
                  else ("p1_win" if replay["rewards"][1] > replay["rewards"][0] else "draw"))
        print(f"[{i+1}/{args.n}] seed={seed} steps={len(state)} result={result}")
        summaries.append({"episode_id": eid, "seed": seed, "result": result,
                          "n_steps": len(state)})

    utc = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps({
        "generated_at_utc": utc,
        "agent": args.agent,
        "n_episodes": args.n,
        "episodes": summaries,
    }, indent=2))
    print(f"\nWrote {args.n} replays to {out_dir}")


if __name__ == "__main__":
    raise SystemExit(main())
