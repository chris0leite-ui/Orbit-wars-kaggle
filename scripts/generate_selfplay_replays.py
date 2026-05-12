"""Generate self-play replays in the live-ladder format for Phase 0 idle-trace.

Writes `episode-<seed>-replay.json` files matching the schema that
`scripts/episode_postmortem.py` consumes (`info.TeamNames`, `rewards`,
`steps[t][seat]["observation"]`), so the postmortem's idle-source
instrumentation can be exercised end-to-end on locally-generated games
when no Kaggle-pulled archive is available.

CLI:
    python -m scripts.generate_selfplay_replays \
        --agent agents/v3_snipe/main.py \
        --out audit/live-episodes/SELFPLAY_PHASE0 \
        --seeds 8 --players 2

Each game runs `players` copies of the same agent. The output directory
mirrors the `audit/live-episodes/<submission_id>/` layout consumed by
`episode_postmortem.py`. Use `--players 2` or `--players 4`.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from kaggle_environments import make  # noqa: E402


def _load_agent(path: Path):
    spec = importlib.util.spec_from_file_location("_selfplay_agent", str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_selfplay_agent"] = mod
    spec.loader.exec_module(mod)
    return mod.agent


def _wrap_replay(env, team_name: str, n_seats: int) -> dict:
    """Inject `info.TeamNames` and top-level `rewards` so the postmortem reader
    sees the same schema as a Kaggle-pulled live replay."""
    payload = env.toJSON()
    final = payload["steps"][-1]
    payload["info"] = {"TeamNames": [team_name] * n_seats}
    payload["rewards"] = [seat.get("reward") for seat in final]
    return payload


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--agent", required=True, help="Path to agent main.py")
    p.add_argument("--out", required=True, help="Output dir")
    p.add_argument("--seeds", type=int, default=8, help="Number of self-play games")
    p.add_argument("--players", type=int, default=2, choices=(2, 4))
    p.add_argument("--seed-base", type=int, default=42)
    args = p.parse_args(argv)

    agent_fn = _load_agent(Path(args.agent))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    team_name = os.environ.get("KAGGLE_USERNAME", "selfplay")

    t0 = time.perf_counter()
    for i in range(args.seeds):
        seed = args.seed_base + i
        env = make("orbit_wars", configuration={"seed": seed}, debug=False)
        env.run([agent_fn] * args.players)
        payload = _wrap_replay(env, team_name, args.players)
        out_path = out_dir / f"episode-{seed:08d}-replay.json"
        out_path.write_text(json.dumps(payload) + "\n")
        print(f"  [{i+1}/{args.seeds}] seed={seed} steps={payload['n_steps' if 'n_steps' in payload else 'steps'] if 'n_steps' in payload else len(payload['steps'])} -> {out_path.name}")
    print(f"elapsed: {time.perf_counter() - t0:.1f}s; out={out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
