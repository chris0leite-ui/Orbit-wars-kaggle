"""Extract a single-step snapshot from a Kaggle live-ladder replay.

Slices `audit/live-episodes/<sub>/episode-<eid>-replay.json` at a chosen
`(step_idx, seat)` and writes a compact JSON file suitable for committing as
a `tests/fixtures/replays/` regression fixture (replays themselves are
~4-15 MB and gitignored).

The produced fixture is structured for `lib.fast_sim.from_obs(obs, config,
episode_seed=seed, num_seats=N)` plus `agents.baseline.main.agent(obs,
config)` — both consume the embedded `obs` dict directly.

CLI:
    python -m scripts.extract_replay_snapshot \\
        --replay audit/live-episodes/52827111/episode-77150441-replay.json \\
        --step 44 --seat 3 \\
        --out tests/fixtures/replays/linrock_77150441_step44.json \\
        --label "linrock home-rush — 78-ship fleet 11u out, our garrison 9"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def extract(
    replay_path: Path,
    step_idx: int,
    seat: int,
    label: str,
) -> dict:
    with replay_path.open() as fh:
        replay = json.load(fh)

    steps = replay.get("steps", [])
    if step_idx < 0 or step_idx >= len(steps):
        raise SystemExit(
            f"step {step_idx} out of range [0, {len(steps)})"
        )
    step_entries = steps[step_idx]
    if seat < 0 or seat >= len(step_entries):
        raise SystemExit(
            f"seat {seat} out of range [0, {len(step_entries)})"
        )

    obs = dict(step_entries[seat]["observation"])
    obs.setdefault("step", step_idx)
    obs.setdefault("player", seat)

    info = replay.get("info", {})
    cfg = replay.get("configuration", {})
    teams = info.get("TeamNames", [])
    num_seats = len(teams) if teams else len(step_entries)

    return {
        "label": label,
        "source_replay": str(replay_path),
        "episode_id": info.get("EpisodeId"),
        "seed": info.get("seed"),
        "team_names": teams,
        "my_seat": seat,
        "num_seats": num_seats,
        "step_idx": step_idx,
        "configuration": cfg,
        "recorded_action": step_entries[seat].get("action", []) or [],
        "obs": obs,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--replay", required=True, type=Path)
    p.add_argument("--step", required=True, type=int)
    p.add_argument("--seat", required=True, type=int)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--label", required=True)
    args = p.parse_args(argv)

    snap = extract(args.replay, args.step, args.seat, args.label)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as fh:
        json.dump(snap, fh, indent=2, sort_keys=False)
    size = args.out.stat().st_size
    print(
        f"wrote {args.out} ({size} bytes)  "
        f"seat={args.seat}/{snap['num_seats']}  step={args.step}  "
        f"label={args.label!r}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
