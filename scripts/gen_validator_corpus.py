"""End-to-end corpus generator for the konbu17-style shot validator MLP.

For each `--pairs A:B:N` spec, plays N games between agent A (seat 0) and
agent B (seat 1) via kaggle_environments. After each game completes, walks
the recorded step-by-step observations and for every shot emitted by either
seat:
  - skips self-reinforcement (target already owned by emitting seat) — the
    most important data-side decision per konbu17's notebook
  - encodes the 24-dim per-shot feature vector via `lib.shot_features`
  - labels 1 iff the target is owned by the emitting seat at
    `min(step + eta + 10, end_of_game)`
  - appends one row `{features, label, game_id, seat, step}` to labels.jsonl

This avoids `scripts/label_shot_outcomes.py`'s focal-team filename-detection
logic (which expects competition-replay naming like `r01-team-2P-W-id.json`)
by labeling both seats directly during play.

Usage:
    python -m scripts.gen_validator_corpus \\
        --pairs agents/baseline/main.py:agents/baseline/main.py:10 \\
        --pairs agents/baseline/main.py:agents/baseline_full/main.py:10 \\
        --pairs agents/baseline/main.py:agents/v3_snipe/main.py:10 \\
        --workers 8 --wallclock-ms 100
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
import time
from multiprocessing import Pool, get_context
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from lib.shot_features import (  # noqa: E402
    NORM,
    encode_features,
    fleet_speed,
    infer_target_pid,
)

DEFAULT_OUT = REPO / "data" / "shot_validator" / "labels.jsonl"
LABEL_BUFFER = 10  # konbu17: 10-turn buffer after eta


def _load_agent_callable(path: str):
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"agent file not found: {path}")
    spec = importlib.util.spec_from_file_location(
        f"_agent_{p.stem}_{abs(hash(path))}", str(p)
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod.agent


def _label_shots_from_game(steps: list, game_id: str) -> list[dict]:
    """For each (step, seat, emit), encode features + lookup ownership at
    step + eta + LABEL_BUFFER. Skip self-reinforcement.

    Phase 2 v2: World + WorldModel are built once per game step and threaded
    into encode_features so the Tier 1+2 features don't pay the ~5 ms build
    per emit (~50 emits / turn would eat 250 ms / turn otherwise).
    """
    rows: list[dict] = []
    n_steps = len(steps)
    if n_steps < 2:
        return rows

    from lib.intent import World  # local import keeps worker boot cheap
    from lib.world_model import WorldModel

    for step_idx, step in enumerate(steps):
        # Build per-step world view once. Both seats observe the same shared
        # planets/fleets in orbit_wars; use seat 0's obs as canonical.
        obs0 = step[0].get("observation", {}) or {}
        if obs0.get("planets"):
            try:
                world = World.from_obs(obs0)
                world_model = WorldModel.from_world(world)
            except Exception:
                world = None
                world_model = None
        else:
            world = None
            world_model = None

        for seat in range(len(step)):
            seat_state = step[seat]
            obs = seat_state.get("observation", {}) or {}
            planets = obs.get("planets", []) or []
            fleets = obs.get("fleets", []) or []
            action = seat_state.get("action") or []
            if not action:
                continue
            by_id = {int(p[0]): p for p in planets}

            for a in action:
                if not a or len(a) < 3:
                    continue
                try:
                    src_pid = int(a[0])
                    angle = float(a[1])
                    ships = float(a[2])
                except (TypeError, ValueError):
                    continue
                src = by_id.get(src_pid)
                if src is None:
                    continue
                target_pid = infer_target_pid(
                    (float(src[2]), float(src[3])), angle, planets
                )
                if target_pid is None:
                    continue
                target = by_id.get(target_pid)
                if target is None:
                    continue

                # Self-reinforcement filter (konbu17): if target.owner == seat
                # at LAUNCH time, skip entirely. These are never filtered at
                # inference either; including them in training degenerates
                # pos_rate to ~0.96 and the validator collapses.
                if int(target[1]) == seat:
                    continue

                d = math.hypot(float(target[2]) - float(src[2]),
                               float(target[3]) - float(src[3]))
                v = fleet_speed(ships)
                if v <= 0:
                    continue
                eta = int(math.ceil(d / max(v, 1e-6)))

                check_step = min(step_idx + eta + LABEL_BUFFER, n_steps - 1)
                if check_step >= n_steps:
                    continue
                check_obs = steps[check_step][seat].get("observation", {}) or {}
                check_planets = check_obs.get("planets", []) or []
                check_by_id = {int(p[0]): p for p in check_planets}
                target_check = check_by_id.get(target_pid)
                if target_check is None:
                    continue
                label = 1 if int(target_check[1]) == seat else 0

                feats = encode_features(
                    src, target, ships, d, eta, v,
                    planets, fleets, seat, step_idx,
                    obs=obs, world=world, world_model=world_model,
                    aim_angle=angle,
                )
                rows.append({
                    "features": feats,
                    "label": label,
                    "game_id": game_id,
                    "seat": seat,
                    "step": step_idx,
                })
    return rows


def _run_one_game(task: tuple) -> list[dict]:
    """Worker: load agents, run one game, return labeled rows. Sets env
    vars (wallclock cap) BEFORE importing the agent so they're respected."""
    seed, a_path, b_path, wallclock_ms, pair_tag = task
    if wallclock_ms is not None:
        os.environ["BASELINE_WALLCLOCK_MS"] = str(int(wallclock_ms))

    from kaggle_environments import make  # noqa: E402

    agent_a = _load_agent_callable(a_path)
    agent_b = _load_agent_callable(b_path)
    env = make("orbit_wars", configuration={"seed": seed}, debug=False)
    env.run([agent_a, agent_b])
    payload = env.toJSON()
    steps = payload.get("steps", [])

    game_id = f"{pair_tag}-{seed:08d}"
    return _label_shots_from_game(steps, game_id)


def _parse_pair_arg(s: str) -> tuple[str, str, int]:
    """`agents/A/main.py:agents/B/main.py:N` -> (A, B, N)."""
    parts = s.rsplit(":", 1)
    if len(parts) != 2 or not parts[1].isdigit():
        raise argparse.ArgumentTypeError(
            f"--pairs spec must end with `:<int>`, got {s!r}"
        )
    n = int(parts[1])
    head = parts[0]
    if ":" not in head:
        raise argparse.ArgumentTypeError(
            f"--pairs spec must be `A:B:N`, got {s!r}"
        )
    a, b = head.rsplit(":", 1)
    return a, b, n


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--pairs", action="append", required=True,
        type=_parse_pair_arg,
        help="`pathA:pathB:n_games`; repeat for multiple pairings",
    )
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--wallclock-ms", type=int, default=100,
                   help="BASELINE_WALLCLOCK_MS for workers (default 100)")
    p.add_argument("--seed-base", type=int, default=10_000)
    p.add_argument("--append", action="store_true",
                   help="Append to --out instead of overwriting")
    args = p.parse_args(argv)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not args.append and out_path.exists():
        out_path.unlink()

    tasks: list[tuple] = []
    for i, (a_path, b_path, n_games) in enumerate(args.pairs):
        pair_tag = f"p{i:02d}-{Path(a_path).parent.name}-vs-{Path(b_path).parent.name}"
        for j in range(n_games):
            seed = args.seed_base + i * 10_000 + j
            tasks.append((seed, a_path, b_path, args.wallclock_ms, pair_tag))

    t0 = time.perf_counter()
    n_total = 0
    n_pos = 0
    per_pair_stats: dict[str, list[int]] = {}

    if args.workers <= 1:
        results = (_run_one_game(t) for t in tasks)
    else:
        ctx = get_context("spawn")
        pool = ctx.Pool(processes=args.workers)
        results = pool.imap_unordered(_run_one_game, tasks, chunksize=1)

    with out_path.open("a") as fh:
        for k, rows in enumerate(results):
            for r in rows:
                fh.write(json.dumps(r) + "\n")
            n_total += len(rows)
            n_pos += sum(r["label"] for r in rows)
            tag = rows[0]["game_id"].rsplit("-", 1)[0] if rows else "(empty)"
            per_pair_stats.setdefault(tag, [0, 0])
            per_pair_stats[tag][0] += len(rows)
            per_pair_stats[tag][1] += sum(r["label"] for r in rows)
            print(f"  [{k+1}/{len(tasks)}] {tag} shots={len(rows)} "
                  f"pos={sum(r['label'] for r in rows)} "
                  f"({time.perf_counter()-t0:.0f}s elapsed)")

    if args.workers > 1:
        pool.close()
        pool.join()

    print()
    print(f"=== summary ===  total={n_total}  pos={n_pos}  "
          f"pos_rate={n_pos/max(1,n_total):.3f}  out={out_path}")
    print()
    for tag, (n, k) in sorted(per_pair_stats.items()):
        print(f"  {tag}: shots={n}  pos={k}  pos_rate={k/max(1,n):.3f}")

    if 0.40 <= n_pos / max(1, n_total) <= 0.85:
        print(f"\npos_rate in healthy range [0.40, 0.85]")
    else:
        print(f"\nWARNING: pos_rate outside healthy range — "
              f"mix opponents or adjust gen to bring it into [0.40, 0.85]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
