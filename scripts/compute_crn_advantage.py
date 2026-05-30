"""Reframe B.3 — CRN-paired advantage label driver.

For every prerank candidate in a B.3 self-play corpus, computes

    A(s, a) = focal_margin(after action a, K) − focal_margin(after idle, K)

where both rollouts step the env with pv_eta as the OPPONENT throughout
and pv_eta as the FOCAL policy from step 1 onward. The focal seat's
action at step 0 is `idle` (no launches) for the baseline rollout and
the candidate action for the action rollout. The idle rollout is
shared across all top-N candidates of the same state (per-state cache).

Inputs (per game-* subdir in --in):
  - replay.jsonl: per-tick full obs (extended schema; needs comets,
                  initial_planets, angular_velocity, next_fleet_id,
                  remainingOverageTime in addition to planets/fleets)
  - prerank.jsonl: one row per scored prerank candidate carrying
                   cheap_delta, leaf_delta, features, etc.

Output: corpus.jsonl with rows
  {game_id, seat, step, src_id, tgt_id, ships, eta, angle, wait_N,
   leaf_delta, features, label=advantage}

Implementation note: env.clone() + env.step() — NO fast_sim. The
2026-05-30 Step 0 bench showed fast_sim offers no speed advantage for
this use case because pv_eta dominates per-rollout cost.

CLI:
    python scripts/compute_crn_advantage.py \
        --in data/value_head/b3-smoke \
        --top-n 5 --K 5 --wallclock-ms 100
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _ships_total(obs: dict, seat: int) -> int:
    """Sum seat's ships on owned planets + in-flight fleets."""
    t = 0
    for p in obs.get("planets", []) or []:
        if int(p[1]) == seat:
            t += int(p[5])
    for f in obs.get("fleets", []) or []:
        if int(f[1]) == seat:
            t += int(f[6])
    return t


def _focal_margin(obs: dict, focal_seat: int) -> int:
    """(focal ships) − (sum of other seats' ships) — at integer precision."""
    others = 0
    for p in obs.get("planets", []) or []:
        if int(p[1]) >= 0 and int(p[1]) != focal_seat:
            others += int(p[5])
    for f in obs.get("fleets", []) or []:
        if int(f[1]) >= 0 and int(f[1]) != focal_seat:
            others += int(f[6])
    focal = _ships_total(obs, focal_seat)
    return focal - others


def _reset_pv_eta(pve) -> None:
    pve._reset_state_for_tests()
    pve._PENDING_LAUNCHES.clear()


def _rollout(env, focal_seat: int, focal_step0_action: list, K: int,
             pve_agent, configuration, pve_mod) -> int:
    """Run K env.step ticks:
      step 0: focal_seat plays focal_step0_action; other seats play pv_eta
      steps 1..K-1: both seats play pv_eta
    Returns focal_margin(terminal_obs, focal_seat) — an int.

    Resets pv_eta module state BEFORE the rollout starts (each rollout
    is an independent game; ledger / recapture-state from prior rollouts
    must not leak).
    """
    _reset_pv_eta(pve_mod)
    n_seats = len(env.state)
    for tick in range(K):
        if env.done:
            break
        acts = []
        for seat in range(n_seats):
            if tick == 0 and seat == focal_seat:
                acts.append(focal_step0_action)
            else:
                acts.append(pve_agent(env.state[seat].observation, configuration))
        env.step(acts)
    return _focal_margin(env.state[0].observation, focal_seat)


def _load_replay(replay_path: Path) -> dict[int, dict]:
    out: dict[int, dict] = {}
    with replay_path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            out[int(r["step"])] = r
    return out


def _load_prerank(prerank_path: Path) -> list[dict]:
    rows: list[dict] = []
    with prerank_path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _group_topn(prerank_rows: list[dict], top_n: int,
                skip_wait_n_gt_0: bool = True
                ) -> dict[tuple[int, int], list[dict]]:
    """Group prerank rows by (step, me), keep top-N by cheap_delta."""
    bag: dict[tuple[int, int], list[dict]] = {}
    for r in prerank_rows:
        if skip_wait_n_gt_0 and int(r.get("wait_N", 0)) != 0:
            continue
        key = (int(r["step"]), int(r["me"]))
        bag.setdefault(key, []).append(r)
    out: dict[tuple[int, int], list[dict]] = {}
    for k, lst in bag.items():
        lst.sort(key=lambda r: float(r["cheap_delta"]), reverse=True)
        out[k] = lst[:top_n]
    return out


def label_one_game(game_dir: Path, K: int, top_n: int,
                   wallclock_ms: int) -> dict:
    """Stage 2 driver for ONE game. Writes <game_dir>/corpus.jsonl,
    returns a small summary dict."""
    # IMPORTANT — set wallclock before any pv_eta import so the module-
    # load env var read picks it up. Worker-process scope.
    os.environ["BASELINE_WALLCLOCK_MS"] = str(int(wallclock_ms))
    from kaggle_environments import make as _make  # noqa: F401  (forces orbit_wars registration)
    import submissions._imported.baseline_pv_eta as pve

    replay_path = game_dir / "replay.jsonl"
    prerank_path = game_dir / "prerank.jsonl"
    if not replay_path.exists() or not prerank_path.exists():
        return {"game_dir": str(game_dir), "n_rows": 0,
                "skipped": "missing replay or prerank"}

    replay = _load_replay(replay_path)
    prerank_rows = _load_prerank(prerank_path)
    groups = _group_topn(prerank_rows, top_n=top_n)

    seed = int(game_dir.name.removeprefix("game-"))
    cfg = {"seed": seed}

    out_path = game_dir / "corpus.jsonl"
    n_rows = 0
    n_states = 0
    n_skipped_truncated = 0
    n_skipped_no_replay = 0
    t0 = time.perf_counter()

    with out_path.open("w") as out_fh:
        for (step, me), candidates in sorted(groups.items()):
            # Truncation: need K more ticks of game runway. Tolerate a
            # short-game; if step+K exceeds the recorded last step, the
            # rollout would walk past the terminal — but env.step() will
            # itself short-circuit on env.done, so labels at end-of-game
            # are still well-defined; just lower variance.
            if step not in replay:
                n_skipped_no_replay += 1
                continue
            obs_at_t = replay[step]
            # Build env from recorded obs.
            env_seed = pve.env_from_obs(obs_at_t, cfg)
            # Idle rollout (shared across top-N candidates).
            env_idle = env_seed.clone()
            margin_idle = _rollout(
                env_idle, focal_seat=me, focal_step0_action=[],
                K=K, pve_agent=pve.agent,
                configuration=env_idle.configuration, pve_mod=pve,
            )
            for cand in candidates:
                env_act = env_seed.clone()
                action = [int(cand["src_id"]), float(cand["angle"]),
                          int(cand["ships"])]
                margin_action = _rollout(
                    env_act, focal_seat=me, focal_step0_action=[action],
                    K=K, pve_agent=pve.agent,
                    configuration=env_act.configuration, pve_mod=pve,
                )
                label = margin_action - margin_idle
                row = {
                    "game_id": game_dir.name,
                    "seat": me,
                    "step": step,
                    "src_id": int(cand["src_id"]),
                    "tgt_id": int(cand["tgt_id"]),
                    "ships": int(cand["ships"]),
                    "angle": float(cand["angle"]),
                    "wait_N": int(cand.get("wait_N", 0)),
                    "eta": int(cand.get("eta", 0)),
                    "cheap_delta": float(cand["cheap_delta"]),
                    "leaf_delta": float(cand["leaf_delta"]),
                    "margin_idle": int(margin_idle),
                    "margin_action": int(margin_action),
                    "label": int(label),
                }
                if "features" in cand:
                    row["features"] = list(cand["features"])
                out_fh.write(json.dumps(row) + "\n")
                n_rows += 1
            n_states += 1

    return {
        "game_dir": str(game_dir),
        "n_rows": n_rows,
        "n_states": n_states,
        "n_skipped_truncated": n_skipped_truncated,
        "n_skipped_no_replay": n_skipped_no_replay,
        "elapsed_s": time.perf_counter() - t0,
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--in", dest="in_dir", required=True, type=Path,
                   help="Self-play corpus dir (contains game-*/ subdirs)")
    p.add_argument("--top-n", type=int, default=5,
                   help="Top-N candidates per (step, seat) to label")
    p.add_argument("--K", type=int, default=5,
                   help="Rollout horizon (ticks)")
    p.add_argument("--wallclock-ms", type=int, default=100,
                   help="BASELINE_WALLCLOCK_MS for pv_eta as policy")
    p.add_argument("--max-games", type=int, default=None,
                   help="Cap game count for smoke tests")
    p.add_argument("--workers", type=int, default=1,
                   help=">1 enables multiprocessing pool (spawn ctx)")
    args = p.parse_args(argv)

    in_dir = Path(args.in_dir)
    game_dirs = sorted(d for d in in_dir.iterdir()
                       if d.is_dir() and d.name.startswith("game-"))
    if args.max_games is not None:
        game_dirs = game_dirs[: args.max_games]
    if not game_dirs:
        print(f"No game-* subdirs in {in_dir}", file=sys.stderr)
        return 2

    print(f"=== B.3 stage 2 — CRN advantage labelling ===")
    print(f"in:  {in_dir}")
    print(f"games: {len(game_dirs)}   top_n: {args.top_n}   K: {args.K}   wc_ms: {args.wallclock_ms}")
    print()

    t_global = time.perf_counter()

    def _task(gd):
        return label_one_game(gd, K=args.K, top_n=args.top_n,
                              wallclock_ms=args.wallclock_ms)

    if args.workers <= 1:
        results = [_task(gd) for gd in game_dirs]
    else:
        from multiprocessing import get_context
        ctx = get_context("spawn")
        with ctx.Pool(processes=args.workers, maxtasksperchild=1) as pool:
            results = list(pool.imap_unordered(
                _label_one_game_worker,
                [(str(gd), args.K, args.top_n, args.wallclock_ms)
                 for gd in game_dirs],
            ))

    # Concatenate per-game corpora into the run-level corpus.jsonl.
    run_corpus = in_dir / "corpus.jsonl"
    total_rows = 0
    with run_corpus.open("w") as out_fh:
        for gd in game_dirs:
            game_corpus = gd / "corpus.jsonl"
            if not game_corpus.exists():
                continue
            with game_corpus.open() as in_fh:
                for line in in_fh:
                    out_fh.write(line)
                    total_rows += 1

    print()
    print("=" * 72)
    for r in results:
        print(f"  {r}")
    print()
    print(f"=== summary === games={len(game_dirs)} rows={total_rows} "
          f"wall={time.perf_counter()-t_global:.0f}s  out={run_corpus}")
    return 0


def _label_one_game_worker(packed):
    """Top-level for spawn ctx pickling."""
    gd_str, K, top_n, wc = packed
    return label_one_game(Path(gd_str), K=K, top_n=top_n, wallclock_ms=wc)


if __name__ == "__main__":
    raise SystemExit(main())
