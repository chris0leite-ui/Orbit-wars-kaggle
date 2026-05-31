"""Decode downloaded 2P replays into (45-d feature, binary label) rows.

For each replay step, for each seat:

  1. Enumerate candidate (src, tgt) pairs:
       owned planets with ships >= MIN_SHIPS  ×  top-K targets by closed-form
       ROI (production / distance, modeling `lite_greedy_policy`'s pruning).
  2. For each candidate, build a synthetic emit [src_pid, straight_line_angle,
     synthetic_ships], encode via `lib.shot_features.encode_shot_features` →
     45-d vector.
  3. Label = 1 iff the seat's actual `action` at this step contains an emit
     whose `(src_id, inferred_target_pid_via_ray_cast)` matches the candidate
     `(src_id, target_pid)`. Multiple emits per seat per turn supported.

Both seats labeled per episode. Game-disjoint train/val split via
`episode_id % 5 < 4` = train (~80/20).

Output: `data/opp_distill/labels.jsonl` — one line per candidate row:
  {"feat": [45 floats], "label": 0|1, "episode": "<id>", "step": int,
   "seat": 0|1, "src": int, "tgt": int, "split": "train"|"val"}

Drops:
  - 4P episodes (defensive — download script already filters but recheck).
  - steps where `obs.planets` is empty (game over / malformed).
  - emits whose target ray-cast returns None (rare; same logic as the encoder
    drops in production).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

# Repo-root import path setup so `lib/...` resolves from script run via
# `python scripts/decode_replays_to_labels.py`.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lib.shot_features import (  # noqa: E402
    encode_shot_features,
    fleet_speed as _fleet_speed,
    infer_target_pid as _infer_tgt,
)
from lib.intent import World as _World  # noqa: E402
from lib.world_model import WorldModel as _WorldModel  # noqa: E402

# Mirrors lite_greedy_policy / top_tier_mirror — only enumerate when the
# source has at least this many ships. Tier 1 also gates on > 5 implicitly
# via aggressive=0.7*src, so 5 here matches the structural floor.
MIN_SRC_SHIPS = 5

# Candidate enumeration: per owned source, take the top-K targets by
# closed-form ROI = production / distance. K=8 is the plan default;
# tune in Phase 1 if pos_rate falls outside [0.05, 0.40].
TOP_K_TARGETS = 8

# Synthetic ships for the candidate emit when computing features.
# Tier 1's aggressive rule: ships = max(0.7 * src.ships, capture_size).
# At training time we use 0.7 * src.ships as a stable proxy; the encoder's
# F2 combat_margin then reflects the typical Tier 1 fleet size.
AGGRESSIVE_FRAC = 0.7


def reconstruct_obs(seat_view: dict, step_idx: int) -> dict:
    """Replay obs for seat 1+ is missing the `step` key (only seat 0 has it).
    Patch it in so downstream feature extraction works."""
    obs = dict(seat_view)
    if "step" not in obs:
        obs["step"] = step_idx
    return obs


def enumerate_candidates(
    obs: dict,
    seat: int,
    top_k: int = TOP_K_TARGETS,
    min_ships: int = MIN_SRC_SHIPS,
) -> list[tuple[int, int, float]]:
    """Return list of (src_pid, tgt_pid, straight_line_angle_rad).

    Matches lite_greedy's pruning: for each owned source with ships >=
    min_ships, score every non-self target by production/distance, keep top-K.
    """
    planets = obs.get("planets") or []
    candidates: list[tuple[int, int, float]] = []
    own = [p for p in planets if int(p[1]) == seat and int(p[5]) >= min_ships]
    others = [p for p in planets if int(p[0]) not in {int(p2[0]) for p2 in own}
              or int(p[1]) != seat]
    # `others` above is loose; the "target != self" filter inside the loop is
    # the strict guard.
    for src in own:
        src_pid = int(src[0])
        sx, sy = float(src[2]), float(src[3])
        scored = []
        for tgt in planets:
            tgt_pid = int(tgt[0])
            if tgt_pid == src_pid:
                continue
            dx, dy = float(tgt[2]) - sx, float(tgt[3]) - sy
            d = math.hypot(dx, dy)
            if d < 1e-6:
                continue
            roi = float(tgt[6]) / (d + 1.0)
            scored.append((roi, tgt_pid, dx, dy, d))
        scored.sort(key=lambda t: -t[0])
        for _, tgt_pid, dx, dy, _d in scored[:top_k]:
            angle = math.atan2(dy, dx)
            candidates.append((src_pid, tgt_pid, angle))
    return candidates


def real_emit_targets(
    actions: list,
    planets: list,
    by_id: dict,
) -> set[tuple[int, int]]:
    """For each real emit in `actions`, ray-cast to infer the target it
    actually heads for, and return the set of (src_pid, tgt_pid) tuples
    representing 'opp emitted here'."""
    out: set[tuple[int, int]] = set()
    if not actions:
        return out
    for emit in actions:
        if not emit or len(emit) < 3:
            continue
        try:
            src_pid = int(emit[0])
            angle = float(emit[1])
        except (TypeError, ValueError):
            continue
        src = by_id.get(src_pid)
        if src is None:
            continue
        tgt_pid = _infer_tgt((float(src[2]), float(src[3])), angle, planets)
        if tgt_pid is None:
            continue
        out.add((src_pid, int(tgt_pid)))
    return out


def decode_episode(
    ep: dict,
    episode_id: str,
    *,
    top_k: int,
) -> list[dict]:
    """Walk every step and seat; emit per-candidate rows."""
    rows: list[dict] = []
    steps = ep.get("steps") or []
    if not steps:
        return rows
    if len(steps[0]) != 2:
        return rows  # 4P defense

    for t, step in enumerate(steps):
        for seat in range(2):
            seat_view = step[seat]
            status = seat_view.get("status")
            if status != "ACTIVE":
                continue
            obs_raw = seat_view.get("observation") or {}
            obs = reconstruct_obs(obs_raw, t)
            planets = obs.get("planets") or []
            if not planets:
                continue
            by_id = {int(p[0]): p for p in planets}

            # Ground-truth positive set for THIS (step, seat).
            actions = seat_view.get("action") or []
            positives = real_emit_targets(actions, planets, by_id)

            # Build world+world_model ONCE per (step, seat) — encoder needs
            # them, and rebuilding per-candidate would be ~5 ms × K_candidates.
            try:
                world = _World.from_obs(obs)
                wm = _WorldModel.from_world(world)
            except Exception:
                continue

            # Enumerate candidates and encode each.
            cands = enumerate_candidates(obs, seat, top_k=top_k)
            for src_pid, tgt_pid, angle in cands:
                src = by_id.get(src_pid)
                if src is None:
                    continue
                synthetic_ships = max(
                    MIN_SRC_SHIPS,
                    int(round(AGGRESSIVE_FRAC * float(src[5]))),
                )
                emit = [src_pid, angle, synthetic_ships]
                try:
                    feat = encode_shot_features(
                        emit, obs, seat, world=world, world_model=wm,
                    )
                except Exception:
                    feat = None
                if feat is None:
                    continue
                label = 1 if (src_pid, tgt_pid) in positives else 0
                rows.append({
                    "feat": feat.tolist(),
                    "label": label,
                    "episode": episode_id,
                    "step": t,
                    "seat": seat,
                    "src": src_pid,
                    "tgt": tgt_pid,
                })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--replays-dir", default="/tmp/ow_replays",
        help="Directory holding episode JSONs (default: /tmp/ow_replays)",
    )
    ap.add_argument(
        "--out", default="data/opp_distill/labels.jsonl",
        help="Output JSONL path (default: data/opp_distill/labels.jsonl)",
    )
    ap.add_argument(
        "--top-k", type=int, default=TOP_K_TARGETS,
        help=f"Top-K targets per source for candidate enumeration "
             f"(default: {TOP_K_TARGETS})",
    )
    ap.add_argument(
        "--max-episodes", type=int, default=None,
        help="Cap on episodes to process (debug; default: all)",
    )
    ap.add_argument(
        "--workers", type=int, default=4,
        help="Parallel worker processes (default: 4)",
    )
    args = ap.parse_args()

    replays_dir = Path(args.replays_dir)
    if not replays_dir.exists():
        print(f"ERROR: {replays_dir} does not exist", file=sys.stderr)
        return 1

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Sorted for reproducibility; the % 5 split is on episode ID hash → stable.
    episodes = sorted(p for p in replays_dir.glob("*.json")
                      if not p.name.startswith("_"))
    if args.max_episodes:
        episodes = episodes[: args.max_episodes]
    if not episodes:
        print(f"ERROR: no *.json files in {replays_dir}", file=sys.stderr)
        return 1

    print(f"decoding {len(episodes)} episodes → {out_path} "
          f"({args.workers} workers)",
          file=sys.stderr, flush=True)

    if args.workers <= 1:
        _decode_serial(episodes, args.top_k, out_path)
    else:
        _decode_parallel(episodes, args.top_k, out_path, args.workers)
    return 0


def _process_episode(arg: tuple[str, int]) -> tuple[int, int, list[dict]]:
    """Worker entry point. Returns (n_rows, n_pos, rows_to_write)."""
    ep_path_str, top_k = arg
    ep_path = Path(ep_path_str)
    try:
        ep = json.loads(ep_path.read_text())
    except Exception:
        return (0, 0, [])
    episode_id = ep_path.stem
    try:
        ep_int = int(episode_id)
    except ValueError:
        ep_int = abs(hash(episode_id))
    split = "train" if (ep_int % 5) < 4 else "val"
    rows = decode_episode(ep, episode_id, top_k=top_k)
    n_pos = sum(r["label"] for r in rows)
    for r in rows:
        r["split"] = split
    return (len(rows), n_pos, rows)


def _decode_serial(episodes: list[Path], top_k: int, out_path: Path) -> None:
    n_rows = 0
    n_pos = 0
    n_eps_decoded = 0
    n_eps_with_positives = 0
    t_start = time.time()
    with out_path.open("w") as f_out:
        for ei, ep_path in enumerate(episodes):
            n, p, rows = _process_episode((str(ep_path), top_k))
            for r in rows:
                f_out.write(json.dumps(r) + "\n")
            n_rows += n
            n_pos += p
            n_eps_decoded += 1
            if p > 0:
                n_eps_with_positives += 1
            if (ei + 1) % 50 == 0 or (ei + 1) == len(episodes):
                _progress(ei + 1, len(episodes), n_rows, n_pos,
                          n_eps_with_positives, n_eps_decoded, t_start)
    _write_summary(out_path, n_rows, n_pos, n_eps_decoded, 0,
                   n_eps_with_positives, top_k)


def _decode_parallel(episodes: list[Path], top_k: int,
                     out_path: Path, workers: int) -> None:
    from concurrent.futures import ProcessPoolExecutor, as_completed

    n_rows = 0
    n_pos = 0
    n_eps_decoded = 0
    n_eps_with_positives = 0
    t_start = time.time()
    work = [(str(p), top_k) for p in episodes]
    with out_path.open("w") as f_out, ProcessPoolExecutor(workers) as ex:
        # Submit all and process as they complete; ordering doesn't matter
        # because the train/val split is encoded in each row's `split` field.
        futures = [ex.submit(_process_episode, w) for w in work]
        for fi, fut in enumerate(as_completed(futures)):
            try:
                n, p, rows = fut.result()
            except Exception as exc:
                print(f"  worker error: {exc}", file=sys.stderr)
                continue
            for r in rows:
                f_out.write(json.dumps(r) + "\n")
            n_rows += n
            n_pos += p
            n_eps_decoded += 1
            if p > 0:
                n_eps_with_positives += 1
            if (fi + 1) % 50 == 0 or (fi + 1) == len(futures):
                _progress(fi + 1, len(futures), n_rows, n_pos,
                          n_eps_with_positives, n_eps_decoded, t_start)
    _write_summary(out_path, n_rows, n_pos, n_eps_decoded, 0,
                   n_eps_with_positives, top_k)


def _progress(done: int, total: int, n_rows: int, n_pos: int,
              n_eps_with_pos: int, n_eps_decoded: int, t_start: float) -> None:
    pos_rate = n_pos / max(1, n_rows)
    pos_ep_rate = n_eps_with_pos / max(1, n_eps_decoded)
    elapsed = time.time() - t_start
    rate = done / max(0.1, elapsed)
    print(
        f"  ep {done}/{total}: rows={n_rows} pos={n_pos} "
        f"pos_rate={pos_rate:.4f} eps_with_pos={pos_ep_rate:.3f} "
        f"({rate:.1f} ep/s, ETA {(total-done)/max(0.1,rate):.0f}s)",
        file=sys.stderr, flush=True,
    )


def _write_summary(out_path: Path, n_rows: int, n_pos: int,
                   n_eps_decoded: int, n_skipped: int,
                   n_eps_with_pos: int, top_k: int) -> None:
    pos_rate = n_pos / max(1, n_rows)
    pos_ep_rate = n_eps_with_pos / max(1, n_eps_decoded)
    summary = {
        "n_episodes_decoded": n_eps_decoded,
        "n_episodes_skipped": n_skipped,
        "n_rows": n_rows,
        "n_positives": n_pos,
        "pos_rate": pos_rate,
        "episodes_with_at_least_one_positive_rate": pos_ep_rate,
        "top_k_targets": top_k,
        "min_src_ships": MIN_SRC_SHIPS,
        "aggressive_frac_for_synthetic_ships": AGGRESSIVE_FRAC,
        "feature_dim": 45,
    }
    summary_path = out_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=1))
    print(
        f"DONE: {n_rows} rows, pos_rate={pos_rate:.4f}, "
        f"eps_with_pos={pos_ep_rate:.3f}; summary → {summary_path}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    sys.exit(main())
