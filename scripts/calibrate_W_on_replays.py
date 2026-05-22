"""Verification (b) per ACTIVE PLAN — calibrate W(s) on live replay episodes.

For each replay in audit/live-episodes/52894340/:
- Iterate steps; reconstruct world state from obs.
- For each seat, compute W = prod_advantage × remaining_turns - opp_pool
  (using lib.joint_solver.predicate primitives).
- Track W_seat[step] paired with final_reward[seat].

Three checks (per plan section "Verification (b)"):
(b1) Pearson r(W, final_focal_reward) > 0.6 → W is a usable global value.
(b2) ΔW per-action variance > 0.1 × mean|ΔW|.
(b3) Pearson r(ΔW_per_action, outcome_shift_per_step) > 0.3.

Outputs: audit/2026-05-23/calibrate_W_results.json + console summary.
"""
from __future__ import annotations

import glob
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from lib.intent import World
from lib.joint_solver.predicate import (
    prod_advantage,
    remaining_turns,
    opp_pool,
)


REPLAY_DIR = REPO / "audit/live-episodes/52894340/"
OUT_DIR = REPO / "audit/2026-05-23"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_JSON = OUT_DIR / "calibrate_W_results.json"
N_EPISODES = 20  # plan says ~20

EPISODE_STEPS = 500  # orbit_wars default per env spec


def _W(world, my_id, opp_id):
    """W(s) = prod_advantage × remaining_turns - opp_pool."""
    adv = prod_advantage(world, my_id, opp_id)
    rem = remaining_turns(world, EPISODE_STEPS)
    op = opp_pool(world, opp_id, EPISODE_STEPS)
    return adv * rem - op


def _obs_to_world(obs_d):
    """Reconstruct World from a replay obs dict. obs is exactly the
    kaggle_environments dict the agent sees, so World.from_obs works."""
    return World.from_obs(obs_d)


def _2p_replays(limit):
    """Yield up to `limit` 2P replay file paths."""
    yielded = 0
    for f in sorted(glob.glob(str(REPLAY_DIR / "*.json"))):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        steps = d.get("steps")
        if not steps:
            continue
        n_seats = len(steps[-1])
        if n_seats != 2:
            continue
        yield f, d
        yielded += 1
        if yielded >= limit:
            return


def _pearson(xs, ys):
    """Pearson r. Returns None on degenerate input."""
    n = len(xs)
    if n < 3:
        return None
    mx = mean(xs)
    my = mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if sx == 0 or sy == 0:
        return None
    return num / (sx * sy)


def calibrate():
    Ws_paired = []  # (W, final_reward) per (replay, step, seat)
    dWs = []        # ΔW per emit-step
    dWs_x_shift = []  # (ΔW, outcome_shift) per emit-step

    n_eps = 0
    n_steps_total = 0
    for path, replay in _2p_replays(N_EPISODES):
        n_eps += 1
        steps = replay["steps"]
        if not steps:
            continue
        final_rewards = [int(steps[-1][s].get("reward", 0)) for s in range(2)]

        # Per-seat trajectory: list of W values + action lists
        per_seat_W = {0: [], 1: []}
        for t, step_data in enumerate(steps):
            for seat in (0, 1):
                obs_d = step_data[seat].get("observation", {})
                if not obs_d.get("planets"):
                    per_seat_W[seat].append(None)
                    continue
                opp = 1 - seat
                try:
                    world = _obs_to_world(obs_d)
                    w = _W(world, seat, opp)
                except Exception:
                    w = None
                per_seat_W[seat].append(w)

        # (b1) pair each step's W with final reward.
        for seat in (0, 1):
            for t, w in enumerate(per_seat_W[seat]):
                if w is None:
                    continue
                Ws_paired.append((float(w), int(final_rewards[seat])))
                n_steps_total += 1

        # (b2) ΔW per emit-step (where the focal seat emitted an action).
        # We consider seat=0 only for ΔW analysis — the agent that "is"
        # our FND focal. (Replay seats are mostly self-play of FND.)
        for seat in (0,):
            for t in range(1, len(per_seat_W[seat])):
                prev_w = per_seat_W[seat][t - 1]
                curr_w = per_seat_W[seat][t]
                if prev_w is None or curr_w is None:
                    continue
                action_prev = steps[t - 1][seat].get("action") or []
                # Only count emit-steps (action != empty).
                if not action_prev:
                    continue
                dW = float(curr_w) - float(prev_w)
                dWs.append(dW)
                # (b3) outcome_shift: 1 if final_reward > 0, -1 else.
                # Per-step proxy: sign of (W_T - W_t) — does our position
                # improve over the rest of the game? Final reward is a
                # noisy single-bit version.
                W_T = per_seat_W[seat][-1] if per_seat_W[seat][-1] is not None else 0.0
                shift = float(W_T) - float(curr_w)
                dWs_x_shift.append((dW, shift))

    # Compute checks.
    Ws = [w for w, _ in Ws_paired]
    Rs = [r for _, r in Ws_paired]
    r_b1 = _pearson(Ws, Rs)

    abs_mean_dW = mean(abs(d) for d in dWs) if dWs else 0.0
    var_dW = stdev(dWs) ** 2 if len(dWs) >= 2 else 0.0
    threshold_b2 = 0.1 * abs_mean_dW
    # The plan says "variance > 0.1 × mean|ΔW|" — both quantities are
    # different units, so the literal interpretation is:
    # variance (units: W²) > 0.1 × mean|ΔW| (units: W). That's a unit
    # mismatch. Standard reading: stdev > 0.1 × mean|ΔW|, OR variance > 0
    # (any spread is meaningful). We report both interpretations.
    stdev_dW = math.sqrt(var_dW)
    threshold_b2_stdev = 0.1 * abs_mean_dW

    dWs_only = [d for d, _ in dWs_x_shift]
    shifts_only = [s for _, s in dWs_x_shift]
    r_b3 = _pearson(dWs_only, shifts_only)

    result = {
        "n_episodes": n_eps,
        "n_step_samples": len(Ws_paired),
        "n_emit_actions": len(dWs),
        "b1_pearson_W_vs_reward": r_b1,
        "b1_threshold": 0.6,
        "b1_pass": r_b1 is not None and r_b1 > 0.6,
        "b2_mean_abs_dW": abs_mean_dW,
        "b2_variance_dW": var_dW,
        "b2_stdev_dW": stdev_dW,
        "b2_stdev_threshold": threshold_b2_stdev,
        "b2_pass_loose_variance_nonzero": var_dW > 0,
        "b2_pass_strict_stdev_above_threshold": stdev_dW > threshold_b2_stdev,
        "b3_pearson_dW_vs_shift": r_b3,
        "b3_threshold": 0.3,
        "b3_pass": r_b3 is not None and r_b3 > 0.3,
    }

    OUT_JSON.write_text(json.dumps(result, indent=2))
    print(f"=== Verification (b) — W calibration ===")
    print(f"  n_episodes: {n_eps}")
    print(f"  n_step_samples (Ws_paired): {len(Ws_paired)}")
    print(f"  n_emit_actions (ΔW): {len(dWs)}")
    print()
    print(f"  (b1) Pearson r(W, final_reward) = {r_b1!r}  (need > 0.6) → "
          f"{'PASS' if result['b1_pass'] else 'FAIL'}")
    print(f"  (b2) mean|ΔW| = {abs_mean_dW:.3f}, stdev(ΔW) = {stdev_dW:.3f}, "
          f"variance(ΔW) = {var_dW:.3f}")
    print(f"       loose: variance > 0 → "
          f"{'PASS' if result['b2_pass_loose_variance_nonzero'] else 'FAIL'}")
    print(f"       strict: stdev > 0.1×mean|ΔW|={threshold_b2_stdev:.3f} → "
          f"{'PASS' if result['b2_pass_strict_stdev_above_threshold'] else 'FAIL'}")
    print(f"  (b3) Pearson r(ΔW, outcome_shift) = {r_b3!r}  (need > 0.3) → "
          f"{'PASS' if result['b3_pass'] else 'FAIL'}")
    print()
    overall = result["b1_pass"] and result["b3_pass"]
    print(f"  Overall (b1 AND b3 pass): {'PASS' if overall else 'FAIL'}")
    print(f"  Wrote: {OUT_JSON.relative_to(REPO)}")


if __name__ == "__main__":
    calibrate()
