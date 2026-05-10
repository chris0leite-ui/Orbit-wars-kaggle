"""Capture-success probe for the roi agent (PI direction: physics correctness).

Per-fleet outcomes (one row per LAUNCHED fleet, both players in roi-vs-roi):

- reached:         arrived at the launch's declared target_id
- sun:             segment-vs-sun intersection on the disappearance step
- oob:             projected next position out of [0, BOARD_SIZE] on both axes
- collided_other:  hit a non-target planet (intercept / wrong aim)
- vanished_unknown: disappeared without sun / oob / planet hit (rare; logged for review)
- alive_at_end:    still present at the final step (counted toward our score either way)

Output: audit/YYYY-MM-DD-capture-success-probe.json with per-game outcome
breakdown and a roll-up across seeds.

The probe does NOT depend on the tournament harness. It instruments
`agents.simple.roi.propose_intents` so the intended target_id (which is
dropped by the env at `realize()` time) is preserved.

Run:
    python -m scripts.capture_probe --seeds 32
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from kaggle_environments import make

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from agents.simple import roi as _roi  # noqa: E402
from lib.fleet import speed as fleet_speed  # noqa: E402
from lib.geometry import (  # noqa: E402
    BOARD_SIZE,
    CENTER,
    SUN_RADIUS,
    dist,
    point_to_segment_distance,
)
from lib.intent import World  # noqa: E402
from lib.mechanism import DEFAULT_MECHANISMS  # noqa: E402

# Module-level launch log; cleared between games.
_LAUNCH_LOG: list[dict] = []


def _instrumented_agent(obs, _config=None):
    """roi.propose_intents + manual mechanism pipeline so we can capture
    post-mechanism target_id alongside the emitted action."""
    player = obs.get("player", 0) if isinstance(obs, dict) else getattr(obs, "player", 0)
    step = (
        int(obs.get("step", 0))
        if isinstance(obs, dict)
        else int(getattr(obs, "step", 0))
    )
    intents = _roi.propose_intents(obs)
    world = World.from_obs(obs)
    for m in DEFAULT_MECHANISMS:
        intents = m(intents, world)
    for it in intents:
        if it.ships > 0 and it.aim_angle is not None:
            _LAUNCH_LOG.append(
                {
                    "step": step,
                    "player": int(player),
                    "src_id": int(it.src_id),
                    "target_id": int(it.target_id),
                    "ships": int(it.ships),
                    "aim_angle": float(it.aim_angle),
                }
            )
    return [
        [i.src_id, i.aim_angle, i.ships]
        for i in intents
        if i.ships > 0 and i.aim_angle is not None
    ]


def _swept_pair_hit(A, B, P0, P1, r) -> bool:
    """Mirror of the env's swept-pair collision (orbit_wars.py).

    True iff a fleet moving A->B and a planet moving P0->P1 come within r
    of each other for some t in [0, 1].
    """
    d0x, d0y = A[0] - P0[0], A[1] - P0[1]
    dvx = (B[0] - A[0]) - (P1[0] - P0[0])
    dvy = (B[1] - A[1]) - (P1[1] - P0[1])
    a = dvx * dvx + dvy * dvy
    b = 2.0 * (d0x * dvx + d0y * dvy)
    c = d0x * d0x + d0y * d0y - r * r
    if a < 1e-12:
        return c <= 0.0
    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        return False
    sq = math.sqrt(disc)
    t1 = (-b - sq) / (2.0 * a)
    t2 = (-b + sq) / (2.0 * a)
    return t2 >= 0.0 and t1 <= 1.0


def _classify_fleet(
    fleet_id: int,
    trajectory: list[dict],
    launch: dict,
    env_steps: list,
) -> dict:
    """Decide outcome for a tracked fleet, mirroring the env's
    planet-then-OOB-then-sun precedence and swept-pair semantics.

    `trajectory` is the per-step record of (step, owner, x, y, angle, from_id,
    ships) entries while the fleet exists. `launch` is the matched launch row.
    """
    last = trajectory[-1]
    n_steps = len(env_steps)
    last_step = last["step"]
    next_step = last_step + 1

    if next_step >= n_steps:
        return {"outcome": "alive_at_end", "last_step": last_step}

    spd = fleet_speed(last["ships"])
    fleet_old = (last["x"], last["y"])
    fleet_new = (
        last["x"] + math.cos(last["angle"]) * spd,
        last["y"] + math.sin(last["angle"]) * spd,
    )

    # Build planet paths (p_old at last_step, p_new at last_step+1) from the
    # two obs snapshots, mirroring the env's `planet_paths` dict.
    step_obs = env_steps[last_step][0].observation
    next_obs = env_steps[next_step][0].observation
    next_planets_by_id = {p[0]: p for p in next_obs.get("planets", [])}
    planets_at_step = step_obs.get("planets", [])

    # 1. Planet hit (env checks planets BEFORE OOB / sun).
    hit_planet_id = None
    for p in planets_at_step:
        pid, _owner, px, py, pr, *_rest = p
        p_old = (px, py)
        next_p = next_planets_by_id.get(pid)
        p_new = (next_p[2], next_p[3]) if next_p is not None else p_old
        if _swept_pair_hit(fleet_old, fleet_new, p_old, p_new, pr):
            hit_planet_id = pid
            break

    if hit_planet_id is not None:
        if hit_planet_id == launch["target_id"]:
            return {"outcome": "reached", "last_step": last_step, "hit": hit_planet_id}
        return {
            "outcome": "collided_other",
            "last_step": last_step,
            "hit": hit_planet_id,
            "intended_target": launch["target_id"],
        }

    # 2. Out-of-bounds (env's endpoint test).
    if (
        fleet_new[0] < 0.0
        or fleet_new[0] > BOARD_SIZE
        or fleet_new[1] < 0.0
        or fleet_new[1] > BOARD_SIZE
    ):
        return {
            "outcome": "oob",
            "last_step": last_step,
            "last_pos": [last["x"], last["y"]],
            "projected": [fleet_new[0], fleet_new[1]],
        }

    # 3. Sun (env uses strict `<` against SUN_RADIUS).
    sun_seg = point_to_segment_distance(
        (CENTER, CENTER), fleet_old, fleet_new
    )
    if sun_seg < SUN_RADIUS:
        return {
            "outcome": "sun",
            "last_step": last_step,
            "sun_seg_dist": sun_seg,
        }

    return {
        "outcome": "vanished_unknown",
        "last_step": last_step,
        "last_pos": [last["x"], last["y"]],
        "projected": [fleet_new[0], fleet_new[1]],
        "sun_seg_dist": sun_seg,
    }


def _analyze_game(env, seed: int, launches: list[dict]) -> dict:
    """Walk env.steps, match each launched fleet to a launch log row, classify."""
    # Build per-fleet trajectory across all steps.
    trajectory: dict[int, list[dict]] = defaultdict(list)
    for step_idx, state in enumerate(env.steps):
        obs0 = state[0].observation
        for f in obs0.get("fleets", []):
            fid, owner, x, y, angle, from_id, ships = f
            trajectory[fid].append(
                {
                    "step": step_idx,
                    "owner": int(owner),
                    "x": float(x),
                    "y": float(y),
                    "angle": float(angle),
                    "from_id": int(from_id),
                    "ships": int(ships),
                }
            )

    # Match each launch to the fleet it produced. The action is submitted at
    # step `launch.step` and the fleet appears in env.steps[launch.step + 1].
    matched: dict[int, dict] = {}
    unmatched: list[dict] = []
    used_ids: set[int] = set()
    for launch in launches:
        birth = launch["step"] + 1
        if birth >= len(env.steps):
            unmatched.append({"reason": "post_terminal_launch", **launch})
            continue
        # Find a fleet with the launch's signature that doesn't appear in
        # env.steps[birth - 1] but does appear in env.steps[birth].
        prev_ids = {
            f[0] for f in env.steps[birth - 1][0].observation.get("fleets", [])
        }
        candidates = []
        for f in env.steps[birth][0].observation.get("fleets", []):
            fid, owner, x, y, angle, from_id, ships = f
            if fid in used_ids or fid in prev_ids:
                continue
            if (
                int(owner) == launch["player"]
                and int(from_id) == launch["src_id"]
                and int(ships) == launch["ships"]
            ):
                candidates.append(fid)
        if not candidates:
            unmatched.append({"reason": "no_birth_match", **launch})
            continue
        # Prefer the candidate with the closest aim_angle.
        if len(candidates) > 1:
            def angle_dist(fid: int) -> float:
                f = env.steps[birth][0].observation["fleets"]
                for entry in f:
                    if entry[0] == fid:
                        return abs(
                            math.atan2(
                                math.sin(entry[4] - launch["aim_angle"]),
                                math.cos(entry[4] - launch["aim_angle"]),
                            )
                        )
                return math.inf

            candidates.sort(key=angle_dist)
        fid = candidates[0]
        used_ids.add(fid)
        matched[fid] = launch

    # Classify each matched fleet.
    outcomes_per_player: dict[int, defaultdict[str, int]] = {
        0: defaultdict(int),
        1: defaultdict(int),
    }
    detail_unknown: list[dict] = []
    detail_collided: list[dict] = []
    for fid, traj in trajectory.items():
        if fid not in matched:
            continue
        launch = matched[fid]
        result = _classify_fleet(fid, traj, launch, env.steps)
        outcomes_per_player[launch["player"]][result["outcome"]] += 1
        if result["outcome"] == "vanished_unknown" and len(detail_unknown) < 20:
            detail_unknown.append({"fleet_id": fid, "launch": launch, "result": result})
        if result["outcome"] == "collided_other" and len(detail_collided) < 20:
            detail_collided.append({"fleet_id": fid, "launch": launch, "result": result})

    n_launches = len(launches)
    n_matched = len(matched)
    n_unmatched = len(unmatched)
    return {
        "seed": seed,
        "n_steps": len(env.steps),
        "n_launches_logged": n_launches,
        "n_matched": n_matched,
        "n_unmatched": n_unmatched,
        "outcomes_per_player": {
            str(k): dict(v) for k, v in outcomes_per_player.items()
        },
        "unmatched_sample": unmatched[:10],
        "vanished_unknown_sample": detail_unknown,
        "collided_other_sample": detail_collided[:5],
    }


def run(seeds: list[int]) -> dict:
    games: list[dict] = []
    for seed in seeds:
        _LAUNCH_LOG.clear()
        env = make("orbit_wars", configuration={"seed": seed}, debug=False)
        env.run([_instrumented_agent, _instrumented_agent])
        games.append(_analyze_game(env, seed, list(_LAUNCH_LOG)))
        # Free env early — env.steps holds the full history.
        del env

    # Aggregate.
    totals: defaultdict[str, int] = defaultdict(int)
    total_launches_matched = 0
    total_launches_logged = 0
    total_launches_unmatched = 0
    for g in games:
        total_launches_logged += g["n_launches_logged"]
        total_launches_matched += g["n_matched"]
        total_launches_unmatched += g["n_unmatched"]
        for player_str, counts in g["outcomes_per_player"].items():
            for k, v in counts.items():
                totals[k] += v
    return {
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "agent": "agents/simple/roi.py",
        "opponent": "agents/simple/roi.py (self-play)",
        "n_seeds": len(seeds),
        "seeds": seeds,
        "totals": dict(totals),
        "total_launches_logged": total_launches_logged,
        "total_launches_matched": total_launches_matched,
        "total_launches_unmatched": total_launches_unmatched,
        "games": games,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--seeds", type=int, default=32, help="number of seeds (1..N)")
    p.add_argument("--seed-list", type=str, default=None,
                   help="comma-separated explicit seed list (overrides --seeds)")
    p.add_argument("--out", type=Path, default=None,
                   help="output JSON path (default: audit/<utc-date>-capture-success-probe.json)")
    args = p.parse_args(argv)

    if args.seed_list:
        seeds = [int(s) for s in args.seed_list.split(",") if s.strip()]
    else:
        seeds = list(range(1, args.seeds + 1))

    out_path = args.out or (
        REPO / "audit"
        / f"{datetime.now(timezone.utc).strftime('%Y-%m-%d')}-capture-success-probe.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[probe] running {len(seeds)} seeds roi-vs-roi self-play...", flush=True)
    result = run(seeds)
    out_path.write_text(json.dumps(result, indent=2))

    totals = result["totals"]
    n_matched = result["total_launches_matched"]
    n_logged = result["total_launches_logged"]
    print(f"[probe] wrote {out_path}")
    print(f"[probe] launches logged={n_logged} matched={n_matched} "
          f"unmatched={result['total_launches_unmatched']}")
    print(f"[probe] outcome totals (across both players, {len(seeds)} seeds):")
    denom = sum(totals.values()) or 1
    for k in ["reached", "sun", "oob", "collided_other", "vanished_unknown", "alive_at_end"]:
        v = totals.get(k, 0)
        pct = 100.0 * v / denom
        print(f"  {k:18s} {v:6d}  ({pct:5.1f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
