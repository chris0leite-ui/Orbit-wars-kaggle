"""Calibration probe for COORD_LAMBDA_W — measures the empirical
distribution of `tier2_score` magnitudes vs ΔW magnitudes on a real
game state.

Picks λ_W such that median |λ_W × ΔW| ≈ 0.3 × median |tier2_score|.
This puts the smooth-ΔW bonus at ~30% of the leaf-Δ, comparable to
phase-α's "200-1500 range alongside topology + prod_stream" calibration.

Usage:
    python scripts/check_coord_endgame_calibration.py [--seeds 0,1]

Output: per-turn magnitudes, the suggested λ_W anchor, and the three
sweep values (anchor × 0.3, anchor, anchor × 3).
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# Ensure DELTA_W is OFF for the probe (we want raw tier2_score values
# and we compute ΔW ourselves to compare).
os.environ["COORD_DELTA_W"] = "0"

from kaggle_environments import make  # noqa: E402
from kaggle_environments.envs.orbit_wars.orbit_wars import Planet  # noqa: E402

from agents.coord.main import (  # noqa: E402
    CHEAP_FILTER_TOP_K,
    BundleKind,
    cheap_filter_bundles,
    enumerate_attack_bundles,
    enumerate_defend_bundles,
    tier2_score_bundles,
    _bundle_endgame_bonus,
    _strongest_opp,
    _largest_threat_owner,
)
from agents.coord._endgame import (  # noqa: E402
    bundle_delta_w_attack,
    bundle_delta_w_defend,
    remaining_turns,
)
from lib.fast_sim import from_obs as fs_from_obs  # noqa: E402
from lib.intent import World  # noqa: E402
from lib.world_model import WorldModel  # noqa: E402


def _compute_delta_w_per_bundle(bundle, world, model, me, num_seats):
    """Re-implement _bundle_endgame_bonus(λ_W=1.0) so we can measure |ΔW|."""
    rem = remaining_turns(world)
    if rem <= 0:
        return 0
    target = world.planets_by_id.get(int(bundle.target_id))
    if target is None:
        return 0
    if bundle.kind == BundleKind.ATTACK:
        cur_owner = int(target.owner)
        if cur_owner == int(me):
            return 0
        if cur_owner >= 0:
            opp_id = cur_owner
        else:
            opp_id = _strongest_opp(world, me, num_seats)
            if opp_id is None:
                return 0
        return bundle_delta_w_attack(target, int(me), int(opp_id), int(rem))
    if bundle.kind == BundleKind.DEFEND:
        opp_threat = _largest_threat_owner(bundle.target_id, model, me)
        if opp_threat is None:
            return 0
        return bundle_delta_w_defend(target, int(me), int(opp_threat), int(rem))
    return 0


def probe_seed(seed: int, max_turns: int = 60):
    """Drive a minimal-vs-minimal game; sample tier2/ΔW magnitudes from
    P0's perspective each turn."""
    env = make("orbit_wars", configuration={"seed": seed})
    env.reset(num_agents=2)
    samples = []  # list of (turn, |t2_score|, |delta_w|, kind)
    for turn in range(max_turns):
        if env.done:
            break
        obs = env.state[0].observation
        obs_d = dict(obs) if not isinstance(obs, dict) else obs
        me = int(obs_d.get("player", 0))
        raw_planets = obs_d.get("planets", []) or []
        planets = [Planet(*p) for p in raw_planets]
        my_planets = [p for p in planets if int(p.owner) == me]
        other_planets = [p for p in planets if int(p.owner) != me]
        if not my_planets:
            break
        world = World.from_obs(obs_d)
        model = WorldModel.from_world(world)
        omega = float(obs_d.get("angular_velocity", 0.0))
        num_seats = 2
        snap_base = fs_from_obs(obs, num_seats=num_seats)

        attacks = enumerate_attack_bundles(
            my_planets, other_planets, world, model, me, omega,
        )
        defends = enumerate_defend_bundles(
            my_planets, world, model, me, omega,
        )
        all_bundles = attacks + defends
        if not all_bundles:
            env.run(["minimal", "minimal"])
            break

        cheap = cheap_filter_bundles(
            all_bundles, world, model, me, num_seats, K=CHEAP_FILTER_TOP_K,
        )
        scored = tier2_score_bundles(
            cheap, snap_base, me, num_seats, world, model,
            wallclock_ms=400.0,
        )
        for b in scored:
            t2 = abs(float(b.tier2_score))
            dw = abs(_compute_delta_w_per_bundle(b, world, model, me, num_seats))
            if t2 == 0 and dw == 0:
                continue
            samples.append((turn, t2, dw, b.kind.name))

        # Step env one turn (minimal acts; we don't care about coord's actual
        # decisions — only the bundle-shape statistics on the state).
        env.step([
            [],  # placeholder; we'll discard this game's outcome
            [],
        ])
    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="0,1",
                        help="comma-separated seed list")
    parser.add_argument("--max-turns", type=int, default=60)
    args = parser.parse_args()

    all_samples = []
    for s in args.seeds.split(","):
        s = int(s.strip())
        print(f"\n=== seed {s} ===", flush=True)
        samples = probe_seed(s, args.max_turns)
        print(f"  collected {len(samples)} bundle-samples", flush=True)
        all_samples.extend(samples)

    if not all_samples:
        print("\n[ERROR] no samples collected — probe failed", file=sys.stderr)
        sys.exit(1)

    t2_mags = [s[1] for s in all_samples]
    dw_mags = [s[2] for s in all_samples if s[2] > 0]
    attack_dws = [s[2] for s in all_samples if s[3] == "ATTACK" and s[2] > 0]
    defend_dws = [s[2] for s in all_samples if s[3] == "DEFEND" and s[2] > 0]

    print("\n=== summary ===")
    print(f"  total samples:     {len(all_samples)}")
    print(f"  tier2_score |.|:   median={statistics.median(t2_mags):.3f}  "
          f"mean={statistics.mean(t2_mags):.3f}  "
          f"max={max(t2_mags):.3f}")
    if dw_mags:
        print(f"  ΔW |.|:            median={statistics.median(dw_mags):.1f}  "
              f"mean={statistics.mean(dw_mags):.1f}  "
              f"max={max(dw_mags):.1f}")
    if attack_dws:
        print(f"  ATTACK ΔW |.|:     median={statistics.median(attack_dws):.1f}  "
              f"n={len(attack_dws)}")
    if defend_dws:
        print(f"  DEFEND ΔW |.|:     median={statistics.median(defend_dws):.1f}  "
              f"n={len(defend_dws)}")

    # Calibration: λ_W × median|ΔW| ≈ 0.3 × median|tier2|
    if dw_mags and t2_mags:
        med_t2 = statistics.median(t2_mags)
        med_dw = statistics.median(dw_mags)
        if med_dw > 0:
            anchor = 0.3 * med_t2 / med_dw
            print(f"\n=== suggested λ_W ===")
            print(f"  anchor:           {anchor:.6f}")
            print(f"  sweep low  ×0.3:  {anchor * 0.3:.6f}")
            print(f"  sweep mid       : {anchor:.6f}")
            print(f"  sweep high ×3   : {anchor * 3:.6f}")
            print(f"\n  COORD_LAMBDA_W={anchor:.6f}  "
                  f"# (median bonus would be {0.3 * med_t2:.3f} = "
                  f"30% of median |tier2|)")


if __name__ == "__main__":
    main()
