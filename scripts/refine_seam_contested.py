"""scripts/refine_seam_contested.py — does the sync-coalition seam reopen vs a strong opponent?

Ledger root-cause (state/MULTI_BRANCH.md, 2026-06-02): `generate_sync_coalitions`
yields 0 raw candidates every turn vs weak opponents (v7_0 / v7_minimax) because
"my planets accumulate enough ships to solo-take their targets" — a 2-source
coalition only forms when NEITHER nearby source can solo-capture a defended
target but both combined can. That regime is near-absent when the champion is
crushing a weak opponent (source-saturated).

UNTESTED: when a STRONG opponent contests and DRAINS my planets, does solo-capture
become infeasible often enough that the 2-source regime reappears?

Method (mirrors the ledger's "instrumented raw-coalition count" — the env.run
output sandbox swallows agent-internal stderr, so we measure the GENERATOR
DIRECTLY): play the champion vs a champion-strength frozen bundle, then walk every
realized board and call `generate_sync_coalitions` on the focal seat. The raw
yield count IS the seam signal. We also report my-source / defended-target counts
to characterise the regime.

Step 2 decision gate: midgame raw_coalitions > 0 → seam reopens (closure does NOT
transfer); ~0 → closure transfers, STOP.
"""
from __future__ import annotations

import os
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# Champion production config (the boards must reflect the real champion strategy).
for k, v in {
    "BASELINE_JOINT_AGGR": "1", "BASELINE_JOINT_TOP_K": "5",
    "BASELINE_JOINT_MAX_PAIRS": "60", "BASELINE_REINFORCE_EMIT": "1",
    "BASELINE_REINFORCE_ANTICIPATE": "1", "BASELINE_NEUTRAL_BONUS": "2.0",
    "BASELINE_NEUTRAL_EARLY_EXTRA": "1.5", "BASELINE_NEUTRAL_EARLY_HORIZON": "50",
    "BASELINE_ORBITAL_SAFETY": "1", "BASELINE_PV_ETA": "1",
    "BASELINE_LAUNCH_RULES": "1", "BASELINE_CAPTURE_HORIZON_K": "10",
    "BASELINE_VALUE_HEAD": "hybrid", "BASELINE_JOINT": "1",
    "BASELINE_CHOOSER": "trajectory",
}.items():
    os.environ.setdefault(k, v)

from kaggle_environments import make  # noqa: E402
from lib.intent import World  # noqa: E402
from lib.world_model import WorldModel  # noqa: E402
from agents.baseline.proposer import MAX_HORIZON, MIN_FLEET_SIZE  # noqa: E402
from agents.baseline.chooser_trajectory import generate_sync_coalitions  # noqa: E402

FOCAL = str(REPO / "agents/baseline/main.py")
OPP = sys.argv[1] if len(sys.argv) > 1 else str(REPO / "submissions/baseline.py")
SEEDS = [int(s) for s in (sys.argv[2].split(",") if len(sys.argv) > 2 else ["0"])]
SEAT_ARG = sys.argv[3] if len(sys.argv) > 3 else "both"

PHASES = [("opening 0-50", 0, 50), ("early-mid 50-150", 50, 150),
          ("MIDGAME 150-300", 150, 300), ("late 300+", 300, 10**9)]


def _bucket(step):
    for name, lo, hi in PHASES:
        if lo <= step < hi:
            return name
    return "late 300+"


def analyze_step(obs, me):
    """Return (raw_coalitions, n_my_sources, n_defended_targets) for one board."""
    try:
        world = World.from_obs(obs)
        model = WorldModel.from_world(world)
    except Exception:
        return None
    raw = 0
    for _launches, _tarr in generate_sync_coalitions(
            world, model, me, MAX_HORIZON, set(), set()):
        raw += 1
    my_srcs = sum(1 for p in world.planets_by_id.values()
                  if int(p.owner) == me and int(p.ships) >= MIN_FLEET_SIZE)
    defended = sum(1 for p in world.planets_by_id.values()
                   if int(p.owner) != me and int(p.owner) >= 0 and int(p.ships) > 0)
    return raw, my_srcs, defended


def run_seed(seed, focal_p0):
    p0, p1 = (FOCAL, OPP) if focal_p0 else (OPP, FOCAL)
    me = 0 if focal_p0 else 1
    env = make("orbit_wars", configuration={"seed": seed}, debug=False)
    env.run([p0, p1])
    final = env.steps[-1]
    r0, r1 = final[0].get("reward"), final[1].get("reward")
    fr, orr = (r0, r1) if focal_p0 else (r1, r0)
    outcome = "WIN" if (fr or 0) > (orr or 0) else ("LOSS" if (orr or 0) > (fr or 0) else "DRAW")
    # by phase: [n_steps, sum_raw, max_raw, sum_my_srcs, sum_defended]
    by_phase = defaultdict(lambda: [0, 0, 0, 0, 0])
    for t, step in enumerate(env.steps):
        obs = step[me].get("observation")
        if not obs or not obs.get("planets"):
            continue
        res = analyze_step(obs, me)
        if res is None:
            continue
        raw, msrc, defd = res
        b = _bucket(t)
        rec = by_phase[b]
        rec[0] += 1; rec[1] += raw; rec[2] = max(rec[2], raw)
        rec[3] += msrc; rec[4] += defd
    return outcome, len(env.steps), by_phase


def main():
    print(f"== sync-coalition generator vs {Path(OPP).name}  seeds={SEEDS}  seat={SEAT_ARG} ==")
    print("   (raw = 2-source coalitions the generator yields on the real board)\n")
    agg = defaultdict(lambda: [0, 0, 0, 0, 0])
    for seed in SEEDS:
        seats = [True, False] if SEAT_ARG == "both" else [SEAT_ARG == "p0"]
        for fp0 in seats:
            outcome, nsteps, by_phase = run_seed(seed, fp0)
            print(f"seed {seed} focal@{'P0' if fp0 else 'P1'}: {outcome:4s} steps={nsteps}")
            for name, _, _ in PHASES:
                n, sraw, mraw, msrc, defd = by_phase[name]
                if n:
                    print(f"    {name:18s} steps={n:3d}  raw_total={sraw:4d}  max_raw={mraw:2d}  "
                          f"avg_my_src={msrc/n:4.1f}  avg_defended_tgt={defd/n:4.1f}")
                for i in range(5):
                    agg[name][i] += by_phase[name][i]
            print(flush=True)
    print("=== AGGREGATE ===")
    for name, _, _ in PHASES:
        n, sraw, mraw, msrc, defd = agg[name]
        if n:
            print(f"  {name:18s} steps={n:4d}  raw_total={sraw:5d}  max_raw={mraw:2d}  "
                  f"avg_my_src={msrc/n:4.1f}  avg_defended_tgt={defd/n:4.1f}")
    mid = agg["MIDGAME 150-300"]
    print(f"\nDECISION: midgame raw coalitions generated total = {mid[1]} "
          f"(over {mid[0]} board-steps, max in any single step = {mid[2]}).")
    print("  >0 → 2-source regime reappears in contested midgames (closure does NOT transfer).")
    print("  ~0 → structural precondition still absent; closure transfers, STOP.")


if __name__ == "__main__":
    main()
