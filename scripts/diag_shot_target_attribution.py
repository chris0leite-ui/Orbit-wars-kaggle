"""Serve-vs-offline encoder fidelity audit for the shot-success filter.

The in-agent filter scores each wave against the PLANNER's true target
(entries.target_slots). The offline counterfactual + live mechanism read
(scripts/label_shot_outcomes.py) only see the emitted (src, angle, ships)
and RAY-CAST the angle to guess a target. If those targets disagree, the
offline "P(success)" is computed against the wrong planet — which would
make both the original counterfactual and the live mechanism read
unreliable.

This reproduces one local game with the 0.15 bundle, dumps the planner's
per-wave target + serve-time P, saves the replay, then ray-casts each
emitted angle the way the labeler does and reports agreement.

Usage:
    python scripts/diag_shot_target_attribution.py [--vs champ] [--seed 7]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.label_shot_outcomes import _infer_target_pid  # noqa: E402

BUNDLE = REPO / "submissions" / "producer_plus_vetorf4p_sync_shotmlp015_on.py"
OPP = {
    "champ": REPO / "submissions" / "champ_adaptiveK_on.py",
    "producer": REPO / "agents" / "producer" / "producer_agent.py",
    "control": REPO / "submissions" / "pp_sync_control.py",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vs", default="champ", choices=list(OPP))
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    dump = Path(tempfile.mkstemp(suffix="-shotdump.jsonl")[1])
    dump.write_text("")
    os.environ["PRODUCER_PLUS_SHOT_MLP_DUMP"] = str(dump)

    from kaggle_environments import make
    env = make("orbit_wars", configuration={"seed": args.seed}, debug=True)
    env.run([str(BUNDLE), str(OPP[args.vs])])
    steps = env.steps
    rewards = [s.reward for s in env.state]
    print(f"game: {len(steps)} steps, rewards {rewards}, "
          f"focal=P0 ({BUNDLE.name})")

    serve = [json.loads(l) for l in dump.read_text().splitlines() if l.strip()]
    # Focal is P0. Keep only P0's waves (the mirror also dumps opp waves).
    serve = [r for r in serve if r["player_id"] == 0]
    if not serve:
        print("no serve records (filter never scored an attack wave) — "
              "local referee blindness; try --vs control or a 4P pool.")
        return 0

    # For each serve record, ray-cast its angle against the planets the
    # focal saw at that step — exactly as the labeler does — and compare.
    agree = 0
    total = 0
    p_low_serve = 0
    raycast_none = 0
    for r in serve:
        step = r["step"]
        if step >= len(steps):
            continue
        obs = steps[step][0].get("observation", {}) or {}
        planets = obs.get("planets", []) or []
        by_id = {int(p[0]): p for p in planets}
        src = by_id.get(r["src_slot"])
        if src is None:
            continue
        raycast_tgt = _infer_target_pid(
            (float(src[2]), float(src[3])), r["angle"], planets)
        total += 1
        if raycast_tgt is None:
            raycast_none += 1
        elif raycast_tgt == r["planner_tgt_slot"]:
            agree += 1
        if r["serve_p"] < 0.15:
            p_low_serve += 1

    print(f"\nserve-time attack waves scored (focal): {len(serve)}")
    print(f"  with serve-P < 0.15 (would be dropped): {p_low_serve} "
          f"({100*p_low_serve/max(1,len(serve)):.1f}%)")
    print(f"\ntarget attribution (offline ray-cast vs planner truth), "
          f"n={total}:")
    print(f"  agree:        {agree} ({100*agree/max(1,total):.1f}%)")
    print(f"  ray-cast None:{raycast_none} ({100*raycast_none/max(1,total):.1f}%)")
    print(f"  DISAGREE:     {total-agree-raycast_none} "
          f"({100*(total-agree-raycast_none)/max(1,total):.1f}%)")
    print("\nRead: low agreement => the offline counterfactual + live "
          "mechanism read score launches against mis-attributed targets; "
          "the filter acts on the planner target, so the two metrics "
          "measure different things.")
    dump.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
