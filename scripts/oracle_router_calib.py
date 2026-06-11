"""Calibrate the oracle_router rush detector.

Plays short probe games (focal = a passive watcher) against candidate
opponents and records the routing features at the decision turn: peak
enemy in-flight mass fraction and peak enemy fleet count over turns
0..DECIDE_T. Prints a table; thresholds go into agents/oracle_router/main.py.

Run AFTER any gating batteries (CPU contention discipline).

Usage:
  python scripts/oracle_router_calib.py --decide-t 12 \
      --opps producer,v7_0_drop_one,submissions/ledger_v1_4.py \
      --seeds 500-507
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from _agent_paths import resolve_agent_path  # noqa: E402

OPP_ALIASES = {
    "producer": "agents/producer/producer_agent.py",
    "champion": "data/external/live_vetorf_1291.py",
}


def probe(opp_path, seed, decide_t):
    from kaggle_environments import make
    feats = {"inflight_frac_peak": 0.0, "fleet_peak": 0}

    def watcher(obs, c=None):
        me = obs["player"] if isinstance(obs, dict) else obs.player
        fleets = (obs.get("fleets") if isinstance(obs, dict)
                  else obs.fleets) or []
        planets = (obs.get("planets") if isinstance(obs, dict)
                   else obs.planets) or []
        enemy_fleets = [f for f in fleets if f[1] >= 0 and f[1] != me]
        e_in = sum(f[6] for f in enemy_fleets)
        e_g = sum(p[5] for p in planets if p[1] >= 0 and p[1] != me)
        tot = e_in + e_g
        if tot > 0:
            feats["inflight_frac_peak"] = max(feats["inflight_frac_peak"],
                                              e_in / tot)
        feats["fleet_peak"] = max(feats["fleet_peak"], len(enemy_fleets))
        return []

    env = make("orbit_wars",
               configuration={"seed": seed, "episodeSteps": decide_t + 2},
               debug=False)
    env.run([watcher, opp_path])
    return feats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decide-t", type=int, default=12)
    ap.add_argument("--opps", default="producer,v7_0_drop_one,"
                    "submissions/ledger_v1_4.py,champion")
    ap.add_argument("--seeds", default="500-507")
    args = ap.parse_args()

    seeds = []
    for part in args.seeds.split(","):
        if "-" in part:
            a, b = part.split("-")
            seeds += list(range(int(a), int(b) + 1))
        else:
            seeds.append(int(part))

    for opp in args.opps.split(","):
        path = OPP_ALIASES.get(opp, opp)
        path = resolve_agent_path(path)
        rows = [probe(path, s, args.decide_t) for s in seeds]
        fr = sorted(r["inflight_frac_peak"] for r in rows)
        fl = sorted(r["fleet_peak"] for r in rows)
        print(f"{opp:30s} inflight_frac peak: min {fr[0]:.2f} "
              f"med {fr[len(fr)//2]:.2f} max {fr[-1]:.2f} | "
              f"fleets peak: min {fl[0]} med {fl[len(fl)//2]} max {fl[-1]}")


if __name__ == "__main__":
    main()
