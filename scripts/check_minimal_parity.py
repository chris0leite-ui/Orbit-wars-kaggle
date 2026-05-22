"""check_minimal_parity — turn-by-turn action-stream parity check.

Verifies that agents/minimal/main.py emits identical action streams to
the orbitfix configuration of agents/baseline/main.py on the same seeds.
This is the "100% accurate" validation gate for the consolidation.

In-process: orbitfix's env vars are set BEFORE any agent import (so the
baseline path reads them as constants at module load); both agents are
then imported and driven on independent env copies for each seed. Action
streams must match byte-for-byte (modulo float-repr rounding to 6 dp).

Usage
-----
    python scripts/check_minimal_parity.py             # default: 4 seeds × 60 turns
    python scripts/check_minimal_parity.py --seeds 8
    python scripts/check_minimal_parity.py --turns 80
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# Mirror submissions/baseline_joint_aggr_consolidated_orbitfix.py BEFORE
# any agent import — these gate module-level constants in chooser_trajectory.
os.environ.setdefault("BASELINE_JOINT_AGGR", "1")
os.environ.setdefault("BASELINE_JOINT_TOP_K", "5")
os.environ.setdefault("BASELINE_JOINT_MAX_PAIRS", "60")
os.environ.setdefault("BASELINE_REINFORCE_EMIT", "1")
os.environ.setdefault("BASELINE_REINFORCE_ANTICIPATE", "1")
os.environ.setdefault("BASELINE_NEUTRAL_BONUS", "2.0")
os.environ.setdefault("BASELINE_NEUTRAL_EARLY_EXTRA", "1.5")
os.environ.setdefault("BASELINE_NEUTRAL_EARLY_HORIZON", "50")
os.environ.setdefault("BASELINE_ORBITAL_SAFETY", "1")

from kaggle_environments import make  # noqa: E402

from agents.baseline.main import agent as orbitfix_agent  # noqa: E402
from agents.minimal.main import agent as minimal_agent  # noqa: E402


def _action_repr(a) -> str:
    if not a:
        return "[]"
    out = []
    for entry in a:
        try:
            sid, ang, ships = entry[0], entry[1], entry[2]
            out.append(f"({int(sid)},{float(ang):.6f},{int(ships)})")
        except (IndexError, TypeError, ValueError):
            out.append(repr(entry))
    return "[" + ",".join(out) + "]"


def run_one(seed: int, agent_fn, turns: int) -> list[tuple]:
    env = make("orbit_wars", configuration={"seed": int(seed)})
    env.reset(num_agents=2)
    log: list[tuple] = []
    for t in range(turns):
        p0_obs = env.state[0].observation
        p1_obs = env.state[1].observation
        a0 = agent_fn(p0_obs)
        a1 = agent_fn(p1_obs)
        log.append((t, _action_repr(a0), _action_repr(a1)))
        env.step([a0, a1])
        if env.done:
            break
    return log


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--turns", type=int, default=60)
    args = ap.parse_args()

    all_match = True
    for s in range(args.seeds):
        print(f"[seed {s}] running...", flush=True)
        mlog = run_one(s, minimal_agent, args.turns)
        olog = run_one(s, orbitfix_agent, args.turns)
        if len(mlog) != len(olog):
            print(f"[seed {s}] LENGTH MISMATCH: minimal={len(mlog)} orbitfix={len(olog)}",
                  flush=True)
            all_match = False
            continue
        diffs = 0
        first_diff = None
        for (mt, ma0, ma1), (ot, oa0, oa1) in zip(mlog, olog):
            if mt != ot or ma0 != oa0 or ma1 != oa1:
                diffs += 1
                if first_diff is None:
                    first_diff = (mt, ma0, oa0, ma1, oa1)
        if diffs == 0:
            print(f"[seed {s}] PARITY OK  ({len(mlog)} turns)", flush=True)
        else:
            all_match = False
            t, ma0, oa0, ma1, oa1 = first_diff
            print(f"[seed {s}] {diffs}/{len(mlog)} turn diffs; FIRST at turn {t}",
                  flush=True)
            print(f"    minimal  p0={ma0}", flush=True)
            print(f"    orbitfix p0={oa0}", flush=True)
            print(f"    minimal  p1={ma1}", flush=True)
            print(f"    orbitfix p1={oa1}", flush=True)
    print()
    print("RESULT:", "PARITY GREEN" if all_match else "PARITY FAIL", flush=True)
    return 0 if all_match else 1


if __name__ == "__main__":
    sys.exit(main())
