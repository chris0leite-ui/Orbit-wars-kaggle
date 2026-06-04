"""protoflow_probe — go/no-go triage for the flow-field "converging streams" agent.

Runs agents/protoflow/main.py against the current champion and the Producer over a
small seed panel (both seats), and reports the two things the probe exists to
answer:

  (A) Does convergence emerge?  -> convergence-turns/game = turns where >=2 of our
      planets launched at the SAME target (the thing the per-launch champion cannot do).
  (B) Is it competitive, or inert like its three analytic ancestors?  -> winrate +
      Wilson lower bound, plus launches/game and idle-fraction (a flat midgame shows
      up as a high idle-fraction).

This is a TRIAGE probe, not a submission gate — Rule 45's n>=32 lift gate does not
apply here. Games run serially (in-process) so the prototype's module-global trace
is readable after each game; keep the seed count small.

Usage:
    python scripts/protoflow_probe.py                 # default: 6 seeds x 2 seats x 2 opponents
    python scripts/protoflow_probe.py --seeds 4
    python scripts/protoflow_probe.py --opponents submissions/champ_refine_adaptivek.py
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import agents.protoflow.main as proto  # noqa: E402  (path set above)

from fast import wilson_ci, _load_callable  # noqa: E402  reuse the repo's loader + Wilson

PROTO_PATH = REPO / "agents" / "protoflow" / "main.py"
DEFAULT_CHAMP = "submissions/champ_refine_adaptivek.py"
DEFAULT_PRODUCER = "agents/producer/producer_agent.py"


def load_callable(path: str):
    # Reuse fast.py's loader: it registers the module in sys.modules BEFORE
    # exec_module, which the bundled champion's module-level @dataclass needs.
    p = Path(path)
    if not p.is_absolute():
        p = REPO / p
    return _load_callable(str(p))


def trace_metrics(trace: list[dict]) -> dict:
    """Per-game convergence / activity metrics from the prototype's turn trace."""
    n_turns = len(trace)
    n_launches = sum(len(t["launches"]) for t in trace)
    idle_turns = sum(1 for t in trace if t["idle"])
    conv_turns = 0
    max_cohort = 0
    for t in trace:
        by_tgt: dict[int, int] = defaultdict(int)
        for _src, tgt, _ships in t["launches"]:
            by_tgt[tgt] += 1
        if by_tgt:
            biggest = max(by_tgt.values())
            max_cohort = max(max_cohort, biggest)
            if biggest >= 2:
                conv_turns += 1
    return {
        "turns": n_turns,
        "launches": n_launches,
        "idle_frac": idle_turns / n_turns if n_turns else 0.0,
        "conv_turns": conv_turns,
        "max_cohort": max_cohort,
    }


def run_game(seed: int, focal, opp, focal_is_p0: bool):
    from kaggle_environments import make

    proto.reset_trace()
    env = make("orbit_wars", configuration={"seed": seed}, debug=False)
    line_up = [focal, opp] if focal_is_p0 else [opp, focal]
    try:
        env.run(line_up)
    except Exception as exc:  # pragma: no cover
        return None, [], f"error: {exc}"
    final = env.steps[-1]
    r0, r1 = final[0].reward, final[1].reward
    focal_r, opp_r = (r0, r1) if focal_is_p0 else (r1, r0)
    if focal_r is None or opp_r is None:
        return None, proto.get_trace(), "error: null reward"
    won = focal_r > opp_r
    return won, proto.get_trace(), "p0" if focal_is_p0 else "p1"


def probe_opponent(name: str, opp_path: str, seeds: int, focal):
    opp = load_callable(opp_path)
    wins = 0
    n = 0
    agg = defaultdict(float)
    print(f"\n=== vs {name}  ({opp_path}) ===")
    for seed in range(seeds):
        for focal_is_p0 in (True, False):
            t0 = time.time()
            won, trace, seat = run_game(seed, focal, opp, focal_is_p0)
            m = trace_metrics(trace)
            n += 1
            if won is True:
                wins += 1
            res = "WIN " if won else ("LOSS" if won is False else "ERR ")
            print(f"  seed {seed:>3} {seat}  {res}  "
                  f"launches={m['launches']:>4}  idle={m['idle_frac']:.2f}  "
                  f"conv_turns={m['conv_turns']:>3}  max_cohort={m['max_cohort']}  "
                  f"({time.time()-t0:.1f}s)")
            if won is not None:
                for k, v in m.items():
                    agg[k] += v
    lo, hi = wilson_ci(wins, n)
    games = max(1, n)
    print(f"  --> {name}: {wins}/{n} ({100*wins/games:.1f}%)  Wilson[{lo:.3f}, {hi:.3f}]")
    print(f"      mean/game: launches={agg['launches']/games:.1f}  "
          f"idle_frac={agg['idle_frac']/games:.2f}  "
          f"conv_turns={agg['conv_turns']/games:.1f}  "
          f"max_cohort_seen={int(agg['max_cohort']/games) if games else 0}")
    return wins, n


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=6, help="seeds per opponent (x2 seats)")
    ap.add_argument("--opponents", default=f"{DEFAULT_CHAMP},{DEFAULT_PRODUCER}",
                    help="comma-separated opponent paths")
    args = ap.parse_args()

    # Use the IMPORTED module's agent (not a fresh _load_callable copy) so the
    # trace we reset/read is the same _TRACE object the running agent writes to.
    focal = proto.agent
    print(f"protoflow probe — {args.seeds} seeds x 2 seats per opponent")
    for opp_path in args.opponents.split(","):
        opp_path = opp_path.strip()
        if not opp_path:
            continue
        name = Path(opp_path).stem
        probe_opponent(name, opp_path, args.seeds, focal)

    print("\nRead: conv_turns>0 and max_cohort>=2 -> convergence emerges; "
          "idle_frac high / launches~0 -> inert (kill); winrate in the fight -> build the full agent.")


if __name__ == "__main__":
    main()
