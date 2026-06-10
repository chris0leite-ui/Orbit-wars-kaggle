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
DEFAULT_LITE_GREEDY = "agents/lite_greedy/main.py"   # fastest opponent; clean alignment baseline
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
    """Per-game convergence / activity / waste metrics from the turn trace.

    The "good properties" lens (PI): a sound flow-field should naturally produce
    aligned, non-wasteful actions. We surface the signals that would expose the
    opposite: tiny fleets (<3 ships), far shots (dist>50, easy for the opponent
    to react to), and split cohorts whose legs do NOT share an arrival turn (two
    small fleets that arrive apart -> they don't sum -> waste).
    """
    n_turns = len(trace)
    all_launches = [lc for t in trace for lc in t["launches"]]
    n_launches = len(all_launches)
    idle_turns = sum(1 for t in trace if t["idle"])
    conv_turns = 0
    max_cohort = 0
    bad_split = 0  # cohort turns where legs to one target span >1 arrival turn
    for t in trace:
        by_tgt: dict[int, list] = defaultdict(list)
        for lc in t["launches"]:
            by_tgt[lc["tgt"]].append(lc)
        for tgt, legs in by_tgt.items():
            max_cohort = max(max_cohort, len(legs))
            if len(legs) >= 2:
                conv_turns += 1
                if len({lc["arrive_turn"] for lc in legs}) > 1:
                    bad_split += 1
    sizes = [lc["ships"] for lc in all_launches]
    dists = [lc["dist"] for lc in all_launches]
    regroup = sum(1 for lc in all_launches if lc.get("kind") == "regroup")
    final = trace[-1] if trace else {}
    return {
        "turns": n_turns,
        "launches": n_launches,
        "idle_frac": idle_turns / n_turns if n_turns else 0.0,
        "conv_turns": conv_turns,
        "max_cohort": max_cohort,
        "tiny_frac": (sum(1 for s in sizes if s < 3) / len(sizes)) if sizes else 0.0,
        "far_frac": (sum(1 for d in dists if d > 50) / len(dists)) if dists else 0.0,
        "bad_split": bad_split,
        "regroup": regroup,
        "end_planets": final.get("my_planets", 0),
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
    # One INDEPENDENT game per seed (a fresh board each time). Seats are balanced
    # ACROSS seeds (even seed -> we play p0, odd -> p1) rather than playing both seats
    # of the SAME seed -- p0/p1 on one seed are mirror games, not independent samples.
    for seed in range(seeds):
        focal_is_p0 = (seed % 2 == 0)
        t0 = time.time()
        won, trace, seat = run_game(seed, focal, opp, focal_is_p0)
        m = trace_metrics(trace)
        n += 1
        if won is True:
            wins += 1
        res = "WIN " if won else ("LOSS" if won is False else "ERR ")
        print(f"  seed {seed:>3} {seat}  {res}  "
              f"launches={m['launches']:>4}  idle={m['idle_frac']:.2f}  "
              f"conv={m['conv_turns']:>3}  cohort={m['max_cohort']}  "
              f"tiny={m['tiny_frac']:.2f}  far={m['far_frac']:.2f}  "
              f"bad_split={m['bad_split']:>2}  regroup={m['regroup']:>3}  "
              f"end_planets={m['end_planets']:>2}  ({time.time()-t0:.1f}s)")
        if won is not None:
            for k, v in m.items():
                agg[k] += v
    lo, hi = wilson_ci(wins, n)
    games = max(1, n)
    print(f"  --> {name}: {wins}/{n} ({100*wins/games:.1f}%)  Wilson[{lo:.3f}, {hi:.3f}]")
    print(f"      mean/game: launches={agg['launches']/games:.1f}  "
          f"idle_frac={agg['idle_frac']/games:.2f}  conv_turns={agg['conv_turns']/games:.1f}  "
          f"tiny_frac={agg['tiny_frac']/games:.2f}  far_frac={agg['far_frac']/games:.2f}  "
          f"bad_split={agg['bad_split']/games:.1f}/game  regroup={agg['regroup']/games:.1f}/game  "
          f"end_planets={agg['end_planets']/games:.1f}")
    return wins, n


def verbose_game(seed: int, opp_path: str, focal, max_turns: int = 60):
    """Run one game and dump per-turn launches so we can EYEBALL action quality:
    are fleets well-sized and concentrated, or tiny / far / split senselessly?"""
    opp = load_callable(opp_path)
    print(f"\n=== verbose game: protoflow (P0) vs {Path(opp_path).stem}  seed {seed} ===")
    won, trace, _seat = run_game(seed, focal, opp, True)
    for t in trace[:max_turns]:
        if not t["launches"]:
            continue
        by_tgt: dict[int, list] = defaultdict(list)
        for lc in t["launches"]:
            by_tgt[lc["tgt"]].append(lc)
        parts = []
        for tgt, legs in by_tgt.items():
            owner = legs[0]["tgt_owner"]
            tag = "ENEMY" if owner not in (-1,) else "neutral"
            kind = legs[0]["kind"]
            srcs = "+".join(f"{lc['src']}({lc['ships']})" for lc in legs)
            arr = legs[0]["arrive_turn"]
            dist = legs[0]["dist"]
            cohort = f" COHORT(arrive@{arr})" if len(legs) >= 2 else ""
            parts.append(f"->{tgt}[{tag},{kind},d={dist},floor={legs[0]['floor']}] {srcs}{cohort}")
        print(f"  step {t['step']:>3}  planets={t['my_planets']} ships={t['my_ships']:>3}  " + "  ".join(parts))
    print(f"  result: {'WIN' if won else 'LOSS'}  (final planets={trace[-1]['my_planets'] if trace else 0})")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=6,
                    help="independent games per opponent (one per seed; seats balanced across seeds)")
    ap.add_argument("--opponents", default=DEFAULT_LITE_GREEDY,
                    help="comma-separated opponent paths (default: the fast light-greedy agent)")
    ap.add_argument("--verbose-seed", type=int, default=None,
                    help="dump per-turn launches for one game vs the first opponent at this seed")
    ap.add_argument("--simulate-value", action="store_true",
                    help="use the simulation-based evaluator (proto.SIMULATE_VALUE = True) for this run")
    ap.add_argument("--drain-cost", action="store_true",
                    help="price the source-drain cost in offense values (proto.SIMVALUE_DRAIN_COST = True)")
    ap.add_argument("--drain-anticipatory", action="store_true",
                    help="charge the drain against the source's anticipated standing counter (proto.SIMVALUE_DRAIN_ANTICIPATORY = True)")
    ap.add_argument("--flowdiff", action="store_true",
                    help="single-currency terminal-wealth evaluator (proto.FLOWDIFF_VALUE = True; supersedes the sim evaluator)")
    ap.add_argument("--flowdiff-tail", action="store_true",
                    help="ownership continuation credit beyond the window (proto.FLOWDIFF_TAIL = True)")
    ap.add_argument("--flowdiff-reaction", action="store_true",
                    help="inject the defender's standing counter into offense rollouts (proto.FLOWDIFF_REACTION = True)")
    ap.add_argument("--flowdiff-reaction-adaptive", action="store_true",
                    help="scale the injected reaction by the opponent's measured reinforcement propensity (proto.FLOWDIFF_REACTION_ADAPTIVE = True)")
    args = ap.parse_args()

    # Evaluator A/B toggles: flip the probe agent's module-level flags before any game runs.
    proto.SIMULATE_VALUE = bool(args.simulate_value)
    proto.SIMVALUE_DRAIN_COST = bool(args.drain_cost)
    proto.SIMVALUE_DRAIN_ANTICIPATORY = bool(args.drain_anticipatory)
    proto.FLOWDIFF_VALUE = bool(args.flowdiff)
    proto.FLOWDIFF_TAIL = bool(args.flowdiff_tail)
    proto.FLOWDIFF_REACTION = bool(args.flowdiff_reaction)
    proto.FLOWDIFF_REACTION_ADAPTIVE = bool(args.flowdiff_reaction_adaptive)
    print(f"SIMULATE_VALUE = {proto.SIMULATE_VALUE}  SIMVALUE_DRAIN_COST = {proto.SIMVALUE_DRAIN_COST}  "
          f"SIMVALUE_DRAIN_ANTICIPATORY = {proto.SIMVALUE_DRAIN_ANTICIPATORY}  FLOWDIFF_VALUE = {proto.FLOWDIFF_VALUE}")

    # Use the IMPORTED module's agent (not a fresh _load_callable copy) so the
    # trace we reset/read is the same _TRACE object the running agent writes to.
    focal = proto.agent
    opps = [o.strip() for o in args.opponents.split(",") if o.strip()]

    if args.verbose_seed is not None:
        verbose_game(args.verbose_seed, opps[0], focal)
        return

    print(f"protoflow probe — {args.seeds} independent games per opponent (one per seed, balanced seats)")
    for opp_path in opps:
        name = Path(opp_path).stem
        probe_opponent(name, opp_path, args.seeds, focal)

    print("\nRead: conv_turns>0 and max_cohort>=2 -> convergence emerges; "
          "idle_frac high / launches~0 -> inert (kill); winrate in the fight -> build the full agent.")


if __name__ == "__main__":
    main()
