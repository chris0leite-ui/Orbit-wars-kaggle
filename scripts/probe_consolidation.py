#!/usr/bin/env python3
"""Plan v5 STEP 1 — mass-to-HOLD consolidation INERT-CHECK (pre-registered gate).

Counts how often the consolidation OPPORTUNITY actually arises in real games
before any mechanism is built. The closest prior (the baseline capture-coalition
refiner) generated ZERO coalitions ever; this probe is the cheap kill for the
HOLD variant.

An opportunity = a high-value ENEMY planet that NO single source can HOLD (some
can only PRESSURE), but the nearest 2..max_legs sources' pooled budget clears the
hold threshold at a synchronised arrival (chooser.consolidation_opportunities).

PRE-REGISTERED thresholds (frozen):
  NO-GO  : opportunities on < 1% of focal turns AND median < 2 / game  -> STOP.
  GO     : >= 3% of focal turns OR median >= 5 / game                  -> build.
  MARGINAL (between)                                                    -> ask PI.

Usage:
  python scripts/probe_consolidation.py                  # 32 geometry seeds vs default panel
  python scripts/probe_consolidation.py --seeds 16 --vs v7_0
"""
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from lib.intent import realize
from lib.mechanism import DEFAULT_MECHANISMS

from agents.holdgrab.chooser import _spendable, consolidation_opportunities, select
from agents.holdgrab.config import DEFAULT
from agents.holdgrab.world_view import build_turn_view
from agents.holdgrab.main import agent as holdgrab_agent
from fast import _load_callable, resolve_agent_spec


def _phase(step: int) -> str:
    if step < 100:
        return "early(<100)"
    if step < 300:
        return "mid(100-300)"
    return "late(>=300)"


def _make_recording_holdgrab(use_rollout: bool):
    """Return (agent, start_game). Each focal turn builds the view ONCE, records
    opportunity stats, then plays a holdgrab move.

    Default plays the CLOSED-FORM path (mirrors main.py lines 30-31), ~20x faster
    than the rollout, so the census stays cheap. The opportunity enumeration is
    rollout-independent, and closed-form holdgrab exercises the same board
    geometry / source-budget structure the opportunity is a property of. Use
    --rollout-focal for full-fidelity board trajectories at ~20x the cost."""
    state = {"turns": []}

    def start_game(turns_list):
        state["turns"] = turns_list

    def recording_agent(obs, configuration=None):
        view = build_turn_view(obs, DEFAULT)
        try:
            if view.my_sources and view.targets:
                spendable = _spendable(view, DEFAULT)
                opps = consolidation_opportunities(view, DEFAULT, spendable)
            else:
                opps = []
            state["turns"].append({
                "step": int(view.step),
                "phase": _phase(int(view.step)),
                "n_opps": len(opps),
                "values": [round(c.value, 1) for c in opps],
                "tgts": [c.tgt_id for c in opps],
            })
        except Exception as exc:  # never let instrumentation crash the game
            state["turns"].append({"step": -1, "phase": "err", "n_opps": 0,
                                   "values": [], "tgts": [], "err": str(exc)})
        if not view.my_sources or not view.targets:
            return []
        if use_rollout:
            return holdgrab_agent(obs, configuration)
        intents = select(view, DEFAULT)
        return realize(intents, view.world.obs_raw,
                       mechanisms=DEFAULT_MECHANISMS, model=view.model)

    return recording_agent, start_game


def run(seeds, opponents, use_rollout: bool) -> int:
    from kaggle_environments import make

    rec_agent, start_game = _make_recording_holdgrab(use_rollout)

    all_turns: list = []          # every focal turn across all games
    per_game_opp_counts: list = []  # opportunities per game

    n_games = 0
    for opp_name, opp_path in opponents:
        opp = _load_callable(opp_path)
        for seed in seeds:
            for focal_is_p0 in (True, False):
                turns: list = []
                start_game(turns)
                env = make("orbit_wars", configuration={"seed": seed}, debug=False)
                players = [rec_agent, opp] if focal_is_p0 else [opp, rec_agent]
                try:
                    env.run(players)
                except Exception as exc:
                    print(f"  [warn] seed {seed} vs {opp_name} "
                          f"(focal_p0={focal_is_p0}) errored: {exc}")
                    continue
                n_games += 1
                all_turns.extend(turns)
                per_game_opp_counts.append(sum(t["n_opps"] for t in turns))

    # ---- tally ----
    total_turns = len(all_turns)
    turns_with_opp = sum(1 for t in all_turns if t["n_opps"] > 0)
    total_opps = sum(t["n_opps"] for t in all_turns)
    pct_turns = (100.0 * turns_with_opp / total_turns) if total_turns else 0.0
    median_per_game = statistics.median(per_game_opp_counts) if per_game_opp_counts else 0.0

    by_phase: dict[str, list[int]] = {}
    for t in all_turns:
        by_phase.setdefault(t["phase"], [0, 0])
        by_phase[t["phase"]][0] += 1
        by_phase[t["phase"]][1] += t["n_opps"]

    print("\n" + "=" * 64)
    print("Plan v5 STEP 1 — mass-to-HOLD consolidation census")
    print("=" * 64)
    print(f"games played          : {n_games}")
    print(f"focal turns total     : {total_turns}")
    print(f"turns with >=1 opp    : {turns_with_opp}  ({pct_turns:.2f}% of turns)")
    print(f"total opportunities   : {total_opps}")
    print(f"opps/game  median     : {median_per_game}")
    if per_game_opp_counts:
        print(f"opps/game  max        : {max(per_game_opp_counts)}   "
              f"mean: {statistics.mean(per_game_opp_counts):.2f}")
    print("\nby game-phase (focal-turns / opportunities):")
    for phase in ("early(<100)", "mid(100-300)", "late(>=300)", "err"):
        if phase in by_phase:
            nturns, nopp = by_phase[phase]
            print(f"  {phase:14s}: {nturns:5d} turns   {nopp:5d} opps")
    all_values = [v for t in all_turns for v in t["values"]]
    if all_values:
        print(f"\ncoalition value (all enemy/double): n={len(all_values)} "
              f"median={statistics.median(all_values):.1f} "
              f"max={max(all_values):.1f}")

    # ---- pre-registered verdict ----
    print("\n" + "-" * 64)
    no_go = pct_turns < 1.0 and median_per_game < 2
    go = pct_turns >= 3.0 or median_per_game >= 5
    if no_go:
        verdict = "NO-GO  -> STOP. Inert like the baseline refiner; close the axis (Rule 37)."
    elif go:
        verdict = "GO     -> build STEP 2+ (the opportunity exists)."
    else:
        verdict = "MARGINAL -> surface this census to the PI before spending build budget."
    print(f"VERDICT: {verdict}")
    print("-" * 64)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="mass-to-HOLD consolidation inert-check")
    ap.add_argument("--seeds", type=int, default=32,
                    help="number of geometry-panel seeds (default 32)")
    ap.add_argument("--range-seeds", action="store_true",
                    help="use range(seeds) instead of the geometry panel")
    ap.add_argument("--vs", type=str, default="v7_0,v4_planner,v3.5.1",
                    help="comma-separated opponents (default: the eval panel)")
    ap.add_argument("--rollout-focal", action="store_true",
                    help="play the focal seat with the full rollout (20x slower, "
                         "full-fidelity trajectories); default plays closed-form")
    args = ap.parse_args(argv)

    if args.range_seeds:
        seeds = list(range(args.seeds))
    else:
        from lib.seed_panel import SEED_PANEL_128_INTERLEAVED
        seeds = list(SEED_PANEL_128_INTERLEAVED[: args.seeds])

    opponents = [resolve_agent_spec(s.strip()) for s in args.vs.split(",") if s.strip()]
    mode = "rollout" if args.rollout_focal else "closed-form"
    print(f"seeds: {len(seeds)}  opponents: {[n for n, _ in opponents]}  "
          f"(both seat orders) => {len(seeds) * len(opponents) * 2} games  "
          f"focal={mode}")
    return run(seeds, opponents, args.rollout_focal)


if __name__ == "__main__":
    raise SystemExit(main())
