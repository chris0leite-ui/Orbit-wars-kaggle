"""Stage 1 verification — does CONSOLIDATION-from-step-0 actually fire?

Plays games with BUILDUP_PLANNER_OPENING_ENABLED=0 and records, per turn
for steps 0..29:
  - emitted actions (src, target, ships, aim)
  - ledger churn (did the target set change vs previous turn?)
  - whether we held (no emit at all)

Two pass criteria:
  (a) Emit rate: fraction of opening turns (0..29) where >=1 action emitted
      should be >= 0.6. Phase-4-era LP saw 0.23 — that's the floor we
      have to clear to falsify the 'changing-mind' risk.
  (b) Ledger churn: fraction of consecutive-turn pairs where the
      emitted-target-set differs in identity (not just count). High churn
      => 'changing-mind' still happening; low churn => stable plan.

Usage:
    python scripts/diag_consolidation_opening.py [seed] [opp]
    python scripts/diag_consolidation_opening.py --panel [opp]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# Apply the env BEFORE importing agent (the gate is read at import time
# via setdefault paths). Hard-set so any inherited "1" gets overridden.
os.environ["BUILDUP_PLANNER_OPENING_ENABLED"] = "0"

from kaggle_environments import make
from kaggle_environments.envs.orbit_wars.orbit_wars import Planet

from agents.buildup_planner import main as bp_main
from fast import _load_callable


PANEL_SEED_PATH = REPO / "data" / "seed_panel_128.json"


def _planet_at_angle(planets, src, aim, ships):
    """Heuristic: find the target planet that best matches the (src, aim)
    direction. Used purely for trace-logging; the agent's intent is the
    aim angle, not a target ID, so we recover the likely target by
    angular distance from src's position to each planet."""
    best = None
    best_d = math.inf
    for p in planets:
        if int(p.id) == int(src.id):
            continue
        dx = float(p.x) - float(src.x)
        dy = float(p.y) - float(src.y)
        ang = math.atan2(dy, dx)
        d = abs((ang - aim + math.pi) % (2 * math.pi) - math.pi)
        if d < best_d:
            best_d = d
            best = p
    return best


def diag_one(seed: int, opp_path: str, max_turns: int = 30, verbose: bool = True):
    """Play one game and return per-turn record + summary stats."""
    env = make("orbit_wars", configuration={"seed": seed, "episodeSteps": 500})
    env.reset(num_agents=2)
    opp_agent = _load_callable(opp_path)

    state = env.steps[0]
    last_target_ids: set[int] = set()
    record = []
    churn_pairs = 0
    pair_total = 0
    emit_turns = 0

    for turn in range(max_turns):
        obs0 = state[0]["observation"] if isinstance(state[0], dict) else state[0].observation
        obs1 = state[1]["observation"] if isinstance(state[1], dict) else state[1].observation
        obs_d = bp_main._as_dict(obs0)
        me = int(obs_d.get("player", 0))
        raw_planets = obs_d.get("planets", []) or []
        planets = [Planet(*p) for p in raw_planets]
        planets_by_id = {int(p.id): p for p in planets}
        my_planets = [p for p in planets if int(p.owner) == me]
        opp_planets = [p for p in planets if int(p.owner) != me and int(p.owner) >= 0]

        my_ships = sum(int(p.ships) for p in my_planets)
        opp_ships = sum(int(p.ships) for p in opp_planets)

        a0 = bp_main.agent(obs0, env.configuration)
        try:
            a1 = opp_agent(obs1, env.configuration)
        except TypeError:
            a1 = opp_agent(obs1)

        # Decode targets from emitted actions
        tgt_ids = []
        emit_detail = []
        for act in (a0 or []):
            src_id, aim, ships = int(act[0]), float(act[1]), int(act[2])
            src = planets_by_id.get(src_id)
            if src is None:
                continue
            tgt = _planet_at_angle(planets, src, aim, ships)
            if tgt is not None:
                tgt_ids.append(int(tgt.id))
                emit_detail.append({
                    "src": src_id, "tgt": int(tgt.id),
                    "tgt_gar": int(tgt.ships), "tgt_prod": int(tgt.production),
                    "ships": ships,
                })
        emitted = bool(a0)
        if emitted:
            emit_turns += 1

        current_target_set = set(tgt_ids)
        if turn > 0:
            pair_total += 1
            if current_target_set != last_target_ids:
                churn_pairs += 1

        record.append({
            "turn": turn, "my_ships": my_ships, "opp_ships": opp_ships,
            "my_p": len(my_planets), "opp_p": len(opp_planets),
            "n_emits": len(a0 or []), "emits": emit_detail,
        })

        if verbose:
            head = (f"  t={turn:>2}  ships me/opp={my_ships:>3}/{opp_ships:<3}  "
                    f"planets={len(my_planets)}/{len(opp_planets)}  emits={len(a0 or [])}")
            print(head, end="")
            if emit_detail:
                print("  →  " + ", ".join(
                    f"p{e['src']}→p{e['tgt']} (g{e['tgt_gar']} pr{e['tgt_prod']}) "
                    f"x{e['ships']}" for e in emit_detail))
            else:
                print()

        last_target_ids = current_target_set
        state = env.step([a0, a1])
        s0 = state[0] if isinstance(state[0], dict) else state[0]
        status = s0.get("status") if isinstance(s0, dict) else getattr(s0, "status", "ACTIVE")
        if status != "ACTIVE":
            if verbose:
                print(f"  GAME OVER at turn={turn} status={status}")
            break

    emit_rate = emit_turns / max(1, len(record))
    churn = churn_pairs / max(1, pair_total)
    return {
        "seed": seed,
        "turns": len(record),
        "emit_turns": emit_turns,
        "emit_rate": emit_rate,
        "churn_pairs": churn_pairs,
        "churn_total": pair_total,
        "churn_rate": churn,
        "record": record,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("seed", nargs="?", type=int, default=1622482326)
    ap.add_argument("--opp", default="nearest",
                    help="opp agent name (nearest, v7_0, v4_planner, or path)")
    ap.add_argument("--panel", action="store_true",
                    help="run 16 panel seeds instead of one")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    opp_map = {
        "nearest": str(REPO / "agents/simple/nearest.py"),
        "v7_0": str(REPO / "submissions/v7_0_drop_one.py"),
        "v4_planner": str(REPO / "submissions/v4_planner.py"),
        "phi1": str(REPO / "submissions/buildup_planner_phi1_only.py"),
        "concentration": str(REPO / "submissions/buildup_planner_concentration.py"),
    }
    opp_path = opp_map.get(args.opp, args.opp)

    if args.panel:
        with open(PANEL_SEED_PATH) as f:
            panel_doc = json.load(f)
        panel = panel_doc["panel"] if isinstance(panel_doc, dict) else panel_doc
        seeds = [int(s["seed"]) for s in panel[:16]]
    else:
        seeds = [args.seed]

    print(f"== diag_consolidation_opening  opp={args.opp}  "
          f"BUILDUP_PLANNER_OPENING_ENABLED={os.environ['BUILDUP_PLANNER_OPENING_ENABLED']} ==\n")

    results = []
    for s in seeds:
        verbose = (not args.quiet) and (len(seeds) <= 2)
        print(f"\n--- seed {s} ---")
        r = diag_one(s, opp_path, verbose=verbose)
        results.append(r)
        if not verbose:
            print(f"  emit_rate={r['emit_rate']:.2f}  "
                  f"churn={r['churn_rate']:.2f}  "
                  f"turns={r['turns']}")

    # Aggregate
    if len(results) > 1:
        em = sum(r["emit_turns"] for r in results) / sum(r["turns"] for r in results)
        cp = sum(r["churn_pairs"] for r in results) / max(1, sum(r["churn_total"] for r in results))
        print(f"\n== AGGREGATE  n_games={len(results)} ==")
        print(f"  pooled emit_rate = {em:.3f}  (pass: >=0.6)")
        print(f"  pooled churn     = {cp:.3f}  (pass: <0.3)")
        print(f"  per-seed emit_rate range: "
              f"{min(r['emit_rate'] for r in results):.2f} – "
              f"{max(r['emit_rate'] for r in results):.2f}")


if __name__ == "__main__":
    main()
