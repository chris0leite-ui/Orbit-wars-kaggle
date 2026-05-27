"""Single-game tracer for `agents/lagrange_simple` against `random`.

Replays one (seed, focal_seat) pair, calls the focal agent's chooser
directly each turn, and dumps:
  - step / my_planets / opp_planets / my_ship_total / opp_ship_total
  - num candidates the enumerator produced; max value; whether ≥1 was non-partial
  - what the dual picked (count, total ships)
  - if no picks emitted AND opp_planets > 0: print one diagnostic line
    asking why (no candidates / no solo-feasible / all filtered).

Designed to answer: "on stuck random-ELIM seeds, what filter is rejecting
the kill move?"

Usage:
    python scripts/inspect_lagrange_simple_game.py --seed 80504 --focal-seat 0
    python scripts/inspect_lagrange_simple_game.py --seed 31448 --focal-seat 0 --start-step 300 --end-step 500
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# Match agent's runtime env (load-bearing).
os.environ.setdefault("BASELINE_ORBITAL_SAFETY", "1")
os.environ.setdefault("KINEMATIC_TABLE_ENABLED", "1")

from kaggle_environments import make
from kaggle_environments.envs.orbit_wars.orbit_wars import Planet

from lib.intent import World
from lib.world_model import WorldModel
from agents.lagrange_simple.dual import solve as solve_dual
from agents.lagrange_simple.score import (
    enumerate_candidates,
    _source_defensive_ok,
    _refine_ships,
    _capture_value,
    MAX_LAUNCH_TICK,
    MIN_FLEET,
    DEFENSE_HORIZON,
)
from agents.baseline.chooser_trajectory import (
    merge_ledgers,
    predict_opp_responses,
)
from agents.baseline.proposer import (
    _target_holdable_after_capture,
    aim_and_eta,
)
from lib.trajectory import predict_fleet_fate
from lib.world_model import predict_garrison_at


def _why_no_capture(world, model, me, omega, comet_ids):
    """When enumerate_candidates returns [], walk the filter chain on the
    "fattest" (src,tgt) pair and report which filter rejected.

    Heuristic: pick our planet with the most ships as src; opp planet with
    least ships as tgt. Try launch_tick=0 only (this is the diagnostic, not
    the optimizer).
    """
    my_planets = [p for p in world.planets_by_id.values() if int(p.owner) == int(me)]
    opps = [p for p in world.planets_by_id.values()
            if int(p.owner) != int(me) and int(p.owner) >= 0
            and int(p.id) not in comet_ids]
    if not my_planets or not opps:
        return "no-targets-or-sources"

    src = max(my_planets, key=lambda p: int(p.ships))
    tgt = min(opps, key=lambda p: int(p.ships))
    projected_opp = predict_opp_responses(world, int(me), num_seats=2)
    enriched = merge_ledgers(model.ledger, projected_opp)
    base_arrivals = list(enriched.get(int(tgt.id), []))

    msg = f"src=P{src.id}(ships={int(src.ships)},prod={int(src.production)}) tgt=P{tgt.id}(ships={int(tgt.ships)},prod={int(tgt.production)})"

    # try launch_tick=0
    lt = 0
    res = aim_and_eta(src, tgt, max(MIN_FLEET, int(tgt.ships) + 1), omega, wait_N=lt)
    if res is None:
        return f"REJ aim_and_eta_None  {msg}"
    angle, eta = res
    arrival = lt + eta
    owner_at_arr, gar_at_arr = predict_garrison_at(tgt, arrival, base_arrivals)
    if int(owner_at_arr) == int(me):
        return f"REJ tgt-already-ours-at-arrival  arr={arrival} {msg}"
    needed = int(math.ceil(float(gar_at_arr))) + 1
    if needed > int(src.ships):
        return f"REJ ships_needed>budget  needed={needed} budget={int(src.ships)} arr={arrival} gar={gar_at_arr:.1f} {msg}"
    fate = predict_fleet_fate(src, tgt, angle, needed, world, wait_N=lt)
    if fate.outcome != "target":
        return f"REJ fate.outcome={fate.outcome!r}  {msg}"
    if fate.hit_planet_id is None or int(fate.hit_planet_id) != int(tgt.id):
        return f"REJ fate.hit_planet_id={fate.hit_planet_id} (want {tgt.id})  {msg}"
    value = _capture_value(tgt, arrival)
    if value <= 0.0:
        return f"REJ value=0 (arrival>=500)  arr={arrival} {msg}"
    # dominant-endgame computed in enumerate_candidates
    dom = (len([p for p in world.planets_by_id.values()
                if int(p.owner) != int(me) and int(p.owner) >= 0])
           and len(my_planets) >= 3 * len(opps))
    if not dom and not _target_holdable_after_capture(
        src, tgt, needed, lt, eta, world, model, me,
    ):
        return f"REJ B1-hold-filter (non-dom-endgame)  {msg}"
    src_arr_list = list(enriched.get(int(src.id), []))
    if not _source_defensive_ok(src, int(needed), int(lt), src_arr_list):
        return f"REJ rear-defense (Phase B)  {msg}"
    return f"???should-have-fired  ships={needed} angle={angle:.3f} arr={arrival} {msg}"


def _obs_dict(obs):
    if isinstance(obs, dict):
        return obs
    return {
        "player": getattr(obs, "player", 0),
        "step": getattr(obs, "step", 0),
        "planets": list(getattr(obs, "planets", []) or []),
        "fleets": list(getattr(obs, "fleets", []) or []),
        "comets": list(getattr(obs, "comets", []) or []),
        "comet_planet_ids": list(getattr(obs, "comet_planet_ids", []) or []),
        "angular_velocity": float(getattr(obs, "angular_velocity", 0.0) or 0.0),
    }


def trace_game(seed: int, focal_seat: int, start_step: int, end_step: int,
               every: int, dump_when_no_emit: bool):
    env = make(
        "orbit_wars",
        configuration={
            "seed": int(seed),
            "actTimeout": 600,    # disable timeout for in-process trace
            "agentTimeout": 600,
            "runTimeout": 3600,
        },
        debug=False,
    )
    env.reset(2)
    # We need to actually run the game, but capture per-turn focal-side
    # state. Easiest path: use env.run with a custom focal agent that
    # delegates to lagrange_simple.agent but also logs.

    log: list[str] = []
    call_counter = [0]

    def _focal_agent(obs, configuration=None):
        call_counter[0] += 1
        try:
            return _focal_agent_inner(obs, configuration)
        except Exception as e:
            log.append(f"[call#{call_counter[0]}] EXCEPTION {type(e).__name__}: {e}")
            return []

    def _focal_agent_inner(obs, configuration=None):
        obs_d = _obs_dict(obs)
        me = int(obs_d.get("player", 0))
        step = int(obs_d.get("step", 0))
        raw_planets = obs_d.get("planets", []) or []
        if not raw_planets:
            return []
        planets = [Planet(*p) for p in raw_planets]
        my_planets = [p for p in planets if int(p.owner) == me]
        opps = [p for p in planets if int(p.owner) != me and int(p.owner) >= 0]
        my_total = sum(int(p.ships) for p in my_planets)
        opp_total = sum(int(p.ships) for p in opps)

        emit_log = (start_step <= step <= end_step) and (step % every == 0)
        # Always log if it looks like a stuck phase: opp planets present
        # AND no fleets to opp territory on our side. Detected later.

        if not my_planets or not opps:
            if emit_log:
                log.append(f"[step {step:>3}] my_p={len(my_planets)} opp_p={len(opps)} "
                           f"my_ships={my_total} opp_ships={opp_total}  → game-ending or 1-sided")
            return []

        world = World.from_obs(obs_d)
        try:
            from lib.kinematic_table import begin_turn as _kt
            _kt(world)
        except Exception:
            pass
        model = WorldModel.from_world(world)
        omega = float(obs_d.get("angular_velocity", 0.0) or 0.0)
        comet_ids = set(int(c) for c in (obs_d.get("comet_planet_ids", []) or []))

        candidates = enumerate_candidates(world, model, me, omega, comet_ids)
        budgets = {int(p.id): int(p.ships) for p in my_planets}
        prods = {int(p.id): int(p.production) for p in my_planets}
        picked = solve_dual(candidates, budgets, prods)

        # Are we dominant?
        dom = len(my_planets) >= 3 * max(1, len(opps))

        # Decide whether to log
        log_this = emit_log
        if dump_when_no_emit and not picked and opps:
            log_this = True

        if log_this:
            solo = [c for c in candidates if not c.is_partial]
            partial = [c for c in candidates if c.is_partial]
            max_val = max((c.value for c in candidates), default=0.0)
            line = (
                f"[step {step:>3}] my_p={len(my_planets)} opp_p={len(opps)} "
                f"my_ships={my_total} opp_ships={opp_total} dom={int(dom)}  "
                f"cands={len(candidates)}(solo={len(solo)},part={len(partial)},maxV={max_val:.0f})  "
                f"picks={len(picked)}({sum(int(c.ships) for c in picked)}ships)"
            )
            log.append(line)
            if not picked and opps:
                # Diagnostic: why didn't anything fire?
                why = _why_no_capture(world, model, me, omega, comet_ids)
                log.append(f"          NO-EMIT-REASON: {why}")

        ret = [[int(c.src_id), float(c.angle), int(c.ships)] for c in picked]
        return ret

    def _opp_random(obs, configuration=None):
        # match the bundled `random` agent behaviour: random per-planet emits.
        import random as _rnd
        obs_d = _obs_dict(obs)
        me = int(obs_d.get("player", 0))
        raw = obs_d.get("planets", []) or []
        planets = [Planet(*p) for p in raw]
        my_planets = [p for p in planets if int(p.owner) == me]
        moves = []
        for p in my_planets:
            if int(p.ships) <= 0:
                continue
            if _rnd.random() < 0.05:
                ships = max(1, int(p.ships * _rnd.random()))
                ang = _rnd.uniform(-math.pi, math.pi)
                moves.append([int(p.id), float(ang), int(ships)])
        return moves

    # Use the bundled "random" agent (matches gate).
    if focal_seat == 0:
        env.run([_focal_agent, "random"])
    else:
        env.run(["random", _focal_agent])

    # Final state
    final = env.steps[-1]
    obs_final = final[0]["observation"]
    planets = obs_final.get("planets", []) if isinstance(obs_final, dict) \
        else getattr(obs_final, "planets", [])
    n_p0 = sum(1 for p in planets if int(p[1]) == 0)
    n_p1 = sum(1 for p in planets if int(p[1]) == 1)
    print(f"\n== seed={seed} focal_seat={focal_seat}  final: P0={n_p0}p, P1={n_p1}p, steps={len(env.steps)}  focal_calls={call_counter[0]} ==")
    for line in log:
        print(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--focal-seat", type=int, default=0, choices=[0, 1])
    ap.add_argument("--start-step", type=int, default=0)
    ap.add_argument("--end-step", type=int, default=500)
    ap.add_argument("--every", type=int, default=20, help="log every Nth step in the window")
    ap.add_argument("--no-emit-only", action="store_true",
                    help="ONLY log turns where the chooser emitted nothing despite opp_planets>0")
    args = ap.parse_args()

    if args.no_emit_only:
        args.start_step = 0
        args.end_step = 500
        args.every = 10**9  # disable periodic logging
    trace_game(
        args.seed, args.focal_seat,
        args.start_step, args.end_step, max(1, args.every),
        dump_when_no_emit=True,
    )


if __name__ == "__main__":
    main()
