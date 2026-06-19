"""One-game worker for the continuous-score A/B harness.

Plays a SINGLE Orbit Wars game in a FRESH process (one variant per process —
honours the DROPOUT_PLAN env-leak rule: producer_plus bundles set knobs via
``os.environ.setdefault`` which leak across variants in one process; here each
subprocess sets exactly one variant's knobs as REAL env vars before import).

Emits one JSON line on stdout with both the binary win AND the continuous
ship-margin outcome. The continuous score is the natural relaxation of the
engine's own win rule: the interpreter declares the winner by
``argmax_i scores[i]`` where ``scores[i]`` = total ships across player i's
planets + fleets at game end. The signed, scale-normalised margin

    margin = (focal_ships - best_rival_ships) / (focal_ships + best_rival_ships)

lies in [-1, 1]; its SIGN reproduces the win/loss exactly, and its MAGNITUDE
distinguishes "barely won" (0.01) from "dominated" (0.95) — a far lower-variance
per-game signal than the binary outcome.

Usage (invoked by scripts/continuous_ab.py, not by hand):
    python scripts/_continuous_game_worker.py \
        --seed 5000 --focal-seat 0 --players 2 \
        --focal agents/producer_plus/main.py \
        --opps audit/external/agents/slawekbiel_the-producer-v2/main.py \
        --knobs '{"PRODUCER_PLUS_DROPOUT": "1", ...}'
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _load_agent(path: str):
    spec = importlib.util.spec_from_file_location(
        "_cg_%d" % abs(hash(path)), path
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    fn = getattr(mod, "agent", None) or getattr(mod, "act", None)
    if fn is None:
        raise AttributeError("%s has no top-level agent/act callable" % path)
    return fn


def _arity_wrap(fn):
    argc = fn.__code__.co_argcount if hasattr(fn, "__code__") else 2
    if argc == 1:
        return lambda o, c=None: fn(o)
    return lambda o, c=None: fn(o, c)


def _player_ship_scores(observation, players: int) -> list[float]:
    """Replicate the interpreter's end-state scoring: total ships across each
    player's planets + fleets. Index conventions match orbit_wars.py
    (planet owner=1 ships=5; fleet owner=1 ships=6)."""
    scores = [0.0] * players
    planets = observation["planets"] if isinstance(observation, dict) else observation.planets
    fleets = observation["fleets"] if isinstance(observation, dict) else observation.fleets
    for p in planets:
        owner = int(p[1])
        if 0 <= owner < players:
            scores[owner] += float(p[5])
    for f in (fleets or []):
        owner = int(f[1])
        if 0 <= owner < players:
            scores[owner] += float(f[6])
    return scores


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--focal-seat", type=int, required=True)
    ap.add_argument("--players", type=int, default=2)
    ap.add_argument("--focal", required=True)
    ap.add_argument("--opps", required=True,
                    help="comma-separated opponent main.py paths (players-1)")
    ap.add_argument("--knobs", default="{}",
                    help="JSON dict of env vars to set for the focal agent")
    args = ap.parse_args()

    # Set the focal variant's knobs as REAL env vars BEFORE importing the focal
    # agent. (producer_plus reads every knob per-call via os.environ.get, so the
    # timing is not strictly load-bearing, but setting them first is the safe,
    # leak-free contract.)
    knobs = json.loads(args.knobs)
    for k, v in knobs.items():
        os.environ[str(k)] = str(v)

    # Both producer_plus and Producer V2 import the `orbit_lite` package that
    # lives under agents/producer/. Put it (and the repo root) on the path.
    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(REPO / "agents" / "producer"))

    # Single-threaded torch so N workers can run in parallel on N cores without
    # BLAS thread oversubscription. (The driver also exports OMP/MKL=1.)
    try:
        import torch as _torch
        _torch.set_num_threads(1)
    except Exception:
        pass

    from kaggle_environments import make  # late import (slow)

    players = int(args.players)
    seat = int(args.focal_seat)
    opp_paths = [p for p in args.opps.split(",") if p]

    try:
        focal_fn = _load_agent(args.focal)
        opp_fns = [_load_agent(p) for p in opp_paths]
    except Exception as e:  # load failure
        print(json.dumps({"seed": args.seed, "seat": seat, "error":
                          "load: " + repr(e)[:120]}))
        return 0

    times: list[float] = []

    def timed_focal(o, c=None):
        t = time.perf_counter()
        try:
            return focal_fn(o)
        finally:
            times.append((time.perf_counter() - t) * 1000.0)

    seats = [None] * players
    seats[seat] = timed_focal
    j = 0
    for i in range(players):
        if i == seat:
            continue
        seats[i] = _arity_wrap(opp_fns[j])
        j += 1

    try:
        env = make("orbit_wars", configuration={"seed": args.seed}, debug=False)
        env.run(seats)
    except Exception as e:
        print(json.dumps({"seed": args.seed, "seat": seat, "error":
                          "run: " + repr(e)[:120]}))
        return 0

    final = env.steps[-1]
    obs0 = final[0].observation
    scores = _player_ship_scores(obs0, players)
    focal_ships = scores[seat]
    rival_ships = max((scores[i] for i in range(players) if i != seat),
                      default=0.0)
    total = focal_ships + rival_ships
    margin = (focal_ships - rival_ships) / total if total > 0 else 0.0
    max_all = max(scores) if scores else 0.0
    win = bool(focal_ships == max_all and max_all > 0)

    # engine reward cross-check + idle guard (a game where a side never launched
    # is degenerate; verify_confirm filters these).
    rewards = [s.reward for s in final]
    # Our re-summed `win` must agree with the engine reward (reward==1 <=> win).
    # If this ever fires, our score replication has drifted from the engine and
    # the continuous margin is no longer a faithful relaxation of the outcome.
    if rewards[seat] is not None:
        assert win == (rewards[seat] == 1), (
            "win/reward mismatch: win=%s reward=%s scores=%s"
            % (win, rewards[seat], scores))
    focal_launches = sum(len(st[seat].action) for st in env.steps if st[seat].action)
    opp_launches = sum(len(st[i].action or []) for st in env.steps
                       for i in range(players) if i != seat)

    print(json.dumps({
        "seed": args.seed,
        "seat": seat,
        "players": players,
        "win": win,
        "margin": round(margin, 6),
        "focal_ships": round(focal_ships, 2),
        "rival_ships": round(rival_ships, 2),
        "scores": [round(s, 2) for s in scores],
        "steps": len(env.steps),
        "max_ms": round(max(times), 1) if times else 0.0,
        "focal_launches": focal_launches,
        "opp_launches": opp_launches,
        "reward": rewards[seat],
        "idle": bool(focal_launches == 0 or opp_launches == 0),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
