#!/usr/bin/env python3
"""Panel / margin / compute-scaling evaluation harness for Orbit Wars agents.

Why this exists: this session's misleading conclusions all came from a 1-bit
metric (win/loss) against a single opponent, plus an in-process knob env-leak.
This harness fixes both:
  * MARGIN, not just win/loss — Σ(my ships+prod) − Σ(opp ships+prod) from the
    final observation (low-variance signal the win-bit throws away).
  * one DISTINCT map-seed per game, agent at P0 (seat is map-determined and
    ~irrelevant here; never condition strength on seat — it confounds with map).
  * FRESH SUBPROCESS per config — agent env-knobs are read at module load, so
    configs must not share a process (the leak that bit us).

Two modes:
  worker  : run ONE config (knobs already in os.environ) over the seeds, print
            a single JSON result line.
  sweep   : driver — for each config, subprocess this script in worker mode with
            the config's env overlaid, collect, tabulate (the compute-scaling
            curve / panel table).

Agents are loaded from a file path via importlib (module registered in
sys.modules so dataclasses resolve); knobs are env vars set before load.
"""
from __future__ import annotations
import argparse, importlib.util, inspect, json, math, os, subprocess, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
AGENTS = {
    "lr": REPO / "agents/least_resistance/main.py",
    "v2": REPO / "audit/external/agents/slawekbiel_the-producer-v2/main.py",
    "producer": REPO / "agents/producer_plus/main.py",
}


def _load_agent(path: str, modname: str):
    spec = importlib.util.spec_from_file_location(modname, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod          # register BEFORE exec so dataclasses resolve
    spec.loader.exec_module(mod)
    fn = mod.agent
    n = len(inspect.signature(fn).parameters)
    return (lambda o, c=None: fn(o, c)) if n >= 2 else (lambda o, c=None: fn(o))


def _margin(obs, my_id: int) -> float:
    """Σ(my ships+prod) − Σ(opp ships+prod) from a final observation."""
    planets = obs["planets"] if isinstance(obs, dict) else obs.planets
    fleets = (obs.get("fleets", []) if isinstance(obs, dict) else getattr(obs, "fleets", [])) or []
    mine = opp = 0.0
    for p in planets:
        o = int(p[1])
        if o == my_id:
            mine += float(p[5]) + float(p[6])
        elif o >= 0:
            opp += float(p[5]) + float(p[6])
    for f in fleets:
        o = int(f[1])
        if o == my_id:
            mine += float(f[6])
        elif o >= 0:
            opp += float(f[6])
    return mine - opp


def _wilson_lo(wins: int, n: int, z: float = 1.96) -> float:
    if n == 0:
        return 0.0
    p = wins / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, (c - m) / d)


def run_worker(focal_path: str, opp_path: str, seeds: list[int], players: int) -> dict:
    from kaggle_environments import make
    sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "agents" / "producer"))
    focal = _load_agent(focal_path, "focal_mod")
    opp = _load_agent(opp_path, "opp_mod")
    wins = 0; margins = []; times = []; det = []
    for seed in seeds:
        turns = []
        def timed(o, c=None):
            t = time.perf_counter(); r = focal(o, c); turns.append((time.perf_counter() - t) * 1000); return r
        agents = [timed] + [opp] * (players - 1)
        env = make("orbit_wars", configuration={"seed": seed}, debug=False)
        env.run(agents)
        last = env.steps[-1]
        rew = [s.reward for s in last]
        obs0 = last[0]["observation"] if isinstance(last[0], dict) else last[0].observation
        m = _margin(obs0, 0)
        won = all(rew[0] > rew[i] for i in range(1, players))
        wins += int(won); margins.append(m); det.append("W" if won else ("D" if rew[0] == max(rew) else "L"))
        if turns: times.append(max(turns))
    n = len(seeds)
    return {
        "n": n, "players": players, "wins": wins, "win_rate": wins / n if n else 0.0,
        "wilson_lo": _wilson_lo(wins, n), "margin_mean": sum(margins) / n if n else 0.0,
        "margin_min": min(margins) if margins else 0.0, "margin_max": max(margins) if margins else 0.0,
        "max_turn_ms": max(times) if times else 0.0, "detail": "".join(det),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--focal", default="lr")
    ap.add_argument("--opp", default="v2")
    ap.add_argument("--seeds", default="5000-5023")
    ap.add_argument("--players", type=int, default=2)
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    a, b = args.seeds.split("-") if "-" in args.seeds else (args.seeds, args.seeds)
    seeds = list(range(int(a), int(b) + 1))
    focal_path = str(AGENTS.get(args.focal, args.focal))
    opp_path = str(AGENTS.get(args.opp, args.opp))

    if args.worker:
        print("RESULT " + json.dumps(run_worker(focal_path, opp_path, seeds, args.players)))
        return

    # sweep driver: configs = (label, env-overrides). Edit here for experiments.
    configs = [
        ("lr_depth0(2ply)", {"LR_ROLLOUT_DEPTH": "0"}),
        ("lr_depth2       ", {"LR_ROLLOUT_DEPTH": "2"}),
        ("lr_depth3       ", {"LR_ROLLOUT_DEPTH": "3"}),
        ("lr_depth2_wide48", {"LR_ROLLOUT_DEPTH": "2", "LR_MAX_CANDIDATES": "48"}),
    ]
    cfg_env = os.environ.get("EVAL_CONFIGS")
    if cfg_env:
        configs = json.loads(cfg_env)  # [[label, {env}], ...]
    print(f"# panel sweep: focal={args.focal} opp={args.opp} seeds={args.seeds} players={args.players}")
    for label, overrides in configs:
        env = {**os.environ, "PYTHONPATH": f"{REPO}:{REPO}/agents/producer"}
        env.update({k: str(v) for k, v in overrides.items()})
        out = subprocess.run(
            [sys.executable, __file__, "--worker", "--focal", args.focal,
             "--opp", args.opp, "--seeds", args.seeds, "--players", str(args.players)],
            capture_output=True, text=True, cwd=str(REPO), env=env,
        )
        line = next((l for l in out.stdout.splitlines() if l.startswith("RESULT ")), None)
        if not line:
            print(f"{label}: ERR {out.stderr.strip()[-200:]}"); continue
        r = json.loads(line[len("RESULT "):])
        print(f"{label}: {r['wins']}/{r['n']} (wlo {r['wilson_lo']:.2f}) "
              f"margin {r['margin_mean']:+.0f} [{r['margin_min']:+.0f},{r['margin_max']:+.0f}] "
              f"maxms {r['max_turn_ms']:.0f}  {r['detail']}")
    print("# DONE")


if __name__ == "__main__":
    main()
