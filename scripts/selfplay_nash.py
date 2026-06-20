#!/usr/bin/env python3
"""Self-play draw analysis + single-player (Nash) deviation probe for Orbit Wars.

The question this answers (PI): a least_resistance SELF-PLAY game (same agent in
every seat) ends in a DRAW. Fix every other seat at the default policy and let
ONE seat deviate -- is there a deviation that lets it WIN, or is the draw a Nash
equilibrium (no profitable unilateral deviation)?

Outcome is read on TWO axes because the env reward is +1 for ANY seat tying the
max final ship-count and -1 otherwise (a symmetric draw gives every tied seat
+1, un-improvable in raw payoff):
  * placement -- does the focal become the SOLE max (a solo win) vs tied vs lose?
  * margin    -- focal final (ships+production) minus the best OTHER seat.

Determinism: least_resistance has no RNG; time only truncates the candidate
list. We set ORBIT_WARS_PARITY_WALLCLOCK_MS huge so the per-turn time-bail never
fires -> every game is exactly reproducible and the deviation test is clean.

The knob-leak trap (same os.environ leak that bit us, now WITHIN one game):
  * import-time knobs (module globals): LR_ROI_FLOOR / LR_TWOPLY / LR_MAX_CANDIDATES
    / LR_HORIZON_* / LR_FRONTIER_REF_SHIPS -- captured at import, so we import the
    focal module under deviant env and the default module under default env.
  * call-time knobs (read live from os.environ every turn, hence SHARED by all
    seats in one process): LR_ENEMY_BOOST / LR_HOLD_MARGIN / LR_DEFEND_RANGE (via
    the module's _f) and _rollout_depth / _leader_relative_4p / _deep_opp (funcs).
    A call-time deviation is applied by MONKEYPATCHING the focal module object
    only; os.environ stays at defaults so the fixed seats are truly default.

Modes:
  selfplay : all seats = default agent; print per-seed JSON (rewards, per-seat
             scores, winners, nsteps).
  deviate  : one focal seat uses a deviation, the rest default; print focal
             placement + per-seat scores + winners.
  driver   : (default) scan self-play seeds for draws, then run the deviation
             family on each draw seed x focal seat, sequentially (one job at a
             time), and tabulate.
"""
from __future__ import annotations
import argparse, importlib.util, inspect, json, os, subprocess, sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LR_PATH = REPO / "agents/least_resistance/main.py"

# Default values of every knob we ever deviate -- restored in os.environ before
# importing the default module so the fixed seats are unambiguously default.
KNOB_DEFAULTS = {
    "LR_ROI_FLOOR": "1.5", "LR_TWOPLY": "1", "LR_MAX_CANDIDATES": "28",
    "LR_HORIZON_2P": "18", "LR_HORIZON_4P": "13", "LR_FRONTIER_REF_SHIPS": "30.0",
    "LR_ENEMY_BOOST": "1.0", "LR_HOLD_MARGIN": "0.5", "LR_DEFEND": "1",
    "LR_DEFEND_RANGE": "35.0", "LR_ROLLOUT_DEPTH": "0", "LR_DEEP_OPP": "0",
    "LR_LEADER_RELATIVE_4P": "0", "LR_SKIP_COMETS": "0", "LR_ITERDEEPEN": "0",
}

# The deviation family. Each: import_env (set before importing the focal module),
# f_over (focal-only _f key overrides), func_over (focal-only function overrides).
DEVIATIONS = {
    "control(default)":   {},
    "aggressive(roi0.3)": {"import_env": {"LR_ROI_FLOOR": "0.3"}},
    "conservative(roi4)": {"import_env": {"LR_ROI_FLOOR": "4.0"}},
    "denial(boost2.5)":   {"f_over": {"LR_ENEMY_BOOST": 2.5}},        # 4P only
    "leader_relative":    {"func_over": {"_leader_relative_4p": True}},  # 4P only
    "expansion(boost0.1)":{"f_over": {"LR_ENEMY_BOOST": 0.1, "LR_HOLD_MARGIN": 0.0}},
    "fortress(hold1)":    {"f_over": {"LR_HOLD_MARGIN": 1.0, "LR_DEFEND_RANGE": 100.0}},
    "deep(depth3)":       {"func_over": {"_rollout_depth": 3}},
    "no_lookahead":       {"import_env": {"LR_TWOPLY": "0"}},
}


def _setup_path():
    for p in (str(REPO), str(REPO / "agents" / "producer")):
        if p not in sys.path:
            sys.path.insert(0, p)


def _import_lr(modname: str):
    """Import least_resistance/main.py as a fresh module object under modname,
    capturing whatever import-time knobs are currently in os.environ."""
    spec = importlib.util.spec_from_file_location(modname, str(LR_PATH))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod


def _as_callable(mod):
    fn = mod.agent
    n = len(inspect.signature(fn).parameters)
    return (lambda o, c=None: fn(o, c)) if n >= 2 else (lambda o, c=None: fn(o))


def _patch_focal(mod, f_over: dict, func_over: dict):
    """Apply focal-ONLY call-time deviations by monkeypatching the module object.
    Only this module's agent() sees them; other seats use their own module."""
    if f_over:
        orig_f = mod._f
        ov = {k: float(v) for k, v in f_over.items()}
        mod._f = (lambda name, default, _o=ov, _orig=orig_f:
                  _o[name] if name in _o else _orig(name, default))
    for fname, val in (func_over or {}).items():
        setattr(mod, fname, (lambda _v=val: _v))


def _seat_scores(obs, num_seats: int):
    """Per-seat final strength: ships+production on owned planets + ships in
    owned fleets. planets/fleets are global (owner id per row) so one obs gives
    every seat's total."""
    planets = obs["planets"] if isinstance(obs, dict) else obs.planets
    fleets = (obs.get("fleets", []) if isinstance(obs, dict) else getattr(obs, "fleets", [])) or []
    score = [0.0] * num_seats
    for p in planets:
        o = int(p[1])
        if 0 <= o < num_seats:
            score[o] += float(p[5]) + float(p[6])
    for f in fleets:
        o = int(f[1])
        if 0 <= o < num_seats:
            score[o] += float(f[6])
    return score


def _winners(rewards):
    mx = max(r for r in rewards if r is not None)
    return [i for i, r in enumerate(rewards) if r == mx]


def _run_game(seat_agents, seed: int):
    from kaggle_environments import make
    env = make("orbit_wars", configuration={"seed": seed}, debug=False)
    env.run(seat_agents)
    last = env.steps[-1]
    rewards = [s.reward for s in last]
    obs0 = last[0]["observation"] if isinstance(last[0], dict) else last[0].observation
    scores = _seat_scores(obs0, len(seat_agents))
    return rewards, scores, len(env.steps)


def run_selfplay(players: int, seeds: list[int]) -> list[dict]:
    _setup_path()
    for k, v in KNOB_DEFAULTS.items():
        os.environ[k] = v
    default = _as_callable(_import_lr("lr_default"))
    out = []
    for seed in seeds:
        rewards, scores, nsteps = _run_game([default] * players, seed)
        winners = _winners(rewards)
        out.append({"seed": seed, "players": players, "rewards": rewards,
                    "scores": [round(s, 1) for s in scores], "winners": winners,
                    "draw": len(winners) >= 2, "nsteps": nsteps})
    return out


def run_deviate(players: int, focal_seat: int, seed: int, dev_name: str) -> dict:
    _setup_path()
    dev = DEVIATIONS[dev_name]
    # 1) focal module imported under its import-time env overrides.
    for k, v in KNOB_DEFAULTS.items():
        os.environ[k] = v
    for k, v in dev.get("import_env", {}).items():
        os.environ[k] = str(v)
    focal_mod = _import_lr("lr_focal")
    _patch_focal(focal_mod, dev.get("f_over", {}), dev.get("func_over", {}))
    focal = _as_callable(focal_mod)
    # 2) default module imported under pure defaults (call-time os.environ stays
    #    default for the fixed seats).
    for k, v in KNOB_DEFAULTS.items():
        os.environ[k] = v
    default = _as_callable(_import_lr("lr_default"))
    seat_agents = [focal if i == focal_seat else default for i in range(players)]
    rewards, scores, nsteps = _run_game(seat_agents, seed)
    winners = _winners(rewards)
    fr = rewards[focal_seat]
    if fr == max(r for r in rewards if r is not None):
        placement = "SOLO" if len(winners) == 1 else "tied"
    else:
        placement = "lose"
    others = [scores[i] for i in range(players) if i != focal_seat]
    return {"seed": seed, "players": players, "focal_seat": focal_seat,
            "deviation": dev_name, "rewards": rewards,
            "scores": [round(s, 1) for s in scores], "winners": winners,
            "placement": placement, "focal_score": round(scores[focal_seat], 1),
            "best_other": round(max(others), 1) if others else 0.0,
            "margin": round(scores[focal_seat] - (max(others) if others else 0.0), 1),
            "nsteps": nsteps}


def _worker_env(extra: dict | None = None) -> dict:
    env = {**os.environ, "PYTHONPATH": f"{REPO}:{REPO}/agents/producer",
           "ORBIT_WARS_PARITY_WALLCLOCK_MS": "100000000"}
    if extra:
        env.update({k: str(v) for k, v in extra.items()})
    return env


def _sub(args: list[str]) -> dict:
    out = subprocess.run([sys.executable, __file__, *args],
                         capture_output=True, text=True, cwd=str(REPO),
                         env=_worker_env())
    line = next((l for l in out.stdout.splitlines() if l.startswith("RESULT ")), None)
    if not line:
        return {"error": out.stderr.strip()[-400:] or out.stdout.strip()[-400:]}
    return json.loads(line[len("RESULT "):])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="driver",
                    choices=["driver", "selfplay", "deviate"])
    ap.add_argument("--players", type=int, default=4)
    ap.add_argument("--seeds", default="5000-5015")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--focal-seat", type=int, default=0)
    ap.add_argument("--deviation", default="control(default)")
    ap.add_argument("--max-draws", type=int, default=3,
                    help="driver: probe at most this many drawn seeds")
    args = ap.parse_args()

    if args.mode == "selfplay":
        a, b = (args.seeds.split("-") if "-" in args.seeds else (args.seeds, args.seeds))
        seeds = list(range(int(a), int(b) + 1))
        print("RESULT " + json.dumps(run_selfplay(args.players, seeds)))
        return
    if args.mode == "deviate":
        print("RESULT " + json.dumps(
            run_deviate(args.players, args.focal_seat, args.seed, args.deviation)))
        return

    # driver: self-play scan -> draws -> deviation sweep (sequential subprocesses)
    for players in ([args.players] if args.players else [3, 4]):
        print(f"\n=== self-play scan: {players} players, seeds {args.seeds} ===")
        rows = _sub(["--mode", "selfplay", "--players", str(players),
                     "--seeds", args.seeds])
        if isinstance(rows, dict) and rows.get("error"):
            print("  ERR", rows["error"]); continue
        draws = []
        for r in rows:
            tag = "DRAW" if r["draw"] else "    "
            print(f"  seed {r['seed']}: {tag} winners={r['winners']} "
                  f"rewards={r['rewards']} scores={r['scores']} steps={r['nsteps']}")
            if r["draw"]:
                draws.append(r["seed"])
        if not draws:
            print("  (no draws found in this band)"); continue
        probe = draws[: args.max_draws]
        print(f"\n=== deviation sweep: {players}p, draw seeds {probe} ===")
        for seed in probe:
            print(f"\n  -- seed {seed} --")
            for dev in DEVIATIONS:
                for fs in range(players):
                    r = _sub(["--mode", "deviate", "--players", str(players),
                              "--seed", str(seed), "--focal-seat", str(fs),
                              "--deviation", dev])
                    if r.get("error"):
                        print(f"    {dev:<20} seat{fs}: ERR {r['error'][-160:]}")
                        continue
                    print(f"    {dev:<20} seat{fs}: {r['placement']:<5} "
                          f"margin {r['margin']:+8.1f}  winners={r['winners']} "
                          f"focal={r['focal_score']} best_other={r['best_other']}")
    print("\n# DONE")


if __name__ == "__main__":
    main()
