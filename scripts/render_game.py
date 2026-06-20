"""render_game.py — run one 4P (or 2P) Orbit Wars game with our agent
(least_resistance) in a chosen seat vs the public panel, print a per-turn trace
of our position (to spot the drain/bounce/lose-largest blunder), and optionally
render the game to a watchable HTML replay.

This is the PI's replay-watching loop (NOT A/B). Env vars passed through the
process configure the agent (e.g. LR_ROBUST=1 for the live 4P probe, or our new
exposure knob). Example:

    LR_ROBUST=1 LR_ROBUST_SAMPLES=8 LR_ROBUST_MS=500 \
      python scripts/render_game.py --seed 1912745358 --seat 0 \
      --out /tmp/g.html --label before

    # seat sweep, trace only (fast), to find the seat that reproduces a blunder
    python scripts/render_game.py --seed 1912745358 --sweep --no-html
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time
from pathlib import Path

# Pin threads so ad-hoc renders match the single-threaded eval workers. The
# 2026-06-19 postmortem caught a multi-threaded render diverging from the
# canonical result (float reductions reorder) -- a rendered "loss" was actually a
# WIN, misleading a diagnosis. Set BEFORE torch is imported by any agent.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
try:
    import torch as _torch
    _torch.set_num_threads(1)
except Exception:
    pass

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "agents" / "producer"))

EXT = REPO / "audit" / "external" / "agents"
LR = str(REPO / "agents" / "least_resistance" / "main.py")
PANEL = {
    "V2": str(EXT / "slawekbiel_the-producer-v2" / "main.py"),
    "Roman": str(EXT / "romantamrazov_orbit-star-wars-lb-max-1224" / "main.py"),
    "konbu": str(EXT / "konbu17_orbit-wars-rule-base-ml-shot-validator-hybrid" / "main.py"),
}


def _load(path):
    spec = importlib.util.spec_from_file_location("_a_%d" % abs(hash(path)), path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return getattr(m, "agent", None) or getattr(m, "act")


def _arity(fn):
    argc = fn.__code__.co_argcount if hasattr(fn, "__code__") else 2
    return (lambda o: fn(o)) if argc == 1 else (lambda o, c=None: fn(o, c))


def _timed(fn, sink):
    """Wrap an arity-normalized agent to record per-turn wallclock ms."""
    def w(o, c=None):
        t = time.perf_counter()
        try:
            return fn(o, c)
        finally:
            sink.append((time.perf_counter() - t) * 1000.0)
    return w


def _prod_split(step_obs, me):
    """Production owned by me / opponents / neutral at this step.
    planet tuple = (id, owner, x, y, ships, production, ...)."""
    planets = step_obs.get("planets", []) or []
    mine = opp = neu = 0.0
    for p in planets:
        owner, prod = int(p[1]), float(p[5])
        if owner == me:
            mine += prod
        elif owner == -1:
            neu += prod
        else:
            opp += prod
    return mine, opp, neu


def _my_inflight(step_obs, me):
    """Count fleets I currently have in flight (scatter indicator).
    fleet tuple = (id, owner, x, y, ships, ...)."""
    return sum(1 for f in (step_obs.get("fleets", []) or []) if int(f[1]) == me)


def _focal_state(step_obs, me):
    planets = step_obs.get("planets", []) or []
    mine = [p for p in planets if int(p[1]) == me]
    if not mine:
        return 0, 0.0, None, 0.0, set()
    total = sum(float(p[4]) for p in mine)
    big = max(mine, key=lambda p: float(p[5]))   # largest by PRODUCTION
    owned = {int(p[0]) for p in mine}
    return len(mine), total, int(big[0]), float(big[5]), owned


def play(seed, seat, players, render_html):
    from kaggle_environments import make

    opp_names = list(PANEL.keys())[: players - 1]
    seats = [None] * players
    focal_ms: list[float] = []
    focal_fn = _load(LR)
    seats[seat] = _timed(_arity(focal_fn), focal_ms)
    j = 0
    for i in range(players):
        if i == seat:
            continue
        seats[i] = _arity(_load(PANEL[opp_names[j]]))
        j += 1

    env = make("orbit_wars", configuration={"seed": seed}, debug=False)
    t0 = time.perf_counter()
    env.run(seats)
    elapsed = time.perf_counter() - t0

    rewards = [s.reward for s in env.steps[-1]]
    valid = [r for r in rewards if r is not None]
    won = rewards[seat] is not None and bool(valid) and rewards[seat] == max(valid)

    # Production-share trajectory (PI metric) + scatter (peak in-flight fleets).
    nsteps = len(env.steps)
    marks = {int(nsteps * f): None for f in (0.25, 0.5, 0.75, 0.999)}
    share_str = []
    peak_inflight = 0
    total_launches = 0
    trace = []
    for t, st in enumerate(env.steps):
        ob = st[seat].observation
        mine, opp, neu = _prod_split(ob, seat)
        inflight = _my_inflight(ob, seat)
        peak_inflight = max(peak_inflight, inflight)
        act = st[seat].action
        total_launches += len(act) if act else 0
        tot = mine + opp + neu
        trace.append((t, mine, opp, neu, inflight))
        if t in marks:
            sh = (100.0 * mine / tot) if tot else 0.0
            share_str.append("s%d:%.0f%%(%.0f/%.0f)" % (t, sh, mine, opp))

    layout = "%dP" % players
    opps = ",".join(opp_names)
    fm, fo, fn = _prod_split(env.steps[-1][seat].observation, seat)
    print("seed=%-11d seat=%d %s vs[%s]  %s  steps=%d  %.1fs  rewards=%s"
          % (seed, seat, layout, opps, "WIN " if won else "loss",
             nsteps, elapsed, rewards))
    print("    prod-share over game: %s  | final prod mine=%.0f opp=%.0f"
          % ("  ".join(share_str), fm, fo))
    print("    scatter: peak in-flight fleets=%d  total launches=%d"
          % (peak_inflight, total_launches))
    nat = focal_fn.__globals__.get("_NATIVE_LEAF_CALLS")
    if nat is not None:
        print("    native_leaf_calls=%d  (proof the native leaf executed in-game)" % nat)
    if focal_ms:
        sm = sorted(focal_ms)
        p95 = sm[min(len(sm) - 1, int(0.95 * len(sm)))]
        over = sum(1 for x in focal_ms if x >= 1000.0)
        print("    turn-ms: p50=%.0f p95=%.0f max=%.0f  over_1000ms=%d"
              % (sm[len(sm) // 2], p95, sm[-1], over))
    blunders = []

    out = None
    if render_html:
        html = env.render(mode="html")
        out = render_html
        Path(out).write_text(html)
        print("    rendered -> %s  (%d bytes)" % (out, len(html)))
    return won, blunders, trace, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--seat", type=int, default=0)
    ap.add_argument("--players", type=int, default=4)
    ap.add_argument("--sweep", action="store_true", help="play all seats 0..players-1")
    ap.add_argument("--out", default=None, help="html output path (single seat)")
    ap.add_argument("--no-html", action="store_true", help="trace only, skip render")
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    if args.label:
        print("== %s ==" % args.label)
    seats = range(args.players) if args.sweep else [args.seat]
    for seat in seats:
        html = None if (args.no_html or args.sweep) else args.out
        play(args.seed, seat, args.players, html)


if __name__ == "__main__":
    raise SystemExit(main())
