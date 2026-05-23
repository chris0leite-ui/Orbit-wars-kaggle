"""Small 4-seed A/B (wave-OFF vs wave-ON) with proper stdout flushing
and one-time import per variant (no per-game module reload)."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import importlib.util
from kaggle_environments import make

SEEDS = [1467, 78, 32, 1900]


def load(path):
    p = Path(path)
    spec = importlib.util.spec_from_file_location(f"_m_{p.stem}_{id(p)}", path)
    m = importlib.util.module_from_spec(spec); sys.modules[spec.name] = m
    spec.loader.exec_module(m); return m.agent


def play(seed, focal, opp):
    env = make("orbit_wars", configuration={"seed": seed}, debug=False)
    t = time.time()
    env.run([focal, opp])
    final = env.steps[-1]
    return (final[0].reward > final[1].reward, time.time() - t, len(env.steps))


if __name__ == "__main__":
    opp = load("submissions/baseline_full.py")

    # wave-OFF first
    os.environ.pop("BASELINE_CONVERGENCE_WAVE", None)
    import agents.baseline.main as bm
    bm.CONVERGENCE_WAVE_ENABLED = False
    off_agent = bm.agent

    print(f"\n=== small 4-seed A/B (P0=focal vs baseline_full) ===", flush=True)
    print(f"  seeds: {SEEDS}", flush=True)
    print(f"\n  --- wave-OFF pass ---", flush=True)
    off_results = {}
    for s in SEEDS:
        w, t, n = play(s, off_agent, opp)
        off_results[s] = (w, t, n)
        print(f"  seed {s}: {'W' if w else 'L'}  ({t:.1f}s, {n} steps)", flush=True)

    # Reload baseline with wave-ON
    os.environ["BASELINE_CONVERGENCE_WAVE"] = "1"
    for k in list(sys.modules):
        if k.startswith("agents.baseline"):
            del sys.modules[k]
    import agents.baseline.main as bm2
    bm2.CONVERGENCE_WAVE_ENABLED = True
    on_agent = bm2.agent

    print(f"\n  --- wave-ON pass ---", flush=True)
    on_results = {}
    for s in SEEDS:
        w, t, n = play(s, on_agent, opp)
        on_results[s] = (w, t, n)
        print(f"  seed {s}: {'W' if w else 'L'}  ({t:.1f}s, {n} steps)", flush=True)

    print(f"\n  === per-seed delta ===", flush=True)
    print(f"  {'seed':>5} | {'OFF':>3} | {'ON':>3} | delta", flush=True)
    print(f"  " + "-" * 50, flush=True)
    regressions = []
    gains = []
    n_off = n_on = 0
    for s in SEEDS:
        ow, *_ = off_results[s]
        nw, *_ = on_results[s]
        n_off += int(ow); n_on += int(nw)
        if ow and not nw:
            d = "REGRESSION"; regressions.append(s)
        elif nw and not ow:
            d = "GAIN"; gains.append(s)
        elif ow and nw:
            d = "both W"
        else:
            d = "both L"
        print(f"  {s:>5} | {'W' if ow else 'L':>3} | "
              f"{'W' if nw else 'L':>3} | {d}", flush=True)
    print(f"\n  totals: wave-OFF {n_off}/4   wave-ON {n_on}/4", flush=True)
    print(f"  regressions: {regressions}   gains: {gains}", flush=True)
