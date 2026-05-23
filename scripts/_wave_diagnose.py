"""Run 8 panel seeds with wave-OFF and wave-ON; report per-seed delta to
identify regression cases for tracing."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import importlib.util
from kaggle_environments import make

SEEDS = [5199, 2083, 3493, 1649, 3233, 405, 3335, 1030]


def load(path):
    p = Path(path)
    spec = importlib.util.spec_from_file_location(f"_m_{p.stem}_{id(p)}", path)
    m = importlib.util.module_from_spec(spec); sys.modules[spec.name] = m
    spec.loader.exec_module(m); return m.agent


def play(seed, wave_on, opp):
    if wave_on:
        os.environ["BASELINE_CONVERGENCE_WAVE"] = "1"
    else:
        os.environ.pop("BASELINE_CONVERGENCE_WAVE", None)
    # Re-import baseline to pick up the env var.
    for k in list(sys.modules):
        if k.startswith("agents.baseline"):
            del sys.modules[k]
    import agents.baseline.main as bm
    bm.CONVERGENCE_WAVE_ENABLED = bool(wave_on)
    env = make("orbit_wars", configuration={"seed": seed}, debug=False)
    t = time.time()
    env.run([bm.agent, opp])
    final = env.steps[-1]
    return (final[0].reward > final[1].reward, time.time() - t, len(env.steps))


if __name__ == "__main__":
    opp_path = "submissions/baseline_full.py"
    opp = load(opp_path)
    print(f"\n=== per-seed wave-OFF vs wave-ON (P0=focal vs baseline_full) ===\n")
    print(f"  {'seed':>5} | {'wave-OFF':>10} | {'wave-ON':>10} | delta")
    print(f"  " + "-" * 60)
    n_off_win = n_on_win = 0
    regressions = []
    gains = []
    for s in SEEDS:
        off_win, off_t, off_n = play(s, False, opp)
        on_win, on_t, on_n = play(s, True, opp)
        n_off_win += int(off_win); n_on_win += int(on_win)
        delta = ""
        if off_win and not on_win:
            delta = "REGRESSION (wave-ON loses)"
            regressions.append(s)
        elif on_win and not off_win:
            delta = "GAIN (wave flipped to win)"
            gains.append(s)
        elif off_win and on_win:
            delta = "both win"
        else:
            delta = "both lose"
        print(f"  {s:>5} | {'W' if off_win else 'L':>10} | "
              f"{'W' if on_win else 'L':>10} | {delta}")
    print()
    print(f"  totals: wave-OFF {n_off_win}/8  wave-ON {n_on_win}/8")
    print(f"  regression seeds (wave hurts): {regressions}")
    print(f"  gain seeds       (wave helps): {gains}")
