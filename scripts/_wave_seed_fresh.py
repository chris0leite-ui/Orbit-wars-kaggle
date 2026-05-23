"""Clean fresh-process A/B for seed 3233 only — both variants in separate
subprocess invocations to rule out reload-contamination."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

RUNNER = textwrap.dedent("""
    import os, sys, time
    sys.path.insert(0, {repo!r})
    wave_on = {wave_on!r}
    if wave_on:
        os.environ["BASELINE_CONVERGENCE_WAVE"] = "1"
    import importlib.util
    from kaggle_environments import make
    import agents.baseline.main as bm
    bm.CONVERGENCE_WAVE_ENABLED = bool(wave_on)
    p = "submissions/baseline_full.py"
    spec = importlib.util.spec_from_file_location("_opp", p)
    m = importlib.util.module_from_spec(spec); sys.modules["_opp"] = m
    spec.loader.exec_module(m); opp = m.agent
    env = make("orbit_wars", configuration={{"seed": {seed}}}, debug=False)
    t = time.time()
    env.run([bm.agent, opp])
    final = env.steps[-1]
    p0_win = final[0].reward > final[1].reward
    print(f"FRESH wave_on={{wave_on}} seed={{ {seed} }} -> "
          f"{{'W' if p0_win else 'L'}}  ({{time.time()-t:.1f}}s, "
          f"{{len(env.steps)}} steps)", flush=True)
""")


def run(seed, wave_on):
    code = RUNNER.format(repo=str(REPO), wave_on=wave_on, seed=seed)
    out = subprocess.run([sys.executable, "-u", "-c", code],
                         capture_output=True, text=True)
    for line in out.stdout.splitlines():
        if line.startswith("FRESH"):
            print(line, flush=True)
    if out.returncode != 0:
        print("STDERR:", out.stderr[-500:], flush=True)


if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 3233
    print(f"\n=== FRESH-PROCESS A/B for seed {seed} ===\n", flush=True)
    run(seed, False)
    run(seed, True)
