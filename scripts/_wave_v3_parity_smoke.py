"""Bit-parity smoke for V3 wave changes (gate off).

With BASELINE_CONVERGENCE_WAVE unset, the baseline agent must produce
byte-identical moves vs HEAD. This is the Rule 38 fix-verification
substrate: the wave changes must NOT alter the default-OFF behavior.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

# Wave OFF, instrumentation OFF, leaf-validate OFF.
os.environ.pop("BASELINE_CONVERGENCE_WAVE", None)
os.environ.pop("BASELINE_WAVE_INSTRUMENT", None)
os.environ.pop("BASELINE_WAVE_LEAF_VALIDATE", None)

import agents.baseline.main as bm
bm.CONVERGENCE_WAVE_ENABLED = False
bm.WAVE_INSTRUMENT = False
bm.WAVE_LEAF_VALIDATE = False

from kaggle_environments import make
import importlib.util


def load(path):
    p = Path(path)
    spec = importlib.util.spec_from_file_location(f"_m_{p.stem}", path)
    m = importlib.util.module_from_spec(spec); sys.modules[spec.name] = m
    spec.loader.exec_module(m); return m.agent


if __name__ == "__main__":
    opp = load("submissions/baseline_full.py")
    seeds = [int(s) for s in sys.argv[1:]] or [42]
    for seed in seeds:
        env = make("orbit_wars", configuration={"seed": seed}, debug=False)
        env.run([bm.agent, opp])
        final = env.steps[-1]
        r0, r1 = final[0].reward, final[1].reward
        outcome = "P0" if r0 > r1 else "P1" if r1 > r0 else "DRAW"
        n_steps = len(env.steps)
        print(f"seed={seed} outcome={outcome} steps={n_steps} "
              f"telemetry_records={len(bm._WAVE_TELEMETRY)} "
              f"ownership_records={len(bm._OWNERSHIP_TRACE)}",
              flush=True)
