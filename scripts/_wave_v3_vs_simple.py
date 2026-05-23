"""V3 wave (leaf-validate ON) vs a simpler opponent — print per-fire
details so we can see what the wave actually does in a clean win.

Usage: python scripts/_wave_v3_vs_simple.py [seed]  # default seed 42
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

os.environ["BASELINE_CONVERGENCE_WAVE"] = "1"
os.environ["BASELINE_WAVE_LEAF_VALIDATE"] = "1"
os.environ["BASELINE_WAVE_INSTRUMENT"] = "1"

import importlib.util
from kaggle_environments import make
import agents.baseline.main as bm
bm.CONVERGENCE_WAVE_ENABLED = True
bm.WAVE_LEAF_VALIDATE = True
bm.WAVE_INSTRUMENT = True


def load(path: str):
    p = Path(path)
    spec = importlib.util.spec_from_file_location(f"_m_{p.stem}", path)
    m = importlib.util.module_from_spec(spec); sys.modules[spec.name] = m
    spec.loader.exec_module(m); return m.agent


if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    opp = load("agents/simple/nearest.py")

    env = make("orbit_wars", configuration={"seed": seed}, debug=False)
    env.run([bm.agent, opp])
    final = env.steps[-1]
    r0, r1 = final[0].reward, final[1].reward
    outcome = "P0" if r0 > r1 else "P1" if r1 > r0 else "DRAW"

    fires = [r for r in bm._WAVE_TELEMETRY if r.get("actual_fire")]
    rejected = [r for r in bm._WAVE_TELEMETRY
                if r.get("gate_rejected_reason")]
    candidates = [r for r in bm._WAVE_TELEMETRY
                  if r.get("gate_delta") is not None]

    print(f"\n=== V3 wave (leaf-validate) vs agents/simple/nearest ===")
    print(f"  seed={seed}  outcome={outcome}  n_steps={len(env.steps)}")
    print(f"  rewards: P0={r0}  P1={r1}")
    print(f"  total wave invocations: {len(bm._WAVE_TELEMETRY)}")
    print(f"  gate candidates (delta computed): {len(candidates)}")
    print(f"  gate-rejected: {len(rejected)}")
    print(f"  actual fires: {len(fires)}")
    print()

    if candidates:
        print(f"  --- gate decisions (chronological) ---")
        for r in candidates:
            step = r.get("step", "?")
            delta = r.get("gate_delta")
            gate_ms = r.get("gate_ms")
            reason = r.get("gate_rejected_reason", "FIRED")
            tgt = r.get("chosen_tgt_id", "?")
            n_sources = r.get("n_sources_in_prefix", 0)
            srcs = r.get("prefix_src_ids", [])
            added = r.get("added_ships", 0)
            tag = "FIRE" if r.get("actual_fire") else f"REJ:{reason}"
            print(f"  step {step:>3}  Δ={delta:>+9.3f}  gate_ms={gate_ms:>5.1f}  "
                  f"tgt={tgt}  n_src={n_sources}  srcs={srcs}  "
                  f"added={added}  [{tag}]")

    walls = [r.get("wall_ms", 0) for r in bm._WAVE_TELEMETRY if r.get("wall_ms")]
    if walls:
        walls_sorted = sorted(walls)
        print(f"\n  wave wall_ms: p50={walls_sorted[len(walls)//2]:.2f}"
              f"  p95={walls_sorted[int(0.95*len(walls))]:.2f}"
              f"  max={max(walls):.2f}")
