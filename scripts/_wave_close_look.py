"""Instrument 2 games to inspect the convergence wave's per-turn fires.

For each seed, runs the wave-on baseline vs baseline_full (the production
candidate's natural opponent for this branch), records:
  - per-turn wave emissions (count, total ships, targets)
  - chooser emissions for context
  - final game outcome and turn count

This is the small-and-close inspection the PI requested before any
A/B at scale.
"""

from __future__ import annotations

import os
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

os.environ["BASELINE_CONVERGENCE_WAVE"] = "1"

import agents.baseline.main as bm
bm.CONVERGENCE_WAVE_ENABLED = True

from kaggle_environments import make

# Capture per-turn wave activity.
_LOG: list[dict] = []
_orig_wave = bm.emit_convergence_wave


def _logged_wave(moves, planets, my_id, world, model):
    moves_in = list(moves)
    out = _orig_wave(moves_in, planets, my_id, world, model)
    extras = out[len(moves_in):]
    if extras:
        # Decode each extra into (src, ships); resolve target by re-aiming
        # from src angle (informational only).
        entry = {
            "step": int(world.step),
            "me": int(my_id),
            "chooser_count": len(moves_in),
            "wave_count": len(extras),
            "wave_total_ships": sum(int(e[2]) for e in extras),
            "wave_srcs": [int(e[0]) for e in extras],
            "wave_ships": [int(e[2]) for e in extras],
        }
        _LOG.append(entry)
    return out


bm.emit_convergence_wave = _logged_wave


def _resolve_opp(path: str):
    """Load an agent callable from a submission file path.

    Mirrors fast.py::_load_callable — registers the temp module in
    sys.modules BEFORE exec so @dataclass / bundle-internal names resolve.
    """
    import importlib.util
    p = Path(path)
    spec = importlib.util.spec_from_file_location(
        f"_opp_{p.stem}_{id(p)}", path,
    )
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m.agent


def run_one(seed: int, opp_path: str):
    _LOG.clear()
    env = make("orbit_wars", configuration={"seed": seed}, debug=False)
    focal = bm.agent
    opp = _resolve_opp(opp_path)
    env.run([focal, opp])
    final = env.steps[-1]
    r0, r1 = final[0].reward, final[1].reward
    outcome = "p0_win" if r0 > r1 else "p1_win" if r1 > r0 else "draw"
    n_steps = len(env.steps)

    wave_turns = len(_LOG)
    wave_launches = sum(e["wave_count"] for e in _LOG)
    wave_ships = sum(e["wave_total_ships"] for e in _LOG)
    avg_bundle = wave_ships / wave_launches if wave_launches else 0.0

    print(f"\n=== seed={seed} vs {Path(opp_path).stem} ===")
    print(f"  outcome: {outcome}  (rewards P0={r0} P1={r1})  n_steps={n_steps}")
    print(f"  wave fired on {wave_turns} turns, {wave_launches} total launches, "
          f"{wave_ships} ships  (avg bundle = {avg_bundle:.1f} ships/launch)")
    if _LOG:
        print(f"  per-fire: step | chooser | wave-launches | wave-ships | sources")
        for e in _LOG[:20]:
            srcs = ",".join(map(str, e["wave_srcs"]))
            ships = ",".join(map(str, e["wave_ships"]))
            print(f"    step={e['step']:>3}  chooser={e['chooser_count']:>2}  "
                  f"wave={e['wave_count']}  ships=[{ships}]  srcs=[{srcs}]")
        if len(_LOG) > 20:
            print(f"    ... and {len(_LOG) - 20} more wave-fire turns")


if __name__ == "__main__":
    opp = sys.argv[1] if len(sys.argv) > 1 else "submissions/baseline_full.py"
    seeds = [int(s) for s in sys.argv[2:]] or [42, 7]
    for s in seeds:
        run_one(s, opp)
