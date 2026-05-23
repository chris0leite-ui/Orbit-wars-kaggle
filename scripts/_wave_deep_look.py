"""Deeper close-look on 2 games — what was the GAME STATE when the wave
fired, what was the target's actual garrison, and what was the launch's
fate (capture vs bounce, kept vs lost)?

For each wave-fire turn, dump:
  - my total ships / planets vs opponent
  - target id, owner, ships, production
  - per-launch: predicted vs ACTUAL outcome (resolve from step+eta in env)
  - was the source planet alive 5/10 turns later?
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

os.environ["BASELINE_CONVERGENCE_WAVE"] = "1"
import agents.baseline.main as bm
bm.CONVERGENCE_WAVE_ENABLED = True

from kaggle_environments import make
import importlib.util

WAVE_EVENTS: list[dict] = []
_orig_wave = bm.emit_convergence_wave

def _logged_wave(moves, planets, my_id, world, model):
    moves_in = list(moves)
    out = _orig_wave(moves_in, planets, my_id, world, model)
    extras = out[len(moves_in):]
    if extras:
        my = [p for p in planets if int(p.owner) == my_id]
        enemy = [p for p in planets if int(p.owner) != my_id and int(p.owner) != -1]
        # Each extra carries (src_id, angle, ships). Resolve target by
        # finding the enemy planet whose angle-from-src matches.
        import math
        for src_id, angle, ships in extras:
            src = next(p for p in planets if int(p.id) == int(src_id))
            # Find the enemy planet whose direction from src best matches angle.
            best_tgt = None
            best_dot = -2.0
            for t in enemy:
                tx = float(t.x) - float(src.x)
                ty = float(t.y) - float(src.y)
                norm = math.hypot(tx, ty)
                if norm == 0:
                    continue
                cos = (math.cos(angle) * tx + math.sin(angle) * ty) / norm
                if cos > best_dot:
                    best_dot = cos
                    best_tgt = t
            WAVE_EVENTS.append({
                "step": int(world.step),
                "me": int(my_id),
                "src_id": int(src_id),
                "src_ships_before": int(src.ships),
                "ships_emit": int(ships),
                "tgt_id": int(best_tgt.id) if best_tgt else None,
                "tgt_owner_before": int(best_tgt.owner) if best_tgt else None,
                "tgt_ships_before": int(best_tgt.ships) if best_tgt else None,
                "tgt_prod": int(best_tgt.production) if best_tgt else None,
                "my_total_ships": sum(int(p.ships) for p in my),
                "my_planet_count": len(my),
                "enemy_total_ships": sum(int(p.ships) for p in enemy),
                "enemy_planet_count": len(enemy),
            })
    return out

bm.emit_convergence_wave = _logged_wave

def load(path):
    p = Path(path)
    spec = importlib.util.spec_from_file_location(f"_opp_{p.stem}_{id(p)}", path)
    m = importlib.util.module_from_spec(spec); sys.modules[spec.name] = m
    spec.loader.exec_module(m); return m.agent

def planet_state_at(env, step, pid):
    """Look up the actual planet state at `step` from env.steps history."""
    if step >= len(env.steps):
        return None
    obs = env.steps[step][0].observation
    for p in obs["planets"]:
        if int(p[0]) == int(pid):
            return {"owner": int(p[1]), "ships": int(p[5])}
    return None

def run(seed):
    WAVE_EVENTS.clear()
    env = make("orbit_wars", configuration={"seed": seed}, debug=False)
    opp = load("submissions/baseline_full.py")
    env.run([bm.agent, opp])
    final = env.steps[-1]
    r0, r1 = final[0].reward, final[1].reward
    outcome = "p0_win" if r0 > r1 else "p1_win" if r1 > r0 else "draw"
    print(f"\n=== seed={seed} vs baseline_full — outcome {outcome} n_steps={len(env.steps)} ===")
    print(f"  total wave fires: {len(WAVE_EVENTS)}\n")
    for e in WAVE_EVENTS:
        step = e["step"]
        tgt_id = e["tgt_id"]
        # Lookup actual target state at step+5 and step+15 to see capture/bounce.
        s5 = planet_state_at(env, step + 5, tgt_id) if tgt_id is not None else None
        s15 = planet_state_at(env, step + 15, tgt_id) if tgt_id is not None else None
        src5 = planet_state_at(env, step + 5, e["src_id"])
        s_at = planet_state_at(env, step, tgt_id)
        # Capture verdict — if at step+15 the target is owned by me (player=e["me"]), capture.
        # If target was originally enemy and stayed enemy, bounce.
        capture = (s15 is not None and s15["owner"] == e["me"])
        src_lost = (src5 is not None and src5["owner"] != e["me"])
        my_share = e["my_total_ships"] / max(1, e["my_total_ships"] + e["enemy_total_ships"])
        verdict = "CAPTURE" if capture else "BOUNCE"
        flag = " SRC-LOST!" if src_lost else ""
        print(f"  step={step:>3}  src={e['src_id']}→tgt={tgt_id}  "
              f"emit={e['ships_emit']:>3}  src_had={e['src_ships_before']:>3}  "
              f"tgt_owner={e['tgt_owner_before']} tgt_ships={e['tgt_ships_before']:>3} prod={e['tgt_prod']}  "
              f"share={my_share:.0%}  → {verdict}{flag}")
        if not capture and tgt_id is not None:
            # show what the target looked like later
            owner_15 = s15["owner"] if s15 else "?"
            ships_15 = s15["ships"] if s15 else "?"
            print(f"      tgt@step+15: owner={owner_15} ships={ships_15}  (we wanted capture)")

if __name__ == "__main__":
    seeds = [int(s) for s in sys.argv[1:]] or [42, 7]
    for s in seeds:
        run(s)
