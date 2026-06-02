"""Time-INDEPENDENT position-parity: does the kinematic table produce the
EXACT same planet_positions dict as the inline build, on a real game state?
This isolates computational correctness from the (desired) timing effect."""
import os
os.environ.setdefault("KAGGLE_ENV_DEBUG", "0")
from kaggle_environments import make
from lib.intent import World
from lib.world_model import _comet_paths_by_id
from lib.orbit import is_orbiting
from lib.trajectory import _predict_relative_window, _table_window_or_none
from lib.kinematic_table import KinematicTable

OFF_BOARD = (-1e6, -1e6)

def inline_positions(world, wait_N, length):
    """Exact copy of the inline build in predict_fleet_fate."""
    max_steps = length - 1
    omega = world.omega
    comet_paths = _comet_paths_by_id(world) if world.comet_ids else {}
    pp = {}
    for pid, p in world.planets_by_id.items():
        if int(pid) in comet_paths:
            path, path_index = comet_paths[int(pid)]
            positions = []
            for t in range(max_steps + 1):
                path_t = int(path_index) + int(wait_N) + t
                if 0 <= path_t < len(path):
                    pt = path[path_t]
                    positions.append((float(pt[0]), float(pt[1])))
                else:
                    positions.append(OFF_BOARD)
            pp[pid] = positions
            continue
        p_tuple = [p.id, p.owner, p.x, p.y, p.radius, p.ships, p.production]
        if is_orbiting(p_tuple) and omega != 0.0:
            pp[pid] = _predict_relative_window(p_tuple, omega, wait_N, max_steps + 1)
        else:
            pp[pid] = [(p.x, p.y)] * (max_steps + 1)
    return pp

env = make("orbit_wars", debug=False)
env.reset()
# step with random actions to reach a state WITH comets (appear ~step 50+)
import random
random.seed(1)
checked = 0
mismatches = 0
for step_i in range(120):
    obs0 = env.state[0].observation
    world = World.from_obs(obs0)
    if step_i in (10, 40, 70, 100):  # sample several states
        table = KinematicTable(); table.begin_turn(world)
        n_comets = len(world.comet_ids)
        n_orbital = sum(1 for p in world.planets_by_id.values()
                        if is_orbiting([p.id,p.owner,p.x,p.y,p.radius,p.ships,p.production]) and world.omega!=0.0)
        for wait_N in (0, 5, 17, 40):
            for length in (50, 201):
                t_dict = _table_window_or_none(table, world, wait_N, length)
                i_dict = inline_positions(world, wait_N, length)
                if t_dict is None:
                    print(f"  step{step_i} wait{wait_N} len{length}: table MISS (fallback)")
                    continue
                checked += 1
                if t_dict != i_dict:
                    mismatches += 1
                    # find first differing pid
                    for pid in i_dict:
                        if t_dict.get(pid) != i_dict[pid]:
                            print(f"  MISMATCH step{step_i} wait{wait_N} len{length} pid{pid}")
                            print(f"    table[0:3]={t_dict[pid][:3]}")
                            print(f"    inline[0:3]={i_dict[pid][:3]}")
                            break
        print(f"step{step_i}: {n_orbital} orbital, {n_comets} comets, checked so far={checked}, mismatches={mismatches}")
    # advance
    acts = [a.action if hasattr(a,'action') else None for a in env.state]
    env.step([None, None])
    if env.done:
        print(f"(game ended at step {step_i})"); break

print(f"\nRESULT: checked={checked} window-comparisons, mismatches={mismatches}")
print("PASS — table is bit-identical to inline" if mismatches == 0 else "FAIL — table DIVERGES from inline")
