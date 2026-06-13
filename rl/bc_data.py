"""Behavior-cloning data: producer replay corpus -> (GameState, labels).

Each dumped step becomes up to 2 samples (one per seat that launched).
Labels live in the policy's action space: per source planet a target
slot (or hold) + fraction index. Producer's raw [pid, angle, ships]
launches are mapped to targets by ray-marching the launch against
predicted planet positions (same math as the feature builder); the
fraction is ships/garrison snapped to {25,50,75,100}%.

Output of `convert_corpus`: stacked GameState + label tensors, ready
for a cross-entropy auxiliary loss alongside PPO.
"""
from __future__ import annotations

import glob
import gzip
import json
import os

import numpy as np

from lib.game.jax.jax_types import (
    GameState, MAX_AGENTS, MAX_COMET_PATH_LEN, MAX_COMET_PATHS_PER_GROUP,
    MAX_FLEETS, MAX_PLANETS, NUM_COMET_SPAWNS,
)
from rl import numpy_infer as ni
from rl.features import N_FRACS

FRACS = np.array([0.25, 0.5, 0.75, 1.0])
COMET_SPAWN_STEPS = (50, 150, 250, 350, 450)


def _step_to_obs(game: dict, st: dict) -> dict:
    return {
        "planets": st["planets"],
        "fleets": st["fleets"],
        "comets": st.get("comets") or [],
        "comet_planet_ids": game.get("comet_planet_ids") or [],
        "initial_planets": game["initial_planets"],
        "angular_velocity": game["angular_velocity"],
        "step": st["step"],
        "player": 0,
    }


def _obs_to_gamestate(obs: dict, num_agents: int = 2) -> GameState:
    """Build a padded GameState from an obs dict (replay step)."""
    P, F, S = MAX_PLANETS, MAX_FLEETS, NUM_COMET_SPAWNS
    L = MAX_COMET_PATH_LEN

    planets = obs["planets"]
    pid_to_idx = {}
    arr = {
        "planets_x": np.zeros(P, np.float32),
        "planets_y": np.zeros(P, np.float32),
        "planets_new_x": np.zeros(P, np.float32),
        "planets_new_y": np.zeros(P, np.float32),
        "planets_id": -np.ones(P, np.int32),
        "planets_owner": -np.ones(P, np.int32),
        "planets_ships": np.zeros(P, np.int32),
        "planets_prod": np.zeros(P, np.int32),
        "planets_radius": np.zeros(P, np.float32),
        "planets_alive": np.zeros(P, bool),
        "initial_x": np.zeros(P, np.float32),
        "initial_y": np.zeros(P, np.float32),
        "is_comet": np.zeros(P, bool),
        "planet_comet_spawn": -np.ones(P, np.int32),
        "planet_comet_path": -np.ones(P, np.int32),
    }
    comet_ids = set(int(c) for c in (obs.get("comet_planet_ids") or []))
    for i, p in enumerate(planets[:P]):
        pid = int(p[0])
        pid_to_idx[pid] = i
        arr["planets_id"][i] = pid
        arr["planets_owner"][i] = int(p[1])
        arr["planets_x"][i] = p[2]; arr["planets_y"][i] = p[3]
        arr["planets_new_x"][i] = p[2]; arr["planets_new_y"][i] = p[3]
        arr["planets_radius"][i] = p[4]
        arr["planets_ships"][i] = p[5]; arr["planets_prod"][i] = p[6]
        arr["planets_alive"][i] = True
        arr["is_comet"][i] = pid in comet_ids
        arr["initial_x"][i] = p[2]; arr["initial_y"][i] = p[3]
    for ip in obs["initial_planets"]:
        idx = pid_to_idx.get(int(ip[0]))
        if idx is not None:
            arr["initial_x"][idx] = ip[2]
            arr["initial_y"][idx] = ip[3]

    f = {
        "fleets_x": np.zeros(F, np.float32),
        "fleets_y": np.zeros(F, np.float32),
        "fleets_angle": np.zeros(F, np.float32),
        "fleets_owner": -np.ones(F, np.int32),
        "fleets_ships": np.zeros(F, np.int32),
        "fleets_from_planet": -np.ones(F, np.int32),
        "fleets_id": -np.ones(F, np.int32),
        "fleets_alive": np.zeros(F, bool),
    }
    for i, fl in enumerate(obs["fleets"][:F]):
        f["fleets_id"][i] = int(fl[0]); f["fleets_owner"][i] = int(fl[1])
        f["fleets_x"][i] = fl[2]; f["fleets_y"][i] = fl[3]
        f["fleets_angle"][i] = fl[4]
        f["fleets_from_planet"][i] = int(fl[5])
        f["fleets_ships"][i] = int(fl[6])
        f["fleets_alive"][i] = True

    cs = {
        "comet_step": np.array(COMET_SPAWN_STEPS, np.int32),
        "comet_paths_xy": np.zeros((S, 4, L, 2), np.float32),
        "comet_paths_len": np.zeros((S, 4), np.int32),
        "comet_ships": np.zeros(S, np.int32),
        "comet_valid": np.zeros(S, bool),
        "comet_path_index": -np.ones(S, np.int32),
        "comet_spawned": np.zeros(S, bool),
        "comet_planet_idx": -np.ones((S, 4), np.int32),
    }
    step = int(obs["step"])
    spawn_step_to_k = {int(s): k for k, s in enumerate(COMET_SPAWN_STEPS)}
    for g in obs.get("comets") or []:
        gidx = int(g["path_index"])
        k = spawn_step_to_k.get(step - gidx)
        if k is None:
            continue
        cs["comet_valid"][k] = True
        cs["comet_spawned"][k] = True
        cs["comet_path_index"][k] = gidx
        # Re-align paths to surviving planet_ids: the engine filters
        # planet_ids on expiry but not paths (interpreter.py:570-573).
        alive_paths = [p for p in g["paths"] if gidx < len(p)]
        for j, pid in enumerate(g["planet_ids"][:MAX_COMET_PATHS_PER_GROUP]):
            slot = pid_to_idx.get(int(pid))
            path = alive_paths[j] if j < len(alive_paths) else g["paths"][j]
            Lp = min(len(path), L)
            cs["comet_paths_len"][k, j] = Lp
            for t in range(Lp):
                cs["comet_paths_xy"][k, j, t, 0] = path[t][0]
                cs["comet_paths_xy"][k, j, t, 1] = path[t][1]
            if slot is not None:
                # The step-0 game-level comet_planet_ids field is empty
                # (comets spawn mid-game), so is_comet was never set from
                # it — flag it here where we actually wire the comet.
                arr["is_comet"][slot] = True
                arr["planet_comet_spawn"][slot] = k
                arr["planet_comet_path"][slot] = j
                cs["comet_planet_idx"][k, j] = slot

    return GameState(
        **arr, **f, **cs,
        step=np.int32(step),
        angular_velocity=np.float32(obs["angular_velocity"]),
        episode_seed=np.int32(0),
        done=np.bool_(False),
        num_agents=np.int32(num_agents),
        next_fleet_id=np.int32(0),
        rewards=np.zeros(4, np.int32),
    )


def _label_launches(obs: dict, actions: list, a_ni: dict):
    """Map raw launches [[pid, angle, ships], ...] -> per-planet
    (tgt_slot, frac_idx). Returns (tgt (P,), frac (P,), has_label (P,)).
    Target = first planet the launch ray hits (predicted positions)."""
    P = MAX_PLANETS
    tgt = np.full(P, P, np.int64)        # default hold
    frac = np.zeros(P, np.int64)
    has = np.zeros(P, bool)

    pid_to_idx = {int(p[0]): i for i, p in enumerate(obs["planets"][:P])}
    best_ships = np.zeros(P)
    for launch in actions or []:
        if len(launch) != 3:
            continue
        pid, angle, ships = int(launch[0]), float(launch[1]), int(launch[2])
        src = pid_to_idx.get(pid)
        if src is None or ships <= 0:
            continue
        if ships < best_ships[src]:
            continue  # keep largest launch per source
        speed = float(ni.fleet_speed(np.array([ships]))[0])
        sx = a_ni["x"][src] + np.cos(angle) * (a_ni["radius"][src] + 0.1)
        sy = a_ni["y"][src] + np.sin(angle) * (a_ni["radius"][src] + 0.1)
        vx, vy = np.cos(angle) * speed, np.sin(angle) * speed
        hit = -1
        for t in range(1, ni.T_HORIZON + 1):
            px = sx + vx * t
            py = sy + vy * t
            pp = ni.planet_pos_at(a_ni, float(t))
            d = np.sqrt((pp[:, 0] - px) ** 2 + (pp[:, 1] - py) ** 2)
            cand = (d < a_ni["radius"] + 0.5 * speed) & a_ni["alive"]
            cand[src] = cand[src] and t > 2  # ignore immediate self-hit
            if cand.any():
                hit = int(np.argmax(cand))
                break
        if hit < 0 or hit == src:
            continue
        g = max(a_ni["ships"][src], 1.0)
        fr = int(np.argmin(np.abs(FRACS - ships / g)))
        tgt[src] = hit
        frac[src] = fr
        has[src] = True
        best_ships[src] = ships
    return tgt, frac, has


def convert_corpus(corpus_dir: str, out_path: str, winners_only: bool = True,
                   max_games: int = 0, stride: int = 2):
    """All games in corpus_dir -> packed npz of samples.

    A sample = one (step, seat) with >=1 labeled launch. winners_only
    keeps the seat that won the game (style worth cloning). stride
    subsamples steps to cut redundancy.
    """
    games = sorted(glob.glob(os.path.join(corpus_dir, "*.json.gz")))
    if max_games:
        games = games[:max_games]
    states, tgts, fracs, masks, seats = [], [], [], [], []
    for gi, path in enumerate(games):
        with gzip.open(path, "rt") as f:
            game = json.load(f)
        rewards = game.get("rewards") or [0, 0]
        keep_seats = [s for s in (0, 1)
                      if (not winners_only) or (rewards[s] == 1)]
        if not keep_seats:
            continue
        for si, st in enumerate(game["steps"]):
            if si % stride:
                continue
            obs = _step_to_obs(game, st)
            a_ni = ni.obs_to_arrays(obs)
            gs = None
            for seat in keep_seats:
                tgt, frac, has = _label_launches(
                    obs, st["actions"][seat], a_ni)
                if not has.any():
                    continue
                if gs is None:
                    gs = _obs_to_gamestate(obs)
                states.append(gs)
                tgts.append(tgt)
                fracs.append(frac)
                masks.append(has)
                seats.append(seat)
        if (gi + 1) % 20 == 0:
            print(f"  {gi + 1}/{len(games)} games, {len(tgts)} samples",
                  flush=True)

    import jax
    stacked = jax.tree.map(lambda *xs: np.stack(xs), *states)
    out = {f"gs_{k}": v for k, v in stacked._asdict().items()}
    out["tgt"] = np.stack(tgts)
    out["frac"] = np.stack(fracs)
    out["mask"] = np.stack(masks)
    out["seat"] = np.array(seats, np.int32)
    np.savez_compressed(out_path, **out)
    print(f"saved {len(tgts)} BC samples -> {out_path}")


def load_bc_npz(path: str):
    z = np.load(path)
    gs = GameState(**{k: z[f"gs_{k}"] for k in GameState._fields})
    return gs, z["tgt"], z["frac"], z["mask"], z["seat"]


if __name__ == "__main__":
    import sys
    convert_corpus(
        sys.argv[1] if len(sys.argv) > 1 else "data/bc_corpus",
        sys.argv[2] if len(sys.argv) > 2 else "data/bc_samples.npz",
    )
