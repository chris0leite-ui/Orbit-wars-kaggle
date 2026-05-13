"""Batched orbit_wars interpreter for multi-candidate agent lookahead.

Runs N independent game states in lockstep, sharing what's deterministic
across lanes (planet positions, comet generation) and vectorising the
fleet × planet sweep collision across all lanes in a single numpy pass.

Byte-exact parity with N sequential scalar `lib.game.interpreter`
calls — enforced by `tests/test_batch_interpreter_parity.py`.

Workload it's tuned for: agent inference, where 4–100 candidate
rollouts from the same starting Snapshot diverge only by their first
action. Single-game per-step cost is the same as scalar (slightly
worse due to dispatch overhead at very small batch sizes); the win
scales with N × F.
"""

from __future__ import annotations

import math
import random
from typing import Sequence

import numpy as np

from lib.fast_sim import Snapshot, clone as fs_clone
from lib.game.interpreter import (
    BOARD_SIZE, CENTER, COMET_PRODUCTION, COMET_RADIUS, COMET_SPAWN_STEPS,
    ROTATION_RADIUS_LIMIT, SUN_RADIUS,
    distance, point_to_segment_distance, swept_pair_hit,
    generate_comet_paths, _get,
    _cos, _sin, _sqrt, _log,
)


def batch_step(
    snaps: list[Snapshot],
    actions_per_lane: Sequence[Sequence[list]],
    *,
    in_place: bool = False,
) -> list[Snapshot]:
    """Advance N snapshots by one tick in lockstep.

    `actions_per_lane[i]` is the list of per-seat actions for snap i.
    Returns a new list of Snapshots unless `in_place=True`.

    Same semantics as `[fast_sim.step(s, a) for s, a in zip(snaps,
    actions)]` but with shared planet-path computation and a batched
    fleet × planet collision check across all lanes.
    """
    if not snaps:
        return list(snaps) if not in_place else snaps
    if not in_place:
        snaps = [fs_clone(s) for s in snaps]
    N = len(snaps)

    # Wire actions onto each lane (same shape as fast_sim.step).
    for i in range(N):
        actions = actions_per_lane[i] if i < len(actions_per_lane) else []
        for seat_idx in range(snaps[i].num_seats):
            snaps[i].state[seat_idx].action = (
                actions[seat_idx] if seat_idx < len(actions) else []
            )

    # Active lanes (skip done ones); they will pass through unchanged.
    active = [i for i in range(N) if not snaps[i].fake_env.done]
    if not active:
        return snaps

    # All lanes were cloned from a single Snapshot at the same step, so
    # `step`, `angular_velocity`, `initial_planets` start identical.
    # Planet positions stay identical across lanes; only ownership and
    # ship counts diverge.
    s0 = snaps[active[0]]
    step = int(_get(s0.state[0].observation, "step", 0))
    configuration = s0.fake_env.configuration
    angular_velocity = float(s0.state[0].observation.angular_velocity)
    max_speed = float(configuration.shipSpeed)
    log1000 = _log(1000)

    # --- Phase: comet expiration (per-lane) -----------------------------
    for i in active:
        _phase_comet_expire(snaps[i].state[0].observation)

    # --- Phase: comet spawn (per-lane; cached) --------------------------
    if (step + 1) in COMET_SPAWN_STEPS:
        for i in active:
            _phase_comet_spawn(snaps[i].state[0].observation, snaps[i].fake_env, step)

    # --- Phase: fleet launch from actions (per-lane) --------------------
    for i in active:
        num_agents = snaps[i].num_seats
        _phase_fleet_launch(snaps[i].state, num_agents)

    # --- Phase: production tick (per-lane) ------------------------------
    for i in active:
        _phase_production(snaps[i].state[0].observation)

    # --- Phase: planet path computation (per-lane) ----------------------
    # We do NOT share `planet_paths` across lanes because callers may pass
    # snapshots with different `initial_planets` / `angular_velocity`. The
    # `planet_position_cache` on each fake_env makes the per-lane compute
    # cheap (dict lookups). When all lanes share a root Snapshot (agent
    # use case) the caches are physically the same object so there's no
    # duplicated work in the cache itself; only the per-lane dict
    # construction is redundant. Phase 3c can detect same-root and share.
    per_lane_paths: list[dict] = []
    per_lane_expired: list[list] = []
    for i in active:
        obs = snaps[i].state[0].observation
        lane_paths = _compute_planet_paths(
            obs,
            snaps[i].fake_env.planet_position_cache,
            float(obs.angular_velocity),
            int(_get(obs, "step", 0)),
        )
        expired = _phase_comet_advance(obs, lane_paths)
        per_lane_paths.append(lane_paths)
        per_lane_expired.append(expired)

    # --- Phase: fleet movement + sweep collision (BATCHED) ---------------
    per_lane_combat: list[tuple[dict, list]] = _phase_batched_fleet_collision(
        snaps, active, per_lane_paths,
        max_speed, log1000,
    )

    # --- Phase: apply planet movement (per-lane) ------------------------
    for ai, i in enumerate(active):
        _phase_apply_planet_movement(
            snaps[i].state[0].observation, per_lane_paths[ai]
        )

    # --- Phase: remove expired comets (per-lane) ------------------------
    for ai, i in enumerate(active):
        _phase_remove_expired_comets(
            snaps[i].state[0].observation, per_lane_expired[ai]
        )

    # --- Phase: fleet removal + combat resolution (per-lane) ------------
    for ai, i in enumerate(active):
        combat_lists, fleets_to_remove = per_lane_combat[ai]
        _phase_combat(
            snaps[i].state[0].observation, combat_lists, fleets_to_remove
        )

    # --- Phase: broadcast to other seats (per-lane) ---------------------
    for i in active:
        _phase_broadcast(snaps[i].state)

    # --- Phase: termination (per-lane) ----------------------------------
    for i in active:
        _phase_terminate(snaps[i].state, snaps[i].fake_env, configuration)

    # --- Bookkeeping (step counter + done propagation) ------------------
    for i in active:
        obs0 = snaps[i].state[0].observation
        new_step = int(_get(obs0, "step", 0)) + 1
        obs0.step = new_step
        for seat_idx in range(1, snaps[i].num_seats):
            snaps[i].state[seat_idx].observation.step = new_step
        if any(s.status == "DONE" for s in snaps[i].state):
            snaps[i].fake_env.done = True

    return snaps


# ---------------------------------------------------------------------------
# Per-phase helpers — extracted from `lib.game.interpreter` and stripped of
# init/done short-circuit handling (the caller filters done lanes). Logic
# is BYTE-EXACT to the scalar interpreter; any deviation breaks the parity
# gate.
# ---------------------------------------------------------------------------


def _phase_comet_expire(obs0) -> None:
    expired_comet_pids = []
    for group in obs0.comets:
        idx = group["path_index"]
        for i, pid in enumerate(group["planet_ids"]):
            if idx >= len(group["paths"][i]):
                expired_comet_pids.append(pid)
    if not expired_comet_pids:
        return
    expired_set = set(expired_comet_pids)
    obs0.planets = [p for p in obs0.planets if p[0] not in expired_set]
    obs0.initial_planets = [
        p for p in obs0.initial_planets if p[0] not in expired_set
    ]
    obs0.comet_planet_ids = [
        pid for pid in obs0.comet_planet_ids if pid not in expired_set
    ]
    for group in obs0.comets:
        group["planet_ids"] = [
            pid for pid in group["planet_ids"] if pid not in expired_set
        ]
    obs0.comets = [g for g in obs0.comets if g["planet_ids"]]


def _phase_comet_spawn(obs0, env, step: int) -> None:
    comet_speed = env.configuration.cometSpeed
    env_info = getattr(env, "info", None) or {}
    episode_seed = env_info.get("seed", 0) or 0
    cache = getattr(env, "comet_path_cache", None)
    cache_key = (episode_seed, step + 1)
    cached = cache.get(cache_key) if cache is not None else None
    if cached is not None:
        comet_paths, comet_ships = cached
    else:
        comet_rng = random.Random(
            f"orbit_wars-comet-{episode_seed}-{step + 1}"
        )
        comet_paths = generate_comet_paths(
            obs0.initial_planets, obs0.angular_velocity, step + 1,
            obs0.comet_planet_ids, comet_speed, rng=comet_rng,
        )
        if comet_paths:
            comet_ships = min(
                comet_rng.randint(1, 99), comet_rng.randint(1, 99),
                comet_rng.randint(1, 99), comet_rng.randint(1, 99),
            )
        else:
            comet_ships = None
        if cache is not None:
            cache[cache_key] = (comet_paths, comet_ships)
    if not comet_paths:
        return
    next_id = max(p[0] for p in obs0.planets) + 1
    group = {"planet_ids": [], "paths": comet_paths, "path_index": -1}
    for i, _path in enumerate(comet_paths):
        pid = next_id + i
        group["planet_ids"].append(pid)
        obs0.comet_planet_ids.append(pid)
        planet = [pid, -1, -99, -99, COMET_RADIUS, comet_ships, COMET_PRODUCTION]
        obs0.planets.append(planet)
        obs0.initial_planets.append(planet[:])
    obs0.comets.append(group)


def _phase_fleet_launch(state: list, num_agents: int) -> None:
    obs0 = state[0].observation
    for player_id in range(num_agents):
        action = state[player_id].action
        if not action or not isinstance(action, list):
            continue
        for move in action:
            if len(move) != 3:
                continue
            from_id, angle, ships = move
            ships = int(ships)
            from_planet = next((p for p in obs0.planets if p[0] == from_id), None)
            if not from_planet or from_planet[1] != player_id:
                continue
            if from_planet[5] < ships or ships <= 0:
                continue
            from_planet[5] -= ships
            start_x = from_planet[2] + _cos(angle) * (from_planet[4] + 0.1)
            start_y = from_planet[3] + _sin(angle) * (from_planet[4] + 0.1)
            obs0.fleets.append(
                [obs0.next_fleet_id, player_id, start_x, start_y,
                 angle, from_id, ships]
            )
            obs0.next_fleet_id += 1


def _phase_production(obs0) -> None:
    for planet in obs0.planets:
        if planet[1] != -1:
            planet[5] += planet[6]


def _compute_planet_paths(obs0, position_cache, angular_velocity, step) -> dict:
    """Compute the {planet_id -> (old_pos, new_pos, check)} dict for
    NON-COMET planets. Comet entries are written per-lane by
    `_phase_comet_advance` (each lane may have its own comet membership).
    """
    paths: dict = {}
    comet_pid_set = set(obs0.comet_planet_ids)
    initial_by_id = {p[0]: p for p in obs0.initial_planets}
    pos_cache = position_cache if position_cache is not None else {}
    for planet in obs0.planets:
        if planet[0] in comet_pid_set:
            continue
        old_pos = (planet[2], planet[3])
        new_pos = old_pos
        cached = pos_cache.get(planet[0])
        if cached is not None and step < len(cached):
            new_pos = cached[step]
        else:
            initial_p = initial_by_id.get(planet[0])
            if initial_p is not None:
                dx = initial_p[2] - CENTER
                dy = initial_p[3] - CENTER
                r = _sqrt(dx * dx + dy * dy)
                if r + planet[4] < ROTATION_RADIUS_LIMIT:
                    current_angle = math.atan2(dy, dx) + angular_velocity * step
                    new_pos = (
                        CENTER + r * _cos(current_angle),
                        CENTER + r * _sin(current_angle),
                    )
        paths[planet[0]] = (old_pos, new_pos, True)
    return paths


def _phase_comet_advance(obs0, lane_paths: dict) -> list:
    """Advance comet paths for THIS lane. Mutates lane_paths to include
    per-lane comet entries (positions and check flags). Returns the list
    of expired comet pids (to be filtered out post-collision)."""
    expired = []
    planet_by_id = {p[0]: p for p in obs0.planets}
    for group in obs0.comets:
        group["path_index"] += 1
        idx = group["path_index"]
        group_paths = group["paths"]
        for i, pid in enumerate(group["planet_ids"]):
            planet = planet_by_id.get(pid)
            if planet is None:
                continue
            p_path = group_paths[i]
            old_pos = (planet[2], planet[3])
            if idx >= len(p_path):
                expired.append(pid)
                lane_paths[pid] = (old_pos, old_pos, True)
            else:
                pp = p_path[idx]
                check = old_pos[0] >= 0
                lane_paths[pid] = (old_pos, (pp[0], pp[1]), check)
    return expired


_BATCH_FLEET_THRESHOLD = 30  # below this, scalar per-fleet beats numpy


def _phase_batched_fleet_collision(
    snaps: list[Snapshot], active: list[int],
    per_lane_paths: list[dict],
    max_speed: float, log1000: float,
) -> list[tuple[dict, list]]:
    """Apply fleet movement + sweep-collision across ALL active lanes in
    one batched pass. Per-lane fleet positions are mutated in place;
    returns a list of (combat_lists, fleets_to_remove) tuples per active
    lane for the combat phase to consume.
    """
    # Collect all lanes' fleets into a flat array.
    lane_ids: list[int] = []  # per-fleet: active-lane index (0..len(active)-1)
    fleet_objs: list = []
    angles: list[float] = []
    ships_list: list[float] = []
    fold_x_list: list[float] = []
    fold_y_list: list[float] = []
    for ai, i in enumerate(active):
        for f in snaps[i].state[0].observation.fleets:
            lane_ids.append(ai)
            fleet_objs.append(f)
            angles.append(f[4])
            ships_list.append(float(f[6]))
            fold_x_list.append(f[2])
            fold_y_list.append(f[3])
    F_total = len(fleet_objs)
    per_lane_results: list[tuple[dict, list]] = []
    if F_total == 0:
        for i in active:
            combat_lists = {
                p[0]: [] for p in snaps[i].state[0].observation.planets
            }
            per_lane_results.append((combat_lists, []))
        return per_lane_results

    # Position update is done with SCALAR math.* to match the scalar
    # interpreter byte-for-byte (numpy's vector cos/sin/log/pow can drift
    # by a single ULP due to different libm paths; the parity test caught
    # it). Only the F×P collision check below is numpy-vectorised.
    fnew_x_list: list[float] = []
    fnew_y_list: list[float] = []
    for k in range(F_total):
        ships = ships_list[k]
        speed = 1.0 + (max_speed - 1.0) * (_log(ships) / log1000) ** 1.5
        if speed > max_speed:
            speed = max_speed
        angle = angles[k]
        f2 = fold_x_list[k]; f3 = fold_y_list[k]
        new_x = f2 + _cos(angle) * speed
        new_y = f3 + _sin(angle) * speed
        fleet_objs[k][2] = new_x
        fleet_objs[k][3] = new_y
        fnew_x_list.append(new_x)
        fnew_y_list.append(new_y)
    fold_x = np.asarray(fold_x_list, dtype=np.float64)
    fold_y = np.asarray(fold_y_list, dtype=np.float64)
    fnew_x = np.asarray(fnew_x_list, dtype=np.float64)
    fnew_y = np.asarray(fnew_y_list, dtype=np.float64)

    # Per-lane: planet arrays for sweep collision.
    # We do the collision check per-lane (not concatenated across lanes)
    # because each lane has its own planets list (different comet pids
    # may exist per lane after captures). For typical batch sizes the
    # outer Python loop is fine; the win is from numpy on F×P per lane.
    for ai, i in enumerate(active):
        obs = snaps[i].state[0].observation
        planets_local = obs.planets
        n_planets = len(planets_local)
        lane_paths = per_lane_paths[ai]
        combat_lists = {p[0]: [] for p in planets_local}
        fleets_to_remove = []

        # Pull this lane's fleets back out of the flat arrays.
        lane_fleet_idx = [k for k, lid in enumerate(lane_ids) if lid == ai]
        if not lane_fleet_idx:
            per_lane_results.append((combat_lists, fleets_to_remove))
            continue

        # Build planet arrays once for this lane.
        pold_x = np.empty(n_planets); pold_y = np.empty(n_planets)
        pnew_x = np.empty(n_planets); pnew_y = np.empty(n_planets)
        pr_arr = np.empty(n_planets)
        pcheck = np.zeros(n_planets, dtype=bool)
        for pj, p in enumerate(planets_local):
            path = lane_paths.get(p[0])
            if path is None:
                continue
            pold_x[pj] = path[0][0]; pold_y[pj] = path[0][1]
            pnew_x[pj] = path[1][0]; pnew_y[pj] = path[1][1]
            pr_arr[pj] = p[4]
            pcheck[pj] = path[2]

        # Indices into the flat arrays for this lane's fleets.
        lf_idx = np.asarray(lane_fleet_idx, dtype=np.intp)
        F_lane = len(lane_fleet_idx)
        lf_old_x = fold_x[lf_idx]; lf_old_y = fold_y[lf_idx]
        lf_new_x = fnew_x[lf_idx]; lf_new_y = fnew_y[lf_idx]

        if F_lane * n_planets >= _BATCH_FLEET_THRESHOLD:
            first_hits = _f_x_p_first_hits(
                lf_old_x, lf_old_y, lf_new_x, lf_new_y,
                pold_x, pold_y, pnew_x, pnew_y, pr_arr, pcheck,
            )
        else:
            first_hits = _scalar_first_hits_per_lane(
                lane_fleet_idx, fleet_objs, fold_x_list, fold_y_list,
                fnew_x_list, fnew_y_list, planets_local, lane_paths,
            )

        for fk, k in enumerate(lane_fleet_idx):
            fleet = fleet_objs[k]
            hit_idx = int(first_hits[fk])
            if hit_idx >= 0:
                combat_lists[planets_local[hit_idx][0]].append(fleet)
                fleets_to_remove.append(fleet)
                continue
            nx = fnew_x_list[k]; ny = fnew_y_list[k]
            if not (0 <= nx <= BOARD_SIZE and 0 <= ny <= BOARD_SIZE):
                fleets_to_remove.append(fleet)
                continue
            if point_to_segment_distance(
                (CENTER, CENTER), (fold_x_list[k], fold_y_list[k]), (nx, ny)
            ) < SUN_RADIUS:
                fleets_to_remove.append(fleet)
                continue

        per_lane_results.append((combat_lists, fleets_to_remove))

    return per_lane_results


def _f_x_p_first_hits(
    fold_x: np.ndarray, fold_y: np.ndarray,
    fnew_x: np.ndarray, fnew_y: np.ndarray,
    pold_x: np.ndarray, pold_y: np.ndarray,
    pnew_x: np.ndarray, pnew_y: np.ndarray,
    pr_arr: np.ndarray, pcheck_arr: np.ndarray,
) -> np.ndarray:
    """Vectorised (F, P) swept-pair test → first-hit index per fleet (or -1)."""
    f_min_x = np.minimum(fold_x, fnew_x)[:, None]
    f_max_x = np.maximum(fold_x, fnew_x)[:, None]
    f_min_y = np.minimum(fold_y, fnew_y)[:, None]
    f_max_y = np.maximum(fold_y, fnew_y)[:, None]
    p_min_x = (np.minimum(pold_x, pnew_x) - pr_arr)[None, :]
    p_max_x = (np.maximum(pold_x, pnew_x) + pr_arr)[None, :]
    p_min_y = (np.minimum(pold_y, pnew_y) - pr_arr)[None, :]
    p_max_y = (np.maximum(pold_y, pnew_y) + pr_arr)[None, :]
    candidate = (
        pcheck_arr[None, :]
        & (f_max_x >= p_min_x) & (f_min_x <= p_max_x)
        & (f_max_y >= p_min_y) & (f_min_y <= p_max_y)
    )

    d0x = fold_x[:, None] - pold_x[None, :]
    d0y = fold_y[:, None] - pold_y[None, :]
    dvx = (fnew_x - fold_x)[:, None] - (pnew_x - pold_x)[None, :]
    dvy = (fnew_y - fold_y)[:, None] - (pnew_y - pold_y)[None, :]
    a = dvx * dvx + dvy * dvy
    b = 2.0 * (d0x * dvx + d0y * dvy)
    c = d0x * d0x + d0y * d0y - (pr_arr * pr_arr)[None, :]
    a_small = a < 1e-12
    disc = b * b - 4.0 * a * c
    disc_ok = disc >= 0.0
    sq = np.sqrt(np.where(disc_ok, disc, 0.0))
    denom = np.where(a_small, 1.0, 2.0 * a)
    t1 = (-b - sq) / denom
    t2 = (-b + sq) / denom
    hit_full = disc_ok & (t2 >= 0.0) & (t1 <= 1.0)
    hit = candidate & np.where(a_small, c <= 0.0, hit_full)

    any_hit = hit.any(axis=1)
    first_hit = hit.argmax(axis=1)
    return np.where(any_hit, first_hit, -1)


def _scalar_first_hits_per_lane(
    lane_fleet_idx: list, fleet_objs: list,
    fold_x_list: list, fold_y_list: list,
    fnew_x_list: list, fnew_y_list: list,
    planets_local: list, lane_paths: dict,
):
    """Fallback scalar path for very small lanes (< _BATCH_FLEET_THRESHOLD)."""
    out = [-1] * len(lane_fleet_idx)
    for fk, k in enumerate(lane_fleet_idx):
        f2 = fold_x_list[k]; f3 = fold_y_list[k]
        nx = fnew_x_list[k]; ny = fnew_y_list[k]
        for p_idx, p in enumerate(planets_local):
            path = lane_paths.get(p[0])
            if path is None or not path[2]:
                continue
            if swept_pair_hit((f2, f3), (nx, ny), path[0], path[1], p[4]):
                out[fk] = p_idx
                break
    return out


def _phase_apply_planet_movement(obs0, lane_paths: dict) -> None:
    for planet in obs0.planets:
        path = lane_paths.get(planet[0])
        if path is not None:
            planet[2], planet[3] = path[1]


def _phase_remove_expired_comets(obs0, expired_comet_pids: list) -> None:
    if not expired_comet_pids:
        return
    expired_set = set(expired_comet_pids)
    obs0.planets = [p for p in obs0.planets if p[0] not in expired_set]
    obs0.initial_planets = [
        p for p in obs0.initial_planets if p[0] not in expired_set
    ]
    obs0.comet_planet_ids = [
        pid for pid in obs0.comet_planet_ids if pid not in expired_set
    ]
    for group in obs0.comets:
        group["planet_ids"] = [
            pid for pid in group["planet_ids"] if pid not in expired_set
        ]
    obs0.comets = [g for g in obs0.comets if g["planet_ids"]]


def _phase_combat(obs0, combat_lists: dict, fleets_to_remove: list) -> None:
    """Apply fleet removal + per-planet combat resolution. Mirrors the
    scalar interpreter's order: remove dead fleets first, then resolve
    combat on planets that took hits.
    """
    # Apply fleet removal first (matches scalar interpreter order).
    if fleets_to_remove:
        remove_ids = {id(f) for f in fleets_to_remove}
        obs0.fleets = [f for f in obs0.fleets if id(f) not in remove_ids]

    for pid, planet_fleets in combat_lists.items():
        planet = next((p for p in obs0.planets if p[0] == pid), None)
        if not planet or not planet_fleets:
            continue
        player_ships: dict = {}
        for fleet in planet_fleets:
            owner = fleet[1]
            player_ships[owner] = player_ships.get(owner, 0) + fleet[6]
        if not player_ships:
            continue
        sorted_players = sorted(
            player_ships.items(), key=lambda item: item[1], reverse=True
        )
        top_player, top_ships = sorted_players[0]
        if len(sorted_players) > 1:
            second_ships = sorted_players[1][1]
            survivor_ships = top_ships - second_ships
            if sorted_players[0][1] == sorted_players[1][1]:
                survivor_ships = 0
            survivor_owner = top_player if survivor_ships > 0 else -1
        else:
            survivor_owner = top_player
            survivor_ships = top_ships
        if survivor_ships > 0:
            if planet[1] == survivor_owner:
                planet[5] += survivor_ships
            else:
                planet[5] -= survivor_ships
                if planet[5] < 0:
                    planet[1] = survivor_owner
                    planet[5] = abs(planet[5])


def _phase_broadcast(state: list) -> None:
    obs0 = state[0].observation
    for i in range(1, len(state)):
        state[i].observation.planets = obs0.planets
        state[i].observation.initial_planets = obs0.initial_planets
        state[i].observation.fleets = obs0.fleets
        state[i].observation.next_fleet_id = obs0.next_fleet_id
        state[i].observation.comets = obs0.comets
        state[i].observation.comet_planet_ids = obs0.comet_planet_ids


def _phase_terminate(state: list, fake_env, configuration) -> None:
    obs0 = state[0].observation
    step = int(_get(obs0, "step", 0))
    terminated = step >= configuration.episodeSteps - 2
    alive_players: set = set()
    for p in obs0.planets:
        if p[1] != -1:
            alive_players.add(p[1])
    for f in obs0.fleets:
        alive_players.add(f[1])
    if len(alive_players) <= 1:
        terminated = True
    if terminated:
        num_agents = len(state)
        for s in state:
            s.status = "DONE"
        scores = [0] * num_agents
        for p in obs0.planets:
            if p[1] != -1:
                scores[p[1]] += p[5]
        for f in obs0.fleets:
            scores[f[1]] += f[6]
        max_score = max(scores)
        for i in range(num_agents):
            state[i].reward = 1 if scores[i] == max_score and max_score > 0 else -1


