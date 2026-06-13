
from __future__ import annotations

import dataclasses
import os
import sys
from dataclasses import dataclass

# Make the sibling ``orbit_lite`` package importable wherever this file runs.
try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _HERE = os.getcwd()
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import torch
from torch import Tensor

from orbit_lite.geometry import fleet_speed
from orbit_lite.intercept_aim import intercept_angle
from orbit_lite.movement import MovementConfig, PlanetMovement
from orbit_lite.movement_step import (
    apply_private_planned_launches,
    concat_launch_entries,
    disambiguate_duplicate_launches,
    ensure_planet_movement,
    infer_planned_launches_from_entries,
)
from orbit_lite.obs import parse_obs
from orbit_lite.distance_cache import build_distance_cache
from orbit_lite.planner_core import (
    _candidate_indices,
    _empty_entries,
    _greedy_select,
    _plan_regroup,
    build_target_shortlist,
    capture_floor,
    empty_action_row,
    entries_to_sparse_payload,
    largest_initial_player_count,
    make_launch_set,
    reachable_mask,
    reinforcement_timing_factor,
    safe_drain,
    score_candidates,
)
from orbit_lite.adapter import single_obs_to_tensor, sparse_action_row_to_moves


@dataclass(frozen=True)
class ProducerLiteConfig:
    """Behaviour knobs — v5 (improved from v4)."""

    # projection window, movement build length, AND target ETA cap
    horizon: int = 20               
    # --- shortlists ---------------------------------------------------------
    max_sources_per_lane: int = 14  
    max_offensive_targets: int = 15 
    max_defensive_targets: int = 6  
    # --- scoring / greedy ---------------------------------------------------
    max_waves_per_turn: int = 8     
    roi_threshold: float = 1.25     
    min_ships_to_launch: float = 3.0  
    # --- regroup ------------------------------------------------------------
    enable_regroup: bool = True
    max_regroup_time: float = 8.0              
    regroup_pressure_delta_min: float = 0.20   
    max_regroup_sources_per_lane: int = 8      
    max_regroup_targets_per_source: int = 10   
    regroup_pressure_norm: str = "none"
    regroup_time_penalty_weight: float = 1e-3
    # --- FFA bonuses --------------------------------------------------------
    ffa_leader_attack_bonus: float = 0.0
    ffa_target_prod_bonus: float = 0.0
    # --- new strategic bonuses ----------------------------------------------
    comet_attack_bonus: float = 3.0   
    high_prod_attack_bonus: float = 0.20  


def _movement_config(config: ProducerLiteConfig, *, player_count: int) -> MovementConfig:
    """MovementConfig: fleet tracking on, horizon = config.horizon."""
    return MovementConfig(
        movement_horizon=int(config.horizon),
        drift_epsilon=1e-3,
        track_fleets=True,
        player_count=int(player_count),
        max_tracked_fleets=128,
    )


def _apply_phase_config(config: ProducerLiteConfig, step: int) -> ProducerLiteConfig:
    """Adjust scoring knobs based on game phase."""
    if step < 80:
        return dataclasses.replace(
            config,
            roi_threshold=max(0.90, config.roi_threshold - 0.20),
            min_ships_to_launch=max(2.0, config.min_ships_to_launch - 1.0),
        )
    elif step > 400:
        return dataclasses.replace(
            config,
            roi_threshold=config.roi_threshold + 0.20,
            max_defensive_targets=min(10, config.max_defensive_targets + 2),
        )
    return config


def _apply_time_budget(config: ProducerLiteConfig, remaining_time: float) -> ProducerLiteConfig:
    """Reduce computation when the overage time budget is running low."""
    if remaining_time < 5.0:
        return dataclasses.replace(
            config,
            max_sources_per_lane=max(4, config.max_sources_per_lane - 5),
            max_offensive_targets=max(4, config.max_offensive_targets - 7),
            max_waves_per_turn=max(3, config.max_waves_per_turn - 3),
        )
    elif remaining_time < 15.0:
        return dataclasses.replace(
            config,
            max_sources_per_lane=max(6, config.max_sources_per_lane - 3),
            max_offensive_targets=max(6, config.max_offensive_targets - 4),
        )
    return config


def cheap_enemy_pressure(obs, cache, *, horizon: float, player_id: int) -> Tensor:
    """Cheap reachable-enemy-mass proxy per planet — ``[P]``."""
    P = int(obs.P)
    device = obs.device
    dtype = obs.ships.dtype
    if P == 0:
        return torch.zeros(P, dtype=dtype, device=device)
    d0 = cache.cross_dist[0].to(dtype)
    ships = obs.ships.to(dtype)
    speeds = fleet_speed(ships.clamp(min=1e-6))
    reach_dist = (speeds.view(P, 1) * float(horizon)).clamp(min=1e-6)
    enemy = obs.alive & (obs.owner_abs >= 0) & (obs.owner_abs != int(player_id))
    eye = torch.eye(P, device=device, dtype=torch.bool)
    valid = enemy.view(P, 1) & obs.alive.view(1, P) & ~eye
    decay = (1.0 - d0 / reach_dist).clamp(min=0.0)
    contrib = torch.where(valid, ships.view(P, 1) * decay, torch.zeros_like(decay))
    return contrib.sum(dim=0)


def plan_lite_waves(
    *,
    movement: PlanetMovement,
    obs,
    obs_tensors: dict,
    cache,
    garrison_status,
    prod: Tensor,
    alive_by_step: Tensor,
    config: ProducerLiteConfig,
    player_count: int,
):
    """Single-size, single-source attack planner + regroup — v5."""
    P = obs.P
    device = obs.device
    dtype = obs.ships.dtype
    pid = int(obs.player_id)

    H_axis = int(garrison_status.ships.shape[-1])
    H = max(H_axis - 1, 0)
    K_eta = max(1, min(int(config.horizon), H))
    W = max(1, int(config.max_waves_per_turn))

    source_mask = obs.owned & obs.alive & (obs.ships >= float(config.min_ships_to_launch))
    if not bool(source_mask.any()):
        return _empty_entries(device, dtype)

    S_cap = max(1, min(int(config.max_sources_per_lane), P))
    source_idx, source_exists = _candidate_indices(obs.ships, source_mask, S_cap)
    target_idx, target_exists = build_target_shortlist(
        obs, obs_tensors, garrison_status, cache,
        config=config, K_eta=K_eta, H=H, prod=prod, source_mask=source_mask,
    )
    if not bool(target_exists.any()):
        return _empty_entries(device, dtype)
    S = int(source_idx.shape[0])
    T = int(target_idx.shape[0])
    target_is_mine = obs.owned[target_idx.clamp(0, P - 1)]

    source_ships = obs.ships[source_idx.clamp(0, P - 1)].to(dtype)
    H_eff = torch.full((), float(H), dtype=dtype, device=device)
    drain = safe_drain(
        garrison_status, source_idx=source_idx, source_ships=source_ships,
        H_eff=H_eff, player_id=pid,
    )

    eta_cap = torch.full((T,), float(K_eta), dtype=dtype, device=device)

    floor = capture_floor(
        garrison_status, target_idx=target_idx, k_max=K_eta,
        capture_overhead=1.0, player_id=pid,
    )
    K = int(floor.shape[-1])

    sizes = drain.view(S, 1).expand(S, T).floor()

    active = reachable_mask(
        movement, source_idx=source_idx, target_idx=target_idx,
        fleet_sizes=sizes.unsqueeze(-1), eta_cap=eta_cap,
    ).squeeze(-1)
    aim = intercept_angle(
        movement,
        source_idx.unsqueeze(1),
        target_idx.unsqueeze(0),
        sizes,
        active=active,
    )
    angle = aim["angle"]
    eta = aim["eta"]
    viable = aim["viable"] & (eta <= eta_cap.view(1, T))

    if K > 0:
        k_arr = (eta.clamp(min=1.0, max=float(K)).ceil().long() - 1).clamp(0, K - 1)
        floor_at_arr = floor.unsqueeze(0).expand(S, T, K).gather(-1, k_arr.unsqueeze(-1)).squeeze(-1)
    else:
        floor_at_arr = torch.ones(S, T, dtype=dtype, device=device)
    clears_floor = sizes >= floor_at_arr

    src_neq_tgt = source_idx.view(S, 1) != target_idx.view(1, T)
    valid = (
        viable & clears_floor & (sizes >= 1.0) & src_neq_tgt
        & source_exists.view(S, 1) & target_exists.view(1, T)
    )

    L = 1
    C = S * T
    cand_src = source_idx.view(S, 1).expand(S, T).reshape(C, L)
    cand_tgt_slot = target_idx.view(1, T).expand(S, T).reshape(C)
    cand_tgt_short = torch.arange(T, device=device).view(1, T).expand(S, T).reshape(C)
    cand_send = torch.where(valid, sizes, torch.zeros_like(sizes)).reshape(C, L)
    cand_angle = angle.reshape(C, L)
    cand_eta = torch.where(valid, eta, torch.ones_like(eta)).reshape(C, L)
    cand_active = valid.reshape(C, L)
    cand_valid = valid.reshape(C)
    cand_is_def = target_is_mine[cand_tgt_short]

    launches = make_launch_set(
        source_slots=cand_src,
        target_slots=cand_tgt_slot.unsqueeze(-1).expand(C, L),
        ships=cand_send,
        eta=cand_eta,
        valid=cand_active & cand_valid.unsqueeze(-1),
        player_id=pid,
    )
    score = score_candidates(
        garrison_status, prod=prod, alive_by_step=alive_by_step,
        player_count=int(player_count), launches=launches, player_id=pid,
    )

    # --- FFA leader-attack bonus + production bonus -------------------------
    if int(player_count) >= 4 and (
        float(config.ffa_leader_attack_bonus) > 0.0
        or float(config.ffa_target_prod_bonus) > 0.0
    ):
        owner = obs.owner_abs.to(torch.long)
        owner_valid = (owner >= 0) & (owner < int(player_count)) & obs.alive
        owner_idx = owner.clamp(min=0, max=max(int(player_count) - 1, 0))
        prod_by_owner = torch.zeros(int(player_count), dtype=dtype, device=device)
        ships_by_owner = torch.zeros(int(player_count), dtype=dtype, device=device)
        prod_by_owner.scatter_add_(
            0, owner_idx,
            torch.where(owner_valid, prod.to(dtype), torch.zeros_like(prod.to(dtype)))
        )
        ships_by_owner.scatter_add_(
            0, owner_idx,
            torch.where(owner_valid, obs.ships.to(dtype), torch.zeros_like(obs.ships.to(dtype)))
        )
        strength = prod_by_owner * 3.0 + 0.03 * ships_by_owner
        my_strength = strength[pid].detach()

        target_owner = owner[target_idx.clamp(0, P - 1)].clamp(
            min=0, max=max(int(player_count) - 1, 0)
        )
        target_owned_enemy = (
            target_exists
            & obs.is_enemy[target_idx.clamp(0, P - 1)]
            & (obs.owner_abs[target_idx.clamp(0, P - 1)] >= 0)
        )
        owner_strength = strength[target_owner]
        leader_delta = (owner_strength - my_strength).clamp(min=0.0)
        target_bonus_short = torch.where(
            target_owned_enemy,
            float(config.ffa_leader_attack_bonus) * leader_delta
            + float(config.ffa_target_prod_bonus) * prod[target_idx.clamp(0, P - 1)].to(dtype),
            torch.zeros_like(owner_strength),
        )
        score = score + target_bonus_short[cand_tgt_short]

    # --- Comet attack bonus -------------------------------------------------
    comet_ids_raw = obs_tensors.get("comet_planet_ids", None)
    if comet_ids_raw is not None and float(config.comet_attack_bonus) > 0.0:
        if isinstance(comet_ids_raw, torch.Tensor):
            comet_ids_list = comet_ids_raw.tolist()
        else:
            comet_ids_list = list(comet_ids_raw)

        if len(comet_ids_list) > 0:
            comet_id_set = set(int(x) for x in comet_ids_list)
            all_abs_ids = obs_tensors["planets"].reshape(-1, 7)[:, 0].long()  # [P]
            target_abs_ids = all_abs_ids[target_idx.clamp(0, P - 1)]          # [T]
            target_not_owned = ~obs.owned[target_idx.clamp(0, P - 1)]         # [T]
            comet_flag = torch.tensor(
                [int(target_abs_ids[i].item()) in comet_id_set for i in range(T)],
                dtype=torch.bool, device=device,
            )
            comet_bonus_t = torch.where(
                comet_flag & target_not_owned & target_exists,
                torch.full((T,), float(config.comet_attack_bonus), dtype=dtype, device=device),
                torch.zeros(T, dtype=dtype, device=device),
            )
            score = score + comet_bonus_t[cand_tgt_short]

    # --- High-production target bonus (2P) ----------------------------------
    if float(config.high_prod_attack_bonus) > 0.0:
        target_prod_vals = prod[target_idx.clamp(0, P - 1)].to(dtype)       # [T]
        target_not_owned2 = ~obs.owned[target_idx.clamp(0, P - 1)]          # [T]
        prod_bonus_t = torch.where(
            target_not_owned2 & target_exists,
            float(config.high_prod_attack_bonus) * (target_prod_vals - 2.0).clamp(min=0.0),
            torch.zeros(T, dtype=dtype, device=device),
        )
        score = score + prod_bonus_t[cand_tgt_short]

    score = torch.where(cand_valid, score, torch.full_like(score, float("-inf")))

    wave_entries, leftover = _greedy_select(
        P=P, W=W, device=device, dtype=dtype, score=score,
        cand_src=cand_src, cand_send=cand_send, cand_angle=cand_angle, cand_eta=cand_eta,
        cand_active=cand_active, cand_tgt_slot=cand_tgt_slot, cand_tgt_short=cand_tgt_short,
        cand_is_def=cand_is_def, source_budget=obs.ships.to(dtype).clone(),
        target_exists=target_exists, roi_threshold=float(config.roi_threshold),
    )

    if not bool(config.enable_regroup):
        return wave_entries
    enemy_mass = cheap_enemy_pressure(obs, cache, horizon=float(K_eta), player_id=pid)
    regroup_entries = _plan_regroup(
        movement=movement, obs=obs, obs_tensors=obs_tensors, garrison_status=garrison_status,
        leftover=leftover, original_ships=obs.ships.to(dtype), pressure=enemy_mass,
        config=config, H=H,
    )
    return concat_launch_entries([wave_entries, regroup_entries])


def run_turn(obs_tensors: dict, *, config: ProducerLiteConfig, player_count: int, memory) -> dict:
    """Full per-turn pipeline — v5."""
    device = obs_tensors["planets"].device
    obs = parse_obs(obs_tensors)
    P = obs.P
    if P == 0:
        return empty_action_row(device)

    base_movement_cfg = _movement_config(config, player_count=int(player_count))

    _step_t = obs_tensors.get("step", torch.tensor(0))
    step = int(_step_t.item()) if hasattr(_step_t, "item") else int(_step_t)
    remaining_time = float(obs_tensors.get("remainingOverageTime", 60.0))
    config = _apply_phase_config(config, step)
    config = _apply_time_budget(config, remaining_time)

    movement = ensure_planet_movement(
        obs_tensors=obs_tensors,
        expected_cfg=base_movement_cfg,
        cached_movement=getattr(memory, "movement", None),
    )
    memory.movement = movement
    cache = build_distance_cache(movement, max_k=int(config.horizon))
    H = int(config.horizon)
    status = movement.garrison_status(max_horizon=H)
    alive_by_step = movement.alive_by_step[: H + 1]

    entries = plan_lite_waves(
        movement=movement, obs=obs, obs_tensors=obs_tensors, cache=cache,
        garrison_status=status, prod=movement.planet_prod,
        alive_by_step=alive_by_step, config=config, player_count=int(player_count),
    )
    entries = disambiguate_duplicate_launches(entries)
    launches = infer_planned_launches_from_entries(
        obs_tensors=obs_tensors, movement=movement, entries=entries, player_id=int(obs.player_id),
    )
    apply_private_planned_launches(
        movement=movement, launches=launches, owner_id=int(obs.player_id),
        obs_tensors=obs_tensors,
    )
    planet_ids = obs_tensors["planets"][..., 0].long()
    return entries_to_sparse_payload(entries, planet_ids=planet_ids)


# ---------------------------------------------------------------------------
# Per-mode presets
# ---------------------------------------------------------------------------

CONFIG_4P = dataclasses.replace(
    ProducerLiteConfig(),
    horizon=16,
    max_sources_per_lane=10,
    max_offensive_targets=12,
    max_defensive_targets=4,
    roi_threshold=1.25,
    min_ships_to_launch=3.0,
    max_regroup_time=8.0,
    max_regroup_sources_per_lane=8,
    max_regroup_targets_per_source=12,
    ffa_leader_attack_bonus=0.12,      
    ffa_target_prod_bonus=0.20,        
    comet_attack_bonus=4.0,            
    high_prod_attack_bonus=0.0,        
)


def _config_for(player_count: int) -> ProducerLiteConfig:
    return CONFIG_4P if int(player_count) >= 4 else ProducerLiteConfig()


# ---------------------------------------------------------------------------
# State / runtime
# ---------------------------------------------------------------------------

class ProducerLiteMemory:
    def __init__(self) -> None:
        self.movement = None
        self.cached_player_count: int | None = None
        self.last_sparse_action_row: dict | None = None

    def reset(self) -> None:
        self.movement = None
        self.cached_player_count = None
        self.last_sparse_action_row = None


class ProducerLiteRuntime:
    def __init__(self, memory: ProducerLiteMemory | None = None) -> None:
        self.memory = memory if memory is not None else ProducerLiteMemory()

    def reset(self) -> None:
        self.memory.reset()

    def tensor_action(self, obs_tensors: dict):
        mem = self.memory
        if bool((obs_tensors["step"] == 0).all()):
            mem.cached_player_count = None
        if mem.cached_player_count is None:
            mem.cached_player_count = largest_initial_player_count(obs_tensors)
        config = _config_for(mem.cached_player_count)
        row = run_turn(
            obs_tensors, config=config,
            player_count=int(mem.cached_player_count), memory=mem,
        )
        mem.last_sparse_action_row = row
        return row


_RUNTIME = ProducerLiteRuntime()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def agent(obs):
    """Single-observation entry point for local play and Kaggle."""
    player = obs.get("player", 0) if isinstance(obs, dict) else obs.player
    player_id = int(player)
    obs_tensors = single_obs_to_tensor(obs, player_id=player_id)

    if isinstance(obs, dict):
        comet_ids = obs.get("comet_planet_ids", [])
        obs_tensors["remainingOverageTime"] = float(obs.get("remainingOverageTime", 60.0))
    else:
        comet_ids = list(getattr(obs, "comet_planet_ids", []))
        obs_tensors["remainingOverageTime"] = float(getattr(obs, "remainingOverageTime", 60.0))

    # Fix: Convert to a PyTorch tensor so orbit_lite can safely call `.to(device=device)`
    obs_tensors["comet_planet_ids"] = torch.tensor(comet_ids, dtype=torch.long)

    with torch.no_grad():
        sparse_row = _RUNTIME.tensor_action(obs_tensors)
    return sparse_action_row_to_moves(sparse_row, obs, player_id=player_id)