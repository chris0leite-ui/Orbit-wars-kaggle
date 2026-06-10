
from __future__ import annotations

import dataclasses
import os
import sys
from dataclasses import dataclass

# Make the sibling ``orbit_lite`` package importable wherever this file runs:
# loaded in place, dropped at a submission-archive root, or exec'd by
# kaggle_environments with no ``__file__`` (fall back to the working dir).
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
from orbit_lite.garrison_launch import LaunchSet
from orbit_lite.opp_projection import predict_opp_launches_via_mirror, MAX_L_OPP
from orbit_lite.recapture import recapture_penalty
from orbit_lite.strategic_value import denial_bonus, opening_bonus
from orbit_lite.planner_core import (
    _candidate_indices,
    _empty_entries,
    _greedy_select,
    _plan_regroup,
    _stable_topk_indices,
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


# Adaptive candidate-arrival horizon K_eta — ported from champion's
# capture_horizon_k (agents/baseline/launch_rules.py). Default OFF
# preserves bit-identical behaviour vs the untouched producer.
# Clamped to H so capture_floor lookups stay inside garrison_status.
def _adaptive_k_enabled() -> bool:
    return os.environ.get("PRODUCER_PLUS_ADAPTIVE_K", "0").strip().lower() in (
        "1", "true", "yes", "on",
    )


# Multi-size enumeration per (source, target): emit three ships variants —
# (capture_floor, 2 × capture_floor, safe_drain) — instead of a single
# safe_drain candidate. Default OFF preserves bit-identical single-size
# behaviour. State/MIGRATION_PLAN.md Step 4.
def _multi_size_enabled() -> bool:
    return os.environ.get("PRODUCER_PLUS_MULTI_SIZE", "0").strip().lower() in (
        "1", "true", "yes", "on",
    )


# Multi-source coalitions: in addition to single-source candidates, emit
# L=2 pairs that combine two source planets on the same target with
# (near-)same arrival tick. Producer's planner already handles L>1
# end-to-end; this step fills the unused L axis. Default OFF preserves
# bit-identical single-source behaviour. state/MIGRATION_PLAN.md Step 5.
def _coalitions_enabled() -> bool:
    return os.environ.get("PRODUCER_PLUS_COALITIONS", "0").strip().lower() in (
        "1", "true", "yes", "on",
    )


# Opponent multi-launch projection: once per turn, project the opponents'
# next 8 ticks of launches and inject them as background LaunchSet slots
# into the per-candidate scorer. The scorer's `sparse_launch_flow_delta`
# natively handles mixed-owner LaunchSets via per-launch `owner`, so
# every candidate is now scored against "do my action AND opp does their
# projected actions" rather than "do my action while opp does nothing".
# Default OFF preserves bit-identical static-opp scoring. Migration plan
# Step 3 (redux). See orbit_lite/opp_projection.py.
def _opp_projection_enabled() -> bool:
    return os.environ.get("PRODUCER_PLUS_OPP_PROJECTION", "0").strip().lower() in (
        "1", "true", "yes", "on",
    )


# Multi-tick opp projection: instead of projecting opp's launches at the
# current tick only, run opp's planner K successive rounds (game-ticks
# 0, 1, ..., K-1) with the cumulative previously-projected opp launches
# passed as ``background`` each round. Each round's launches are
# eta-shifted by +k turns before merging. Default 0/1 preserves the
# single-pass byte-identical behaviour. Player-count-suffixed knobs
# override the common one; 3P games fall back to the _2P suffix (3P is
# untested terrain on this comp; tune via the base var if a 3P-specific
# value is needed). NOTE: multi-tick is silently a no-op when opp_proj
# is OFF (the value is read only inside the opp_proj-gated branch in
# run_turn). Set PRODUCER_PLUS_OPP_PROJECTION=1 to activate. See
# knowledge-base/thoughts/2026-06-05-cycle-stalemate-and-horizon-
# scaling.md for the structural-defect diagnosis.
def _multi_tick_opp_k(player_count: int) -> int:
    suffix = "_4P" if int(player_count) >= 4 else "_2P"
    raw = os.environ.get(f"PRODUCER_PLUS_MULTI_TICK_OPP_K{suffix}")
    if raw is None:
        raw = os.environ.get("PRODUCER_PLUS_MULTI_TICK_OPP_K", "0")
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        return 0


# Recapture penalty: per-candidate leaf-scorer discount for thin captures
# the opponent can plausibly recapture. Composes additively with
# competitive_score; the term is in ship units so the weight is a pure
# multiplier. Multi-tick opp_proj already debits for opp launches inside
# its projection window; to avoid double-counting we clip the recapture
# window via K_recap_eff = max(1, K_recap - K_opp). Default OFF preserves
# byte-identical static behaviour. See knowledge-base/thoughts/
# 2026-06-05-cycle-stalemate-and-horizon-scaling.md for motivation.
def _recapture_penalty_enabled() -> bool:
    return os.environ.get("PRODUCER_PLUS_RECAPTURE_PENALTY", "0").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _recapture_penalty_weight() -> float:
    raw = os.environ.get("PRODUCER_PLUS_RECAPTURE_PENALTY_WEIGHT", "1.0")
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        return 1.0


def _recapture_k(player_count: int) -> int:
    suffix = "_4P" if int(player_count) >= 4 else "_2P"
    raw = os.environ.get(f"PRODUCER_PLUS_RECAPTURE_K{suffix}")
    if raw is None:
        raw = os.environ.get("PRODUCER_PLUS_RECAPTURE_K", "8")
    try:
        return max(1, int(raw))
    except (TypeError, ValueError):
        return 8


def _recapture_safety_reserve() -> float:
    raw = os.environ.get("PRODUCER_PLUS_RECAPTURE_SAFETY_RESERVE", "0.5")
    try:
        return min(1.0, max(0.0, float(raw)))
    except (TypeError, ValueError):
        return 0.5


# Strategic-value bonuses: per-candidate scorer credits for the
# production we'd accrue past the scorer horizon (H=18 in 2P / 13 in 4P).
#
# Two opt-in mechanisms, each in ship units so the weight is a pure
# multiplier:
#   denial_bonus  — captures of targets the opponent values (currently
#                   owns OR predicted to attack via opp_proj's
#                   background LaunchSet). Encodes "block the opponent's
#                   biggest bet."  Opp-aware: depends on opp_proj.
#   opening_bonus — captures during the early-game phase, linearly
#                   decaying to zero at ``opening_window``. Opp-agnostic.
# Both default OFF preserves byte-identical static behaviour. Share the
# game-length estimate knob (``PRODUCER_PLUS_GAME_LENGTH_EST``).
def _denial_bonus_enabled() -> bool:
    return os.environ.get("PRODUCER_PLUS_DENIAL_BONUS", "0").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _denial_bonus_weight() -> float:
    raw = os.environ.get("PRODUCER_PLUS_DENIAL_WEIGHT", "0.1")
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        return 0.1


def _opening_bonus_enabled() -> bool:
    return os.environ.get("PRODUCER_PLUS_OPENING_BONUS", "0").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _opening_bonus_weight() -> float:
    raw = os.environ.get("PRODUCER_PLUS_OPENING_WEIGHT", "0.1")
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        return 0.1


def _opening_window() -> int:
    raw = os.environ.get("PRODUCER_PLUS_OPENING_WINDOW", "30")
    try:
        return max(1, int(raw))
    except (TypeError, ValueError):
        return 30


def _game_length_est() -> int:
    raw = os.environ.get("PRODUCER_PLUS_GAME_LENGTH_EST", "200")
    try:
        return max(1, int(raw))
    except (TypeError, ValueError):
        return 200


# Force-concentration: relax the target mutex inside _greedy_select so up to
# ``max_waves_per_target`` waves can land on the same target per turn. Between
# waves we re-score candidates with the just-fired wave appended to the
# scoring LaunchSet (owner=pid) so wave 2 sees wave 1's capture/reinforcement
# and does NOT double-count. Default OFF preserves byte-identical single-wave
# behaviour (no rescore closure built, max_waves_per_target=1 passed through).
# See knowledge-base for the architectural diagnosis: scorer tuning can never
# reach candidates the chooser refuses to enumerate.
def _force_concentration_enabled() -> bool:
    return os.environ.get("PRODUCER_PLUS_FORCE_CONCENTRATION", "0").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _force_concentration_max_waves() -> int:
    raw = os.environ.get("PRODUCER_PLUS_FORCE_CONCENTRATION_MAX_WAVES", "2")
    try:
        return max(1, int(raw))
    except (TypeError, ValueError):
        return 2


# --- FFA-aware competitive score (4P objective fix) -------------------------
# Live-replay diagnosis (knowledge-base 2026-06-10): 82 of 83 4P losses end
# with us ELIMINATED, carved by 2+ opponents mid-game, while the legacy score
# (my delta minus the SUM of all opponents' deltas) scores mutual-damage
# trades positive — it optimizes total damage dealt, which in a 4-player
# free-for-all leaves both fighters weaker relative to the bystanders. The
# fix weights each opponent's delta by their strength share (weights sum to
# 1), so trades are valued by how much they shift my standing against the
# rivals that actually threaten me. 2P is byte-identical: weights are only
# built when player_count >= 3.
def _ffa_score_enabled() -> bool:
    return os.environ.get("PRODUCER_PLUS_FFA_SCORE", "0").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _ffa_weight_mode() -> str:
    """``strength`` (default): weights ∝ rival planet+fleet ships.
    ``uniform``: equal weight per living rival — tests whether the
    trade-devaluation alone helps without the hit-the-leader tilt."""
    raw = os.environ.get("PRODUCER_PLUS_FFA_WEIGHTS", "strength").strip().lower()
    return raw if raw in ("strength", "uniform") else "strength"


def _ffa_opp_weights(obs_tensors: dict, *, player_id: int, player_count: int):
    """Per-opponent weights ∝ current total strength (planet + fleet ships),
    or equal-per-living-rival under ``PRODUCER_PLUS_FFA_WEIGHTS=uniform``.

    Returns a ``[player_count]`` float tensor with 0 at ``player_id``,
    summing to 1 over living opponents (all-zero if every opponent is dead).
    """
    planets = obs_tensors["planets"]            # [P, 7]: owner=col1, ships=col5
    device = planets.device
    a = int(player_count)
    strength = torch.zeros(a, dtype=planets.dtype, device=device)
    p_owner = planets[:, 1].long()
    p_mask = (planets[:, 0] >= 0) & (p_owner >= 0) & (p_owner < a)
    if bool(p_mask.any()):
        strength.scatter_add_(0, p_owner[p_mask], planets[p_mask, 5])
    fleets = obs_tensors.get("fleets")
    if fleets is not None and fleets.numel():
        f_owner = fleets[:, 1].long()           # [F, 7]: owner=col1, ships=col6
        f_mask = (fleets[:, 0] >= 0) & (f_owner >= 0) & (f_owner < a)
        if bool(f_mask.any()):
            strength.scatter_add_(0, f_owner[f_mask], fleets[f_mask, 6])
    strength[int(player_id)] = 0.0
    if _ffa_weight_mode() == "uniform":
        strength = (strength > 0).to(planets.dtype)
    total = float(strength.sum())
    if total <= 0.0:
        return torch.zeros(a, dtype=planets.dtype, device=device)
    return strength / total


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def compute_k_eta_for_step(step: int, *, H: int) -> int:
    H_int = max(1, int(H))
    if not _adaptive_k_enabled():
        return H_int
    floor = _env_int("PRODUCER_PLUS_ADAPTIVE_K_FLOOR", 10)
    k_open = _env_int("PRODUCER_PLUS_ADAPTIVE_K_OPEN", 20)
    t_settle = _env_int("PRODUCER_PLUS_ADAPTIVE_K_TSETTLE", 30)
    floor = max(1, floor)
    if t_settle <= 0 or int(step) >= t_settle or k_open <= floor:
        decayed = floor
    else:
        raw = k_open - (k_open - floor) * int(step) / float(t_settle)
        decayed = max(floor, int(round(raw)))
    return max(1, min(H_int, decayed))


@dataclass(frozen=True)
class ProducerLiteConfig:
    """Behaviour knobs.  """

    
    # the projection window, the movement build length, AND the target ETA cap 
    horizon: int = 18
    # --- shortlists ------------------------------------------------------
    max_sources_per_lane: int = 12
    max_offensive_targets: int = 12         # enemy/neutral proximity targets
    max_defensive_targets: int = 4          
    # --- scoring / greedy ------------------------------------------------
    max_waves_per_turn: int = 6
    roi_threshold: float = 1.5              # fire if score > this
    min_ships_to_launch: float = 4.0
    # --- regroup  ------------------------------
    enable_regroup: bool = True
    max_regroup_time: float = 7.0
    regroup_pressure_delta_min: float = 0.25
    max_regroup_sources_per_lane: int = 6
    max_regroup_targets_per_source: int = 7
    regroup_pressure_norm: str = "none"
    regroup_time_penalty_weight: float = 1e-3


def _movement_config(config: ProducerLiteConfig, *, player_count: int) -> MovementConfig:
    """MovementConfig: fleet tracking on, horizon = config.horizon."""
    return MovementConfig(
        movement_horizon=int(config.horizon),
        drift_epsilon=1e-3,
        track_fleets=True,
        player_count=int(player_count),
        max_tracked_fleets=128,
    )


def cheap_enemy_pressure(obs, cache, *, horizon: float, player_id: int) -> Tensor:
    """Cheap reachable-enemy-mass proxy per planet — ``[P]``.

    Consumed only as the **regroup gradient** (rank owned planets by how stressed
    they are, move ships up the gradient). For each planet ``t``, sums a
    distance-decayed share of every enemy source's **current** garrison that could
    straight-line reach ``t`` within ``horizon`` turns, using the step-0 centre
    distance ``cross_dist[0]``. The decay ``(1 - d/(speed·H))₊`` weights nearer
    enemies more, giving a graded frontline signal in ship-mass units.

    Approximations: ignores target orbital drift over the horizon, production
    accrued in flight, the per-owner split, and in-flight enemy fleets. Pure
    arithmetic on cached tensors
    """
    P = int(obs.P)
    device = obs.device
    dtype = obs.ships.dtype
    if P == 0:
        return torch.zeros(P, dtype=dtype, device=device)
    d0 = cache.cross_dist[0].to(dtype)                                   # [src, tgt] current centre dist
    ships = obs.ships.to(dtype)
    speeds = fleet_speed(ships.clamp(min=1e-6))                          # [P]
    reach_dist = (speeds.view(P, 1) * float(horizon)).clamp(min=1e-6)    # [src, 1]
    enemy = obs.alive & (obs.owner_abs >= 0) & (obs.owner_abs != int(player_id))  # [P]
    eye = torch.eye(P, device=device, dtype=torch.bool)
    valid = enemy.view(P, 1) & obs.alive.view(1, P) & ~eye              # [src, tgt]
    decay = (1.0 - d0 / reach_dist).clamp(min=0.0)                       # nearer enemy -> heavier
    contrib = torch.where(valid, ships.view(P, 1) * decay, torch.zeros_like(decay))
    return contrib.sum(dim=0)                                            # [P] summed over sources


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
    K_eta_override: int | None = None,
    background: LaunchSet | None = None,
    force_concentration: bool | None = None,
    opp_weights: Tensor | None = None,
):
    """Single-size, single-source attack planner + regroup.

    Builds exactly one candidate per ``(source, target)`` shortlist pair — fleet
    size = the source's max garrison launch (``safe_drain``) — scores them with the
    exact competitive flow diff, and greedily fires the best wave per target up to
    ``max_waves_per_turn``. Returns the combined ``LaunchEntries`` (attack waves ++
    regroup).
    """
    P = obs.P
    device = obs.device
    dtype = obs.ships.dtype
    pid = int(obs.player_id)

    H_axis = int(garrison_status.ships.shape[-1])
    H = max(H_axis - 1, 0)
    K_eta_raw = int(K_eta_override) if K_eta_override is not None else int(config.horizon)
    K_eta = max(1, min(K_eta_raw, H))
    W = max(1, int(config.max_waves_per_turn))

    source_mask = obs.owned & obs.alive & (obs.ships >= float(config.min_ships_to_launch))
    if not bool(source_mask.any()):
        return _empty_entries(device, dtype)

    S_cap = max(1, min(int(config.max_sources_per_lane), P))
    source_idx, source_exists = _candidate_indices(obs.ships, source_mask, S_cap)
    target_idx, target_exists = build_target_shortlist(
        obs, obs_tensors, garrison_status, cache,
        config=config, K_eta=K_eta, H=H, prod=prod, source_mask=source_mask,
        background=background,
    )
    if not bool(target_exists.any()):
        return _empty_entries(device, dtype)
    S = int(source_idx.shape[0])
    T = int(target_idx.shape[0])
    target_is_mine = obs.owned[target_idx.clamp(0, P - 1)]                       # [T]

    source_ships = obs.ships[source_idx.clamp(0, P - 1)].to(dtype)                # [S]
    H_eff = torch.full((), float(H), dtype=dtype, device=device)
    drain = safe_drain(
        garrison_status, source_idx=source_idx, source_ships=source_ships,
        H_eff=H_eff, player_id=pid,
    )                                                                            # [S]

    # Uniform reach cap = K_eta (= horizon).
    eta_cap = torch.full((T,), float(K_eta), dtype=dtype, device=device)          # [T]

    floor = capture_floor(
        garrison_status, target_idx=target_idx, k_max=K_eta,
        capture_overhead=1.0, player_id=pid,
    )                                                                            # [T, K]
    K = int(floor.shape[-1])

    src_neq_tgt = source_idx.view(S, 1) != target_idx.view(1, T)

    if _multi_size_enabled() and _coalitions_enabled():
        # --- Step 4 + Step 5 composed: multi-size single-source + L=2 coalitions
        # Single-source: 3 size variants per (s, t), padded to L=2 with slot 1
        # inactive. Coalitions: safe_drain per contributor (no size variation
        # on the coalition side — that's the controlled compose), L=2.
        # C_total = S*T*N + T*C(K_src, 2). Target mutex blocks losing
        # variants/coalitions once any candidate wins a target.
        L = 2
        N = 3

        # ===== Stage A: Step 4 multi-size variants =====
        sizes_hi = drain.view(S, 1).expand(S, T).floor()                          # [S, T]
        active_hi = reachable_mask(
            movement, source_idx=source_idx, target_idx=target_idx,
            fleet_sizes=sizes_hi.unsqueeze(-1), eta_cap=eta_cap,
        ).squeeze(-1)
        aim_hi = intercept_angle(
            movement,
            source_idx.unsqueeze(1), target_idx.unsqueeze(0),
            sizes_hi, active=active_hi,
        )
        eta_hi = aim_hi["eta"]                                                    # [S, T]
        if K > 0:
            k_arr_hi = (eta_hi.clamp(min=1.0, max=float(K)).ceil().long() - 1).clamp(0, K - 1)
            floor_at_arr_hi = (
                floor.unsqueeze(0).expand(S, T, K).gather(-1, k_arr_hi.unsqueeze(-1)).squeeze(-1)
            )
        else:
            floor_at_arr_hi = torch.ones(S, T, dtype=dtype, device=device)

        sizes_lo = torch.minimum(floor_at_arr_hi.ceil().clamp(min=1.0), sizes_hi)
        sizes_mid = torch.minimum(2.0 * sizes_lo, sizes_hi)
        sizes_3 = torch.stack([sizes_lo, sizes_mid, sizes_hi], dim=-1)            # [S, T, N]

        active_3 = reachable_mask(
            movement, source_idx=source_idx, target_idx=target_idx,
            fleet_sizes=sizes_3, eta_cap=eta_cap,
        )
        aim_3 = intercept_angle(
            movement,
            source_idx.view(S, 1, 1), target_idx.view(1, T, 1),
            sizes_3, active=active_3,
        )
        angle_3 = aim_3["angle"]                                                  # [S, T, N]
        eta_3 = aim_3["eta"]                                                      # [S, T, N]
        viable_3 = aim_3["viable"] & (eta_3 <= eta_cap.view(1, T, 1))

        if K > 0:
            k_arr_3 = (eta_3.clamp(min=1.0, max=float(K)).ceil().long() - 1).clamp(0, K - 1)
            floor_at_arr_3 = (
                floor.view(1, T, 1, K).expand(S, T, N, K)
                .gather(-1, k_arr_3.unsqueeze(-1)).squeeze(-1)
            )
        else:
            k_arr_3 = torch.zeros(S, T, N, dtype=torch.long, device=device)
            floor_at_arr_3 = torch.ones(S, T, N, dtype=dtype, device=device)
        clears_floor_3 = sizes_3 >= floor_at_arr_3
        ships_ok_3 = sizes_3 <= source_ships.view(S, 1, 1)
        # Leg-level viability WITHOUT the floor gate (for Fix-A coalitions).
        viable_only_3 = (
            viable_3 & (sizes_3 >= 1.0) & ships_ok_3
            & src_neq_tgt.unsqueeze(-1)
            & source_exists.view(S, 1, 1) & target_exists.view(1, T, 1)
        )                                                                         # [S, T, N]
        valid_3 = viable_only_3 & clears_floor_3                                  # [S, T, N]

        # Pack multi-size single-source into [C_ms, L=2] padded.
        C_ms = S * T * N
        ms_src_planet = source_idx.view(S, 1, 1).expand(S, T, N).reshape(C_ms)
        ms_src_padded = torch.stack([ms_src_planet, ms_src_planet], dim=-1)       # [C_ms, 2]
        ms_send_flat = torch.where(valid_3, sizes_3, torch.zeros_like(sizes_3)).reshape(C_ms)
        ms_send_padded = torch.stack(
            [ms_send_flat, torch.zeros_like(ms_send_flat)], dim=-1)
        ms_angle_padded = torch.stack(
            [angle_3.reshape(C_ms), torch.zeros_like(ms_send_flat)], dim=-1)
        ms_eta_flat = torch.where(valid_3, eta_3, torch.ones_like(eta_3)).reshape(C_ms)
        ms_eta_padded = torch.stack([ms_eta_flat, torch.ones_like(ms_eta_flat)], dim=-1)
        ms_valid_flat = valid_3.reshape(C_ms)                                     # [C_ms]
        ms_active = torch.stack(
            [ms_valid_flat, torch.zeros_like(ms_valid_flat)], dim=-1)
        ms_tgt_slot = target_idx.view(1, T, 1).expand(S, T, N).reshape(C_ms)
        ms_tgt_short = torch.arange(T, device=device).view(1, T, 1).expand(S, T, N).reshape(C_ms)

        # ===== Stage B: Step 5 coalitions (safe_drain per contributor) =====
        # Source ranking + per-pair aim use the safe_drain variant (sizes_hi).
        # Use viable_only_base (not valid_base) for ranking so sources that
        # CAN'T clear floor alone still enter the coalition pool — those are
        # the ones a Fix-A coalition actually helps.
        viable_only_base = viable_only_3[..., -1]                                 # [S, T]
        clears_floor_base = clears_floor_3[..., -1]                               # [S, T]
        eta_base = eta_3[..., -1]                                                 # [S, T]
        angle_base = angle_3[..., -1]                                             # [S, T]
        k_arr_base = k_arr_3[..., -1]                                             # [S, T]
        floor_at_arr_base = floor_at_arr_3[..., -1]                               # [S, T]

        K_src = min(_env_int("PRODUCER_PLUS_COALITION_K", 6), int(S))
        K_src = max(0, K_src)

        if S >= 2 and K_src >= 2:
            ranked_per_tgt = torch.where(
                viable_only_base, -eta_base, torch.full_like(eta_base, float("-inf"))
            ).transpose(0, 1)
            top_src_per_tgt = _stable_topk_indices(ranked_per_tgt, K_src)
            pair_idx = torch.triu_indices(K_src, K_src, offset=1, device=device)
            pair_a = pair_idx[0]
            pair_b = pair_idx[1]
            P_pairs = int(pair_a.numel())
        else:
            P_pairs = 0

        if P_pairs > 0:
            T_idx_pair = torch.arange(T, device=device).view(T, 1).expand(T, P_pairs)
            sA = top_src_per_tgt[:, pair_a]
            sB = top_src_per_tgt[:, pair_b]
            sizesA = sizes_hi[sA, T_idx_pair]
            sizesB = sizes_hi[sB, T_idx_pair]
            etaA = eta_base[sA, T_idx_pair]
            etaB = eta_base[sB, T_idx_pair]
            angleA = angle_base[sA, T_idx_pair]
            angleB = angle_base[sB, T_idx_pair]
            viableA = viable_only_base[sA, T_idx_pair]
            viableB = viable_only_base[sB, T_idx_pair]
            clearsA = clears_floor_base[sA, T_idx_pair]
            clearsB = clears_floor_base[sB, T_idx_pair]
            k_arr_A = k_arr_base[sA, T_idx_pair]
            k_arr_B = k_arr_base[sB, T_idx_pair]
            floor_joint = floor_at_arr_base[sA, T_idx_pair]   # == [sB] when k_arr_A==k_arr_B

            # Fix-A gate: coalitions only fire on targets where NEITHER source
            # clears floor alone, but their joint same-tick arrival does.
            eta_strict = k_arr_A == k_arr_B
            neither_alone = ~clearsA & ~clearsB
            joint_clears = (sizesA + sizesB) >= floor_joint
            distinct_src = sA != sB
            valid_pair = (
                viableA & viableB & neither_alone & joint_clears
                & eta_strict & distinct_src
            )                                                                    # [T, P_pairs]

            C_coal = T * P_pairs
            coal_src = torch.stack(
                [source_idx[sA].reshape(C_coal), source_idx[sB].reshape(C_coal)],
                dim=-1,
            )
            sendA = torch.where(valid_pair, sizesA, torch.zeros_like(sizesA))
            sendB = torch.where(valid_pair, sizesB, torch.zeros_like(sizesB))
            coal_send = torch.stack(
                [sendA.reshape(C_coal), sendB.reshape(C_coal)], dim=-1)
            coal_angle = torch.stack(
                [angleA.reshape(C_coal), angleB.reshape(C_coal)], dim=-1)
            etaA_safe = torch.where(valid_pair, etaA, torch.ones_like(etaA))
            etaB_safe = torch.where(valid_pair, etaB, torch.ones_like(etaB))
            coal_eta = torch.stack(
                [etaA_safe.reshape(C_coal), etaB_safe.reshape(C_coal)], dim=-1)
            coal_active = torch.stack(
                [valid_pair.reshape(C_coal), valid_pair.reshape(C_coal)], dim=-1)
            coal_tgt_short = torch.arange(T, device=device).view(T, 1).expand(T, P_pairs).reshape(C_coal)
            coal_tgt_slot = target_idx[coal_tgt_short]
            coal_valid = valid_pair.reshape(C_coal)

            cand_src = torch.cat([ms_src_padded, coal_src], dim=0)
            cand_send = torch.cat([ms_send_padded, coal_send], dim=0)
            cand_angle = torch.cat([ms_angle_padded, coal_angle], dim=0)
            cand_eta = torch.cat([ms_eta_padded, coal_eta], dim=0)
            cand_active = torch.cat([ms_active, coal_active], dim=0)
            cand_tgt_slot = torch.cat([ms_tgt_slot, coal_tgt_slot], dim=0)
            cand_tgt_short = torch.cat([ms_tgt_short, coal_tgt_short], dim=0)
            cand_valid = torch.cat([ms_valid_flat, coal_valid], dim=0)
        else:
            cand_src = ms_src_padded
            cand_send = ms_send_padded
            cand_angle = ms_angle_padded
            cand_eta = ms_eta_padded
            cand_active = ms_active
            cand_tgt_slot = ms_tgt_slot
            cand_tgt_short = ms_tgt_short
            cand_valid = ms_valid_flat

        cand_is_def = target_is_mine[cand_tgt_short]
        C = int(cand_src.shape[0])
    elif _coalitions_enabled():
        # --- Step 5: single-size base + L=2 multi-source coalitions ------------
        # Stage 1 — per-(s, t) single-size base (mirrors the else: branch).
        sizes = drain.view(S, 1).expand(S, T).floor()                            # [S, T]
        active = reachable_mask(
            movement, source_idx=source_idx, target_idx=target_idx,
            fleet_sizes=sizes.unsqueeze(-1), eta_cap=eta_cap,
        ).squeeze(-1)                                                            # [S, T]
        aim = intercept_angle(
            movement,
            source_idx.unsqueeze(1),
            target_idx.unsqueeze(0),
            sizes,
            active=active,
        )
        angle = aim["angle"]                                                     # [S, T]
        eta = aim["eta"]                                                         # [S, T]
        viable = aim["viable"] & (eta <= eta_cap.view(1, T))
        if K > 0:
            k_arr = (eta.clamp(min=1.0, max=float(K)).ceil().long() - 1).clamp(0, K - 1)
            floor_at_arr = (
                floor.unsqueeze(0).expand(S, T, K).gather(-1, k_arr.unsqueeze(-1)).squeeze(-1)
            )
        else:
            k_arr = torch.zeros(S, T, dtype=torch.long, device=device)
            floor_at_arr = torch.ones(S, T, dtype=dtype, device=device)
        clears_floor = sizes >= floor_at_arr
        # Leg-level viability WITHOUT the floor gate. Used by Fix-A coalitions:
        # a coalition needs both legs to be physically viable (aim works, src
        # exists, src != tgt, ships >= 1) but is NOT required to clear floor
        # alone — that's the whole point of overlapping fleets.
        viable_only = (
            viable & (sizes >= 1.0) & src_neq_tgt
            & source_exists.view(S, 1) & target_exists.view(1, T)
        )                                                                        # [S, T]
        valid = viable_only & clears_floor                                       # [S, T]

        L = 2
        C_base = S * T

        # Stage 2 — top-K_src viable sources per target, ranked by -eta (fast
        # arrivers first). Use `viable_only` (not `valid`) so sources that
        # CAN'T clear floor alone still enter the coalition pool — those are
        # the ones a coalition actually helps.
        K_src = min(_env_int("PRODUCER_PLUS_COALITION_K", 6), int(S))
        K_src = max(0, K_src)
        if S >= 2 and K_src >= 2:
            ranked_per_tgt = torch.where(
                viable_only, -eta, torch.full_like(eta, float("-inf"))
            ).transpose(0, 1)                                                    # [T, S]
            top_src_per_tgt = _stable_topk_indices(ranked_per_tgt, K_src)        # [T, K_src]

            # Stage 3 — enumerate (a < b) pairs across the K_src pool.
            pair_idx = torch.triu_indices(K_src, K_src, offset=1, device=device) # [2, P_pairs]
            pair_a = pair_idx[0]
            pair_b = pair_idx[1]
            P_pairs = int(pair_a.numel())
        else:
            P_pairs = 0

        if P_pairs > 0:
            T_idx = torch.arange(T, device=device).view(T, 1).expand(T, P_pairs)
            sA = top_src_per_tgt[:, pair_a]                                      # [T, P_pairs]
            sB = top_src_per_tgt[:, pair_b]
            sizesA = sizes[sA, T_idx]                                            # [T, P_pairs]
            sizesB = sizes[sB, T_idx]
            etaA = eta[sA, T_idx]
            etaB = eta[sB, T_idx]
            angleA = angle[sA, T_idx]
            angleB = angle[sB, T_idx]
            viableA = viable_only[sA, T_idx]
            viableB = viable_only[sB, T_idx]
            clearsA = clears_floor[sA, T_idx]
            clearsB = clears_floor[sB, T_idx]
            k_arr_A = k_arr[sA, T_idx]
            k_arr_B = k_arr[sB, T_idx]
            floor_joint = floor_at_arr[sA, T_idx]   # == floor_at_arr[sB, T_idx] when k_arr_A == k_arr_B

            # Stage 4 — Fix-A gate: coalitions only cover targets where NEITHER
            # single source can clear floor alone, but the combined same-tick
            # arrival can. This makes coalitions a true superset extension
            # rather than overlap with single-source captures.
            eta_strict = k_arr_A == k_arr_B
            neither_alone = ~clearsA & ~clearsB
            joint_clears = (sizesA + sizesB) >= floor_joint
            distinct_src = sA != sB
            valid_pair = (
                viableA & viableB & neither_alone & joint_clears
                & eta_strict & distinct_src
            )                                                                    # [T, P_pairs]

        # Stage 5 — pack [C_total, L=2]. Single-source candidates pad slot 1
        # with active=False; greedy's `~cand_active` short-circuit + the
        # send=0 mask make padded slots no-op.
        base_src_planet = source_idx.view(S, 1).expand(S, T).reshape(C_base)
        base_src_padded = torch.stack(
            [base_src_planet, base_src_planet], dim=-1)                          # [C_base, 2]
        base_send_flat = torch.where(valid, sizes, torch.zeros_like(sizes)).reshape(C_base)
        base_send_padded = torch.stack(
            [base_send_flat, torch.zeros_like(base_send_flat)], dim=-1)
        base_angle_padded = torch.stack(
            [angle.reshape(C_base), torch.zeros_like(base_send_flat)], dim=-1)
        base_eta_flat = torch.where(valid, eta, torch.ones_like(eta)).reshape(C_base)
        base_eta_padded = torch.stack(
            [base_eta_flat, torch.ones_like(base_eta_flat)], dim=-1)
        base_valid_flat = valid.reshape(C_base)                                  # [C_base]
        base_active = torch.stack(
            [base_valid_flat, torch.zeros_like(base_valid_flat)], dim=-1)
        base_tgt_slot = target_idx.view(1, T).expand(S, T).reshape(C_base)
        base_tgt_short = torch.arange(T, device=device).view(1, T).expand(S, T).reshape(C_base)

        if P_pairs > 0:
            C_coal = T * P_pairs
            # source_idx maps S-axis index → planet slot.
            coal_src = torch.stack(
                [source_idx[sA].reshape(C_coal), source_idx[sB].reshape(C_coal)],
                dim=-1,
            )                                                                    # [C_coal, 2]
            sendA = torch.where(valid_pair, sizesA, torch.zeros_like(sizesA))
            sendB = torch.where(valid_pair, sizesB, torch.zeros_like(sizesB))
            coal_send = torch.stack(
                [sendA.reshape(C_coal), sendB.reshape(C_coal)], dim=-1)
            coal_angle = torch.stack(
                [angleA.reshape(C_coal), angleB.reshape(C_coal)], dim=-1)
            etaA_safe = torch.where(valid_pair, etaA, torch.ones_like(etaA))
            etaB_safe = torch.where(valid_pair, etaB, torch.ones_like(etaB))
            coal_eta = torch.stack(
                [etaA_safe.reshape(C_coal), etaB_safe.reshape(C_coal)], dim=-1)
            coal_active = torch.stack(
                [valid_pair.reshape(C_coal), valid_pair.reshape(C_coal)], dim=-1)
            coal_tgt_short = torch.arange(T, device=device).view(T, 1).expand(T, P_pairs).reshape(C_coal)
            coal_tgt_slot = target_idx[coal_tgt_short]
            coal_valid = valid_pair.reshape(C_coal)

            cand_src = torch.cat([base_src_padded, coal_src], dim=0)
            cand_send = torch.cat([base_send_padded, coal_send], dim=0)
            cand_angle = torch.cat([base_angle_padded, coal_angle], dim=0)
            cand_eta = torch.cat([base_eta_padded, coal_eta], dim=0)
            cand_active = torch.cat([base_active, coal_active], dim=0)
            cand_tgt_slot = torch.cat([base_tgt_slot, coal_tgt_slot], dim=0)
            cand_tgt_short = torch.cat([base_tgt_short, coal_tgt_short], dim=0)
            cand_valid = torch.cat([base_valid_flat, coal_valid], dim=0)
        else:
            cand_src = base_src_padded
            cand_send = base_send_padded
            cand_angle = base_angle_padded
            cand_eta = base_eta_padded
            cand_active = base_active
            cand_tgt_slot = base_tgt_slot
            cand_tgt_short = base_tgt_short
            cand_valid = base_valid_flat

        cand_is_def = target_is_mine[cand_tgt_short]
        C = int(cand_src.shape[0])
    elif _multi_size_enabled():
        # --- Three fleet sizes per (source, target): capture_floor, 2× floor,
        # safe_drain. Each variant gets its own aim (fleet speed depends on
        # ships, so eta differs). Packed as [C=S*T*N, L=1] so greedy's target
        # mutex blocks losing variants once one wins the wave.
        N = 3
        # Step 1 — compute the largest variant (safe_drain) first and use its
        # eta to read the capture floor; size_lo = floor at that eta.
        sizes_hi = drain.view(S, 1).expand(S, T).floor()                          # [S, T]
        active_hi = reachable_mask(
            movement, source_idx=source_idx, target_idx=target_idx,
            fleet_sizes=sizes_hi.unsqueeze(-1), eta_cap=eta_cap,
        ).squeeze(-1)                                                             # [S, T]
        aim_hi = intercept_angle(
            movement,
            source_idx.unsqueeze(1), target_idx.unsqueeze(0),
            sizes_hi, active=active_hi,
        )
        eta_hi = aim_hi["eta"]                                                    # [S, T]
        if K > 0:
            k_arr_hi = (eta_hi.clamp(min=1.0, max=float(K)).ceil().long() - 1).clamp(0, K - 1)
            floor_at_arr_hi = (
                floor.unsqueeze(0).expand(S, T, K).gather(-1, k_arr_hi.unsqueeze(-1)).squeeze(-1)
            )                                                                     # [S, T]
        else:
            floor_at_arr_hi = torch.ones(S, T, dtype=dtype, device=device)

        # Floor at hi's eta gives the minimum to capture; cap by drain so a
        # single launch can never over-drain the source.
        sizes_lo = torch.minimum(floor_at_arr_hi.ceil().clamp(min=1.0), sizes_hi) # [S, T]
        sizes_mid = torch.minimum(2.0 * sizes_lo, sizes_hi)                       # [S, T]
        sizes_3 = torch.stack([sizes_lo, sizes_mid, sizes_hi], dim=-1)            # [S, T, N]

        # Step 2 — recompute reachability + aim per variant (each variant's eta
        # depends on its own fleet speed via fleet_speed(ships)).
        active_3 = reachable_mask(
            movement, source_idx=source_idx, target_idx=target_idx,
            fleet_sizes=sizes_3, eta_cap=eta_cap,
        )                                                                         # [S, T, N]
        aim_3 = intercept_angle(
            movement,
            source_idx.view(S, 1, 1), target_idx.view(1, T, 1),
            sizes_3, active=active_3,
        )
        angle_3 = aim_3["angle"]                                                  # [S, T, N]
        eta_3 = aim_3["eta"]                                                      # [S, T, N]
        viable_3 = aim_3["viable"] & (eta_3 <= eta_cap.view(1, T, 1))

        # Step 3 — capture-floor gate at each variant's own arrival turn.
        if K > 0:
            k_arr_3 = (eta_3.clamp(min=1.0, max=float(K)).ceil().long() - 1).clamp(0, K - 1)
            floor_at_arr_3 = (
                floor.view(1, T, 1, K).expand(S, T, N, K)
                .gather(-1, k_arr_3.unsqueeze(-1)).squeeze(-1)
            )                                                                     # [S, T, N]
        else:
            floor_at_arr_3 = torch.ones(S, T, N, dtype=dtype, device=device)
        clears_floor_3 = sizes_3 >= floor_at_arr_3                                # [S, T, N]
        ships_ok_3 = sizes_3 <= source_ships.view(S, 1, 1)                        # [S, T, N]

        valid_3 = (
            viable_3 & clears_floor_3 & (sizes_3 >= 1.0) & ships_ok_3
            & src_neq_tgt.unsqueeze(-1)
            & source_exists.view(S, 1, 1) & target_exists.view(1, T, 1)
        )                                                                         # [S, T, N]

        L = 1
        C = S * T * N
        cand_src = source_idx.view(S, 1, 1).expand(S, T, N).reshape(C, L)
        cand_tgt_slot = target_idx.view(1, T, 1).expand(S, T, N).reshape(C)
        cand_tgt_short = (
            torch.arange(T, device=device).view(1, T, 1).expand(S, T, N).reshape(C)
        )
        cand_send = torch.where(valid_3, sizes_3, torch.zeros_like(sizes_3)).reshape(C, L)
        cand_angle = angle_3.reshape(C, L)
        cand_eta = torch.where(valid_3, eta_3, torch.ones_like(eta_3)).reshape(C, L)
        cand_active = valid_3.reshape(C, L)
        cand_valid = valid_3.reshape(C)
        cand_is_def = target_is_mine[cand_tgt_short]                              # [C]
    else:
        # --- single fleet size = the max garrison launch (safe_drain) -----------
        # Engine needs integer ship counts; floor (never exceed what's available).
        sizes = drain.view(S, 1).expand(S, T).floor()                            # [S, T]

        # Strict-superset reachability precheck (always on): defers the body screen to
        # candidates that can physically reach the target in time.
        active = reachable_mask(
            movement, source_idx=source_idx, target_idx=target_idx,
            fleet_sizes=sizes.unsqueeze(-1), eta_cap=eta_cap,
        ).squeeze(-1)                                                            # [S, T]
        aim = intercept_angle(
            movement,
            source_idx.unsqueeze(1),                                             # [S, 1]
            target_idx.unsqueeze(0),                                             # [1, T]
            sizes,                                                               # [S, T]
            active=active,
        )
        angle = aim["angle"]                                                     # [S, T]
        eta = aim["eta"]
        viable = aim["viable"] & (eta <= eta_cap.view(1, T))

        # Capture-floor gate at each fleet's arrival turn (defenders grow with k). The
        # single size must clear the defender it lands on (size >= floor_at_arr). Owned
        # targets have floor 1 (reinforcement), so any positive send clears.
        if K > 0:
            k_arr = (eta.clamp(min=1.0, max=float(K)).ceil().long() - 1).clamp(0, K - 1)  # [S,T]
            floor_at_arr = floor.unsqueeze(0).expand(S, T, K).gather(-1, k_arr.unsqueeze(-1)).squeeze(-1)
        else:
            floor_at_arr = torch.ones(S, T, dtype=dtype, device=device)
        clears_floor = sizes >= floor_at_arr                                     # [S, T]

        valid = (
            viable & clears_floor & (sizes >= 1.0) & src_neq_tgt
            & source_exists.view(S, 1) & target_exists.view(1, T)
        )                                                                        # [S, T]

        # --- pack one candidate per (source, target); contributor axis L = 1 ----
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
        cand_is_def = target_is_mine[cand_tgt_short]                             # [C]

    launches = make_launch_set(
        source_slots=cand_src,
        target_slots=cand_tgt_slot.unsqueeze(-1).expand(C, L),
        ships=cand_send,
        eta=cand_eta,
        valid=cand_active & cand_valid.unsqueeze(-1),
        player_id=pid,
    )
    # If opp_projection is on, broadcast the projected opp launches to the
    # candidate axis and concat them onto launches' L axis BEFORE scoring.
    # The scorer's per-launch `owner` + `_per_step_survivor`'s owner-axis
    # topk handle the mixed-owner combat correctly. Greedy below still
    # operates on the ORIGINAL [C, L_my] tensors — opp slots never enter
    # greedy's budget / role-mutex view.
    scoring_launches = launches
    if background is not None:
        L_opp = int(background.source_slots.shape[-1])
        if L_opp > 0:
            def _bg(t):
                return t.unsqueeze(0).expand(C, -1)
            scoring_launches = LaunchSet(
                source_slots=torch.cat([launches.source_slots, _bg(background.source_slots)], dim=-1),
                target_slots=torch.cat([launches.target_slots, _bg(background.target_slots)], dim=-1),
                ships=torch.cat([launches.ships, _bg(background.ships)], dim=-1),
                eta=torch.cat([launches.eta, _bg(background.eta)], dim=-1),
                owner=torch.cat([launches.owner, _bg(background.owner)], dim=-1),
                valid=torch.cat([launches.valid, _bg(background.valid)], dim=-1),
            )
    score = score_candidates(
        garrison_status, prod=prod, alive_by_step=alive_by_step,
        player_count=int(player_count), launches=scoring_launches, player_id=pid,
        opp_weights=opp_weights,
    )                                                                            # [C]
    # Capture the base competitive score before additive terms so the
    # force-concentration rescore can re-derive the addon contribution per
    # iteration without recomputing recapture/denial/opening. Only allocated
    # when force-concentration is ON — OFF path is byte-identical. The kwarg
    # override lets the opp-projection inner calls disable FC explicitly so
    # the rescore closure doesn't run inside K-round opponent simulation
    # (where the K_opp x num_opps multiplier would blow up wallclock).
    if force_concentration is None:
        _fc_enabled = _force_concentration_enabled()
    else:
        _fc_enabled = bool(force_concentration)
    _fc_base_score = score.clone() if _fc_enabled else None
    if _recapture_penalty_enabled():
        # Subtract a non-negative recapture discount per candidate. The
        # penalty is in ship units (prod[T] * turns_lost), additive with
        # competitive_score. K_opp is read from the multi-tick env knob
        # only when opp projection is on; otherwise pass 0 so we don't
        # subtract a window the scorer never modeled.
        pen = recapture_penalty(
            obs=obs, cache=cache, garrison_status=garrison_status,
            cand_tgt_slot=cand_tgt_slot, cand_tgt_short=cand_tgt_short,
            cand_send=cand_send, cand_eta=cand_eta,
            cand_valid=cand_valid, cand_is_def=cand_is_def,
            capture_floor_TK=floor,
            prod=prod, H=H,
            K_recap=_recapture_k(int(player_count)),
            K_opp=(
                _multi_tick_opp_k(int(player_count)) if _opp_projection_enabled() else 0
            ),
            safety_reserve=_recapture_safety_reserve(),
            player_id=pid,
        )
        score = score - pen * float(_recapture_penalty_weight())
    if _denial_bonus_enabled() or _opening_bonus_enabled():
        # Resolve current_step once (cheap) for both bonuses.
        _cur_step = int(obs_tensors["step"].max().item())
        if _denial_bonus_enabled():
            # Rewards captures of targets opp values (currently owns OR
            # opp_proj's background launches target it). Encodes
            # "blocking the opponent's biggest bet."
            d_bonus = denial_bonus(
                obs=obs, background=background,
                cand_tgt_slot=cand_tgt_slot, cand_tgt_short=cand_tgt_short,
                cand_send=cand_send, cand_eta=cand_eta,
                cand_valid=cand_valid, cand_is_def=cand_is_def,
                capture_floor_TK=floor, prod=prod,
                garrison_status=garrison_status,
                H=H, current_step=_cur_step,
                game_length_est=_game_length_est(),
                weight=_denial_bonus_weight(),
                player_id=pid,
            )
            score = score + d_bonus
        if _opening_bonus_enabled():
            # Opp-agnostic early-game boost: linearly decays from full at
            # step 0 to zero at ``opening_window`` (default 30). Encodes
            # the horizon-too-short defect during the opening expansion.
            o_bonus = opening_bonus(
                obs=obs,
                cand_tgt_slot=cand_tgt_slot, cand_tgt_short=cand_tgt_short,
                cand_send=cand_send, cand_eta=cand_eta,
                cand_valid=cand_valid, cand_is_def=cand_is_def,
                capture_floor_TK=floor, prod=prod,
                garrison_status=garrison_status,
                H=H, current_step=_cur_step,
                game_length_est=_game_length_est(),
                opening_window=_opening_window(),
                weight=_opening_bonus_weight(),
                player_id=pid,
            )
            score = score + o_bonus
    # Force-concentration rescore closure: between greedy waves, re-score
    # every candidate against the just-fired waves so wave 2 at a target sees
    # wave 1's commitment (no double-counting). Uses the same `scoring_launches`
    # the initial score saw, plus the committed waves appended to the L axis
    # owner=pid. Add-on terms (recapture/denial/opening) depend only on
    # per-candidate state, so we precompute their offset once and add it back.
    _fc_rescore_fn = None
    _fc_max_waves = 1
    if _fc_enabled:
        _fc_addon_offset = score - _fc_base_score
        _fc_max_waves = _force_concentration_max_waves()
        _fc_C = int(cand_src.shape[0])
        _fc_scoring_launches = scoring_launches

        def _fc_rescore(c_src, c_send, c_eta, c_tgt, c_active):
            flat_src = c_src.reshape(-1)
            flat_send = c_send.reshape(-1)
            flat_eta = c_eta.reshape(-1)
            flat_tgt = c_tgt.reshape(-1)
            flat_active = c_active.reshape(-1)
            L_done = int(flat_src.shape[0])
            if L_done == 0:
                _new = _fc_base_score + _fc_addon_offset
                return torch.where(cand_valid, _new, torch.full_like(_new, float("-inf")))
            flat_owner = torch.full(
                (L_done,), int(pid), dtype=torch.long, device=device,
            )

            def _bc(t):
                return t.unsqueeze(0).expand(_fc_C, -1)

            merged = LaunchSet(
                source_slots=torch.cat(
                    [_fc_scoring_launches.source_slots, _bc(flat_src)], dim=-1,
                ),
                target_slots=torch.cat(
                    [_fc_scoring_launches.target_slots, _bc(flat_tgt)], dim=-1,
                ),
                ships=torch.cat([_fc_scoring_launches.ships, _bc(flat_send)], dim=-1),
                eta=torch.cat([_fc_scoring_launches.eta, _bc(flat_eta)], dim=-1),
                owner=torch.cat([_fc_scoring_launches.owner, _bc(flat_owner)], dim=-1),
                valid=torch.cat([_fc_scoring_launches.valid, _bc(flat_active)], dim=-1),
            )
            new_base = score_candidates(
                garrison_status, prod=prod, alive_by_step=alive_by_step,
                player_count=int(player_count), launches=merged, player_id=pid,
                opp_weights=opp_weights,
            )
            _new = new_base + _fc_addon_offset
            return torch.where(cand_valid, _new, torch.full_like(_new, float("-inf")))

        _fc_rescore_fn = _fc_rescore
    score = torch.where(cand_valid, score, torch.full_like(score, float("-inf")))

    # Cross-wave over-drain guard for multi-size: single-size's accidental
    # invariant — at-most-one wave per source, because cand_send==drain — does
    # NOT hold for multi-size, where size_lo<drain lets multiple small launches
    # from the same source fire across waves and total more than safe_drain.
    # Cap the source budget at drain so the cumulative sent per source stays
    # safe. OFF path unchanged to preserve bit-identical OFF parity.
    source_budget = obs.ships.to(dtype).clone()
    if _multi_size_enabled() or _coalitions_enabled():
        src_planet = source_idx.clamp(0, P - 1)
        source_budget[src_planet] = torch.minimum(source_budget[src_planet], drain)

    wave_entries, leftover = _greedy_select(
        P=P, W=W, device=device, dtype=dtype, score=score,
        cand_src=cand_src, cand_send=cand_send, cand_angle=cand_angle, cand_eta=cand_eta,
        cand_active=cand_active, cand_tgt_slot=cand_tgt_slot, cand_tgt_short=cand_tgt_short,
        cand_is_def=cand_is_def, source_budget=source_budget,
        target_exists=target_exists, roi_threshold=float(config.roi_threshold),
        rescore_fn=_fc_rescore_fn, max_waves_per_target=_fc_max_waves,
    )

    if not bool(config.enable_regroup):
        return wave_entries
    enemy_mass = cheap_enemy_pressure(obs, cache, horizon=float(K_eta), player_id=pid)  # [P]
    regroup_entries = _plan_regroup(
        movement=movement, obs=obs, obs_tensors=obs_tensors, garrison_status=garrison_status,
        leftover=leftover, original_ships=obs.ships.to(dtype), pressure=enemy_mass,
        config=config, H=H,
    )
    return concat_launch_entries([wave_entries, regroup_entries])


def _score_do_nothing(
    *,
    status,
    prod: Tensor,
    alive_by_step: Tensor,
    player_count: int,
    background: LaunchSet,
    player_id: int,
    opp_weights: Tensor | None = None,
) -> Tensor:
    """Score of NOT launching anything while opp executes their projected
    plan. Returns a scalar tensor. Used to renormalize roi_threshold under
    opp-aware scoring.
    """
    L_opp = int(background.source_slots.shape[-1])
    if L_opp == 0:
        return torch.tensor(0.0, device=background.source_slots.device)
    # 1 candidate with the background's L slots, all owner=opp_id, our
    # contribution = nothing.
    bg = LaunchSet(
        source_slots=background.source_slots.unsqueeze(0),
        target_slots=background.target_slots.unsqueeze(0),
        ships=background.ships.unsqueeze(0),
        eta=background.eta.unsqueeze(0),
        owner=background.owner.unsqueeze(0),
        valid=background.valid.unsqueeze(0),
    )
    score = score_candidates(
        status, prod=prod, alive_by_step=alive_by_step,
        player_count=int(player_count), launches=bg, player_id=int(player_id),
        opp_weights=opp_weights,
    )
    return score.flatten()[0]


def run_turn(obs_tensors: dict, *, config: ProducerLiteConfig, player_count: int, memory) -> dict:
    """Full per-turn pipeline: build movement → plan single-size waves + regroup → emit.

    ``memory`` must expose a mutable ``movement`` attribute (the rolling cache).
    """
    device = obs_tensors["planets"].device
    obs = parse_obs(obs_tensors)
    P = obs.P
    if P == 0:
        return empty_action_row(device)

    movement = ensure_planet_movement(
        obs_tensors=obs_tensors,
        expected_cfg=_movement_config(config, player_count=int(player_count)),
        cached_movement=getattr(memory, "movement", None),
    )
    memory.movement = movement
    cache = build_distance_cache(movement, max_k=int(config.horizon))
    H = int(config.horizon)
    status = movement.garrison_status(max_horizon=H)
    alive_by_step = movement.alive_by_step[: H + 1]

    current_step = int(obs_tensors["step"].max().item())
    K_eta_override = compute_k_eta_for_step(current_step, H=H)

    # Opponent model: run Producer's own planner from each opp seat with
    # background=None (one-step best response, opp assumes we do nothing
    # this turn). Returns the opp's predicted launches for this turn as a
    # padded LaunchSet that we pass as `background` to our own planner.
    # Default OFF preserves bit-identical static-opp scoring.
    #
    # The roi_threshold needs a per-turn shift because competitive_score is
    # measured against a do-nothing-by-everyone baseline (garrison_status).
    # In the static-opp world, do-nothing-by-me also means do-nothing-by-
    # opp, so do_nothing_score = 0 and a 1.5 absolute threshold == "1.5
    # ships of differential gain over not firing." In the opp-aware world,
    # do-nothing-by-me still leaves opp's projected launches running, so
    # do_nothing_score = -opp_gain (a per-turn constant, usually < 0).
    # To preserve the threshold's semantic meaning -- "fire only if you
    # gain >= roi_threshold ships over doing nothing" -- shift the
    # absolute threshold by do_nothing_score.
    background = None
    cfg = config
    # FFA objective weights: only built for 3+ player games so the 2P path
    # stays byte-identical (None -> legacy equal-weight opponent sum).
    opp_w = None
    if _ffa_score_enabled() and int(player_count) >= 3:
        opp_w = _ffa_opp_weights(
            obs_tensors, player_id=int(obs.player_id), player_count=int(player_count),
        )
    if _opp_projection_enabled():
        opp_ids = [
            pid for pid in range(int(player_count)) if pid != int(obs.player_id)
        ]
        K_opp = max(1, _multi_tick_opp_k(int(player_count)))
        background = predict_opp_launches_via_mirror(
            plan_fn=plan_lite_waves,
            obs_tensors=obs_tensors, movement=movement, cache=cache,
            garrison_status=status, prod=movement.planet_prod,
            alive_by_step=alive_by_step,
            opp_ids=opp_ids, config=config, player_count=int(player_count),
            K_eta_override=K_eta_override,
            pad_to=_env_int("PRODUCER_PLUS_OPP_MAX_L", MAX_L_OPP),
            K=K_opp,
            H=H,
        )
        do_nothing_score = float(_score_do_nothing(
            status=status, prod=movement.planet_prod,
            alive_by_step=alive_by_step, player_count=int(player_count),
            background=background, player_id=int(obs.player_id),
            opp_weights=opp_w,
        ))
        cfg = dataclasses.replace(
            config, roi_threshold=do_nothing_score + float(config.roi_threshold),
        )

    entries = plan_lite_waves(
        movement=movement, obs=obs, obs_tensors=obs_tensors, cache=cache,
        garrison_status=status, prod=movement.planet_prod,
        alive_by_step=alive_by_step, config=cfg, player_count=int(player_count),
        K_eta_override=K_eta_override,
        background=background,
        opp_weights=opp_w,
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


# 4P FFA preset — only the knobs that differ from the 2P default. 
CONFIG_4P = dataclasses.replace(
    ProducerLiteConfig(),
    horizon=13,
    max_sources_per_lane=6,
    max_defensive_targets=2,
    max_regroup_time=6.0,
    max_regroup_targets_per_source=8,
)


def _config_for(player_count: int) -> ProducerLiteConfig:
    cfg = CONFIG_4P if int(player_count) >= 4 else ProducerLiteConfig()
    # Optional override of the scorer's lookahead horizon. Bumping H lets
    # the scorer see longer-term outcomes (e.g. the recapture leg of an
    # exchange cycle) and properly value stockpiling vs cyclical attacks.
    # Cost scales linearly in H; defaults unchanged when env unset.
    env_h = os.environ.get(
        "PRODUCER_PLUS_HORIZON_4P" if int(player_count) >= 4 else "PRODUCER_PLUS_HORIZON_2P"
    )
    if env_h:
        try:
            cfg = dataclasses.replace(cfg, horizon=int(env_h))
        except ValueError:
            pass
    return cfg


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
    with torch.no_grad():
        sparse_row = _RUNTIME.tensor_action(obs_tensors)
    return sparse_action_row_to_moves(sparse_row, obs, player_id=player_id)

