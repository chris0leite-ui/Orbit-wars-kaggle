
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

from opp_projector import (
    _debug_log,
    _none_projector,
    affected_slots,
    arrivals_tuples_to_buckets_delta,
    get_projector,
)


# Adaptive candidate-arrival horizon K_eta — ported from champion's
# capture_horizon_k (agents/baseline/launch_rules.py). Default OFF
# preserves bit-identical behaviour vs the untouched producer.
# Clamped to H so capture_floor lookups stay inside garrison_status.
def _adaptive_k_enabled() -> bool:
    return os.environ.get("PRODUCER_PLUS_ADAPTIVE_K", "0").strip().lower() in (
        "1", "true", "yes", "on",
    )


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


def _invalidate_garrison_cache(movement, *, planet_slots=None) -> None:
    """Single chokepoint for the vendored producer's private invalidation API.

    If ``planet_slots`` is None: invalidate every planet from step 0
    (full rebuild). Otherwise: invalidate only the named slots. Isolated
    here so a future re-vendor that renames the underscore-prefixed
    method is a one-line fix, not a grep across the host.
    """
    if planet_slots is None:
        movement._mark_garrison_dirty_all(0)
    else:
        movement._mark_garrison_dirty(planet_slots, 0)


# Opp-foresight env gates — mechanisms that LEVERAGE the opp projection.
# All default OFF; bit-identical to vanilla Producer when unset.
def _source_exposure_enabled() -> bool:
    return os.environ.get("PRODUCER_PLUS_SOURCE_EXPOSURE", "0").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _counter_capture_enabled() -> bool:
    return os.environ.get("PRODUCER_PLUS_COUNTER_CAPTURE", "0").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _opp_owner_mask(*, my_id: int, num_seats: int, device, dtype) -> Tensor:
    """Bool mask ``[A]`` selecting all seats != ``my_id``.

    Used to slice the opponent axes out of ``fleet_buckets`` and
    ``garrison_status`` tensors so the foresight mechanisms aggregate
    threats from every non-self seat (works for 2P and 4P).
    """
    A = max(1, int(num_seats))
    mask = torch.ones(A, dtype=torch.bool, device=device)
    if 0 <= int(my_id) < A:
        mask[int(my_id)] = False
    return mask


def _fresh_opp_capture_mask(
    garrison_status, *, my_id: int, K_eta: int,
) -> Tensor:
    """Return ``[P]`` bool: True for planets the opp captures fresh in K_eta.

    A "fresh opp capture" is a planet whose owner at some tick
    ``k in [1, K_eta]`` is an opponent AND differs from the owner at
    step 0. Planets the opp already controls at step 0 do not qualify.
    The defender count after such a capture is ``1 + production`` —
    very cheap to take back.
    """
    owner_tl = garrison_status.owner  # [P, H+1]
    P = int(owner_tl.shape[0])
    H_axis = int(owner_tl.shape[-1])
    K = max(0, min(int(K_eta), H_axis - 1))
    if K == 0 or P == 0:
        return torch.zeros(P, dtype=torch.bool, device=owner_tl.device)
    future = owner_tl[:, 1 : K + 1]                                    # [P, K]
    initial = owner_tl[:, 0:1]                                         # [P, 1]
    is_opp = (future >= 0) & (future != int(my_id))
    is_fresh = future != initial
    return (is_opp & is_fresh).any(dim=-1)


def _append_counter_capture_targets(
    target_idx: Tensor,
    target_exists: Tensor,
    *,
    fresh_capture_mask: Tensor,
) -> tuple[Tensor, Tensor]:
    """Append fresh-opp-capture planet slots to the shortlist.

    Skips planets already present in ``target_idx`` to avoid duplicates.
    Returns the (possibly-expanded) ``(target_idx, target_exists)``.
    """
    if not bool(fresh_capture_mask.any()):
        return target_idx, target_exists
    device = target_idx.device
    capture_slots = torch.nonzero(fresh_capture_mask, as_tuple=False).flatten()
    if target_idx.numel() > 0:
        existing = set(target_idx.detach().to("cpu").tolist())
        new_slots = [int(s) for s in capture_slots.detach().to("cpu").tolist() if int(s) not in existing]
        if not new_slots:
            return target_idx, target_exists
        addition = torch.tensor(new_slots, dtype=target_idx.dtype, device=device)
    else:
        addition = capture_slots.to(dtype=target_idx.dtype)
    new_target_idx = torch.cat([target_idx, addition], dim=0)
    new_target_exists = torch.cat(
        [target_exists, torch.ones(addition.shape[0], dtype=target_exists.dtype, device=device)],
        dim=0,
    )
    return new_target_idx, new_target_exists


def _apply_source_exposure_penalty(
    score: Tensor,
    *,
    cand_src: Tensor,
    cand_send: Tensor,
    cand_eta: Tensor,
    fleet_buckets: Tensor,
    opp_owner_mask: Tensor,
    source_ships_per_planet: Tensor,
    safety_margin: float,
) -> Tensor:
    """Set score to -inf where residual < safety_margin × opp_arrivals_at_src.

    For each candidate ``c``:
      residual = source_ships[src_c] - ships_sent_c
      opp_arrivals_at_src = sum over [tick 0..eta_c] and over opp axes
                            of fleet_buckets[src_c, k, owner]
    Reject ``c`` (score = -inf) if ``residual < safety_margin × opp_arrivals``.

    Batched: pure tensor gather + masked sum + comparison.
    """
    C = int(score.shape[0])
    L = int(cand_send.shape[-1]) if cand_send.dim() >= 2 else 1
    if C == 0:
        return score
    # Sum ships and eta across the contributor axis to get per-candidate scalars.
    ships_total = cand_send.reshape(C, L).sum(dim=-1)                  # [C]
    # Use the LATEST arrival tick across contributors as the exposure window.
    eta_latest = cand_eta.reshape(C, L).max(dim=-1).values             # [C]
    # All contributors of a candidate share the same source slot by construction
    # (single-source per candidate today); take the first.
    src_slot = cand_src.reshape(C, L)[:, 0]                            # [C], long
    src_slot = src_slot.clamp(min=0, max=max(int(fleet_buckets.shape[0]) - 1, 0))
    P, H_buckets, A = int(fleet_buckets.shape[0]), int(fleet_buckets.shape[1]), int(fleet_buckets.shape[2])
    if P == 0 or H_buckets == 0 or A == 0:
        return score
    # eta_latest can be float; convert to bucket-window length (ceil) and clamp to H.
    window = eta_latest.ceil().long().clamp(min=0, max=H_buckets)      # [C]
    # Build a [C, H_buckets] mask over the tick axis: True for k < window[c].
    ks = torch.arange(H_buckets, device=fleet_buckets.device).view(1, H_buckets)
    tick_mask = ks < window.view(C, 1)                                 # [C, H_buckets]
    # Sum opp arrivals at src up to eta. Gather [C, H_buckets, A] then mask + sum.
    arrivals_at_src = fleet_buckets[src_slot]                          # [C, H_buckets, A]
    opp_axes = opp_owner_mask.view(1, 1, A)
    opp_at_src = (arrivals_at_src * opp_axes.to(arrivals_at_src.dtype)).sum(dim=-1)  # [C, H_buckets]
    opp_total = (opp_at_src * tick_mask.to(opp_at_src.dtype)).sum(dim=-1)            # [C]
    # residual = source_ships[src] - ships_total
    source_ships_c = source_ships_per_planet[src_slot].to(opp_total.dtype)
    residual = source_ships_c - ships_total.to(opp_total.dtype)
    exposed = residual < (float(safety_margin) * opp_total)
    return torch.where(exposed, torch.full_like(score, float("-inf")), score)


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
    )
    # Mechanism 3 — counter-capture target seeding. APPEND planets that
    # the opp is projected to capture inside K_eta but that the shortlist
    # left out. The scorer (now fed augmented buckets) will rank a
    # recapture against the post-capture defender count (1 + production).
    if _counter_capture_enabled():
        fresh_capture_mask = _fresh_opp_capture_mask(
            garrison_status, my_id=pid, K_eta=K_eta,
        )
        target_idx, target_exists = _append_counter_capture_targets(
            target_idx, target_exists, fresh_capture_mask=fresh_capture_mask,
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

    # --- single fleet size = the max garrison launch (safe_drain) ---------------
    # Engine needs integer ship counts; floor (never exceed what's available).
    sizes = drain.view(S, 1).expand(S, T).floor()                                # [S, T]

    # Strict-superset reachability precheck (always on): defers the body screen to
    # candidates that can physically reach the target in time.
    active = reachable_mask(
        movement, source_idx=source_idx, target_idx=target_idx,
        fleet_sizes=sizes.unsqueeze(-1), eta_cap=eta_cap,
    ).squeeze(-1)                                                                # [S, T]
    aim = intercept_angle(
        movement,
        source_idx.unsqueeze(1),                                                 # [S, 1]
        target_idx.unsqueeze(0),                                                 # [1, T]
        sizes,                                                                    # [S, T]
        active=active,
    )
    angle = aim["angle"]                                                         # [S, T]
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
    clears_floor = sizes >= floor_at_arr                                         # [S, T]

    src_neq_tgt = source_idx.view(S, 1) != target_idx.view(1, T)
    valid = (
        viable & clears_floor & (sizes >= 1.0) & src_neq_tgt
        & source_exists.view(S, 1) & target_exists.view(1, T)
    )                                                                            # [S, T]

    # --- pack one candidate per (source, target); contributor axis L = 1 --------
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
    cand_is_def = target_is_mine[cand_tgt_short]                                  # [C]

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
    )                                                                            # [C]
    score = torch.where(cand_valid, score, torch.full_like(score, float("-inf")))

    # Mechanism 1 — source-exposure penalty. Hard-reject any candidate
    # whose launch would strip the source planet below the projected opp
    # arrival force by the candidate's eta. (Race-loss / Mechanism 2 was
    # ablated out — redundant in this pipeline because the augmented
    # scorer already reflects opp captures via the owner timeline; M2
    # double-penalised candidates the scorer had correctly valued.)
    if _source_exposure_enabled():
        opp_mask = _opp_owner_mask(
            my_id=pid, num_seats=int(player_count),
            device=movement.device, dtype=dtype,
        )
        margin = _env_float("PRODUCER_PLUS_SOURCE_EXPOSURE_MARGIN", 1.0)
        score = _apply_source_exposure_penalty(
            score,
            cand_src=cand_src,
            cand_send=cand_send,
            cand_eta=cand_eta,
            fleet_buckets=movement.fleet_buckets,
            opp_owner_mask=opp_mask,
            source_ships_per_planet=obs.ships.to(dtype),
            safety_margin=margin,
        )

    wave_entries, leftover = _greedy_select(
        P=P, W=W, device=device, dtype=dtype, score=score,
        cand_src=cand_src, cand_send=cand_send, cand_angle=cand_angle, cand_eta=cand_eta,
        cand_active=cand_active, cand_tgt_slot=cand_tgt_slot, cand_tgt_short=cand_tgt_short,
        cand_is_def=cand_is_def, source_budget=obs.ships.to(dtype).clone(),
        target_exists=target_exists, roi_threshold=float(config.roi_threshold),
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


def run_turn(obs_tensors: dict, *, config: ProducerLiteConfig, player_count: int, memory) -> dict:
    """Full per-turn pipeline: build movement → plan single-size waves + regroup → emit.

    ``memory`` must expose a mutable ``movement`` attribute (the rolling cache).
    Reads ``memory.raw_obs`` for the opponent projector (Step 3); falls back to
    the no-op projector if absent.
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

    # Opponent projection: augment fleet_buckets with projected opp arrivals
    # so the scorer sees a non-static opponent. Default (env unset) is
    # _none_projector → empty delta → bit-identical to vanilla Producer.
    raw_obs = getattr(memory, "raw_obs", None)
    projector_name = os.environ.get("PRODUCER_PLUS_OPP_PROJECTOR", "none")
    projector = get_projector(projector_name)
    original_buckets = movement.fleet_buckets
    augmented = False
    touched_slots = None
    if projector is not _none_projector and raw_obs is not None and original_buckets is not None:
        try:
            tuples = projector(
                raw_obs, my_id=int(obs.player_id),
                num_seats=int(player_count), horizon=H,
            )
        except Exception as exc:
            _debug_log(f"projector={projector_name}", exc)
            tuples = []
        if tuples:
            A_dim = int(original_buckets.shape[-1])
            H_dim = int(original_buckets.shape[-2])
            delta = arrivals_tuples_to_buckets_delta(
                tuples, movement.planet_ids, A=A_dim, H=H_dim,
                device=movement.device, dtype=original_buckets.dtype,
            )
            if bool(delta.any()):
                slots_cpu = affected_slots(
                    tuples, movement.planet_ids, H=H_dim, A=A_dim,
                )
                touched_slots = slots_cpu.to(movement.device)
                movement.fleet_buckets = original_buckets + delta
                _invalidate_garrison_cache(movement, planet_slots=touched_slots)
                augmented = True

    # Restore fleet_buckets BEFORE apply_private_planned_launches so the
    # engine's own state-update path (which writes through fleet_buckets)
    # sees authoritative state, not our scoring augmentation.
    try:
        status = movement.garrison_status(max_horizon=H)
        alive_by_step = movement.alive_by_step[: H + 1]

        current_step = int(obs_tensors["step"].max().item())
        K_eta_override = compute_k_eta_for_step(current_step, H=H)

        entries = plan_lite_waves(
            movement=movement, obs=obs, obs_tensors=obs_tensors, cache=cache,
            garrison_status=status, prod=movement.planet_prod,
            alive_by_step=alive_by_step, config=config, player_count=int(player_count),
            K_eta_override=K_eta_override,
        )
    finally:
        if augmented:
            movement.fleet_buckets = original_buckets
            # MUST re-invalidate touched_slots: garrison_status above just
            # marked them clean, but the cache now reflects AUGMENTED data.
            # apply_private_planned_launches only dirties launch sources /
            # targets, so augmented-but-not-launched planets would otherwise
            # serve stale-augmented data on the next read.
            _invalidate_garrison_cache(movement, planet_slots=touched_slots)
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
    return CONFIG_4P if int(player_count) >= 4 else ProducerLiteConfig()


class ProducerLiteMemory:
    def __init__(self) -> None:
        self.movement = None
        self.cached_player_count: int | None = None
        self.last_sparse_action_row: dict | None = None
        # Raw obs dict stashed by ``agent()`` so the opponent projector
        # (Step 3) can build a champion-style World without re-plumbing
        # every signature down the call chain.
        self.raw_obs = None

    def reset(self) -> None:
        self.movement = None
        self.cached_player_count = None
        self.last_sparse_action_row = None
        self.raw_obs = None


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
    _RUNTIME.memory.raw_obs = obs
    with torch.no_grad():
        sparse_row = _RUNTIME.tensor_action(obs_tensors)
    return sparse_action_row_to_moves(sparse_row, obs, player_id=player_id)

