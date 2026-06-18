
from __future__ import annotations

import dataclasses
import math
import os
import sys
import time
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
    LaunchEntries,
    apply_private_planned_launches,
    concat_launch_entries,
    disambiguate_duplicate_launches,
    ensure_planet_movement,
    infer_planned_launches_from_entries,
)
from orbit_lite.obs import parse_obs
from orbit_lite.distance_cache import build_distance_cache, min_distance_to_targets
from orbit_lite.garrison_launch import LaunchSet, _run_exact_recurrence
from orbit_lite.movement import PlanetGarrisonStatus
from orbit_lite.opp_projection import predict_opp_launches_via_mirror, MAX_L_OPP
from orbit_lite.recapture import recapture_penalty
from orbit_lite.strategic_value import denial_bonus, opening_bonus
from orbit_lite.planner_core import (
    _candidate_indices,
    _empty_entries,
    _greedy_select,
    _plan_regroup,
    _stable_argmax,
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
# --- Holding-time-priced capture credit (PRODUCER_PLUS_HOLD_VALUE) -------------
# The decision-trace finding (Gregor Lied loss): the in-horizon flow scorer
# truncates capture payoffs at H, so every expansion scores ~0 against the
# fire threshold and the agent banks instead (paralysis). A FLAT terminal
# credit (TERMINAL_PROD_VALUE=12) was refuted on both referee classes — it
# rewards expansion the opponent punishes before payback. This version
# credits post-horizon production ONLY for captures the opponent cannot
# feasibly retake inside the lookahead: project the captured garrison
# (survivors + production) against the enemy's FULL routable mass at every
# later tick; any deficit ⇒ no credit. Safe rear expansions unlock;
# contested grabs stay priced by raw flow. Default 0 = byte-identical.


# --- Source-safety drain cap (PRODUCER_PLUS_SOURCE_SAFETY) ----------------------
# The economy-credit refutation chain (3 mirror routs, all decided ~step 29)
# localized the true blindspot: ``safe_drain`` caps drain by the DO-NOTHING
# projection (in-flight fleets + production), so the enemy's uncommitted
# standing reserve is invisible — the punisher simply strikes whichever home
# planet the expander just thinned. Symmetric counterpart of the reactive
# capture floor, for SOURCES: a source may shed only what keeps it able to
# survive the enemy's routable mass at every tick of the window, crediting
# its own production growth and friendly garrisons that can route help in
# time:  drain ≤ g_s + min_k( prod_s·k + help(s,k) − w·threat(s,k) ).
# Default 0 = byte-identical.


def _source_safety_weight() -> float:
    raw = os.environ.get("PRODUCER_PLUS_SOURCE_SAFETY", "0")
    try:
        return max(0.0, float(raw))
    except ValueError:
        return 0.0


def _source_safety_lag() -> float:
    raw = os.environ.get("PRODUCER_PLUS_SOURCE_SAFETY_LAG", "0")
    try:
        return float(raw)
    except ValueError:
        return 0.0


def _friendly_support_margin(
    obs, cache, source_idx: Tensor, K: int, *, lag: float = 0.0,
):
    """``[S, K]`` friendly garrison mass routable to each source by tick k.

    Mirror of ``_reactive_reinforcement_margin`` over OUR planets. Each
    helper keeps 1 ship (can't fully strip a held planet), and the source
    itself is excluded — its own garrison is already the defender.
    """
    mine = obs.owned & obs.alive
    q_idx = mine.nonzero(as_tuple=True)[0]
    Q = int(q_idx.shape[0])
    S = int(source_idx.shape[0])
    if Q == 0 or S == 0 or K <= 0:
        return None
    dtype = obs.ships.dtype
    g_q = (obs.ships[q_idx].to(dtype) - 1.0).clamp(min=0.0)          # [Q]
    speed_q = fleet_speed(g_q.clamp(min=1.0))                        # [Q]
    reach = _margin_reach(cache, q_idx, source_idx, speed_q, K,
                          float(lag), int(obs.P))                    # [Q, S, K]
    self_mask = q_idx.view(Q, 1) == source_idx.view(1, S)
    reach = reach & ~self_mask.unsqueeze(-1)
    return (g_q.view(Q, 1, 1) * reach.to(dtype)).sum(dim=0)          # [S, K]


def _source_safety_allowance(
    obs, cache, *, source_idx: Tensor, prod: Tensor, K: int,
):
    """``[S]`` max drain that keeps each source locally defensible, or None.

    None means no constraint applies (gate off, no enemies, or empty window).
    """
    w = _source_safety_weight()
    S = int(source_idx.shape[0])
    if w <= 0.0 or S == 0 or K <= 0:
        return None
    threat = _reactive_reinforcement_margin(
        obs, cache, source_idx, K, weight=w, lag=_source_safety_lag(),
    )                                                                # [S, K] | None
    if threat is None:
        return None
    dtype = obs.ships.dtype
    src = source_idx.clamp(0, int(obs.P) - 1)
    help_sk = _friendly_support_margin(obs, cache, source_idx, K)
    if help_sk is None:
        help_sk = torch.zeros_like(threat)
    k_grid = torch.arange(1, K + 1, device=obs.device, dtype=dtype).view(1, K)
    slack = prod[src].to(dtype).unsqueeze(-1) * k_grid + help_sk - threat  # [S, K]
    allowed = obs.ships[src].to(dtype) + slack.min(dim=-1).values    # [S]
    return allowed.clamp(min=0.0)


def _hold_value() -> float:
    raw = os.environ.get("PRODUCER_PLUS_HOLD_VALUE", "0")
    try:
        return max(0.0, float(raw))
    except ValueError:
        return 0.0


def _hold_value_lag() -> float:
    raw = os.environ.get("PRODUCER_PLUS_HOLD_VALUE_LAG", "2.0")
    try:
        return float(raw)
    except ValueError:
        return 2.0


def _hold_value_bonus(
    *,
    obs,
    cache,
    target_idx: Tensor,        # [T] shortlist slots
    cand_tgt_slot: Tensor,     # [C]
    cand_tgt_short: Tensor,    # [C]
    cand_send: Tensor,         # [C, L]
    cand_eta: Tensor,          # [C, L]
    cand_valid: Tensor,        # [C]
    cand_is_def: Tensor,       # [C]
    capture_floor_TK: Tensor,  # [T, K]
    prod: Tensor,              # [P]
    K: int,
) -> Tensor:
    """Per-candidate post-horizon production credit, ``[C]`` (≥ 0)."""
    device = cand_send.device
    dtype = cand_send.dtype
    C = int(cand_send.shape[0])
    lam = _hold_value()
    if lam <= 0.0 or C == 0 or K <= 0:
        return torch.zeros(C, dtype=dtype, device=device)
    P = int(obs.P)
    tgt = cand_tgt_slot.clamp(0, P - 1)
    neutral_now = obs.is_neutral[tgt] & obs.alive[tgt]
    gate = cand_valid & ~cand_is_def & neutral_now                    # [C]
    if not bool(gate.any()):
        return torch.zeros(C, dtype=dtype, device=device)

    send_tot = cand_send.sum(dim=-1)                                  # [C]
    eta_max = cand_eta.max(dim=-1).values                             # [C]
    k_arr = (eta_max.clamp(min=1.0, max=float(K)).ceil().long() - 1).clamp(0, K - 1)
    floor_at_arr = (
        capture_floor_TK[cand_tgt_short.clamp(0, capture_floor_TK.shape[0] - 1)]
        .gather(-1, k_arr.unsqueeze(-1)).squeeze(-1)
    )                                                                 # [C]
    # capture_floor = defenders + 1 (overhead); conquered garrison =
    # send − defenders = send − floor + 1.
    survivors = (send_tot - floor_at_arr + 1.0).clamp(min=1.0)        # [C]
    prod_t = prod[tgt].to(dtype)                                      # [C]

    margin = _reactive_reinforcement_margin(
        obs, cache, target_idx, K, weight=1.0, lag=_hold_value_lag(),
    )                                                                 # [T, K] | None
    if margin is None:
        safe = gate
    else:
        m_c = margin[cand_tgt_short.clamp(0, margin.shape[0] - 1)].to(dtype)  # [C, K]
        k_grid = torch.arange(K, device=device, dtype=dtype).view(1, K)
        dk = k_grid - k_arr.to(dtype).view(C, 1)                      # ticks after arrival
        garrison = survivors.view(C, 1) + prod_t.view(C, 1) * dk.clamp(min=0.0)
        threat = (m_c >= garrison) & (dk > 0)                         # [C, K]
        safe = gate & ~threat.any(dim=-1)
    return torch.where(
        safe, lam * prod_t, torch.zeros(C, dtype=dtype, device=device))


# --- Garrison-deficit reinforcement value (PRODUCER_PLUS_GARRISON_VALUE) -------
# Live war-ledger finding (audit 2026-06-11 night): at the 1300+ band, the
# 4P winner is whoever reinforces more (our wins: we out-garrison the top
# rival 58%/33%; our losses: 46%/61%; the Blu3s siege: 15,868 vs 942
# reinforcement ships, 42/42 vs 31/39 wave success). The flow scorer values
# reinforcement only when a known IN-FLIGHT wave makes a planet savable —
# by then the avalanche is launched and no single send parries it. This
# term prices PROACTIVE garrisoning: an own-target send earns the planet's
# holding value when the planet's local balance against the enemy's
# UNCOMMITTED reserve is negative and the send covers the deficit. Same
# balance-of-force model as the source-safety cap (push side); this is the
# pull side, chooser-internal per the three-falsifications friction note
# (thin post-pass regroup lanes land ships the chooser never uses).
# Default 0 = byte-identical.


def _living_rival_count(obs) -> int:
    """Rivals with at least one living planet (planet-only proxy)."""
    rivals = obs.owner_abs[obs.is_enemy & obs.alive]
    if rivals.numel() == 0:
        return 0
    return int(torch.unique(rivals).numel())


def _garrison_value() -> float:
    raw = os.environ.get("PRODUCER_PLUS_GARRISON_VALUE", "0")
    try:
        return max(0.0, float(raw))
    except ValueError:
        return 0.0


def _garrison_value_threat_w() -> float:
    """Threat weight INSIDE the garrison-value deficit. Default: fall back
    to the source-safety weight (legacy coupling). The RYOTA loss (seed
    1493019744): a 135-ship massing was halved to ~67 by the 0.5 coupling
    and never registered as a deficit at the 96-garrison planet it killed.
    The rival-cap already guards over-insurance structurally — the threat
    MAGNITUDE should be what the enemy can actually send. Set 1.0 in the
    variant."""
    raw = os.environ.get("PRODUCER_PLUS_GARRISON_VALUE_THREAT_W", "")
    if not raw.strip():
        w = _source_safety_weight()
        return w if w > 0.0 else 1.0
    try:
        return max(0.0, float(raw))
    except ValueError:
        return 1.0


def _garrison_value_from_step() -> int:
    """Opening gate: no proactive-garrison credit before this step. The
    land-grab phase decides production rank (wins are production-ahead@40
    in 16/17); insurance bought then costs expansion tempo (seed-6 panel
    wipe: stalled at 3 planets by t=60, dead by 170)."""
    raw = os.environ.get("PRODUCER_PLUS_GARRISON_VALUE_FROM_STEP", "0")
    try:
        return max(0, int(float(raw)))
    except ValueError:
        return 0


def _garrison_value_bonus(
    *,
    obs,
    cache,
    target_idx: Tensor,        # [T] shortlist slots
    cand_tgt_slot: Tensor,     # [C]
    cand_tgt_short: Tensor,    # [C]
    cand_send: Tensor,         # [C, L]
    cand_eta: Tensor,          # [C, L]
    cand_valid: Tensor,        # [C]
    cand_is_def: Tensor,       # [C]
    prod: Tensor,              # [P]
    K: int,
) -> Tensor:
    """Per-candidate proactive-garrison credit, ``[C]`` (>= 0).

    Deficit of own planet t over the window, judged at/after the send's
    arrival: D(t) = max_{k >= eta} [ w*threat(t,k) - (g_t + prod_t*k +
    help(t,k)) ]. A send earns lambda_g * prod_t when D > 0 (the planet is
    expected to fall to a feasible strike) and the send covers D.
    """
    device = cand_send.device
    dtype = cand_send.dtype
    C = int(cand_send.shape[0])
    lam = _garrison_value()
    if lam <= 0.0 or C == 0 or K <= 0:
        return torch.zeros(C, dtype=dtype, device=device)
    gate = cand_valid & cand_is_def                                   # [C]
    if not bool(gate.any()):
        return torch.zeros(C, dtype=dtype, device=device)
    threat = _reactive_reinforcement_margin(
        obs, cache, target_idx, K,
        weight=_garrison_value_threat_w(), lag=_source_safety_lag(),
        concentration_speed=True,
    )                                                                 # [T, K] | None
    if threat is None:
        return torch.zeros(C, dtype=dtype, device=device)
    help_tk = _friendly_support_margin(obs, cache, target_idx, K)
    if help_tk is None:
        help_tk = torch.zeros_like(threat)
    P = int(obs.P)
    tgt_safe = target_idx.clamp(0, P - 1)
    k_grid = torch.arange(1, K + 1, device=device, dtype=dtype).view(1, K)
    base = (
        obs.ships[tgt_safe].to(dtype).unsqueeze(-1)
        + prod[tgt_safe].to(dtype).unsqueeze(-1) * k_grid
        + help_tk
    )                                                                 # [T, K]
    deficit_tk = threat - base                                        # [T, K]
    # The enemy reserve is ONE resource per rival — it cannot strike every
    # deficit simultaneously. Pricing every worst-case deficit as certain
    # turns the agent into a turtle (seed-6 panel wipe: 36 reinforce
    # launches vs 4 neutral grabs, stalled at 3 planets while rivals took
    # 12+). Credit at most R targets per turn (R = living rivals), ranked
    # by strike attractiveness to the enemy (production of the planet it
    # could feasibly take).
    n_rivals = 0
    owner_alive = obs.is_enemy & obs.alive
    if bool(owner_alive.any()):
        n_rivals = max(int(_living_rival_count(obs)), 1)
    deficit_t = deficit_tk.max(dim=-1).values                         # [T]
    prod_T = prod[tgt_safe].to(dtype)                                 # [T]
    # Enemy's pick: production first, then how badly underdefended the
    # planet is (lexicographic via scaling) — a prod tie must resolve to
    # the planet the strike actually wins (RYOTA replay: 17 and 19 tie on
    # prod; 19 was the crushable one).
    attract = torch.where(
        deficit_t > 0.0, prod_T * 1.0e6 + deficit_t,
        torch.full_like(prod_T, float("-inf")))
    T = int(attract.shape[0])
    R = min(max(n_rivals, 1), T)
    top_idx = attract.topk(R).indices
    eligible_T = torch.zeros(T, dtype=torch.bool, device=device)
    eligible_T[top_idx] = True
    eligible_T &= deficit_t > 0.0                                     # [T]
    t_c = cand_tgt_short.clamp(0, deficit_tk.shape[0] - 1)
    d_c = deficit_tk[t_c]                                             # [C, K]
    eta_max = cand_eta.max(dim=-1).values                             # [C]
    k_arr = (eta_max.clamp(min=1.0, max=float(K)).ceil() - 1.0).view(C, 1)
    at_or_after = (
        torch.arange(K, device=device, dtype=dtype).view(1, K) >= k_arr
    )                                                                 # [C, K]
    neg_fill = torch.full_like(d_c, float("-inf"))
    D = torch.where(at_or_after, d_c, neg_fill).max(dim=-1).values    # [C]
    send_tot = cand_send.sum(dim=-1)                                  # [C]
    covers = (D > 0.0) & (send_tot >= D) & eligible_T[t_c]
    prod_t = prod[cand_tgt_slot.clamp(0, P - 1)].to(dtype)
    return torch.where(
        gate & covers, lam * prod_t,
        torch.zeros(C, dtype=dtype, device=device))


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
    trade-devaluation alone helps without the hit-the-leader tilt.
    ``weakness``: weights ∝ (total_rival_strength − this rival's strength),
    a bounded anti-leader tilt that pays MOST for converting the weakest
    rival's mass. Motivated by the live kingmaker-tax finding
    (audit/2026-06-13-kingmaker-tax.md): in 4P losses we capture from the
    strongest rival (mean ship-rank 1.76) while the eventual winner feeds
    the weak (rank 2.66); strength-weighting actively pays us to attack the
    leader's most expensive planets."""
    raw = os.environ.get("PRODUCER_PLUS_FFA_WEIGHTS", "strength").strip().lower()
    return raw if raw in ("strength", "uniform", "weakness") else "strength"


def _ffa_opp_weights(obs_tensors: dict, *, player_id: int, player_count: int):
    """Per-opponent weights over living rivals (0 at ``player_id``, summing to
    1; all-zero if every opponent is dead). Mode (``PRODUCER_PLUS_FFA_WEIGHTS``):

    - ``strength`` (default): weights ∝ rival planet+fleet ships — tilts
      valuation toward damaging the strongest.
    - ``uniform``: equal weight per living rival — removes the leader tilt.
    - ``weakness``: weights ∝ (Σ_living strength − own strength), a bounded
      complement that leans toward the weakest living rival without the
      blow-ups of a raw 1/strength inverse.

    Returns a ``[player_count]`` float tensor.
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
    mode = _ffa_weight_mode()
    if mode == "uniform":
        weights = (strength > 0).to(planets.dtype)
    elif mode == "weakness":
        living = (strength > 0).to(planets.dtype)
        # complement: total living strength minus each rival's own strength,
        # masked to living rivals so the weakest survivor earns the most.
        weights = (float(strength.sum()) - strength) * living
        # With a single living rival the complement is identically zero;
        # fall back to full weight on that rival (the endgame duel must
        # still be valued) rather than nulling the opponent term.
        if float(weights.sum()) <= 0.0:
            weights = living
    else:  # strength
        weights = strength
    total = float(weights.sum())
    if total <= 0.0:
        return torch.zeros(a, dtype=planets.dtype, device=device)
    return weights / total


# --- Commitment cost ----------------------------------------------------------
# Ported insight from the ledger branch (audit/2026-06-10-ledger-agent-from-
# first-principles.md): in-flight ships cannot change course, so committed
# capital is the army you lack when the opponent's wave lands. Our own
# evidence agrees from two sides — top teams strike at flight-time 4-5 vs
# our 7-8, and the replan/redirect family measured that ships held home
# beat every scheme for spending them. Price it: each candidate pays
# eps x ships x flight-turns (per contributing leg). Tempo tie-break toward
# near targets falls out; distant marginal attacks stop clearing the bar.


def _commit_cost_eps() -> float:
    raw = os.environ.get("PRODUCER_PLUS_COMMIT_COST", "0")
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        return 0.0


def _commit_flight_cost(cand_send: Tensor, cand_eta: Tensor, cand_active: Tensor) -> Tensor:
    """Σ_legs ships x eta over active legs. ``[C]`` (ship-turn units)."""
    cost = torch.where(
        cand_active, cand_send * cand_eta, torch.zeros_like(cand_send),
    )
    return cost.sum(dim=-1)


# --- Reinforcement deficit floor (defense candidate sizing fix) -------------
# capture_floor returns 1 for targets we own at the arrival tick ("arriving
# ships add to the garrison, nothing to clear"), so the multi-size enumeration
# for a defensive target is (1, 2, safe_drain) — the "exactly enough to HOLD
# the planet" size is never a candidate, and trickle sends below it are junk
# the greedy must price out one by one. For an owned target the do-nothing
# projection shows flipping at tick k_f, any reinforcement arriving at k <=
# k_f holds the planet iff it adds at least the attacker's projected margin,
# and that margin IS the projection's post-flip ship count at k_f (engine
# survivor = top1 - top2). This floor replaces 1 with (margin + overhead) on
# the pre-flip cells, giving the chooser the right-sized defense candidate
# and invalidating doomed under-sized ones. Default OFF = byte-identical.
def _reinforce_deficit_enabled() -> bool:
    return os.environ.get("PRODUCER_PLUS_REINFORCE_DEFICIT", "0").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _apply_reinforce_deficit_floor(
    floor: Tensor,                  # [T, K] from capture_floor
    *,
    garrison_status,
    target_idx: Tensor,             # [T] long
    player_id: int,
    capture_overhead: float = 1.0,
) -> Tensor:
    """Raise pre-flip reinforcement floors to the hold-the-planet deficit.

    For each target currently ours whose do-nothing projection flips it at
    tick ``k_f`` (first ``owner != me`` within the floor's K window), every
    cell ``k < k_f`` (where the planet is still ours at arrival) gets
    ``floor = max(1, ceil(ships_at_flip + overhead))`` — ``ships_at_flip``
    is the projected post-combat survivor of the new owner, i.e. exactly
    the margin our reinforcement must add to keep the planet. Cells at and
    after ``k_f`` already carry the retake floor from capture_floor's
    not-mine branch. Targets with no projected flip are untouched.
    """
    T, K = int(floor.shape[0]), int(floor.shape[-1])
    if T == 0 or K == 0:
        return floor
    owner = garrison_status.owner
    ships = garrison_status.ships
    P = int(owner.shape[0])
    pid = int(player_id)
    tgt = target_idx.clamp(0, max(P - 1, 0))
    owner_g = owner[tgt]                                    # [T, H+1]
    ships_g = ships[tgt]
    mine_now = owner_g[..., 0] == pid                       # [T]
    not_mine_k = owner_g[..., 1 : K + 1] != pid             # [T, K]
    any_flip = not_mine_k.any(dim=-1) & mine_now            # [T]
    # first flip tick (0-based index into k=1..K), device-stable on ties.
    k_f_idx = _stable_argmax(not_mine_k.to(torch.int64))    # [T]
    ships_at_flip = ships_g.gather(
        -1, (k_f_idx + 1).clamp(max=int(ships_g.shape[-1]) - 1).unsqueeze(-1)
    ).squeeze(-1)                                           # [T]
    deficit = (ships_at_flip + float(capture_overhead)).clamp(min=1.0).ceil()
    k_grid = torch.arange(K, device=floor.device).view(1, K)
    pre_flip = any_flip.view(T, 1) & (k_grid < k_f_idx.view(T, 1))   # [T, K]
    return torch.where(
        pre_flip, torch.maximum(floor, deficit.view(T, 1)), floor,
    )


# --- Overkill factor (mass-concentration attack sizing) ---------------------
# Top-ladder behavioral mining (audit/2026-06-10-top-ladder-behavior.md):
# the 1600-1750 agents launch ~half as often as we do with 2-4x the fleet
# mass (median 36-83 ships vs our 21), expand faster, and hold 2-4x our ship
# count by step 80. In our own 2P losses the opponent's median fleet is 30+
# vs 16 in our wins. The engine's multi-size lo/mid variants are sized at the
# bare capture floor — the minimal send that flips the planet — which wins
# the combat but leaves a 1-ship garrison the opponent retakes past the
# scorer horizon. OVERKILL_FACTOR scales the SIZING of the lo variant
# (floor*F, capped by safe_drain) so enumerated attacks are decisive instead
# of marginal. The floor VALIDITY gate is unchanged (a drain-sized send that
# clears the true floor stays valid), and 1.0 is byte-identical.
def _overkill_factor() -> float:
    raw = os.environ.get("PRODUCER_PLUS_OVERKILL_FACTOR", "1.0")
    try:
        return max(1.0, float(raw))
    except (TypeError, ValueError):
        return 1.0


# --- Mass tie-break ----------------------------------------------------------
# The exact flow scorer values a minimal capture and an overwhelming capture
# of the same target almost identically (both flip the planet within H; the
# surplus ships survive either way), and _stable_argmax then resolves the tie
# toward the LOWEST index — which is the smallest size variant. Retention
# beyond the horizon favors the larger send (the surplus garrisons the
# capture against the counter the scorer can't see). Add an epsilon-scale
# size preference (1e-4 score per ship sent) so near-ties resolve toward
# mass without distorting genuinely different scores. Default OFF.
def _mass_tiebreak_enabled() -> bool:
    return os.environ.get("PRODUCER_PLUS_MASS_TIEBREAK", "0").strip().lower() in (
        "1", "true", "yes", "on",
    )


# --- Regroup convoying -------------------------------------------------------
# 71% of the champion's launches are own-planet transfers at median 18 ships
# (the regroup lane fires near-continuously), while top-ladder agents move
# mass in large convoys (their overall fleet median is 36-83 vs our 21).
# Small parcels are also strictly SLOWER (fleet speed rises with ship
# count). With a positive threshold, regroup entries below it are dropped —
# ships stay garrisoned and accumulate until a convoy-sized transfer fires.
# 0 = OFF, byte-identical.
def _regroup_min_send() -> float:
    raw = os.environ.get("PRODUCER_PLUS_REGROUP_MIN_SEND", "0")
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        return 0.0


# --- Terminal production value -----------------------------------------------
# The flow scorer truncates a captured planet's payoff at the horizon
# (H=18 2P / 13 4P), so a neutral whose in-horizon production only repays its
# garrison cost scores ~0 and never clears the 1.5-ship roi threshold — the
# seed-7 expansion probe shows the planner offered dozens of valid neutral
# captures every opening turn at best-score 0..1 while the bank climbed to
# ~300 ships, expanding only on turns where the opponent projection shifted
# the do-nothing baseline negative. The weight is the number of post-horizon
# steps the production owned at the horizon's final step is credited for.


def _terminal_neutral_only() -> bool:
    return os.environ.get("PRODUCER_PLUS_TERMINAL_PROD_NEUTRAL_ONLY", "0").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _terminal_prod_value() -> float:
    raw = os.environ.get("PRODUCER_PLUS_TERMINAL_PROD_VALUE", "0")
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        return 0.0


# --- Response veto -------------------------------------------------------------
# The opp projection predicts the opponent's plan ASSUMING WE DO NOTHING, so
# reactive defense to our own waves is invisible to the scorer. Live mining
# (audit/2026-06-10-top-ladder-behavior.md): 30% of our capture-sized attacks
# fail to flip, and 65% of those failures die to defense that arrived while
# our fleet was in flight — ~321 ships/game thrown into parries a
# producer-like opponent visibly prepares. One extra mirror pass with OUR
# chosen waves as background yields each opponent's predicted REPLY; attack
# waves whose flow score under that reply is worse than doing nothing (by
# more than the margin) are dropped before dispatch.


def _response_veto_enabled() -> bool:
    return os.environ.get("PRODUCER_PLUS_RESPONSE_VETO", "0").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _response_veto_2p_only() -> bool:
    """Player-count gate so a composed bundle can run the veto in 2P while
    keeping the 4P action stream byte-identical to a measured 4P bundle."""
    return os.environ.get("PRODUCER_PLUS_RESPONSE_VETO_2P_ONLY", "0").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _response_veto_active(player_count: int) -> bool:
    return _response_veto_enabled() and (
        (not _response_veto_2p_only()) or int(player_count) == 2
    )


def _response_veto_upsize_enabled() -> bool:
    """\"Beat the parry\": when the predicted reply kills a wave, retry the
    same target with the source's full spare budget (new aim/eta for the
    bigger, faster fleet) before dropping. The flow scorer judges whether
    over-draining the source is safe — no separate cap."""
    return os.environ.get("PRODUCER_PLUS_RESPONSE_VETO_UPSIZE", "0").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _response_veto_margin(default: float) -> float:
    """Veto margin; defaults to the planner's own roi threshold so the wave
    must clear the SAME gain bar under the predicted reply that it cleared
    without it. (A clean parry is a material-neutral trade — score exactly 0
    — so a zero margin would never veto it.)"""
    raw = os.environ.get("PRODUCER_PLUS_RESPONSE_VETO_MARGIN")
    if raw is None or not raw.strip():
        return float(default)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return float(default)


def _replan_enabled() -> bool:
    """One-ply replan: after our waves are chosen, predict the opponent's
    reply (same mirror as the veto) and run our WHOLE planner a second time
    with that reply as background. Where the veto only drops doomed waves
    (the ships idle), the replan redirects them to the next-best action,
    plans reinforcements against the predicted counter (the reply feeds the
    defensive shortlist), and re-judges every wave with the reply's flow
    consequences in the diff."""
    return os.environ.get("PRODUCER_PLUS_REPLAN", "0").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _replan_2p_only() -> bool:
    return os.environ.get("PRODUCER_PLUS_REPLAN_2P_ONLY", "0").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _replan_active(player_count: int) -> bool:
    return _replan_enabled() and (
        (not _replan_2p_only()) or int(player_count) == 2
    )


# --- Background-aware floors -------------------------------------------------
# The flow SCORER sees the opponent's predicted launches (they're merged into
# every candidate's diff), but the SIZING subsystem — capture_floor, the
# defensive shortlist, safe_drain — reads the frozen do-nothing projection.
# Three measured behaviours trace to that inconsistency: attacks sized for
# garrisons that get reinforced mid-flight (the scorer then rejects the
# right-sized wave it was never offered), no toll-sniping of predicted
# captures (after THEIR fleet annihilates against a neutral, the survivor is
# cheap — invisible to static floors), and drains/regroups out of planets a
# predicted strike is about to hit. Fix: re-project the garrison trajectories
# ONCE with the background launches applied (exact engine recurrence — the
# same one the scorer trusts) and let the sizing subsystem read that.


def _bg_floors_enabled() -> bool:
    return os.environ.get("PRODUCER_PLUS_BG_FLOORS", "0").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _background_adjusted_status(
    garrison_status, *, background: LaunchSet, prod: Tensor, alive_by_step: Tensor,
):
    """Garrison trajectories with the background launches applied. New status.

    Sources are debited at step 0 (a launch leaves now even if it lands past
    the horizon); arrivals land at ``ceil(eta)`` like the scorer's hypothesis
    axis; the exact production→combat recurrence is replayed. ``alive_by_step``
    is ``[H+1, P]`` (run_turn's orientation).
    """
    owner0 = garrison_status.owner[..., 0]                       # [P]
    ships0 = garrison_status.ships[..., 0]                       # [P]
    arr = garrison_status.arrivals_by_owner                      # [P, H+1, A]
    P, H1, A = int(arr.shape[0]), int(arr.shape[1]), int(arr.shape[2])
    H = H1 - 1
    fdtype = ships0.dtype if ships0.is_floating_point() else torch.float32

    sel = background.valid
    src = background.source_slots[sel].clamp(0, max(P - 1, 0))
    tgt = background.target_slots[sel].clamp(0, max(P - 1, 0))
    ships = background.ships[sel].to(fdtype)
    own = background.owner[sel].clamp(0, max(A - 1, 0))
    tick = background.eta[sel].ceil().long().clamp(min=1)

    init_ships = ships0.to(fdtype).clone()
    init_ships.index_add_(0, src, -ships)
    init_ships = init_ships.clamp(min=0.0)

    arr_delta = arr[:, 1:, :].to(fdtype).clone()                  # [P, H, A]
    in_h = tick <= H
    if bool(in_h.any()):
        arr_delta.index_put_(
            (tgt[in_h], tick[in_h] - 1, own[in_h]), ships[in_h], accumulate=True,
        )

    owner_t, ships_t, pre_o, pre_s = _run_exact_recurrence(
        init_owner=owner0.unsqueeze(0),
        init_ships=init_ships.unsqueeze(0),
        prod=prod.to(fdtype).unsqueeze(0),
        alive=alive_by_step.transpose(0, 1).unsqueeze(0),
        arrivals=arr_delta.unsqueeze(0),
    )
    return PlanetGarrisonStatus(
        owner=owner_t[0], ships=ships_t[0],
        pre_combat_owner=pre_o[0], pre_combat_ships=pre_s[0],
        arrivals_by_owner=torch.cat([arr[:, :1, :].to(fdtype), arr_delta], dim=1),
    )


def _entries_to_launch_set(entries, *, pid: int, device, dtype) -> LaunchSet:
    """Valid rows of a LaunchEntries table as a LaunchSet owned by ``pid``."""
    sel = entries.valid.nonzero(as_tuple=True)[0]
    return LaunchSet(
        source_slots=entries.source_slots[sel].to(torch.long),
        target_slots=entries.target_slots[sel].to(torch.long),
        ships=entries.ships[sel].to(dtype),
        eta=entries.eta[sel].to(dtype),
        owner=torch.full((int(sel.shape[0]),), pid, dtype=torch.long, device=device),
        valid=torch.ones(int(sel.shape[0]), dtype=torch.bool, device=device),
    )


def _reply_seq_enabled() -> bool:
    """Sequential multi-rival reply conditioning — see comment in
    _predict_reply. No-op with a single opponent (2P byte-identical)."""
    return os.environ.get("PRODUCER_PLUS_REPLY_SEQ", "0").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _predict_reply(
    mine: LaunchSet,
    *,
    movement,
    obs_tensors: dict,
    cache,
    garrison_status,
    prod: Tensor,
    alive_by_step: Tensor,
    config,
    player_count: int,
    pid: int,
    K_eta_override,
    H: int,
) -> LaunchSet:
    """Each opponent's predicted reply to OUR launches, merged on the L axis.

    Mirror each opponent seat separately WITH the roi normalization their
    planner needs: with our waves as background, every opp candidate's
    flow diff inherits our attacks' damage as a large negative constant,
    so against the absolute 1.5 threshold the simulated opponent is
    paralyzed and "replies" with nothing (seed-0 instrumented game: 15
    predicted reply launches across 107 turns, 0 vetoes). Shift the
    threshold by THEIR do-nothing score, exactly as run_turn does for us.
    """
    opp_ids = [q for q in range(int(player_count)) if q != int(pid)]
    seq = _reply_seq_enabled() and len(opp_ids) > 1
    reply_parts = []
    pad = _env_int("PRODUCER_PLUS_OPP_MAX_L", MAX_L_OPP)
    base = mine
    for opp_id in opp_ids:
        dn_opp = float(_score_do_nothing(
            status=garrison_status, prod=prod, alive_by_step=alive_by_step,
            player_count=int(player_count), background=base,
            player_id=int(opp_id), opp_weights=None,
        ))
        cfg_opp = dataclasses.replace(
            config, roi_threshold=dn_opp + float(config.roi_threshold),
        )
        part = predict_opp_launches_via_mirror(
            plan_fn=plan_lite_waves,
            obs_tensors=obs_tensors, movement=movement, cache=cache,
            garrison_status=garrison_status, prod=prod, alive_by_step=alive_by_step,
            opp_ids=[int(opp_id)], config=cfg_opp, player_count=int(player_count),
            K_eta_override=K_eta_override,
            pad_to=pad,
            K=1, H=H,
            base_background=base,
        )
        reply_parts.append(part)
        if seq:
            # Sequential conditioning (PRODUCER_PLUS_REPLY_SEQ): later rivals
            # see earlier rivals' predicted launches, not just ours. The
            # independent merge prices every attack as if ALL rivals parry
            # it simultaneously with full attention (defense counted once
            # per rival) — measured to make the ungated 4P veto chronically
            # passive (eliminated by step ~200; panel 1/16). Conditioning
            # divides each rival's attention by the threats already on the
            # board, like the K-round projection does for one opponent.
            base = _cat_launch_sets([base, part])
    if len(reply_parts) == 1:
        return reply_parts[0]
    return _cat_launch_sets(reply_parts)


def _cat_launch_sets(parts: list) -> LaunchSet:
    """Concatenate LaunchSets along the L axis."""
    if len(parts) == 1:
        return parts[0]
    return LaunchSet(
        source_slots=torch.cat([r.source_slots for r in parts]),
        target_slots=torch.cat([r.target_slots for r in parts]),
        ships=torch.cat([r.ships for r in parts]),
        eta=torch.cat([r.eta for r in parts]),
        owner=torch.cat([r.owner for r in parts]),
        valid=torch.cat([r.valid for r in parts]),
    )


def _apply_response_veto(
    entries,
    *,
    movement,
    obs,
    obs_tensors: dict,
    cache,
    garrison_status,
    prod: Tensor,
    alive_by_step: Tensor,
    config,
    player_count: int,
    K_eta_override,
    H: int,
    opp_weights,
    reply_out: list | None = None,
    reply_trust: float | None = None,
):
    """Drop attack waves the opponent's predicted reply kills. See gate note.

    ``reply_out``: optional mutable list; the predicted reply LaunchSet is
    appended when the mirror runs, so a downstream pass (the redirect) can
    reuse it without a second mirror. ``reply_trust``: certainty-equivalent
    scaling of the reply's ships (None = full trust).
    """
    pid = int(obs.player_id)
    P = int(obs.P)
    valid = entries.valid
    if int(valid.sum().item()) == 0:
        return entries
    device = obs.device
    dtype = obs.ships.dtype
    tgt_safe = entries.target_slots.clamp(0, P - 1)
    is_attack = valid & ~obs.owned[tgt_safe]
    idx = is_attack.nonzero(as_tuple=True)[0]
    C = int(idx.shape[0])
    if C == 0:
        return entries

    sel = valid.nonzero(as_tuple=True)[0]
    mine = _entries_to_launch_set(entries, pid=pid, device=device, dtype=dtype)
    reply = _predict_reply(
        mine,
        movement=movement, obs_tensors=obs_tensors, cache=cache,
        garrison_status=garrison_status, prod=prod, alive_by_step=alive_by_step,
        config=config, player_count=int(player_count), pid=pid,
        K_eta_override=K_eta_override, H=H,
    )
    if reply_trust is not None:
        reply = _scale_launch_set_ships(reply, float(reply_trust))
    if reply_out is not None:
        reply_out.append(reply)

    # Score each attack wave alone under the predicted reply, against the
    # do-nothing-under-reply baseline (same normalization as roi_threshold).
    cand = make_launch_set(
        source_slots=entries.source_slots[idx].view(C, 1),
        target_slots=entries.target_slots[idx].view(C, 1),
        ships=entries.ships[idx].view(C, 1).to(dtype),
        eta=entries.eta[idx].view(C, 1).to(dtype),
        valid=torch.ones(C, 1, dtype=torch.bool, device=device),
        player_id=pid,
    )

    def _bg(t: Tensor) -> Tensor:
        return t.unsqueeze(0).expand(C, -1)

    merged = LaunchSet(
        source_slots=torch.cat([cand.source_slots, _bg(reply.source_slots)], dim=-1),
        target_slots=torch.cat([cand.target_slots, _bg(reply.target_slots)], dim=-1),
        ships=torch.cat([cand.ships, _bg(reply.ships)], dim=-1),
        eta=torch.cat([cand.eta, _bg(reply.eta)], dim=-1),
        owner=torch.cat([cand.owner, _bg(reply.owner)], dim=-1),
        valid=torch.cat([cand.valid, _bg(reply.valid)], dim=-1),
    )
    scores = score_candidates(
        garrison_status, prod=prod, alive_by_step=alive_by_step,
        player_count=int(player_count), launches=merged, player_id=pid,
        opp_weights=opp_weights, terminal_prod_weight=_terminal_prod_value(),
        terminal_neutral_only=_terminal_neutral_only(),
    )
    if _hold_value() > 0.0:
        # Price the holding-time capture credit consistently in the veto:
        # without this the veto re-scores hold-value-justified captures at
        # their raw ~0 flow and drops every launch the credit enabled
        # (verified on the Gregor Lied trace: 4 waves pre-veto, 0 post).
        tgt_e = entries.target_slots[idx].clamp(0, P - 1)
        K_v = max(1, min(
            int(K_eta_override) if K_eta_override is not None else int(config.horizon),
            H,
        ))
        _rf_w_v = _reactive_floor_for(int(player_count))
        _rf_m_v = (
            _reactive_reinforcement_margin(obs, cache, tgt_e, K_v, weight=_rf_w_v)
            if _rf_w_v > 0.0 else None
        )
        floor_e = capture_floor(
            garrison_status, target_idx=tgt_e, k_max=K_v,
            capture_overhead=1.0, player_id=pid, reinforcement=_rf_m_v,
        )                                                            # [C, K_b]
        K_b = int(floor_e.shape[-1])
        if K_b > 0:
            scores = scores + _hold_value_bonus(
                obs=obs, cache=cache, target_idx=tgt_e,
                cand_tgt_slot=tgt_e,
                cand_tgt_short=torch.arange(C, device=device),
                cand_send=entries.ships[idx].view(C, 1).to(dtype),
                cand_eta=entries.eta[idx].view(C, 1).to(dtype),
                cand_valid=torch.ones(C, dtype=torch.bool, device=device),
                cand_is_def=obs.owned[tgt_e],
                capture_floor_TK=floor_e, prod=prod, K=K_b,
            )
    dn = _score_do_nothing(
        status=garrison_status, prod=prod, alive_by_step=alive_by_step,
        player_count=int(player_count), background=reply, player_id=pid,
        opp_weights=opp_weights,
    )
    margin = _response_veto_margin(float(config.roi_threshold))
    keep = (scores - dn) >= margin
    if bool(keep.all()):
        return entries

    new_valid = entries.valid.clone()
    new_ships = entries.ships.clone()
    new_angle = entries.angle.clone()
    new_eta = entries.eta.clone()
    drop = idx[~keep]
    new_valid[drop] = False

    if _response_veto_upsize_enabled() and movement is not None and int(drop.shape[0]) > 0:
        # "Beat the parry": retry each killed wave at the source's full
        # spare budget (everything not already committed by the plan),
        # with aim/eta recomputed for the bigger — and therefore FASTER —
        # fleet. The flow scorer judges whether stripping the source is
        # safe (the debit is part of the diff); only drop when even the
        # full send fails the margin under the reply.
        committed = torch.zeros(P, dtype=dtype, device=device)
        committed.scatter_add_(
            0, entries.source_slots[sel].clamp(0, P - 1), entries.ships[sel].to(dtype),
        )
        D = int(drop.shape[0])
        d_src = entries.source_slots[drop].clamp(0, P - 1)
        d_tgt = entries.target_slots[drop].clamp(0, P - 1)
        spare = (obs.ships.to(dtype)[d_src] - committed[d_src]).clamp(min=0.0).floor()
        up_size = entries.ships[drop].to(dtype) + spare
        aim = intercept_angle(movement, d_src, d_tgt, up_size)
        up_viable = (spare >= 1.0) & aim["viable"] & (aim["eta"] <= float(H))
        if bool(up_viable.any()):
            cand_up = make_launch_set(
                source_slots=d_src.view(D, 1),
                target_slots=d_tgt.view(D, 1),
                ships=up_size.view(D, 1),
                eta=aim["eta"].to(dtype).view(D, 1),
                valid=up_viable.view(D, 1),
                player_id=pid,
            )

            def _bgD(t: Tensor) -> Tensor:
                return t.unsqueeze(0).expand(D, -1)

            merged_up = LaunchSet(
                source_slots=torch.cat([cand_up.source_slots, _bgD(reply.source_slots)], dim=-1),
                target_slots=torch.cat([cand_up.target_slots, _bgD(reply.target_slots)], dim=-1),
                ships=torch.cat([cand_up.ships, _bgD(reply.ships)], dim=-1),
                eta=torch.cat([cand_up.eta, _bgD(reply.eta)], dim=-1),
                owner=torch.cat([cand_up.owner, _bgD(reply.owner)], dim=-1),
                valid=torch.cat([cand_up.valid, _bgD(reply.valid)], dim=-1),
            )
            scores_up = score_candidates(
                garrison_status, prod=prod, alive_by_step=alive_by_step,
                player_count=int(player_count), launches=merged_up, player_id=pid,
                opp_weights=opp_weights, terminal_prod_weight=_terminal_prod_value(),
        terminal_neutral_only=_terminal_neutral_only(),
            )
            keep_up = up_viable & ((scores_up - dn) >= margin)
            if bool(keep_up.any()):
                ui = drop[keep_up]
                new_valid[ui] = True
                new_ships[ui] = up_size[keep_up].to(new_ships.dtype)
                new_angle[ui] = aim["angle"][keep_up].to(new_angle.dtype)
                new_eta[ui] = aim["eta"][keep_up].to(new_eta.dtype)

    return LaunchEntries(
        source_slots=entries.source_slots, target_slots=entries.target_slots,
        ships=new_ships, angle=new_angle, eta=new_eta,
        valid=new_valid,
    )


def _apply_replan(
    entries,
    *,
    movement,
    obs,
    obs_tensors: dict,
    cache,
    garrison_status,
    prod: Tensor,
    alive_by_step: Tensor,
    config,
    player_count: int,
    K_eta_override,
    H: int,
    opp_weights,
):
    """One-ply replan: re-run the planner with the predicted reply as background.

    Pass 1 (the caller's ``entries``) planned against the opponent's
    do-nothing-conditioned launches. This pass predicts each opponent's
    best response to OUR pass-1 waves and plans from scratch against it:
    waves the reply kills are not just dropped but their ships redirected,
    reinforcements appear against the predicted counter (the reply feeds
    ``friendly_flip_targets``), and every candidate's flow diff carries the
    reply's consequences. The roi threshold is re-normalized by our
    do-nothing-under-reply score, mirroring run_turn's opp-projection shift.

    Skips (returns pass 1 unchanged) when pass 1 fired nothing — the reply
    to an empty plan is what pass 1 already planned against — or when the
    predicted reply is empty (nothing to adapt to).
    """
    pid = int(obs.player_id)
    if int(entries.valid.sum().item()) == 0:
        return entries
    device = obs.device
    dtype = obs.ships.dtype
    mine = _entries_to_launch_set(entries, pid=pid, device=device, dtype=dtype)
    reply = _predict_reply(
        mine,
        movement=movement, obs_tensors=obs_tensors, cache=cache,
        garrison_status=garrison_status, prod=prod, alive_by_step=alive_by_step,
        config=config, player_count=int(player_count), pid=pid,
        K_eta_override=K_eta_override, H=H,
    )
    if int(reply.valid.sum().item()) == 0:
        return entries
    dn = float(_score_do_nothing(
        status=garrison_status, prod=prod, alive_by_step=alive_by_step,
        player_count=int(player_count), background=reply, player_id=pid,
        opp_weights=opp_weights,
    ))
    cfg2 = dataclasses.replace(
        config, roi_threshold=dn + float(config.roi_threshold),
    )
    return plan_lite_waves(
        movement=movement, obs=obs, obs_tensors=obs_tensors, cache=cache,
        garrison_status=garrison_status, prod=prod,
        alive_by_step=alive_by_step, config=cfg2,
        player_count=int(player_count), K_eta_override=K_eta_override,
        background=reply, opp_weights=opp_weights,
    )


# --- Redirect ---------------------------------------------------------------
# The veto is a filter: when the predicted reply kills a wave, the freed
# ships idle. The full one-ply replan fixed that but measured 2-2 on paired
# seeds with a clear failure mode (decision_diff seed 0: 16 capture-sized
# launches vs the live stack's 24) — pass 2 treats predicted PARRIES as
# fixed background even for attacks it then doesn't make, so the whole plan
# goes conservative. The redirect keeps pass 1 + veto untouched and re-plans
# ONLY the freed budget: surviving waves are committed (sources debited,
# their effects + the reply in the scorer background), and one extra planner
# pass spends what the veto freed on next-best actions. No reopened
# commitments -> no phantom-parry suppression of the plan.


def _redirect_enabled() -> bool:
    return os.environ.get("PRODUCER_PLUS_REDIRECT", "0").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _redirect_2p_only() -> bool:
    return os.environ.get("PRODUCER_PLUS_REDIRECT_2P_ONLY", "0").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _redirect_active(player_count: int) -> bool:
    return _redirect_enabled() and (
        (not _redirect_2p_only()) or int(player_count) == 2
    )


def _apply_redirect(
    entries,
    *,
    reply: LaunchSet,
    movement,
    obs,
    obs_tensors: dict,
    cache,
    garrison_status,
    prod: Tensor,
    alive_by_step: Tensor,
    config,
    player_count: int,
    K_eta_override,
    H: int,
    opp_weights,
):
    """Spend the veto-freed budget on next-best actions. Appends new waves.

    ``entries`` is the post-veto table (some rows invalidated). The surviving
    waves are committed: their sends are debited from the planner's view of
    our garrisons and their effects ride in the scorer background alongside
    the predicted ``reply``, so a second wave at an already-attacked target
    scores ~0 marginal and is naturally suppressed. The roi threshold is
    re-normalized by the do-nothing score under that combined background.
    """
    pid = int(obs.player_id)
    P = int(obs.P)
    device = obs.device
    dtype = obs.ships.dtype
    committed = _entries_to_launch_set(entries, pid=pid, device=device, dtype=dtype)
    if int(committed.source_slots.shape[-1]) > 0:
        debit = torch.zeros_like(obs.ships)
        debit.scatter_add_(
            0, committed.source_slots.clamp(0, P - 1), committed.ships.to(obs.ships.dtype),
        )
        obs2 = dataclasses.replace(obs, ships=(obs.ships - debit).clamp(min=0.0))
        bg2 = _cat_launch_sets([reply, committed])
    else:
        obs2 = obs
        bg2 = reply
    dn = float(_score_do_nothing(
        status=garrison_status, prod=prod, alive_by_step=alive_by_step,
        player_count=int(player_count), background=bg2, player_id=pid,
        opp_weights=opp_weights,
    ))
    cfg2 = dataclasses.replace(
        config, roi_threshold=dn + float(config.roi_threshold),
    )
    extra = plan_lite_waves(
        movement=movement, obs=obs2, obs_tensors=obs_tensors, cache=cache,
        garrison_status=garrison_status, prod=prod,
        alive_by_step=alive_by_step, config=cfg2,
        player_count=int(player_count), K_eta_override=K_eta_override,
        background=bg2, opp_weights=opp_weights,
    )
    if int(extra.valid.sum().item()) == 0:
        return entries
    return LaunchEntries(
        source_slots=torch.cat([entries.source_slots, extra.source_slots]),
        target_slots=torch.cat([entries.target_slots, extra.target_slots]),
        ships=torch.cat([entries.ships, extra.ships]),
        angle=torch.cat([entries.angle, extra.angle]),
        eta=torch.cat([entries.eta, extra.eta]),
        valid=torch.cat([entries.valid, extra.valid]),
    )


# --- Reply trust --------------------------------------------------------------
# Everything reply-conditioned (the veto, the projection background) assumes
# the rivals run our planner. Against producer-derived opponents that mirror
# is near-exact; against originals it is confidently wrong, and a wrong
# parry prediction vetoes good attacks. Honest fix: VERIFY the model online.
# Each turn, check whether last turn's predicted launches materialized as
# real fleets (matched by source planet + owner, ships within 2x), keep an
# exponential moving accuracy, and price replies at trust-scaled strength
# (certainty-equivalent: a reply believed with p=0.4 carries 0.4x ships).
# Producer-likes: trust stays high, behavior unchanged. Originals: the veto
# degrades gracefully toward the unconditioned stack instead of parrying
# ghosts.


def _reply_trust_enabled() -> bool:
    return os.environ.get("PRODUCER_PLUS_REPLY_TRUST", "0").strip().lower() in (
        "1", "true", "yes", "on",
    )


_REPLY_TRUST_FLOOR = 0.25
_REPLY_TRUST_ALPHA = 0.2


def _record_reply_prediction(memory, background: LaunchSet | None, obs_tensors: dict) -> None:
    """Stash this turn's predicted opp launches + current fleet ids for
    next turn's verification."""
    fleets = obs_tensors["fleets"]
    ids = fleets[..., 0].long()
    memory.trust_fleet_ids = set(int(i) for i in ids.tolist() if int(i) >= 0)
    preds = []
    if background is not None and int(background.source_slots.shape[-1]) > 0:
        planets = obs_tensors["planets"]
        pid_of_slot = planets[..., 0].long()
        sel = background.valid.nonzero(as_tuple=True)[0]
        for i in sel.tolist():
            src_slot = int(background.source_slots[i].item())
            preds.append((
                int(pid_of_slot[src_slot].item()),          # source planet id
                int(background.owner[i].item()),
                float(background.ships[i].item()),
            ))
    memory.trust_predictions = preds


def _update_reply_trust(memory, obs_tensors: dict, *, pid: int) -> float:
    """EMA prediction recall; returns current trust in [floor, 1]."""
    trust = getattr(memory, "trust_ema", None)
    if trust is None:
        trust = 1.0                       # start trusting (the live behavior)
    preds = getattr(memory, "trust_predictions", None)
    known_ids = getattr(memory, "trust_fleet_ids", None)
    if preds:
        fleets = obs_tensors["fleets"]
        new_enemy = []
        for row in fleets.tolist():
            fleet_id, owner, _x, _y, _ang, from_id, ships = row[:7]
            if int(fleet_id) < 0 or int(owner) == int(pid):
                continue
            if known_ids is not None and int(fleet_id) in known_ids:
                continue
            new_enemy.append((int(from_id), int(owner), float(ships)))
        matched = 0
        pool = list(new_enemy)
        for p_src, p_owner, p_ships in preds:
            hit = None
            for j, (f_src, f_owner, f_ships) in enumerate(pool):
                if f_src == p_src and f_owner == p_owner and (
                    0.5 * p_ships <= f_ships <= 2.0 * p_ships
                ):
                    hit = j
                    break
            if hit is not None:
                matched += 1
                pool.pop(hit)
        recall = matched / len(preds)
        trust = (1.0 - _REPLY_TRUST_ALPHA) * trust + _REPLY_TRUST_ALPHA * recall
    memory.trust_ema = trust
    return max(_REPLY_TRUST_FLOOR, min(1.0, trust))


def _scale_launch_set_ships(launches: LaunchSet, factor: float) -> LaunchSet:
    if factor >= 1.0:
        return launches
    return LaunchSet(
        source_slots=launches.source_slots, target_slots=launches.target_slots,
        ships=launches.ships * float(factor), eta=launches.eta,
        owner=launches.owner, valid=launches.valid,
    )


# --- Opening search ----------------------------------------------------------
# The pre-contact opening is a deterministic single-player scheduling problem
# (PI thesis; neutral garrisons are static, planet motion is rigid rotation,
# production compounds, and the total-ship lead stops changing hands by step
# 30-54 — so opening production IS the game). The greedy planner expands "by
# accident" (horizon-truncated capture payoffs). This ports the beam search
# from scripts/opening_optimum.py in-agent: each turn while step < window,
# search capture schedules maximizing total production by the opening
# horizon, and emit the launches due NOW. The rest of the pipeline (defense
# lane, veto, regroup) runs as usual on the remaining budget. Pure Python,
# time-boxed; turn budget headroom is ~800 ms.


def _opening_search_window() -> int:
    return max(0, _env_int("PRODUCER_PLUS_OPENING_SEARCH", 0))


def _opening_search_horizon() -> int:
    return max(10, _env_int("PRODUCER_PLUS_OPENING_HORIZON", 40))


def _opening_search_beam() -> int:
    return max(8, _env_int("PRODUCER_PLUS_OPENING_BEAM", 64))


_OPENING_TIMEBOX_S = 0.25
_LOG1000 = math.log(1000.0)
_BOARD_CENTER = 50.0


def _fleet_speed_py(s: float) -> float:
    if s <= 1:
        return 1.0
    return 1.0 + 5.0 * (math.log(min(s, 1000.0)) / _LOG1000) ** 1.5


class _OpeningBoard:
    """Deterministic kinematics + static garrisons from the CURRENT obs.

    t=0 is *now*: angles are taken from current positions, so re-planning
    every turn stays consistent as planets rotate.
    """

    def __init__(self, obs_tensors: dict, pid: int):
        planets = obs_tensors["planets"].detach().cpu()
        self.angvel = float(obs_tensors["angular_velocity"].flatten()[0].item())
        self.planets: dict[int, dict] = {}
        self.mine: list[int] = []
        self.enemy: list[int] = []
        self.neutrals: list[int] = []
        for row in planets.tolist():
            planet_id, owner, x, y, r, ships, prod = row[:7]
            planet_id = int(planet_id)
            if planet_id < 0:
                continue
            ox, oy = x - _BOARD_CENTER, y - _BOARD_CENTER
            orb_r = math.hypot(ox, oy)
            self.planets[planet_id] = dict(
                owner=int(owner), r=float(r), ships=float(ships),
                prod=float(prod), orb_r=orb_r, a0=math.atan2(oy, ox),
                orbiting=(orb_r + float(r)) < _BOARD_CENTER,
            )
            if int(owner) == int(pid):
                self.mine.append(planet_id)
            elif int(owner) >= 0:
                self.enemy.append(planet_id)
            else:
                self.neutrals.append(planet_id)

    def pos(self, planet_id: int, t: float):
        p = self.planets[planet_id]
        a = p["a0"] + (self.angvel * t if p["orbiting"] else 0.0)
        return (_BOARD_CENTER + p["orb_r"] * math.cos(a),
                _BOARD_CENTER + p["orb_r"] * math.sin(a))

    def eta(self, src: int, tgt: int, size: float, t: float) -> int:
        sp = _fleet_speed_py(size)
        sx, sy = self.pos(src, t)
        e = 1.0
        for _ in range(4):
            tx, ty = self.pos(tgt, t + e)
            d = math.hypot(tx - sx, ty - sy) - self.planets[tgt]["r"]
            e = max(1.0, d / sp)
        return max(1, math.ceil(e))


def _opening_search_plan(
    obs_tensors: dict, *, pid: int, claimed: set[int],
    horizon: int, beam_width: int, timebox_s: float = _OPENING_TIMEBOX_S,
) -> list[tuple[int, int, float]]:
    """Beam-search the capture schedule; return launches due NOW.

    Returns ``[(src_planet_id, tgt_planet_id, size)]`` for schedule entries
    with launch time 0. ``claimed`` excludes neutrals already targeted by
    our in-flight opening waves (they're treated as spoken for).
    Safe-only filter: only neutrals at least as reachable by us as by any
    enemy planet (race-losing targets are the midgame planner's problem).
    """
    board = _OpeningBoard(obs_tensors, pid)
    if not board.mine:
        return []
    neutrals = []
    safe_margin = float(_env_int("PRODUCER_PLUS_OPENING_SAFE_MARGIN", 3))
    for n in board.neutrals:
        if n in claimed:
            continue
        g = board.planets[n]["ships"] + 1.0
        ours = min(board.eta(s, n, g, 0) for s in board.mine)
        if board.enemy:
            # Contested-race guard (canon: safe/contested/unsafe neutrals;
            # measured: lane losses on collapse seeds are race losses,
            # 182 wasted ships vs the referee's converging fleets). The
            # bare ours<=theirs test uses the enemy's CURRENT planets,
            # but their reach grows as they expand — demand a margin.
            theirs = min(board.eta(s, n, g, 0) for s in board.enemy)
            if ours + safe_margin > theirs:
                continue
        neutrals.append(n)
    if not neutrals:
        return []

    t_start = time.perf_counter()
    # State: (t, owned dict items, captured frozenset, produced, flights, plan)
    start = (0.0, tuple((p, board.planets[p]["ships"]) for p in board.mine),
             frozenset(), 0.0, (), ())

    def advance(state, until):
        t, owned_t, captured, produced, flights, plan = state
        owned = dict(owned_t)
        fl = sorted(flights)
        while t < until:
            step_to = until
            if fl and fl[0][0] < step_to:
                step_to = fl[0][0]
            dt = step_to - t
            for p in owned:
                owned[p] += board.planets[p]["prod"] * dt
            produced += sum(board.planets[p]["prod"] for p in owned) * dt
            t = step_to
            while fl and fl[0][0] <= t:
                _at, tgt, size = fl.pop(0)
                g = board.planets[tgt]["ships"]
                owned[tgt] = max(1.0, size - g)
                captured = captured | {tgt}
        return (t, tuple(sorted(owned.items())), captured, produced,
                tuple(fl), plan)

    def held_value(state):
        fin = advance(state, float(horizon))
        return fin[3]

    best_value = held_value(start)
    best_plan: tuple = ()
    frontier = [start]
    for _depth in range(10):
        if time.perf_counter() - t_start > timebox_s:
            break
        nxt = []
        for state in frontier:
            t, owned_t, captured, produced, flights, plan = state
            owned = dict(owned_t)
            for n in neutrals:
                if n in captured or n in owned:
                    continue
                g = board.planets[n]["ships"] + 1.0
                for src in owned:
                    have = owned[src]
                    prod_src = board.planets[src]["prod"]
                    need = g + 1.0          # keep 1 ship home
                    wait = 0.0 if have >= need else (
                        math.inf if prod_src <= 0
                        else math.ceil((need - have) / prod_src))
                    t_launch = t + wait
                    if t_launch >= horizon:
                        continue
                    e = board.eta(src, n, g, t_launch)
                    if t_launch + e >= horizon + 10:
                        continue
                    s2 = advance(state, t_launch)
                    t2, owned2_t, cap2, prod2, fl2, plan2 = s2
                    owned2 = dict(owned2_t)
                    if owned2.get(src, 0.0) < need:
                        continue
                    owned2[src] -= g
                    fl3 = tuple(sorted(fl2 + ((t_launch + e, n, g),)))
                    plan3 = plan2 + ((t_launch, src, n, g),)
                    nxt.append((t2, tuple(sorted(owned2.items())), cap2,
                                prod2, fl3, plan3))
        if not nxt:
            break

        def h(s):
            t, owned_t, _cap, produced, fl, _plan = s
            rate = sum(board.planets[p]["prod"] for p, _ in owned_t)
            opt = produced + rate * (horizon - t)
            for at, tgt, _sz in fl:
                if at < horizon:
                    opt += board.planets[tgt]["prod"] * (horizon - at)
            return opt

        nxt.sort(key=h, reverse=True)
        frontier = nxt[:beam_width]
        for state in frontier:
            v = held_value(state)
            if v > best_value:
                best_value = v
                best_plan = state[5]

    return [(src, tgt, size) for (t_launch, src, tgt, size) in best_plan
            if t_launch <= 0.5]


def _opening_reserve_k() -> int:
    """Worst-case reserve window in turns (0 = off). Planet Wars canon
    (Melis's full-attack future): ships may leave only if the source
    survives a POSSIBLE strike, not just the fleets already in flight —
    the do-nothing projection is blind pre-contact, which is exactly when
    the searcher launches. The reserve = enemy garrison mass that could
    reach the source within this window (full-garrison fleet speed)."""
    return max(0, _env_int("PRODUCER_PLUS_OPENING_RESERVE_K", 8))


def _opening_reserve_filter(
    rows: list[tuple[int, int, float]],
    ships_by_slot: dict[int, float],
    reserve_by_slot: dict[int, float],
) -> list[tuple[int, int, float]]:
    """Drop launches whose source would dip below its worst-case reserve."""
    return [
        (s, t, size) for (s, t, size) in rows
        if ships_by_slot.get(s, 0.0) - size >= reserve_by_slot.get(s, 0.0)
    ]


def _opening_hold_filter(
    rows: list[tuple[int, int, float]], drain_by_slot: dict[int, float],
) -> list[tuple[int, int, float]]:
    """Drop scheduled launches the source can't afford under hold discipline.

    The searcher's keep-1-home rule is a single-player safety model — its
    first measured composition stripped sources bare and was punished
    (attribution leg: -27% @120, one map dead by step 115). A capture wave
    is all-or-nothing: clamping below the garrison floor just annihilates,
    so unaffordable launches are SKIPPED (the per-turn re-plan retries when
    the garrison has grown).
    """
    return [
        (s, t, size) for (s, t, size) in rows
        if size <= drain_by_slot.get(s, 0.0)
    ]


def _emit_opening_entries(
    due: list[tuple[int, int, float]], *, movement, obs, obs_tensors: dict,
    garrison_status, H: int, cache=None,
):
    """Aim the due launches with the REAL intercept solver. LaunchEntries."""
    device = obs.device
    dtype = obs.ships.dtype
    pid = int(obs.player_id)
    planet_ids = obs_tensors["planets"][..., 0].long()
    P = int(obs.P)
    slot_of = {int(planet_ids[i].item()): i for i in range(P)}
    rows = []
    for src_pid, tgt_pid, size in due:
        s, t = slot_of.get(src_pid), slot_of.get(tgt_pid)
        if s is None or t is None:
            continue
        size = float(min(size, max(float(obs.ships[s].item()) - 1.0, 0.0)))
        if size < 1.0:
            continue
        rows.append((s, t, size))
    if rows:
        src_slots = torch.tensor([r[0] for r in rows], dtype=torch.long, device=device)
        drains = safe_drain(
            garrison_status, source_idx=src_slots,
            source_ships=obs.ships[src_slots].to(dtype),
            H_eff=torch.full((), float(H), dtype=dtype, device=device),
            player_id=pid,
        )
        drain_by_slot = {
            int(src_slots[i].item()): float(drains[i].item())
            for i in range(int(src_slots.shape[0]))
        }
        rows = _opening_hold_filter(rows, drain_by_slot)
    _rk = _opening_reserve_k()
    if rows and _rk > 0 and cache is not None:
        src_slots = torch.tensor([r[0] for r in rows], dtype=torch.long, device=device)
        margin = _reactive_reinforcement_margin(
            obs, cache, src_slots, _rk, weight=1.0, lag=0.0,
        )
        if margin is not None:
            reserve_by_slot = {
                int(src_slots[i].item()): float(margin[i, _rk - 1].item())
                for i in range(int(src_slots.shape[0]))
            }
            ships_by_slot = {
                int(src_slots[i].item()): float(obs.ships[src_slots[i]].item())
                for i in range(int(src_slots.shape[0]))
            }
            rows = _opening_reserve_filter(rows, ships_by_slot, reserve_by_slot)
    if not rows:
        return None
    src = torch.tensor([r[0] for r in rows], dtype=torch.long, device=device)
    tgt = torch.tensor([r[1] for r in rows], dtype=torch.long, device=device)
    ships = torch.tensor([r[2] for r in rows], dtype=dtype, device=device)
    aim = intercept_angle(movement, src, tgt, ships)
    ok = aim["viable"]
    if not bool(ok.any()):
        return None
    return LaunchEntries(
        source_slots=src, target_slots=tgt, ships=ships,
        angle=aim["angle"].to(dtype), eta=aim["eta"].to(dtype),
        valid=ok,
    )


# --- Neutral shortlist quota -----------------------------------------------------
# The offensive shortlist is the N nearest enemy-or-neutral planets with NO
# class quota: once a frontline forms, the nearest non-owned planets are
# mostly ENEMY planets and neutral expansion targets are crowded out of the
# candidate set entirely (seed-7 probe: neutral candidate counts collapse
# from 20-45 to 0-8 at first contact; SiestaGuru loss: zero neutral captures
# for 40 steps). A visibility defect independent of valuation — the scorer
# never sees the option. The quota appends the Q nearest neutral targets not
# already shortlisted.


def _neutral_shortlist_quota() -> int:
    return max(0, _env_int("PRODUCER_PLUS_NEUTRAL_SHORTLIST", 0))


def _append_neutral_quota(
    target_idx: Tensor, target_exists: Tensor, *, obs, cache, source_mask,
    K_eta: int, quota: int,
):
    """Append the nearest `quota` neutral targets missing from the shortlist."""
    neutral_mask = obs.is_neutral & obs.alive
    if not bool(neutral_mask.any()):
        return target_idx, target_exists
    proximity = min_distance_to_targets(cache, source_mask, neutral_mask, max_k=int(K_eta))
    pref = torch.where(
        neutral_mask, -proximity, torch.full_like(proximity, float("-inf")))
    n_idx, n_exists = _candidate_indices(pref, neutral_mask, int(quota))
    dup = (n_idx.view(-1, 1) == target_idx.view(1, -1)).any(dim=-1)
    n_exists = n_exists & ~dup
    return (
        torch.cat([target_idx, n_idx], dim=0),
        torch.cat([target_exists, n_exists], dim=0),
    )


def _append_deficit_targets(
    target_idx: Tensor, target_exists: Tensor, *, obs, cache, prod, K_eta: int,
):
    """Append own planets under STANDING-reserve deficit to the shortlist.

    friendly_flip_targets only admits planets projected to flip from
    IN-FLIGHT fleets / predicted launches — a planet facing a visibly
    massing but uncommitted reserve never becomes a candidate, so the
    garrison-value bonus has nothing to credit and pre-positioning is
    structurally impossible (RYOTA loss: the 135-massing was watchable for
    30 ticks; planner had zero defensive candidates for the victim).
    Appends the top-R (R = living rivals) positive-deficit own planets,
    ranked by production. Only active when garrison value is on.
    """
    own_idx = (obs.owned & obs.alive).nonzero(as_tuple=True)[0]
    n_own = int(own_idx.shape[0])
    if n_own == 0:
        return target_idx, target_exists
    K = int(K_eta)
    threat = _reactive_reinforcement_margin(
        obs, cache, own_idx, K,
        weight=_garrison_value_threat_w(), lag=_source_safety_lag(),
        concentration_speed=True,
    )
    if threat is None:
        return target_idx, target_exists
    dtype = obs.ships.dtype
    help_tk = _friendly_support_margin(obs, cache, own_idx, K)
    if help_tk is None:
        help_tk = torch.zeros_like(threat)
    k_grid = torch.arange(1, K + 1, device=obs.device, dtype=dtype).view(1, K)
    base = (
        obs.ships[own_idx].to(dtype).unsqueeze(-1)
        + prod[own_idx].to(dtype).unsqueeze(-1) * k_grid
        + help_tk
    )
    deficit = (threat - base).max(dim=-1).values                      # [n_own]
    R = max(int(_living_rival_count(obs)), 1)
    pref = torch.where(
        deficit > 0.0, prod[own_idx].to(dtype) * 1.0e6 + deficit,
        torch.full_like(deficit, float("-inf")))
    take = min(R, n_own)
    top = pref.topk(take)
    d_idx = own_idx[top.indices]
    d_exists = top.values > float("-inf")
    dup = (d_idx.view(-1, 1) == target_idx.view(1, -1)).any(dim=-1)
    d_exists = d_exists & ~dup
    return (
        torch.cat([target_idx, d_idx], dim=0),
        torch.cat([target_exists, d_exists], dim=0),
    )


# --- Reactive floor -------------------------------------------------------------
# capture_floor's `reinforcement` margin hook has been dormant (always None):
# enemy floors assume the defender's garrison sits still while our fleet
# flies. SiestaGuru loss anatomy (episode 79438024): 9 capture-sized strikes
# failed — 700 ships — to defense routed in during our eta-7..8 flights,
# which the 1-ply response veto cannot see either. This margin adds, per
# target t and arrival turn k, the garrison the defender can ROUTE to t by
# k: enemy planets q whose travel q->t fits within (k − reaction lag), at
# the speed of their full garrison (big fleets fly faster). Weight scales
# the margin (1.0 = assume full rerouting).


def _reactive_floor_weight() -> float:
    raw = os.environ.get("PRODUCER_PLUS_REACTIVE_FLOOR", "0")
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        return 0.0


def _reactive_floor_for(player_count: int) -> float:
    """Player-count gate (same pattern as the veto's): a composed bundle can
    run the reactive floor in 2P while keeping 4P byte-identical."""
    only2p = os.environ.get(
        "PRODUCER_PLUS_REACTIVE_FLOOR_2P_ONLY", "0").strip().lower() in (
        "1", "true", "yes", "on")
    if only2p and int(player_count) != 2:
        return 0.0
    return _reactive_floor_weight()


def _reactive_floor_lag() -> float:
    """Reaction lag in turns before rerouted defense starts counting
    (PRODUCER_PLUS_REACTIVE_FLOOR_LAG, default 2.0 — the value the floor
    shipped with; exposed for joint knob tuning)."""
    raw = os.environ.get("PRODUCER_PLUS_REACTIVE_FLOOR_LAG", "2.0")
    try:
        return float(raw)
    except (TypeError, ValueError):
        return 2.0


def _rotation_aware_margins() -> bool:
    """Gate: margin reach tests use the per-k intercept distance
    ``cross_dist[k]`` (planet positions rotate; striking the approaching
    neighbor is ~50% faster at 90-degree separation on a 36-radius ring)
    instead of the static ``cross_dist[0]``. The static path systematically
    misjudges who can reach whom in time, in BOTH directions. Default 0 =
    byte-identical."""
    return os.environ.get("PRODUCER_PLUS_ROTATION_AWARE_MARGINS", "0").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _margin_reach(cache, q_idx: Tensor, target_idx: Tensor, speed_q: Tensor,
                  K: int, lag: float, P: int):
    """``[Q, T, K]`` bool — can q's garrison reach t by arrival turn k.

    Rotation-aware when gated: q reaches t at turn k iff
    ``dist(q@0, t@k) <= speed * (k - lag)`` using the cache's per-k slices.
    Static path (default): one distance, one eta, threshold per k.
    """
    tgt = target_idx.clamp(0, P - 1)
    dtype = speed_q.dtype
    device = speed_q.device
    k_grid = torch.arange(1, K + 1, device=device, dtype=dtype)      # [K]
    budget = (k_grid.view(1, 1, K) - float(lag)).clamp(min=0.0) * speed_q.view(-1, 1, 1)
    if _rotation_aware_margins():
        K_cache = int(cache.cross_dist.shape[0]) - 1
        ks = torch.arange(1, K + 1, device=device).clamp(max=K_cache)  # [K]
        d_k = cache.cross_dist[ks][:, q_idx][:, :, tgt]              # [K, Q, T]
        return d_k.permute(1, 2, 0) <= budget                        # [Q, T, K]
    d = cache.cross_dist[0][q_idx][:, tgt]                           # [Q, T]
    return d.unsqueeze(-1) <= budget


def _reactive_reinforcement_margin(
    obs, cache, target_idx: Tensor, K: int, *, weight: float, lag: float | None = None,
    concentration_speed: bool = False,
):
    """``[T, K]`` reroutable enemy support per target/arrival-turn, or None.

    ``concentration_speed``: price reach at the speed of the enemy's
    COMBINED strength rather than each garrison's own size. Fleet speed
    grows with mass, so a relayed concentration (backline -> staging ->
    strike) flies faster than its components — the RYOTA loss pattern: an
    84-ship backline garrison was "out of range" at its own speed yet
    arrived inside the window after merging into a 135-stack. Used by the
    garrison-value deficit (a planet must be defensible against what the
    enemy CAN concentrate, not what currently sits in range).
    """
    if lag is None:
        lag = _reactive_floor_lag()
    enemy = obs.is_enemy & obs.alive
    q_idx = enemy.nonzero(as_tuple=True)[0]
    Q = int(q_idx.shape[0])
    T = int(target_idx.shape[0])
    if Q == 0 or T == 0 or K <= 0:
        return None
    dtype = obs.ships.dtype
    g_q = obs.ships[q_idx].to(dtype).clamp(min=1.0)                  # [Q]
    if concentration_speed:
        speed_q = fleet_speed(g_q.sum().clamp(min=1.0).view(1)).expand(Q)
    else:
        speed_q = fleet_speed(g_q)                                   # [Q]
    reach = _margin_reach(cache, q_idx, target_idx, speed_q, K,
                          float(lag), int(obs.P))                    # [Q, T, K]
    # The target's own garrison is already the defender — exclude q == t.
    self_mask = q_idx.view(Q, 1) == target_idx.view(1, T)
    reach = reach & ~self_mask.unsqueeze(-1)
    support = (g_q.view(Q, 1, 1) * reach.to(dtype)).sum(dim=0)       # [T, K]
    return float(weight) * support


# --- Forward redistribution ----------------------------------------------------


def _regroup_forward_enabled() -> bool:
    return os.environ.get("PRODUCER_PLUS_REGROUP_FORWARD", "0").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _regroup_forward_time(default: float) -> float:
    raw = os.environ.get("PRODUCER_PLUS_REGROUP_FORWARD_TIME")
    if raw is None or not raw.strip():
        return max(float(default), 12.0)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return max(float(default), 12.0)


# --- 2P-only gate for the mass mechanisms ------------------------------------
# Local evidence splits by player count: mass beats the champion head-to-head
# in 2P (35/64) and holds vs producer (22/32), but costs first-place rate in
# the 4P pool. With this gate set, MASS_TIEBREAK / REGROUP_MIN_SEND /
# OVERKILL_FACTOR apply only when player_count == 2; 3+ player games keep
# champion behavior (and compose with the 4P-only FFA objective fix).
def _mass_2p_only() -> bool:
    return os.environ.get("PRODUCER_PLUS_MASS_2P_ONLY", "0").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _mass_active(player_count: int) -> bool:
    return (not _mass_2p_only()) or int(player_count) == 2


def _overkill_factor_for(player_count: int) -> float:
    return _overkill_factor() if _mass_active(player_count) else 1.0


# --- Class-split overkill -----------------------------------------------------
# Replay mining of the top-3 teams (mine_decision_rules.py, appended to
# audit/2026-06-10-top-ladder-behavior.md): attack sizing is CLASS-dependent —
# ~1.3x the garrison on neutral targets (cheap, front-loaded expansion) but
# 2.6-4.6x at median (7.5-10x at p75) on enemy planets, with 60-89-ship
# median fleets. A single overkill factor over-sizes neutral grabs and
# under-sizes enemy strikes. Unset -> the single-knob path, bit-identical.


def _overkill_factor_enemy() -> float | None:
    raw = os.environ.get("PRODUCER_PLUS_OVERKILL_FACTOR_ENEMY")
    if raw is None or not raw.strip():
        return None
    try:
        return max(1.0, float(raw))
    except (TypeError, ValueError):
        return None


def _overkill_for_targets(obs, target_idx: Tensor, player_count: int, dtype):
    """Scalar (legacy) or per-target ``[T]`` overkill multiplier for sizes_lo."""
    base = _overkill_factor_for(player_count)
    enemy = _overkill_factor_enemy() if _mass_active(player_count) else None
    if enemy is None:
        return base
    is_enemy_t = obs.is_enemy[target_idx.clamp(0, int(obs.P) - 1)]
    return torch.where(
        is_enemy_t,
        torch.full_like(is_enemy_t, enemy, dtype=dtype),
        torch.full_like(is_enemy_t, base, dtype=dtype),
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


# --- Smart dropout (model-free robustness) ----------------------------------
# Instead of MODELLING the opponent's launches, perturb our own forward
# rollout: assume that a planet we CAPTURE this turn is taken back by the
# opponent a few turns later, garrisoned with the enemy's physically-routable
# mass. Each candidate is then scored TWICE — once with the capture held
# (optimist) and once with it dropped — and the two scores are averaged
# (PI sign-off 2026-06-18: "average, keep optimist"). Captures the opponent
# could never reach are not dropped, so the perturbation is grounded in
# reachable enemy mass rather than a guessed constant. This replaces the
# opponent-modelling layers (opp projection / response veto) with one rollout
# perturbation that propagates through the exact production->combat recurrence
# the scorer already trusts. Default OFF preserves byte-identical scoring.
#
# Engine mapping: a "drop" is a credit-only enemy arrival injected at the
# captured planet (no source debit -> source_slots set out of range so the
# scorer's source-validity gate drops the debit). The minimal flip size
# (our projected garrison + 1) is used because the competitive penalty is
# INVARIANT to any larger enemy arrival (surplus enemy ships are relocated,
# not produced or lost) — minimal sizing flips the planet without
# over-crediting enemy combat losses.
def _dropout_enabled() -> bool:
    return os.environ.get("PRODUCER_PLUS_DROPOUT", "0").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _dropout_weight() -> float:
    """Weight on the dropped (pessimistic) scenario in the average. 0.5 = the
    PI's "average, keep optimist": equal weight on hold and drop. 0 reproduces
    the clean static score; 1 is fully pessimistic (worst-case-ish)."""
    raw = os.environ.get("PRODUCER_PLUS_DROPOUT_WEIGHT", "0.5")
    try:
        return min(1.0, max(0.0, float(raw)))
    except (TypeError, ValueError):
        return 0.5


def _dropout_lag() -> float:
    """Turns after our capture-arrival at which the opponent's reflip lands.
    A short recapture delay; the enemy needs flight time to reach the planet
    we just took."""
    raw = os.environ.get("PRODUCER_PLUS_DROPOUT_LAG", "2.0")
    try:
        return max(1.0, float(raw))
    except (TypeError, ValueError):
        return 2.0


def _dropout_reflip_legs(
    *,
    obs,
    cache,
    prod: Tensor,
    target_idx: Tensor,        # [T] shortlist slots
    cand_tgt_slot: Tensor,     # [C] planet slot each candidate targets
    cand_tgt_short: Tensor,    # [C] index into target_idx
    cand_send: Tensor,         # [C, L]
    cand_eta: Tensor,          # [C, L]
    cand_active: Tensor,       # [C, L] bool
    cand_is_def: Tensor,       # [C] bool (target is ours = reinforcement)
    cand_valid: Tensor,        # [C] bool
    floor: Tensor,             # [T, K] capture_floor (defenders + overhead)
    K: int,
    K_eta: int,
    player_count: int,
    pid: int,
    device,
    dtype,
):
    """Per-candidate enemy reflip leg as a ``[C, 1]`` LaunchSet, or ``None``.

    A leg is valid only for candidates that CAPTURE an enemy/neutral planet
    (not reinforcements of our own) AND where the strongest living rival's
    physically-routable mass to that planet, by the reflip tick, can beat the
    garrison we'd be holding there. The leg is credited to the strongest
    rival's owner id, lands ``_dropout_lag`` turns after our arrival, and is
    sized to the minimal flip.
    """
    C = int(cand_send.shape[0])
    P = int(obs.P)
    if C == 0 or K <= 0 or int(K_eta) <= 0:
        return None
    A = int(player_count)

    # Strongest living rival -> the owner id the reflip is credited to.
    enemy = obs.is_enemy & obs.alive
    if not bool(enemy.any()):
        return None
    strength = torch.zeros(A, dtype=dtype, device=device)
    strength.scatter_add_(
        0, obs.owner_abs[enemy].long().clamp(0, A - 1), obs.ships[enemy].to(dtype),
    )
    strength[int(pid)] = float("-inf")
    rival = int(torch.argmax(strength).item())
    if float(strength[rival]) <= 0.0:
        return None

    # Enemy mass that can physically arrive at each shortlist target by tick k
    # (no reaction lag — this is the worst-case reachable mass).
    margin = _reactive_reinforcement_margin(
        obs, cache, target_idx, int(K_eta), weight=1.0, lag=0.0,
    )                                                                  # [T, K_eta] | None
    if margin is None:
        return None
    K_marg = int(margin.shape[-1])

    send_tot = (cand_send * cand_active.to(dtype)).sum(dim=-1)         # [C]
    eta_active = torch.where(cand_active, cand_eta, torch.zeros_like(cand_eta))
    eta_max = eta_active.max(dim=-1).values                            # [C] our capture arrival
    k_arr = (eta_max.clamp(min=1.0, max=float(K)).ceil().long() - 1).clamp(0, K - 1)  # [C]
    floor_at_arr = (
        floor[cand_tgt_short.clamp(0, floor.shape[0] - 1)]
        .gather(-1, k_arr.unsqueeze(-1)).squeeze(-1)
    )                                                                  # [C]
    # capture_floor = defenders + overhead(1); garrison we hold just after
    # capture = send - defenders = send - floor + 1.
    survivors = (send_tot - floor_at_arr + 1.0).clamp(min=1.0)         # [C]

    lag = _dropout_lag()
    reflip_eta = (eta_max + float(lag)).clamp(min=1.0, max=float(K))   # [C]
    k_reflip = (reflip_eta.ceil().long() - 1).clamp(0, K - 1)          # [C] 0-based
    tgt_safe = cand_tgt_slot.clamp(0, P - 1)
    prod_t = prod[tgt_safe].to(dtype)                                  # [C]
    dk = (k_reflip - k_arr).clamp(min=0).to(dtype)                     # turns held before reflip
    garrison_at_reflip = survivors + prod_t * dk                       # [C]

    k_marg = k_reflip.clamp(0, K_marg - 1)
    enemy_mass = (
        margin[cand_tgt_short.clamp(0, margin.shape[0] - 1)]
        .gather(-1, k_marg.unsqueeze(-1)).squeeze(-1).to(dtype)
    )                                                                  # [C]

    # Only CAPTURES of an enemy/neutral planet are droppable; reinforcements
    # of our own planets are not (cand_is_def). And only where the rival can
    # actually overpower the garrison we'd hold.
    is_capture = cand_valid & ~cand_is_def & ~obs.owned[tgt_safe]       # [C]
    can_flip = is_capture & (enemy_mass >= garrison_at_reflip + 1.0)    # [C]

    reflip_ships = torch.where(
        can_flip, garrison_at_reflip + 1.0,
        torch.zeros(C, dtype=dtype, device=device),
    )
    # source_slots = -1 -> the scorer's source-validity gate (src >= 0) drops
    # the debit, so this is a credit-only flip (no friendly planet is drained).
    return LaunchSet(
        source_slots=torch.full((C, 1), -1, dtype=torch.long, device=device),
        target_slots=tgt_safe.view(C, 1),
        ships=reflip_ships.view(C, 1),
        eta=reflip_eta.view(C, 1),
        owner=torch.full((C, 1), int(rival), dtype=torch.long, device=device),
        valid=can_flip.view(C, 1),
    )


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
    sync_sink: list | None = None,
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
    # Background-aware floors: the sizing subsystem (shortlist flips, drain,
    # capture floors) reads trajectories with the predicted opponent launches
    # applied; the SCORER below keeps the static baseline because it merges
    # the background into every candidate's diff itself. ``bg_flip=None``
    # because the adjusted trajectories already contain the predicted flips.
    status_sizing = garrison_status
    bg_flip = background
    if (
        _bg_floors_enabled() and background is not None
        and int(background.source_slots.shape[-1]) > 0
        and bool(background.valid.any())
    ):
        status_sizing = _background_adjusted_status(
            garrison_status, background=background, prod=prod,
            alive_by_step=alive_by_step,
        )
        bg_flip = None
    target_idx, target_exists = build_target_shortlist(
        obs, obs_tensors, status_sizing, cache,
        config=config, K_eta=K_eta, H=H, prod=prod, source_mask=source_mask,
        background=bg_flip,
    )
    _nq = _neutral_shortlist_quota()
    if _nq > 0:
        target_idx, target_exists = _append_neutral_quota(
            target_idx, target_exists, obs=obs, cache=cache,
            source_mask=source_mask, K_eta=K_eta, quota=_nq,
        )
    if _garrison_value() > 0.0:
        # Standing-reserve deficits must be visible as defensive targets,
        # not just projected flips, or pre-positioning can never fire.
        target_idx, target_exists = _append_deficit_targets(
            target_idx, target_exists, obs=obs, cache=cache,
            prod=prod, K_eta=K_eta,
        )
    if not bool(target_exists.any()):
        return _empty_entries(device, dtype)
    S = int(source_idx.shape[0])
    T = int(target_idx.shape[0])
    target_is_mine = obs.owned[target_idx.clamp(0, P - 1)]                       # [T]

    source_ships = obs.ships[source_idx.clamp(0, P - 1)].to(dtype)                # [S]
    H_eff = torch.full((), float(H), dtype=dtype, device=device)
    drain = safe_drain(
        status_sizing, source_idx=source_idx, source_ships=source_ships,
        H_eff=H_eff, player_id=pid,
    )                                                                            # [S]
    _ss_allow = _source_safety_allowance(
        obs, cache, source_idx=source_idx, prod=prod, K=int(K_eta),
    )
    if _ss_allow is not None:
        # Second cap: keep every source locally defensible against the
        # enemy's UNCOMMITTED reserve (safe_drain only sees in-flight).
        drain = torch.minimum(drain, _ss_allow)

    # Uniform reach cap = K_eta (= horizon).
    eta_cap = torch.full((T,), float(K_eta), dtype=dtype, device=device)          # [T]

    _rf_w = _reactive_floor_for(int(player_count))
    _rf_margin = (
        _reactive_reinforcement_margin(
            obs, cache, target_idx, int(K_eta), weight=_rf_w,
        ) if _rf_w > 0.0 else None
    )
    floor = capture_floor(
        status_sizing, target_idx=target_idx, k_max=K_eta,
        capture_overhead=1.0, player_id=pid,
        reinforcement=_rf_margin,
    )                                                                            # [T, K]
    if _reinforce_deficit_enabled():
        floor = _apply_reinforce_deficit_floor(
            floor, garrison_status=status_sizing, target_idx=target_idx,
            player_id=pid, capture_overhead=1.0,
        )
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

        sizes_lo = torch.minimum((floor_at_arr_hi * _overkill_for_targets(obs, target_idx, player_count, dtype)).ceil().clamp(min=1.0), sizes_hi)
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
        sizes_lo = torch.minimum((floor_at_arr_hi * _overkill_for_targets(obs, target_idx, player_count, dtype)).ceil().clamp(min=1.0), sizes_hi) # [S, T]
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

        # --- Sync-pair stage (PRODUCER_PLUS_SYNC; only on the REAL planning
        # pass, sync_sink=None on mirror/replan passes keeps them unchanged).
        # Two-source candidates on targets NEITHER source cracks alone but a
        # joint same-tick arrival does: leg sizes = safe_drain, joint floor
        # read at the LATER leg's arrival tick, the nearer leg scored at the
        # far leg's eta (its launch is deferred via a memory hold; d=0 pairs
        # are plain same-tick coalitions and launch immediately).
        _sy_ksrc = min(_sync_k_src(), S)
        if sync_sink is not None and K > 0 and _sy_ksrc >= 2:
            sizes_hi_v = sizes_3[..., -1]                                         # [S, T]
            eta_hi_v = eta_3[..., -1]
            angle_hi_v = angle_3[..., -1]
            k_arr_hi = k_arr_3[..., -1]
            clears_hi = clears_floor_3[..., -1]
            viable_only_hi = (
                viable_3[..., -1] & (sizes_hi_v >= 1.0) & ships_ok_3[..., -1]
                & src_neq_tgt & source_exists.view(S, 1) & target_exists.view(1, T)
            )                                                                     # [S, T]
            ranked = torch.where(
                viable_only_hi, -eta_hi_v, torch.full_like(eta_hi_v, float("-inf"))
            ).transpose(0, 1)                                                     # [T, S]
            top_src = _stable_topk_indices(ranked, _sy_ksrc)                       # [T, Ksrc]
            pair_idx = torch.triu_indices(_sy_ksrc, _sy_ksrc, offset=1, device=device)
            pa, pb = pair_idx[0], pair_idx[1]
            Pp = int(pa.numel())
            if Pp > 0:
                Tidx = torch.arange(T, device=device).view(T, 1).expand(T, Pp)
                sA = top_src[:, pa]                                               # [T, Pp]
                sB = top_src[:, pb]
                kA = k_arr_hi[sA, Tidx]
                kB = k_arr_hi[sB, Tidx]
                a_is_far = kA >= kB
                k_sync = torch.maximum(kA, kB)
                d_gap = (kA - kB).abs()
                floor_sync = floor.gather(-1, k_sync.clamp(0, K - 1))             # [T, Pp]
                szA = sizes_hi_v[sA, Tidx]
                szB = sizes_hi_v[sB, Tidx]
                valid_pair = (
                    viable_only_hi[sA, Tidx] & viable_only_hi[sB, Tidx]
                    & ~clears_hi[sA, Tidx] & ~clears_hi[sB, Tidx]
                    & ((szA + szB) >= floor_sync)
                    & (source_idx[sA] != source_idx[sB])
                    & (d_gap <= _sync_dmax())
                )                                                                 # [T, Pp]
                # --- floor-proportional pair sizing. Full safe_drain per leg
                # doubles committed capital on one target (the week's known
                # disease — confirmed by the first mirror leg: 1/12, in-flight
                # share 67% vs 45%). Right-size the pair to the joint floor ×
                # overkill, split proportionally to each leg's drain; smaller
                # fleets fly SLOWER, so re-aim and re-check the floor at the
                # later arrival. Pairs where the lo sizing fails the re-check
                # fall back to full drain (the gate proved drain clears).
                ov_t = _overkill_for_targets(obs, target_idx, player_count, dtype)
                if not torch.is_tensor(ov_t):   # scalar (legacy) form
                    ov_t = torch.full((T,), float(ov_t), dtype=dtype, device=device)
                need = (floor_sync * ov_t.view(T, 1)).ceil().clamp(min=2.0)        # [T, Pp]
                pair_sum = (szA + szB).clamp(min=1.0)
                nA = (need * szA / pair_sum).ceil().clamp(min=1.0)
                nB = (need * szB / pair_sum).ceil().clamp(min=1.0)
                src_a_slots = source_idx[sA]                                       # [T, Pp]
                src_b_slots = source_idx[sB]
                tgt_pp = target_idx.view(T, 1).expand(T, Pp)
                aim_loA = intercept_angle(movement, src_a_slots, tgt_pp, nA, active=valid_pair)
                aim_loB = intercept_angle(movement, src_b_slots, tgt_pp, nB, active=valid_pair)
                etaLA = aim_loA["eta"]
                etaLB = aim_loB["eta"]
                kLA = (etaLA.clamp(min=1.0, max=float(K)).ceil().long() - 1).clamp(0, K - 1)
                kLB = (etaLB.clamp(min=1.0, max=float(K)).ceil().long() - 1).clamp(0, K - 1)
                k_sync_lo = torch.maximum(kLA, kLB)
                floor_lo = floor.gather(-1, k_sync_lo)
                cap_pp = eta_cap.view(T, 1)
                lo_ok = (
                    valid_pair & aim_loA["viable"] & aim_loB["viable"]
                    & (etaLA <= cap_pp) & (etaLB <= cap_pp)
                    & ((nA + nB) >= floor_lo)
                )
                szA_f = torch.where(lo_ok, nA, szA)
                szB_f = torch.where(lo_ok, nB, szB)
                etaA_f = torch.where(lo_ok, etaLA, eta_hi_v[sA, Tidx])
                etaB_f = torch.where(lo_ok, etaLB, eta_hi_v[sB, Tidx])
                angA_f = torch.where(lo_ok, aim_loA["angle"], angle_hi_v[sA, Tidx])
                angB_f = torch.where(lo_ok, aim_loB["angle"], angle_hi_v[sB, Tidx])
                kA_f = torch.where(lo_ok, kLA, kA)
                kB_f = torch.where(lo_ok, kLB, kB)
                a_is_far = kA_f >= kB_f
                k_sync = torch.maximum(kA_f, kB_f)
                d_gap = (kA_f - kB_f).abs()
                valid_pair = valid_pair & (d_gap <= _sync_dmax())
                if bool((valid_pair & (d_gap > 0)).any()):
                    # Delayed pairs telegraph: the far leg is visible for the
                    # whole hold window, so the joint size must clear the
                    # FULL-reaction floor (reinforcement weight 1.0, no
                    # reaction lag) at the synced arrival — not the live
                    # stack's discounted 0.5/lag-2 floor. This is what the
                    # -46% mirror rout of unconditioned holds was made of.
                    margin_full = _reactive_reinforcement_margin(
                        obs, cache, target_idx, int(K_eta), weight=1.0, lag=0.0,
                    )
                    if margin_full is not None:
                        floor_full = capture_floor(
                            status_sizing, target_idx=target_idx, k_max=K_eta,
                            capture_overhead=1.0, player_id=pid,
                            reinforcement=margin_full,
                        )
                        ff_sync = floor_full.gather(-1, k_sync.clamp(0, K - 1))
                        valid_pair = valid_pair & (
                            (d_gap == 0) | ((szA_f + szB_f) >= ff_sync))
                eta_far = torch.where(a_is_far, etaA_f, etaB_f)
                delayed_a = (~a_is_far) & (d_gap > 0)
                delayed_b = a_is_far & (d_gap > 0)
                etaA_eff = torch.where(delayed_a, eta_far, etaA_f)
                etaB_eff = torch.where(delayed_b, eta_far, etaB_f)

                C_sy = T * Pp
                m = valid_pair
                zero = torch.zeros_like(szA)
                one = torch.ones_like(etaA_f)
                sy_src = torch.stack(
                    [src_a_slots.reshape(C_sy), src_b_slots.reshape(C_sy)], dim=-1)
                sy_send = torch.stack(
                    [torch.where(m, szA_f, zero).reshape(C_sy),
                     torch.where(m, szB_f, zero).reshape(C_sy)], dim=-1)
                sy_angle = torch.stack(
                    [angA_f.reshape(C_sy), angB_f.reshape(C_sy)], dim=-1)
                sy_eta = torch.stack(
                    [torch.where(m, etaA_eff, one).reshape(C_sy),
                     torch.where(m, etaB_eff, one).reshape(C_sy)], dim=-1)
                sy_active = torch.stack([m.reshape(C_sy), m.reshape(C_sy)], dim=-1)
                sy_tgt_short = (
                    torch.arange(T, device=device).view(T, 1).expand(T, Pp).reshape(C_sy)
                )
                sy_tgt_slot = target_idx[sy_tgt_short]
                sy_valid = m.reshape(C_sy)

                for t_i, p_i in (m & (d_gap > 0)).nonzero(as_tuple=False).tolist():
                    far_a = bool(a_is_far[t_i, p_i].item())
                    sz_far, sz_near = (szA_f, szB_f) if far_a else (szB_f, szA_f)
                    sl_far, sl_near = (src_a_slots, src_b_slots) if far_a else (src_b_slots, src_a_slots)
                    sync_sink.append({
                        "near_src": int(sl_near[t_i, p_i].item()),
                        "far_src": int(sl_far[t_i, p_i].item()),
                        "tgt": int(target_idx[t_i].item()),
                        "eta": float(eta_far[t_i, p_i].item()),
                        "near_ships": float(sz_near[t_i, p_i].item()),
                        "far_ships": float(sz_far[t_i, p_i].item()),
                        "arrival_dt": int(k_sync[t_i, p_i].item()) + 1,
                    })

                base0 = cand_src[:, 0]
                cand_src = torch.cat([torch.stack([base0, base0], dim=-1), sy_src], dim=0)
                cand_send = torch.cat(
                    [torch.cat([cand_send, torch.zeros(C, 1, dtype=dtype, device=device)], dim=-1),
                     sy_send], dim=0)
                cand_angle = torch.cat(
                    [torch.cat([cand_angle, torch.zeros(C, 1, dtype=dtype, device=device)], dim=-1),
                     sy_angle], dim=0)
                cand_eta = torch.cat(
                    [torch.cat([cand_eta, torch.ones(C, 1, dtype=dtype, device=device)], dim=-1),
                     sy_eta], dim=0)
                cand_active = torch.cat(
                    [torch.cat([cand_active, torch.zeros(C, 1, dtype=torch.bool, device=device)], dim=-1),
                     sy_active], dim=0)
                cand_tgt_slot = torch.cat([cand_tgt_slot, sy_tgt_slot], dim=0)
                cand_tgt_short = torch.cat([cand_tgt_short, sy_tgt_short], dim=0)
                cand_valid = torch.cat([cand_valid, sy_valid], dim=0)
                L = 2
                C = int(cand_src.shape[0])

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
        terminal_prod_weight=_terminal_prod_value(),
        terminal_neutral_only=_terminal_neutral_only(),
    )                                                                            # [C]
    if _dropout_enabled():
        # Smart dropout: re-score every candidate in a world where the planet
        # it captures is reflipped to the strongest rival a few turns later,
        # then average the held (optimist) and dropped (pessimist) scores. The
        # reflip legs are enemy-owned arrivals appended to the candidate's
        # scoring LaunchSet (exactly how `background` opp launches merge); the
        # greedy selector below still operates on the clean [C, L] tensors, so
        # the enemy legs never enter its budget / role-mutex view.
        _drop = _dropout_reflip_legs(
            obs=obs, cache=cache, prod=prod, target_idx=target_idx,
            cand_tgt_slot=cand_tgt_slot, cand_tgt_short=cand_tgt_short,
            cand_send=cand_send, cand_eta=cand_eta, cand_active=cand_active,
            cand_is_def=cand_is_def, cand_valid=cand_valid, floor=floor,
            K=K, K_eta=int(K_eta), player_count=int(player_count), pid=pid,
            device=device, dtype=dtype,
        )
        if _drop is not None and bool(_drop.valid.any()):
            scoring_drop = LaunchSet(
                source_slots=torch.cat([scoring_launches.source_slots, _drop.source_slots], dim=-1),
                target_slots=torch.cat([scoring_launches.target_slots, _drop.target_slots], dim=-1),
                ships=torch.cat([scoring_launches.ships, _drop.ships], dim=-1),
                eta=torch.cat([scoring_launches.eta, _drop.eta], dim=-1),
                owner=torch.cat([scoring_launches.owner, _drop.owner], dim=-1),
                valid=torch.cat([scoring_launches.valid, _drop.valid], dim=-1),
            )
            score_drop = score_candidates(
                garrison_status, prod=prod, alive_by_step=alive_by_step,
                player_count=int(player_count), launches=scoring_drop, player_id=pid,
                opp_weights=opp_weights,
                terminal_prod_weight=_terminal_prod_value(),
                terminal_neutral_only=_terminal_neutral_only(),
            )                                                                    # [C]
            _w = _dropout_weight()
            score = (1.0 - _w) * score + _w * score_drop
    _cc = _commit_cost_eps()
    if _cc > 0.0:
        score = score - _cc * _commit_flight_cost(cand_send, cand_eta, cand_active)
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
    if _hold_value() > 0.0:
        # Holding-time-priced capture credit: post-horizon production for
        # captures the opponent cannot feasibly retake within the window.
        score = score + _hold_value_bonus(
            obs=obs, cache=cache, target_idx=target_idx,
            cand_tgt_slot=cand_tgt_slot, cand_tgt_short=cand_tgt_short,
            cand_send=cand_send, cand_eta=cand_eta,
            cand_valid=cand_valid, cand_is_def=cand_is_def,
            capture_floor_TK=floor, prod=prod, K=K,
        )
    if _garrison_value() > 0.0 and (
        int(obs_tensors["step"].max().item()) >= _garrison_value_from_step()
    ):
        # Proactive-garrison credit: reinforcing an own planet whose local
        # balance vs the enemy's uncommitted reserve is negative.
        score = score + _garrison_value_bonus(
            obs=obs, cache=cache, target_idx=target_idx,
            cand_tgt_slot=cand_tgt_slot, cand_tgt_short=cand_tgt_short,
            cand_send=cand_send, cand_eta=cand_eta,
            cand_valid=cand_valid, cand_is_def=cand_is_def,
            prod=prod, K=K,
        )
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
                terminal_prod_weight=_terminal_prod_value(),
        terminal_neutral_only=_terminal_neutral_only(),
            )
            _new = new_base + _fc_addon_offset
            return torch.where(cand_valid, _new, torch.full_like(_new, float("-inf")))

        _fc_rescore_fn = _fc_rescore
    if _mass_tiebreak_enabled() and _mass_active(player_count):
        total_send = (cand_send * cand_active.to(cand_send.dtype)).sum(dim=-1)  # [C]
        score = score + 1e-4 * total_send
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
    # Forward redistribution (Planet Wars canon, confirmed by the del Toro
    # loss: 121 idle rear garrison vs their 39 at step 40): the default
    # "materially more stressed" gate (delta 0.25) strands leftover ships on
    # rear planets with no local gradient. The forward gate lowers the delta
    # to "any strictly forward flow" and extends the flight cap, so rear
    # garrisons stream toward the frontier turn after turn. Strictly-positive
    # gap keeps the flow one-directional (no backwash loops).
    cfg_regroup = config
    if _regroup_forward_enabled():
        cfg_regroup = dataclasses.replace(
            config,
            regroup_pressure_delta_min=0.0,
            max_regroup_time=_regroup_forward_time(float(config.max_regroup_time)),
        )
    regroup_entries = _plan_regroup(
        movement=movement, obs=obs, obs_tensors=obs_tensors, garrison_status=garrison_status,
        leftover=leftover, original_ships=obs.ships.to(dtype), pressure=enemy_mass,
        config=cfg_regroup, H=H,
    )
    _convoy_min = _regroup_min_send() if _mass_active(player_count) else 0.0
    if _convoy_min > 0.0 and int(regroup_entries.ships.shape[0]) > 0:
        keep = regroup_entries.ships >= _convoy_min
        regroup_entries = LaunchEntries(
            source_slots=regroup_entries.source_slots,
            target_slots=regroup_entries.target_slots,
            ships=regroup_entries.ships,
            angle=regroup_entries.angle,
            eta=regroup_entries.eta,
            valid=regroup_entries.valid & keep,
        )
    if _snipe_hold_enabled():
        reserved = _snipe_hold_reserved_sources(
            obs=obs, garrison_status=garrison_status, background=background,
            wave_entries=wave_entries, H=H, movement=movement,
        )
        if reserved is not None and int(regroup_entries.ships.shape[0]) > 0:
            keep_r = ~reserved[regroup_entries.source_slots.clamp(0, P - 1)]
            regroup_entries = LaunchEntries(
                source_slots=regroup_entries.source_slots,
                target_slots=regroup_entries.target_slots,
                ships=regroup_entries.ships,
                angle=regroup_entries.angle,
                eta=regroup_entries.eta,
                valid=regroup_entries.valid & keep_r,
            )
    return concat_launch_entries([wave_entries, regroup_entries])


# --- Snipe-hold (toll-sniping reservation) -------------------------------------
# The planner is now-or-never: when the projection shows an opponent flipping
# a planet at tick k_f, arriving at k_f+1 costs survivor+1 ships (a fraction
# of the pre-flip garrison — "let them pay the toll"), but if our fleet would
# arrive too early there is no way to WAIT, and the regroup lane drains the
# idle ships away before the window opens. v1: detect flip events, find owned
# sources that can afford the snipe from ships remaining after this turn's
# waves and reach the target around k_f+1, and RESERVE them — their regroup
# entries are filtered this turn. Re-planning fires the actual snipe when the
# timing lines up (capture_floor at that arrival already reflects the thin
# survivor). Design: kb/thoughts/2026-06-10-snipe-hold-design.md.


def _snipe_hold_enabled() -> bool:
    return os.environ.get("PRODUCER_PLUS_SNIPE_HOLD", "0").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _snipe_hold_reserved_sources(
    *, obs, garrison_status, background, wave_entries, H: int, movement,
):
    """Mask [P] of sources to hold home for a dated snipe, or None."""
    P = int(obs.P)
    pid = int(obs.player_id)
    device = obs.device
    dtype = obs.ships.dtype
    traj = garrison_status.owner                       # [P, H+1]
    init = traj[:, 0]
    is_opp = (traj != pid) & (traj >= 0)
    became_opp = is_opp & (init.unsqueeze(-1) != traj)
    flips = became_opp & ~obs.owned.unsqueeze(-1)
    flip_any = flips.any(dim=-1)
    if not bool(flip_any.any()):
        return None
    # First flip tick per planet and the survivor garrison at that tick.
    k_idx = torch.arange(traj.shape[-1], device=device).expand_as(flips)
    k_f = torch.where(flips, k_idx, torch.full_like(k_idx, traj.shape[-1])).min(dim=-1).values
    surv = garrison_status.ships.gather(
        -1, k_f.clamp(0, traj.shape[-1] - 1).unsqueeze(-1)).squeeze(-1)
    # Ships left at each source after this turn's committed waves.
    committed = torch.zeros(P, dtype=dtype, device=device)
    if int(wave_entries.ships.shape[0]) > 0:
        v = wave_entries.valid
        committed.scatter_add_(
            0, wave_entries.source_slots[v].clamp(0, P - 1), wave_entries.ships[v].to(dtype))
    remaining = (obs.ships.to(dtype) - committed).clamp(min=0.0)
    reserved = torch.zeros(P, dtype=torch.bool, device=device)
    flip_planets = flip_any.nonzero(as_tuple=True)[0]
    src_planets = (obs.owned & obs.alive).nonzero(as_tuple=True)[0]
    if int(src_planets.shape[0]) == 0:
        return None
    for fp in flip_planets.tolist():
        kf = int(k_f[fp].item())
        if kf <= 0 or kf >= H:
            continue
        cost = float(surv[fp].item()) + 2.0
        for sp in src_planets.tolist():
            if reserved[sp] or float(remaining[sp].item()) < cost:
                continue
            aim = intercept_angle(
                movement,
                torch.tensor([sp], device=device),
                torch.tensor([fp], device=device),
                torch.tensor([cost], dtype=dtype, device=device),
            )
            if not bool(aim["viable"][0]):
                continue
            eta_now = float(aim["eta"][0].item())
            # Reserve only when we would arrive EARLY if we launched now —
            # i.e. waiting is exactly what unlocks the cheap capture.
            if eta_now < (kf + 1):
                reserved[sp] = True
                break
    return reserved if bool(reserved.any()) else None


# --- Synchronized multi-source arrivals (delayed launches) ---------------------
# Planet Wars canon: staggered waves die piecemeal to the 1:1 garrison trade;
# multi-source SAME-TICK arrivals are the capture mechanism for targets no
# single source can crack. The planner is now-or-never, so the second half of
# the mechanism is a HOLD: the nearer source's leg is scored at the far leg's
# arrival tick (exact under the flow scorer — arrival credit lands at
# ceil(eta); the tick-0 source debit makes the score conservative, since the
# held ships actually keep defending home), then diverted post-veto into a
# memory-held schedule and launched on the last turn that still makes the
# arrival date (re-aimed fresh, so orbit drift cannot desynchronize it).
# Default OFF preserves byte-identical behaviour.


def _sync_enabled() -> bool:
    return os.environ.get("PRODUCER_PLUS_SYNC", "0").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _sync_dmax() -> int:
    """Max hold length in ticks (gap between the pair's natural arrivals).

    Default 0 = same-tick coalitions only, NO holds. Delayed legs (d > 0)
    measured 3/12 (-46% @250) vs the live stack on 2026-06-11: the far leg
    telegraphs the attack for the whole hold window and a reply-aware
    defender reinforces past the pair's joint size, while the d=0 ablation
    sat at exact mirror parity (7/12, -0.7%). Holds stay opt-in for
    redemption experiments.
    """
    return max(0, _env_int("PRODUCER_PLUS_SYNC_DMAX", 0))


def _sync_k_src() -> int:
    """Per-target source pool for pair enumeration (nearest-first)."""
    return max(2, _env_int("PRODUCER_PLUS_SYNC_K", 6))


def _sync_max_holds() -> int:
    """Cap on concurrently pending holds (commitment-exposure guard)."""
    return max(1, _env_int("PRODUCER_PLUS_SYNC_MAX_HOLDS", 4))


def _sync_entry_key(src_slot: int, tgt_slot: int, eta: float, ships: float):
    return (int(src_slot), int(tgt_slot), round(float(eta), 4), int(round(float(ships))))


def _process_sync_holds(memory, *, obs, obs_tensors: dict, movement, current_step: int):
    """Advance pending holds one turn. Returns ``(exec_entries, debit)``.

    Per hold: cancel if the source is lost/drained or the target died or
    flipped to us; launch NOW (fresh aim) if waiting one more turn would miss
    the arrival date; release if the date became unreachable (orbit drift
    beyond 1 tick of slack); otherwise keep holding. ``debit`` reserves both
    kept and just-launched ships against the planner's budget view.
    """
    holds = getattr(memory, "sync_holds", None)
    if current_step == 0 or holds is None:
        holds = []
    if not holds:
        memory.sync_holds = []
        return None, None
    device = obs.device
    dtype = obs.ships.dtype
    P = int(obs.P)
    planet_ids = obs_tensors["planets"][..., 0].long()
    slot_of = {int(planet_ids[i].item()): i for i in range(P)}
    kept: list = []
    exec_rows: list = []
    debit = torch.zeros_like(obs.ships)
    for h in holds:
        s = slot_of.get(int(h["src_id"]))
        t = slot_of.get(int(h["tgt_id"]))
        if s is None or t is None:
            continue
        if not bool(obs.owned[s]) or not bool(obs.alive[s]) or not bool(obs.alive[t]):
            continue
        if bool(obs.owned[t]):
            continue  # captured by other means — release the reserve
        ships = float(h["ships"])
        if float(obs.ships[s].item()) < ships:
            continue  # combat ate the reserve — the sized pair is broken
        remaining = int(h["arrival_step"]) - current_step
        aim = intercept_angle(
            movement,
            torch.tensor([s], dtype=torch.long, device=device),
            torch.tensor([t], dtype=torch.long, device=device),
            torch.tensor([ships], dtype=dtype, device=device),
        )
        if not bool(aim["viable"][0]):
            continue
        eta_now = float(aim["eta"][0].item())
        if math.ceil(eta_now) >= remaining:
            # Last turn that can make the date (1 tick of late slack).
            if math.ceil(eta_now) <= remaining + 1:
                exec_rows.append(
                    (s, t, ships, float(aim["angle"][0].item()), eta_now))
                debit[s] += ships
            continue
        kept.append(h)
        debit[s] += ships
    memory.sync_holds = kept
    entries = None
    if exec_rows:
        entries = LaunchEntries(
            source_slots=torch.tensor([r[0] for r in exec_rows], dtype=torch.long, device=device),
            target_slots=torch.tensor([r[1] for r in exec_rows], dtype=torch.long, device=device),
            ships=torch.tensor([r[2] for r in exec_rows], dtype=dtype, device=device),
            angle=torch.tensor([r[3] for r in exec_rows], dtype=dtype, device=device),
            eta=torch.tensor([r[4] for r in exec_rows], dtype=dtype, device=device),
            valid=torch.ones(len(exec_rows), dtype=torch.bool, device=device),
        )
    if not bool((debit > 0).any()):
        debit = None
    return entries, debit


def _divert_sync_entries(entries, *, sink: list, obs_tensors: dict, current_step: int, memory):
    """Post-veto: convert chosen delayed near-legs into memory holds.

    A near-leg entry is identified by its (source, target, scored eta, ships)
    signature from the generation sink. It is NEVER launched this turn (its
    angle/eta describe the future synced flight, not a launch-now one); it
    becomes a hold only if its far partner survived selection + veto —
    otherwise it is dropped outright, because its size only makes sense
    jointly. NOTE: assumes the veto did not resize entries (upsize default
    OFF); a resized leg simply fails the signature match and launches as-is.
    """
    if not sink or int(entries.valid.shape[0]) == 0:
        return entries
    by_key: dict = {}
    for r in sink:
        by_key.setdefault(
            _sync_entry_key(r["near_src"], r["tgt"], r["eta"], r["near_ships"]), []
        ).append(r)
    E = int(entries.valid.shape[0])
    sig = set()
    for j in range(E):
        if bool(entries.valid[j]):
            sig.add(_sync_entry_key(
                entries.source_slots[j], entries.target_slots[j],
                entries.eta[j], entries.ships[j]))
    valid = entries.valid.clone()
    planet_ids = obs_tensors["planets"][..., 0].long()
    holds = list(getattr(memory, "sync_holds", []) or [])
    max_h = _sync_max_holds()
    changed = False
    for j in range(E):
        if not bool(valid[j]):
            continue
        key = _sync_entry_key(
            entries.source_slots[j], entries.target_slots[j],
            entries.eta[j], entries.ships[j])
        rs = by_key.get(key)
        if not rs:
            continue
        valid[j] = False
        changed = True
        partner = next(
            (r for r in rs if _sync_entry_key(
                r["far_src"], r["tgt"], r["eta"], r["far_ships"]) in sig),
            None,
        )
        if partner is not None and len(holds) < max_h:
            holds.append({
                "src_id": int(planet_ids[int(entries.source_slots[j].item())].item()),
                "tgt_id": int(planet_ids[int(entries.target_slots[j].item())].item()),
                "ships": float(entries.ships[j].item()),
                "arrival_step": current_step + int(partner["arrival_dt"]),
            })
    memory.sync_holds = holds
    if not changed:
        return entries
    return LaunchEntries(
        source_slots=entries.source_slots, target_slots=entries.target_slots,
        ships=entries.ships, angle=entries.angle, eta=entries.eta, valid=valid,
    )


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
        terminal_prod_weight=_terminal_prod_value(),
        terminal_neutral_only=_terminal_neutral_only(),
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
    reply_trust = None
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
        if _reply_trust_enabled():
            # Verify last turn's prediction first, then stash this turn's
            # RAW prediction for next turn, then price at trusted strength.
            reply_trust = _update_reply_trust(
                memory, obs_tensors, pid=int(obs.player_id))
            _record_reply_prediction(memory, background, obs_tensors)
            background = _scale_launch_set_ships(background, reply_trust)
        do_nothing_score = float(_score_do_nothing(
            status=status, prod=movement.planet_prod,
            alive_by_step=alive_by_step, player_count=int(player_count),
            background=background, player_id=int(obs.player_id),
            opp_weights=opp_w,
        ))
        cfg = dataclasses.replace(
            config, roi_threshold=do_nothing_score + float(config.roi_threshold),
        )

    opening_entries = None
    obs_for_plan = obs
    _osw = _opening_search_window()
    if _osw > 0 and current_step < _osw:
        claimed = getattr(memory, "opening_claimed", None)
        if claimed is None or current_step == 0:
            claimed = set()
        due = _opening_search_plan(
            obs_tensors, pid=int(obs.player_id), claimed=claimed,
            horizon=_opening_search_horizon(), beam_width=_opening_search_beam(),
        )
        if due:
            opening_entries = _emit_opening_entries(
                due, movement=movement, obs=obs, obs_tensors=obs_tensors,
                garrison_status=status, H=H, cache=cache,
            )
        if opening_entries is not None:
            # Claim targets across turns; debit the planner's budget view so
            # the greedy pass can't double-spend the opening sends.
            planet_ids_now = obs_tensors["planets"][..., 0].long()
            sel = opening_entries.valid.nonzero(as_tuple=True)[0]
            for i in sel.tolist():
                claimed.add(int(planet_ids_now[int(opening_entries.target_slots[i].item())].item()))
            debit = torch.zeros_like(obs.ships)
            debit.scatter_add_(
                0, opening_entries.source_slots[sel].clamp(0, int(obs.P) - 1),
                opening_entries.ships[sel].to(obs.ships.dtype),
            )
            obs_for_plan = dataclasses.replace(
                obs, ships=(obs.ships - debit).clamp(min=0.0))
        memory.opening_claimed = claimed

    # Sync holds: advance pending delayed launches (execute / keep / release)
    # and reserve their ships against the planner's budget view. The sink
    # collects this turn's delayed-leg signatures for post-veto diversion.
    sync_sink = None
    sync_exec_entries = None
    if _sync_enabled():
        sync_sink = []
        sync_exec_entries, sync_debit = _process_sync_holds(
            memory, obs=obs, obs_tensors=obs_tensors, movement=movement,
            current_step=current_step,
        )
        if sync_debit is not None:
            obs_for_plan = dataclasses.replace(
                obs_for_plan, ships=(obs_for_plan.ships - sync_debit).clamp(min=0.0))

    entries = plan_lite_waves(
        movement=movement, obs=obs_for_plan, obs_tensors=obs_tensors, cache=cache,
        garrison_status=status, prod=movement.planet_prod,
        alive_by_step=alive_by_step, config=cfg, player_count=int(player_count),
        K_eta_override=K_eta_override,
        background=background,
        opp_weights=opp_w,
        sync_sink=sync_sink,
    )
    if sync_exec_entries is not None:
        entries = concat_launch_entries([sync_exec_entries, entries])
    if opening_entries is not None:
        entries = LaunchEntries(
            source_slots=torch.cat([opening_entries.source_slots, entries.source_slots]),
            target_slots=torch.cat([opening_entries.target_slots, entries.target_slots]),
            ships=torch.cat([opening_entries.ships, entries.ships]),
            angle=torch.cat([opening_entries.angle, entries.angle]),
            eta=torch.cat([opening_entries.eta, entries.eta]),
            valid=torch.cat([opening_entries.valid, entries.valid]),
        )
    if _replan_active(int(player_count)) and _opp_projection_enabled():
        # Raw config (not the roi-shifted cfg): the replan re-normalizes the
        # roi threshold itself against its own reply background.
        entries = _apply_replan(
            entries,
            movement=movement, obs=obs, obs_tensors=obs_tensors, cache=cache,
            garrison_status=status, prod=movement.planet_prod,
            alive_by_step=alive_by_step, config=config,
            player_count=int(player_count), K_eta_override=K_eta_override,
            H=H, opp_weights=opp_w,
        )
    if _response_veto_active(int(player_count)) and _opp_projection_enabled():
        # Raw config (not the roi-shifted cfg): the reply mirror plans the
        # opponent exactly like the original projection pass did, and the
        # veto margin's do-nothing normalization is computed fresh here.
        _reply_box: list = []
        _valid_before = int(entries.valid.sum().item())
        entries = _apply_response_veto(
            entries,
            movement=movement, obs=obs, obs_tensors=obs_tensors, cache=cache,
            garrison_status=status, prod=movement.planet_prod,
            alive_by_step=alive_by_step, config=config,
            player_count=int(player_count), K_eta_override=K_eta_override,
            H=H, opp_weights=opp_w,
            reply_out=_reply_box,
            reply_trust=reply_trust,
        )
        # Redirect: only when the veto actually freed budget (waves dropped)
        # — otherwise pass 1 already spent everything it wanted to.
        if (
            _redirect_active(int(player_count)) and _reply_box
            and int(entries.valid.sum().item()) < _valid_before
        ):
            entries = _apply_redirect(
                entries,
                reply=_reply_box[0],
                movement=movement, obs=obs, obs_tensors=obs_tensors, cache=cache,
                garrison_status=status, prod=movement.planet_prod,
                alive_by_step=alive_by_step, config=config,
                player_count=int(player_count), K_eta_override=K_eta_override,
                H=H, opp_weights=opp_w,
            )
    if sync_sink:
        # After all entry filters: chosen delayed legs become memory holds
        # (or are dropped if their far partner did not survive the veto) —
        # they must never reach the payload or the private-fleet cache.
        entries = _divert_sync_entries(
            entries, sink=sync_sink, obs_tensors=obs_tensors,
            current_step=current_step, memory=memory,
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
        self.opening_claimed: set | None = None
        self.trust_ema: float | None = None
        self.trust_predictions: list | None = None
        self.trust_fleet_ids: set | None = None
        self.sync_holds: list | None = None

    def reset(self) -> None:
        self.movement = None
        self.cached_player_count = None
        self.last_sparse_action_row = None
        self.opening_claimed = None
        self.trust_ema = None
        self.trust_predictions = None
        self.trust_fleet_ids = None
        self.sync_holds = None


class ProducerLiteRuntime:
    def __init__(self, memory: ProducerLiteMemory | None = None) -> None:
        self.memory = memory if memory is not None else ProducerLiteMemory()

    def reset(self) -> None:
        self.memory.reset()

    def tensor_action(self, obs_tensors: dict):
        mem = self.memory
        if bool((obs_tensors["step"] == 0).all()):
            mem.cached_player_count = None
            mem.opening_claimed = None
            mem.trust_ema = None
            mem.trust_predictions = None
            mem.trust_fleet_ids = None
            mem.sync_holds = None
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

