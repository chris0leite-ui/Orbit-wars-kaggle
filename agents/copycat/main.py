"""copycat — mimic a roster of strong opponents, search for clearly better.

Strategy (plain English):

  Each turn, ask each member of a strong-agent roster what THEY would do
  from our current observation. Score every roster output with the v7
  fast brain (`lib/v7_search.score_candidate`, K-step lookahead on
  `lib/fast_sim`'s 183x-faster-than-env.clone snapshot engine). The
  highest-scoring roster output is the "floor" — what we play unless we
  find something measurably better. Then enumerate sigma-equivariant
  perturbations of the floor (drop {M, sigma(M)} together, swap targets
  in sigma-paired pairs). Score those too. Take the best alternative
  only if it beats the floor by `tau * tau_unit`; otherwise play the
  floor.

Game-theory framing:
  - Roster gives a per-state-adaptive Nash floor — strategy-agnostic.
  - sigma-equivariance keeps every candidate inside the v3-class draw
    lock basin (lib/planner.py + sym_hypot + SCORE_ROUND=6 patches —
    100% self-play draw rate verified for v3-class scalar policies in
    audit/2026-05-11-cannot-lose-final-finding.md).
  - tau gate prevents asymmetric overlays from cascading. The historical
    overlay failure (same audit) was overlays firing without a
    significance gate; tau is the structural fix.

Why scalar, not JAX, for the MVP brain:
  - We measured that JAX `policy_step_jax(aggressive=True)` diverges
    from scalar v3.5.1 on ~11% of turns by ~4 ships each, due to the
    arrival_size approximation in `lib/game/jax/jax_mechanisms.py`
    (documented as "Parity hit <= 1 ship; acceptable").
  - That breaks sigma-equivariance (postmortem: "Even 1 divergent launch
    breaks the symmetric self-play draw lock").
  - The scalar pipeline (lib.opp_model.top_tier_mirror_policy +
    lib.v7_search.score_candidate) inherits the sigma-equivariance
    patches and is what v7 already calls "the fast brain" (vs the
    old env.clone path).
  - JAX vmap brain is a Phase 3 upgrade once we've validated the
    architecture and have a parity-preserving JAX path.

Build-on / not-rebuild:
  - lib.opp_model.{mirror_self_policy, top_tier_mirror_policy} — v7_0
    base / v3.5.1 roster members.
  - lib.fast_sim.from_obs — Snapshot construction.
  - lib.v7_search.score_candidate — K-step rollout scorer.
  - lib.mirror.{build_bijection, rotate_angle, diagonal_opponent} —
    sigma-pair primitives.

Configurable env vars:
  COPYCAT_TAU       float (default 1.0)  — deviation gate in tau-units.
  COPYCAT_K         int   (default 10)   — lookahead depth.
  COPYCAT_TAU_UNIT  float (default 1.0)  — score-noise unit. Phase-0 fits.
                                            Default 1 = "1 ship of advantage".
  COPYCAT_ROSTER    csv   (default "v3_5_1,v7_0_base")
                     Members: v3_5_1 (=top_tier_mirror_policy),
                              v7_0_base (=mirror_self_policy).
  COPYCAT_WALLCLOCK_MS  float (default 700.0) — per-turn budget.
  COPYCAT_MAX_CANDS     int   (default 16)    — cap on sigma-pair variants.
  COPYCAT_OPP_TIER  int   (default 1)    — opp tier in score_candidate
                                            (1 = top_tier_mirror, 0 = mirror_self).

2P only for the deviation search. In 4P we fall back to the
highest-scoring roster member with no perturbation search.
"""

from __future__ import annotations

import math
import os
import time
from typing import Callable, Optional


# ---------------------------------------------------------------------------
# Config.
# ---------------------------------------------------------------------------


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


_TAU = _env_float("COPYCAT_TAU", 1.0)
_K = _env_int("COPYCAT_K", 10)
_TAU_UNIT = _env_float("COPYCAT_TAU_UNIT", 1.0)
_ROSTER = tuple(
    s.strip() for s in _env_str("COPYCAT_ROSTER", "v3_5_1,v7_0_base").split(",")
    if s.strip()
)
_WALLCLOCK_MS = _env_float("COPYCAT_WALLCLOCK_MS", 700.0)
_MAX_CANDS = _env_int("COPYCAT_MAX_CANDS", 16)
_OPP_TIER = _env_int("COPYCAT_OPP_TIER", 1)


# ---------------------------------------------------------------------------
# Roster registry: name -> callable(obs) -> action list.
# ---------------------------------------------------------------------------


def _roster_policies() -> dict[str, Callable]:
    """Resolve roster members lazily so unit tests can stub them."""
    from lib.opp_model import mirror_self_policy, top_tier_mirror_policy
    return {
        "v3_5_1":     top_tier_mirror_policy,
        "v7_0_base":  mirror_self_policy,
    }


# ---------------------------------------------------------------------------
# sigma-bijection cache (built lazily from the first obs's initial_planets).
# ---------------------------------------------------------------------------


_BIJECTION_CACHE: dict[tuple, dict[int, int]] = {}


def _planets_from_obs(obs):
    if isinstance(obs, dict):
        return obs.get("planets", [])
    return list(getattr(obs, "planets", []) or [])


def _initial_planets_from_obs(obs):
    """Prefer `obs.initial_planets` (set by fast_sim and the live env).
    Fall back to the current `obs.planets` snapshot (the bijection is
    time-invariant under uniform orbital rotation; see lib.mirror)."""
    if isinstance(obs, dict):
        ip = obs.get("initial_planets")
    else:
        ip = getattr(obs, "initial_planets", None)
    if ip:
        return list(ip)
    return _planets_from_obs(obs)


def _bijection_for(obs) -> dict[int, int]:
    """Return the 180-deg sigma-bijection {planet_id -> sigma(planet_id)}.

    Cached per-episode keyed on the sorted planet-id tuple. Built from
    initial planet positions; lib.mirror.build_bijection guarantees
    that orbital pairs remain mirror images for the whole episode.
    """
    from lib.mirror import build_bijection

    planets = _initial_planets_from_obs(obs)
    if not planets:
        return {}
    key = tuple(sorted(int(p[0]) for p in planets))
    cached = _BIJECTION_CACHE.get(key)
    if cached is not None:
        return cached
    bij = build_bijection(planets, tol=1.0)
    _BIJECTION_CACHE[key] = bij
    return bij


# ---------------------------------------------------------------------------
# Action helpers (kaggle-action format: [[src_pid, angle, ships], ...]).
# ---------------------------------------------------------------------------


def _action_key(action: list) -> tuple:
    return tuple(
        (int(m[0]), round(float(m[1]), 5), int(m[2])) for m in action
    )


# ---------------------------------------------------------------------------
# sigma-equivariant perturbations on the kaggle action format.
# ---------------------------------------------------------------------------


def _sigma_paired_drops(action: list, bij: dict[int, int]) -> list[list]:
    """Drop each sigma-paired source pair from the action.

    For an action with launches from sources {s, sigma(s)}, build the
    variant that drops both pair-members. If only one of (s, sigma(s))
    appears in the action, we still drop just s (it's an asymmetric
    drop, but the floor's own sigma-equivariance means s and sigma(s)
    will both be in the action when the policy itself is sigma-equiv).
    """
    if not action:
        return []
    by_src: dict[int, list[int]] = {}
    for i, row in enumerate(action):
        by_src.setdefault(int(row[0]), []).append(i)
    sources = sorted(by_src.keys())
    seen_pairs: set[tuple[int, int]] = set()
    out: list[list] = []
    for s in sources:
        sigma_s = bij.get(s, s)
        key = (min(s, sigma_s), max(s, sigma_s))
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        to_drop = set(by_src.get(s, [])) | set(by_src.get(sigma_s, []))
        variant = [row for j, row in enumerate(action) if j not in to_drop]
        if 0 < len(variant) < len(action):
            out.append(variant)
    return out


def _sigma_pair_angle_perturb(action: list, bij: dict[int, int],
                              delta: float) -> list[list]:
    """Generate variants that perturb the launch angle in sigma-paired pairs.

    For each (s, sigma(s)) pair with both endpoints firing, generate ONE
    variant that nudges the s-source's angle by +delta and the sigma(s)-
    source's angle by -delta (the sigma-conjugate perturbation, preserving
    180-deg symmetry).
    """
    if not action:
        return []
    out: list[list] = []
    seen_pairs: set[tuple[int, int]] = set()
    indexed_by_src: dict[int, int] = {}
    for i, row in enumerate(action):
        indexed_by_src.setdefault(int(row[0]), i)
    for src, i in sorted(indexed_by_src.items()):
        sigma_src = bij.get(src, src)
        if sigma_src == src or sigma_src not in indexed_by_src:
            continue
        pair_key = (min(src, sigma_src), max(src, sigma_src))
        if pair_key in seen_pairs:
            continue
        seen_pairs.add(pair_key)
        j = indexed_by_src[sigma_src]
        v = [list(row) for row in action]
        # Conjugate sign: applying +delta on src and -delta on sigma(src)
        # keeps the launch pair sigma-equivariant under rotate_angle
        # (theta -> theta + pi, so the pair's relative orientation is
        # preserved). Yes, this is a small angle nudge; the lookahead
        # decides whether it helps.
        v[i][1] = float(v[i][1]) + delta
        v[j][1] = float(v[j][1]) - delta
        out.append(v)
    return out


def _sigma_pair_ship_perturb(action: list, bij: dict[int, int],
                             frac: float) -> list[list]:
    """Generate variants that scale ship counts on sigma-paired pairs.

    For each (s, sigma(s)) pair both firing, build ONE variant with
    ships scaled by `frac` (clamped to >= 1) on both pair-members.
    """
    if not action or frac <= 0:
        return []
    out: list[list] = []
    seen_pairs: set[tuple[int, int]] = set()
    indexed_by_src: dict[int, list[int]] = {}
    for i, row in enumerate(action):
        indexed_by_src.setdefault(int(row[0]), []).append(i)
    for src in sorted(indexed_by_src.keys()):
        sigma_src = bij.get(src, src)
        if sigma_src == src or sigma_src not in indexed_by_src:
            continue
        pair_key = (min(src, sigma_src), max(src, sigma_src))
        if pair_key in seen_pairs:
            continue
        seen_pairs.add(pair_key)
        v = [list(row) for row in action]
        for i in indexed_by_src[src] + indexed_by_src[sigma_src]:
            new_ships = max(1, int(round(float(v[i][2]) * frac)))
            v[i][2] = new_ships
        out.append(v)
    return out


# ---------------------------------------------------------------------------
# Scoring (scalar fast brain).
# ---------------------------------------------------------------------------


def _score_action(snap, action, K: int, my_id: int, opp_tier: int) -> Optional[float]:
    """Score one candidate action. Returns None on failure."""
    from lib.v7_search import score_candidate
    try:
        return float(score_candidate(
            snap, action, my_id=my_id, K=K, opp_tier=opp_tier,
        ))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Entry point.
# ---------------------------------------------------------------------------


def _player_id(obs) -> int:
    if isinstance(obs, dict):
        return int(obs.get("player", 0))
    return int(getattr(obs, "player", 0))


def _num_seats(obs) -> int:
    from lib.mirror import detect_num_players
    n = detect_num_players(_planets_from_obs(obs))
    return n if n in (2, 4) else 2


def _fallback(obs):
    """Pure v3.5.1 fallback when anything fails. Always sigma-equivariant."""
    from lib.opp_model import top_tier_mirror_policy
    try:
        return top_tier_mirror_policy(obs)
    except Exception:
        return []


def agent(obs, configuration=None):
    t_start = time.perf_counter()

    num_seats = _num_seats(obs)
    my_id = _player_id(obs)

    # ----- Roster evaluation (kaggle action format throughout) -------
    policies = _roster_policies()
    roster_actions: list[tuple[str, list]] = []
    for member in _ROSTER:
        fn = policies.get(member)
        if fn is None:
            continue
        try:
            action = fn(obs) or []
        except Exception:
            continue
        roster_actions.append((member, action))

    if not roster_actions:
        return _fallback(obs)

    # 4P short-path: pick by simple heuristic (no Snapshot scoring; the
    # K-step rollout is 2P-only in score_candidate). Return the first
    # roster member's action as a reasonable default. v3.5.1 already
    # has 4P logic (LEADER_MULTIPLIER spoiler).
    if num_seats != 2:
        for _, action in roster_actions:
            if action:
                return action
        return roster_actions[0][1]

    # ----- Build Snapshot for scoring --------------------------------
    try:
        from lib.fast_sim import from_obs as fs_from_obs
        snap = fs_from_obs(obs, configuration, num_seats=num_seats)
    except Exception:
        # Snapshot build failed — fall back to the first roster member.
        return roster_actions[0][1]

    # Dedup roster actions (it's common for v7_0_base and v3_5_1 to
    # produce identical output on quiet states).
    seen: set[tuple] = set()
    deduped: list[tuple[str, list]] = []
    for name, action in roster_actions:
        k = _action_key(action)
        if k in seen:
            continue
        seen.add(k)
        deduped.append((name, action))
    roster_actions = deduped

    # ----- Score roster, pick the floor ------------------------------
    scored: list[tuple[float, str, list]] = []
    for name, action in roster_actions:
        s = _score_action(snap, action, K=_K, my_id=my_id, opp_tier=_OPP_TIER)
        if s is None:
            continue
        scored.append((s, name, action))
        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        if elapsed_ms > _WALLCLOCK_MS:
            break

    if not scored:
        return roster_actions[0][1]

    # Sigma-canonical tie-break: highest score first, then by roster
    # order (stable from _ROSTER tuple).
    member_rank = {m: i for i, m in enumerate(_ROSTER)}
    scored.sort(key=lambda r: (-r[0], member_rank.get(r[1], 999)))
    floor_score, floor_name, floor_action = scored[0]

    # ----- sigma-paired perturbations of the floor -------------------
    bij = _bijection_for(obs)
    perturbations: list[list] = []
    if bij:
        perturbations.extend(_sigma_paired_drops(floor_action, bij))
        perturbations.extend(_sigma_pair_angle_perturb(floor_action, bij, delta=0.10))
        perturbations.extend(_sigma_pair_angle_perturb(floor_action, bij, delta=-0.10))
        perturbations.extend(_sigma_pair_ship_perturb(floor_action, bij, frac=0.85))
        perturbations.extend(_sigma_pair_ship_perturb(floor_action, bij, frac=1.15))

    # Dedup against floor + already-tried.
    seen.clear()
    seen.update(_action_key(a) for _, _, a in scored)
    candidates: list[list] = []
    for variant in perturbations:
        k = _action_key(variant)
        if k in seen:
            continue
        seen.add(k)
        candidates.append(variant)
        if len(candidates) >= _MAX_CANDS:
            break

    # ----- Score perturbations with wallclock guard ------------------
    best_alt_score = -math.inf
    best_alt_action = floor_action
    for variant in candidates:
        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        if elapsed_ms > _WALLCLOCK_MS:
            break
        s = _score_action(snap, variant, K=_K, my_id=my_id, opp_tier=_OPP_TIER)
        if s is None:
            continue
        if s > best_alt_score:
            best_alt_score = s
            best_alt_action = variant

    # ----- tau deviation gate ----------------------------------------
    if math.isinf(_TAU) or math.isinf(best_alt_score):
        return floor_action

    if (best_alt_score - floor_score) > _TAU * _TAU_UNIT:
        return best_alt_action
    return floor_action
