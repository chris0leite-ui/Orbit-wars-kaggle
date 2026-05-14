"""behavioral_mimic — copy what the OPPONENT just did, deviate only when search finds clearly better.

Strategy in plain English:

  Each turn, watch what fleets the opponent emitted on the previous turn
  (by diffing their fleet IDs between turns). Mirror each of their
  launches into our seat through the board's 180-degree symmetry:
  the source planet they launched from maps to its sigma-pair planet
  (which we own), the angle gets rotated by pi, the ship count stays
  the same. The set of mirrored launches IS our default action — we
  literally copy them.

  Then use the fast K=10 look-ahead brain to score:
    - the mimic action (our baseline / floor),
    - sigma-equivariant perturbations of it (drop a sigma-pair, swap
      sigma-paired targets, etc.).

  Take a perturbation only when the look-ahead says it beats the
  mimic by tau * tau_unit ships. Otherwise just copy the opponent.

Game-theory framing:
  - The mimic floor IS playing their policy from our seat (1 turn
    lagged). In a symmetric 2P zero-sum game, mimicking the opponent's
    strategy is the simplest exploitation-free policy: expected value
    is whatever they'd get from our seat, which they're not.
  - But: there's a 1-turn lag (we observe their move AFTER they made
    it). The lag is the documented failure mode of the historical
    mirror lineage (audit/2026-05-11-cannot-lose-final-finding.md).
    Their defender at sigma(target) gets +production during the lag,
    our copy under-ships, the asymmetry cascades.
  - Mitigation here: the tau gate. When the K=10 lookahead sees the
    lagged mimic is dominated (e.g., defender is fortified, target
    flipped), we deviate. The historical overlays did NOT have this
    gate; that's the architectural difference.

Build-on / reuse:
  - lib.mirror.{build_bijection, rotate_angle, diff_new_fleets,
    diagonal_opponent} — sigma symmetry primitives.
  - lib.fast_sim.from_obs - Snapshot construction.
  - lib.v7_search.score_candidate - K=10 fast-brain rollout score.
  - lib.opp_model.top_tier_mirror_policy - safe fallback when no
    mimic is recoverable (first turn, no opp fleets, illegal mirror).

State across calls:
  Module-level cache of (episode signature, previous fleet IDs).
  Per-episode reset on obs.step == 0 OR new planet-id signature.

Env vars:
  MIMIC_TAU       float (default 1.0)  - deviation gate in tau_unit ships.
  MIMIC_K         int   (default 10)   - lookahead depth.
  MIMIC_TAU_UNIT  float (default 1.0)  - ship-units of the gate.
  MIMIC_WALLCLOCK_MS  float (default 800)
  MIMIC_MAX_CANDS int   (default 12)   - cap on sigma-pair variants.

2P only for the deviation search. 4P falls back to v3.5.1 (mimic
in 4P is ambiguous: which opponent do you copy?).
"""

from __future__ import annotations

import math
import os
import time
from typing import Optional


# ---------------------------------------------------------------------------
# Config (read once at module import).
# ---------------------------------------------------------------------------


def _env_float(name, default):
    v = os.environ.get(name)
    try:
        return float(v) if v is not None else default
    except ValueError:
        return default


def _env_int(name, default):
    v = os.environ.get(name)
    try:
        return int(v) if v is not None else default
    except ValueError:
        return default


_TAU = _env_float("MIMIC_TAU", 1.0)
_K = _env_int("MIMIC_K", 10)
_TAU_UNIT = _env_float("MIMIC_TAU_UNIT", 1.0)
_WALLCLOCK_MS = _env_float("MIMIC_WALLCLOCK_MS", 800.0)
_MAX_CANDS = _env_int("MIMIC_MAX_CANDS", 12)


# ---------------------------------------------------------------------------
# Per-episode state (lives at module scope; reset on new-episode detection).
# ---------------------------------------------------------------------------


_PREV_FLEET_IDS: set[int] = set()
_EPISODE_SIG: Optional[tuple] = None
_BIJECTION: dict[int, int] = {}


def _planets_from_obs(obs):
    if isinstance(obs, dict):
        return obs.get("planets", []) or []
    return list(getattr(obs, "planets", []) or [])


def _fleets_from_obs(obs):
    if isinstance(obs, dict):
        return obs.get("fleets", []) or []
    return list(getattr(obs, "fleets", []) or [])


def _initial_planets_from_obs(obs):
    if isinstance(obs, dict):
        ip = obs.get("initial_planets")
    else:
        ip = getattr(obs, "initial_planets", None)
    return list(ip) if ip else _planets_from_obs(obs)


def _obs_step(obs) -> int:
    if isinstance(obs, dict):
        return int(obs.get("step", 0))
    return int(getattr(obs, "step", 0))


def _player_id(obs) -> int:
    if isinstance(obs, dict):
        return int(obs.get("player", 0))
    return int(getattr(obs, "player", 0))


def _episode_signature(obs) -> tuple:
    """Stable per-episode key derived from initial planet ids + positions."""
    ip = _initial_planets_from_obs(obs)
    return tuple(sorted((int(p[0]), round(float(p[2]), 2), round(float(p[3]), 2)) for p in ip))


def _maybe_reset_episode(obs) -> None:
    """Detect new-episode boundary and reset our per-episode caches."""
    global _PREV_FLEET_IDS, _EPISODE_SIG, _BIJECTION
    sig = _episode_signature(obs)
    if sig != _EPISODE_SIG:
        # New episode. Reset.
        _EPISODE_SIG = sig
        _PREV_FLEET_IDS = set()
        _BIJECTION = {}
    elif _obs_step(obs) == 0:
        # Same planet layout but step reset — defensive double-check.
        _PREV_FLEET_IDS = set()


def _bijection_for(obs) -> dict[int, int]:
    """Lazy-cache the sigma-bijection per episode."""
    global _BIJECTION
    if _BIJECTION:
        return _BIJECTION
    from lib.mirror import build_bijection
    planets = _initial_planets_from_obs(obs)
    if not planets:
        return {}
    _BIJECTION = build_bijection(planets, tol=1.0)
    return _BIJECTION


# ---------------------------------------------------------------------------
# Mimic extraction: read opp's last-turn launches via fleet-diff and mirror.
# ---------------------------------------------------------------------------


def _extract_opp_new_fleets(obs, my_id: int) -> list:
    """New fleets visible this turn that weren't visible last turn,
    owned by an opponent. These are the opp launches we'll mirror."""
    from lib.mirror import diff_new_fleets
    curr = _fleets_from_obs(obs)
    new = diff_new_fleets(curr, _PREV_FLEET_IDS)
    return [f for f in new if int(f[1]) != my_id]


def _mirror_opp_fleets_to_my_action(obs, opp_new_fleets, my_id: int,
                                    bij: dict[int, int]) -> list:
    """For each opp fleet F = [id, owner, x, y, angle, from_pid, ships],
    build our sigma-mirrored launch [sigma(from_pid), angle+pi, ships].
    Filter to legal launches (our source, has enough ships)."""
    if not opp_new_fleets or not bij:
        return []

    from lib.mirror import rotate_angle

    planets = _planets_from_obs(obs)
    by_id: dict[int, list] = {int(p[0]): list(p) for p in planets}

    out: list = []
    src_remaining: dict[int, int] = {}
    for f in opp_new_fleets:
        # f = [id, owner, x, y, angle, from_pid, ships]
        try:
            opp_src = int(f[5])
            opp_angle = float(f[4])
            ships = int(f[6])
        except (IndexError, ValueError, TypeError):
            continue
        if ships <= 0:
            continue
        my_src = bij.get(opp_src)
        if my_src is None:
            continue
        my_planet = by_id.get(my_src)
        if my_planet is None:
            continue
        # Must be owned by us with enough ships.
        if int(my_planet[1]) != my_id:
            continue
        available = src_remaining.get(my_src)
        if available is None:
            available = int(my_planet[5])
        # Sigma-conjugate the angle: theta -> theta + pi.
        my_angle = rotate_angle(opp_angle)
        send = min(ships, available)
        if send <= 0:
            continue
        out.append([my_src, my_angle, send])
        src_remaining[my_src] = available - send
    return out


# ---------------------------------------------------------------------------
# sigma-paired perturbations on the kaggle action format.
# ---------------------------------------------------------------------------


def _action_key(action) -> tuple:
    return tuple((int(r[0]), round(float(r[1]), 5), int(r[2])) for r in action)


def _sigma_paired_drops(action, bij):
    if not action:
        return []
    by_src: dict[int, list[int]] = {}
    for i, r in enumerate(action):
        by_src.setdefault(int(r[0]), []).append(i)
    seen: set[tuple[int, int]] = set()
    out: list[list] = []
    for s in sorted(by_src.keys()):
        sig = bij.get(s, s)
        k = (min(s, sig), max(s, sig))
        if k in seen:
            continue
        seen.add(k)
        drop = set(by_src.get(s, [])) | set(by_src.get(sig, []))
        variant = [r for j, r in enumerate(action) if j not in drop]
        if 0 < len(variant) < len(action):
            out.append(variant)
    return out


def _sigma_pair_ship_perturb(action, bij, frac: float):
    if not action or frac <= 0:
        return []
    by_src: dict[int, list[int]] = {}
    for i, r in enumerate(action):
        by_src.setdefault(int(r[0]), []).append(i)
    seen: set[tuple[int, int]] = set()
    out: list[list] = []
    for s in sorted(by_src.keys()):
        sig = bij.get(s, s)
        if sig == s or sig not in by_src:
            continue
        k = (min(s, sig), max(s, sig))
        if k in seen:
            continue
        seen.add(k)
        v = [list(r) for r in action]
        for i in by_src[s] + by_src[sig]:
            v[i][2] = max(1, int(round(float(v[i][2]) * frac)))
        out.append(v)
    return out


def _sigma_pair_angle_perturb(action, bij, delta: float):
    if not action:
        return []
    indexed: dict[int, int] = {}
    for i, r in enumerate(action):
        indexed.setdefault(int(r[0]), i)
    seen: set[tuple[int, int]] = set()
    out: list[list] = []
    for s, i in sorted(indexed.items()):
        sig = bij.get(s, s)
        if sig == s or sig not in indexed:
            continue
        k = (min(s, sig), max(s, sig))
        if k in seen:
            continue
        seen.add(k)
        j = indexed[sig]
        v = [list(r) for r in action]
        v[i][1] = float(v[i][1]) + delta
        v[j][1] = float(v[j][1]) - delta
        out.append(v)
    return out


# ---------------------------------------------------------------------------
# Scoring (scalar fast brain).
# ---------------------------------------------------------------------------


def _score(snap, action, K, my_id) -> Optional[float]:
    from lib.v7_search import score_candidate
    try:
        return float(score_candidate(snap, action, my_id=my_id, K=K, opp_tier=1))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Fallback (when mimic is unavailable or anything fails).
# ---------------------------------------------------------------------------


def _fallback(obs):
    from lib.opp_model import top_tier_mirror_policy
    try:
        return top_tier_mirror_policy(obs)
    except Exception:
        return []


def _num_seats(obs) -> int:
    from lib.mirror import detect_num_players
    n = detect_num_players(_planets_from_obs(obs))
    return n if n in (2, 4) else 2


# ---------------------------------------------------------------------------
# Agent entry.
# ---------------------------------------------------------------------------


def agent(obs, configuration=None):
    t0 = time.perf_counter()
    _maybe_reset_episode(obs)

    num_seats = _num_seats(obs)
    my_id = _player_id(obs)

    # Update fleet-id cache AT THE END regardless; but compute mimic first
    # using the PREVIOUS-turn snapshot.
    curr_fleets = _fleets_from_obs(obs)
    curr_ids = {int(f[0]) for f in curr_fleets}

    if num_seats != 2:
        # 4P: mimic ambiguous. Use v3.5.1.
        _PREV_FLEET_IDS.clear()
        _PREV_FLEET_IDS.update(curr_ids)
        return _fallback(obs)

    bij = _bijection_for(obs)

    opp_new = _extract_opp_new_fleets(obs, my_id)
    mimic_action = _mirror_opp_fleets_to_my_action(obs, opp_new, my_id, bij)

    if not mimic_action:
        # No opponent move observed (turn 0, or they stood pat). Fallback.
        _PREV_FLEET_IDS.clear()
        _PREV_FLEET_IDS.update(curr_ids)
        return _fallback(obs)

    # ----- Build Snapshot --------------------------------------------
    try:
        from lib.fast_sim import from_obs as fs_from_obs
        snap = fs_from_obs(obs, configuration, num_seats=2)
    except Exception:
        _PREV_FLEET_IDS.clear()
        _PREV_FLEET_IDS.update(curr_ids)
        return mimic_action  # play the mimic raw if scoring fails

    # ----- Score the mimic (floor) -----------------------------------
    mimic_score = _score(snap, mimic_action, K=_K, my_id=my_id)
    if mimic_score is None:
        _PREV_FLEET_IDS.clear()
        _PREV_FLEET_IDS.update(curr_ids)
        return mimic_action

    # ----- sigma-paired perturbations of the mimic -------------------
    perturbations: list[list] = []
    if bij:
        perturbations.extend(_sigma_paired_drops(mimic_action, bij))
        perturbations.extend(_sigma_pair_ship_perturb(mimic_action, bij, 0.85))
        perturbations.extend(_sigma_pair_ship_perturb(mimic_action, bij, 1.15))
        perturbations.extend(_sigma_pair_angle_perturb(mimic_action, bij, 0.10))
        perturbations.extend(_sigma_pair_angle_perturb(mimic_action, bij, -0.10))

    # Also consider the empty action ("stand pat"). Surprisingly often
    # this is a winning play vs an over-extended opponent.
    perturbations.append([])

    seen: set[tuple] = {_action_key(mimic_action)}
    candidates: list[list] = []
    for v in perturbations:
        k = _action_key(v)
        if k in seen:
            continue
        seen.add(k)
        candidates.append(v)
        if len(candidates) >= _MAX_CANDS:
            break

    # ----- Score perturbations (wallclock-gated) ---------------------
    best_alt = (-math.inf, mimic_action)
    for v in candidates:
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        if elapsed_ms > _WALLCLOCK_MS:
            break
        s = _score(snap, v, K=_K, my_id=my_id)
        if s is None:
            continue
        if s > best_alt[0]:
            best_alt = (s, v)

    # ----- Update fleet cache ----------------------------------------
    _PREV_FLEET_IDS.clear()
    _PREV_FLEET_IDS.update(curr_ids)

    # ----- tau deviation gate ----------------------------------------
    if math.isinf(_TAU) or math.isinf(best_alt[0]):
        return mimic_action
    if (best_alt[0] - mimic_score) > _TAU * _TAU_UNIT:
        return best_alt[1]
    return mimic_action
