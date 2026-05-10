"""Behavioural fingerprint — small-dim feature vector over a K-turn prefix.

See plan: /root/.claude/plans/read-the-handover-next-imperative-whisper.md
Phase 1 §architectural-extensions.2.

A *fingerprint* maps `(replay, player_id, prefix_turns)` to a fixed-length
`numpy.ndarray[float]`. Features are hand-designed to distinguish strategy
classes from short play, without needing to read the agent's policy directly.

The prior art (Grover et al., ICML 2018, "Learning Policy Representations in
Multiagent Systems") shows that this kind of short-trajectory embedding is a
viable input for a downstream best-response classifier — provided the
hand-designed features (or, in the Phase-1-fails fallback, a learned
embedding) capture enough between-strategy variance.

Design constraints:
- Pure function. No mutable state across calls; trivially testable.
- Symmetric across player_id. The same strategy played as P0 or P1 should
  produce a comparable fingerprint (subject to seat-asymmetry noise).
- Robust to short prefixes. If `prefix_turns` exceeds the replay length,
  we use what's available; missing-feature fallbacks are 0.0 so downstream
  classifiers see consistent shapes.
- No env imports beyond `lib/{geometry,fleet}` to keep the bundling path
  open if a Phase-3 meta-strategy needs `live_fingerprint` at game time.

Replay format expected (produced by `scripts/tournament.py::_build_replay`):

    {
      "seed": int, "agent_p0": str, "agent_p1": str, "n_steps": int,
      "rewards": [float, float], "statuses": [str, str],
      "steps": [
          {"step": int,
           "planets": [[id, owner, x, y, radius, ships, prod], ...],
           "fleets":  [[id, owner, x, y, angle, from_pid, ships], ...],
           "action_p0": [[from_pid, angle, ships], ...],
           "action_p1": [[from_pid, angle, ships], ...]},
          ...
      ]
    }
"""

from __future__ import annotations

import math
import statistics
from typing import Sequence

import numpy as np

from lib.fleet import speed as fleet_speed
from lib.geometry import dist, path_clears_sun


# Order is load-bearing — downstream classifier indexes by position, not name,
# so any reordering / removal is a breaking change. To extend, append at the
# end and bump the FEATURE_VERSION.
FEATURE_NAMES: list[str] = [
    "launches_per_turn",
    "mean_fleet_size",
    "p95_fleet_size",
    "mean_target_distance",
    "mean_target_production",
    "mean_target_garrison",
    "mean_garrison_at_launch",
    "targets_neutral_fraction",
    "targets_enemy_fraction",
    "launch_angle_var",
    "sun_clip_launch_rate",
    "mean_planets_owned",
    "mean_total_ships",
    "ships_growth_per_turn",
    "multi_launch_turn_rate",
]

FEATURE_VERSION = 1


# Constants from the comp spec (data/README.md).
SUN_CENTER = (50.0, 50.0)
SUN_RADIUS = 10.0


def _percentile(xs: list[float], p: float) -> float:
    """Cheap percentile that doesn't require numpy. p in [0, 100]."""
    if not xs:
        return 0.0
    if len(xs) == 1:
        return float(xs[0])
    s = sorted(xs)
    k = (len(s) - 1) * (p / 100.0)
    lo = int(math.floor(k))
    hi = int(math.ceil(k))
    if lo == hi:
        return float(s[lo])
    return float(s[lo] * (hi - k) + s[hi] * (k - lo))


def _infer_target(
    src_xy: tuple[float, float], angle: float, planets: list[list]
) -> list | None:
    """Project a ray from src at `angle` and return the planet it most likely
    targets — proxied as the nearest planet to the ray's first 100 units.

    The action [from_pid, angle, ships] does not record the intended target
    id, so we approximate. The proxy is consistent across same-strategy
    agents on the same seed (same approximation noise), so the classifier
    sees stable signal even if individual labels misclassify.
    """
    if not planets:
        return None
    sx, sy = src_xy
    dx, dy = math.cos(angle), math.sin(angle)
    # Project candidate planets onto the ray; pick smallest perpendicular
    # distance among those with positive forward projection.
    best = None
    best_perp = float("inf")
    for p in planets:
        # Skip the source itself (radius could be 0; identify via coords).
        px, py = p[2], p[3]
        if abs(px - sx) < 1e-6 and abs(py - sy) < 1e-6:
            continue
        # forward projection
        rx, ry = px - sx, py - sy
        forward = rx * dx + ry * dy
        if forward <= 0:
            continue
        # perpendicular distance to the ray
        perp_x, perp_y = rx - forward * dx, ry - forward * dy
        perp = math.hypot(perp_x, perp_y)
        # Tie-break with forward distance to favour closer targets.
        score = perp + 0.001 * forward
        if score < best_perp:
            best_perp = score
            best = p
    return best


def _planets_by_id(planets: list[list]) -> dict[int, list]:
    return {int(p[0]): p for p in planets}


def fingerprint(
    replay: dict,
    player_id: int,
    prefix_turns: int,
) -> np.ndarray:
    """Compute a fixed-length feature vector for `player_id` from the first
    `prefix_turns` of `replay`.

    Returns a `np.ndarray[float64]` of length `len(FEATURE_NAMES)`, in the
    order above. Missing-data fallbacks are 0.0.
    """
    if player_id not in (0, 1):
        raise ValueError(f"player_id must be 0 or 1; got {player_id!r}")
    steps = replay.get("steps", [])
    K = min(prefix_turns, len(steps))
    if K <= 0:
        return np.zeros(len(FEATURE_NAMES), dtype=np.float64)

    actions_key = f"action_p{player_id}"
    fleet_sizes: list[float] = []
    target_distances: list[float] = []
    target_productions: list[float] = []
    target_garrisons: list[float] = []
    src_garrisons: list[float] = []
    target_neutral = 0
    target_enemy = 0
    target_total = 0
    launch_angles: list[float] = []
    sun_clip_count = 0
    sun_clip_total = 0
    planets_owned_per_turn: list[float] = []
    total_ships_per_turn: list[float] = []
    multi_launch_turns = 0
    n_turns_with_action = 0
    n_launches = 0

    for i in range(K):
        step = steps[i]
        planets = step.get("planets", [])
        actions = step.get(actions_key, []) or []
        by_id = _planets_by_id(planets)

        # state-trajectory features: how many planets / total ships does
        # `player_id` own at the start of this step?
        my_planets = [p for p in planets if int(p[1]) == player_id]
        my_total_ships = sum(float(p[5]) for p in my_planets)
        # Add ships in own fleets (in-flight ships still count).
        for f in step.get("fleets", []):
            if int(f[1]) == player_id:
                my_total_ships += float(f[6])
        planets_owned_per_turn.append(float(len(my_planets)))
        total_ships_per_turn.append(my_total_ships)

        if actions:
            n_turns_with_action += 1
            if len(actions) > 1:
                multi_launch_turns += 1

        for action in actions:
            if not action or len(action) < 3:
                continue
            n_launches += 1
            from_pid_raw, angle, ships_raw = action[0], action[1], action[2]
            try:
                from_pid = int(from_pid_raw)
                ships = float(ships_raw)
            except (TypeError, ValueError):
                continue
            fleet_sizes.append(ships)
            launch_angles.append(float(angle))
            src = by_id.get(from_pid)
            if src is None:
                continue
            sx, sy = float(src[2]), float(src[3])
            src_garrisons.append(float(src[5]))
            # Sun-clip rate: project the launch ray a long way and check
            # whether the resulting segment crosses the sun disc. The probe
            # length (200 board-units) exceeds the board diagonal.
            sun_clip_total += 1
            probe_x = sx + 200.0 * math.cos(float(angle))
            probe_y = sy + 200.0 * math.sin(float(angle))
            if not path_clears_sun((sx, sy), (probe_x, probe_y), safety=0.0):
                sun_clip_count += 1

            target = _infer_target((sx, sy), float(angle), planets)
            if target is None:
                continue
            tx, ty = float(target[2]), float(target[3])
            target_distances.append(dist((sx, sy), (tx, ty)))
            target_productions.append(float(target[6]))
            target_garrisons.append(float(target[5]))
            target_total += 1
            owner = int(target[1])
            if owner == -1:
                target_neutral += 1
            elif owner != player_id:
                target_enemy += 1

    # Aggregate.
    feats: list[float] = []
    feats.append(n_launches / K if K else 0.0)
    feats.append(statistics.fmean(fleet_sizes) if fleet_sizes else 0.0)
    feats.append(_percentile(fleet_sizes, 95.0) if fleet_sizes else 0.0)
    feats.append(statistics.fmean(target_distances) if target_distances else 0.0)
    feats.append(statistics.fmean(target_productions) if target_productions else 0.0)
    feats.append(statistics.fmean(target_garrisons) if target_garrisons else 0.0)
    feats.append(statistics.fmean(src_garrisons) if src_garrisons else 0.0)
    feats.append((target_neutral / target_total) if target_total else 0.0)
    feats.append((target_enemy / target_total) if target_total else 0.0)
    feats.append(statistics.pvariance(launch_angles) if len(launch_angles) >= 2 else 0.0)
    feats.append((sun_clip_count / sun_clip_total) if sun_clip_total else 0.0)
    feats.append(statistics.fmean(planets_owned_per_turn) if planets_owned_per_turn else 0.0)
    feats.append(statistics.fmean(total_ships_per_turn) if total_ships_per_turn else 0.0)
    # Linear slope of total_ships over turns — a 1-feature growth signal.
    if len(total_ships_per_turn) >= 2:
        xs = list(range(len(total_ships_per_turn)))
        n = len(xs)
        mean_x = sum(xs) / n
        mean_y = sum(total_ships_per_turn) / n
        num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, total_ships_per_turn))
        den = sum((x - mean_x) ** 2 for x in xs)
        slope = num / den if den else 0.0
    else:
        slope = 0.0
    feats.append(slope)
    feats.append((multi_launch_turns / n_turns_with_action) if n_turns_with_action else 0.0)

    return np.asarray(feats, dtype=np.float64)


def batch_fingerprints(
    replays: Sequence[dict],
    prefix_turns: int,
) -> tuple[np.ndarray, list[str], list[int], list[int]]:
    """Compute fingerprints for both players of every replay.

    Returns:
      X       — np.ndarray of shape (2*N, len(FEATURE_NAMES))
      labels  — list[str] of length 2*N (the agent's name)
      seeds   — list[int] (the replay seed; useful for CV-by-seed)
      players — list[int] (0 or 1)
    """
    X_rows: list[np.ndarray] = []
    labels: list[str] = []
    seeds: list[int] = []
    players: list[int] = []
    for rep in replays:
        for p in (0, 1):
            X_rows.append(fingerprint(rep, p, prefix_turns))
            labels.append(rep.get(f"agent_p{p}", "?"))
            seeds.append(int(rep.get("seed", -1)))
            players.append(p)
    if not X_rows:
        return np.zeros((0, len(FEATURE_NAMES))), [], [], []
    return np.vstack(X_rows), labels, seeds, players


# Used by tests + manifold_check.
__all__ = [
    "FEATURE_NAMES",
    "FEATURE_VERSION",
    "fingerprint",
    "batch_fingerprints",
]
