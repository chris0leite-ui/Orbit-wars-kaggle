"""Per-planet 8-class binning (production × kinematic × proximity).

Built for the 2026-05-19 archetype × planet-class rollup audit (plan:
``/root/.claude/plans/bootstrap-properly-your-task-sunny-puddle.md``).

Median splits are **per-board** at turn 0 so classes are comparable across
seeds within the same archetype: every game has roughly equal planets in
each split. Kinematic is the only absolute axis (rotates iff
``orbital_radius + planet_radius < 50``; see ``lib.orbit.is_orbiting``).

Class label format: ``{prod}_{kin}_{prox}`` — eight distinct strings.
"""
from __future__ import annotations

import math
import statistics
from typing import Iterable

from lib.fingerprint import _infer_target
from lib.orbit import is_orbiting


SUN_X = SUN_Y = 50.0


def _orb_radius(planet: list) -> float:
    return math.hypot(float(planet[2]) - SUN_X, float(planet[3]) - SUN_Y)


def compute_board_medians(planets: list[list]) -> dict[str, float]:
    """Turn-0 median splits for the two continuous axes (production, orbital
    radius). Kinematic is binary so no median is needed.
    """
    if not planets:
        return {"prod_median": 0.0, "orb_radius_median": 0.0}
    prods = [float(p[6]) for p in planets]
    radii = [_orb_radius(p) for p in planets]
    return {
        "prod_median": float(statistics.median(prods)),
        "orb_radius_median": float(statistics.median(radii)),
    }


def classify_planet(planet: list, board_medians: dict[str, float]) -> str:
    """Return one of 8 class labels for ``planet``.

    Class is computed from turn-0 invariants (production, planet radius,
    orbital radius), so the label is stable across all turns even when
    the planet rotates.
    """
    prod = float(planet[6])
    prod_label = "high_prod" if prod >= board_medians["prod_median"] else "low_prod"
    kin_label = "rotating" if is_orbiting(planet) else "static"
    r = _orb_radius(planet)
    prox_label = "inner" if r <= board_medians["orb_radius_median"] else "outer"
    return f"{prod_label}_{kin_label}_{prox_label}"


ALL_CLASS_LABELS: tuple[str, ...] = tuple(
    f"{p}_{k}_{x}"
    for p in ("high_prod", "low_prod")
    for k in ("rotating", "static")
    for x in ("inner", "outer")
)


def _empty_breakdown() -> dict[str, dict]:
    return {
        c: {
            "n_planets": 0,
            "home_planet_count": 0,
            "target_count": 0,
            "first_capture_turn": None,
            "end_owned": 0,
        }
        for c in ALL_CLASS_LABELS
    }


def per_planet_breakdown(
    flat_replay: dict, player_id: int, prefix_turns: int
) -> dict[str, dict]:
    """Aggregate per-class behavioural counts for ``player_id`` in one replay.

    Expects the *flat* replay schema (same shape ``lib.fingerprint.fingerprint``
    consumes — see ``scripts/fingerprint_external.ke_to_flat``).

    Returns ``{class_label: {n_planets, home_planet_count, target_count,
    first_capture_turn, end_owned}}`` for THIS single replay:

    - ``n_planets`` — count of planets of this class on the turn-0 board.
    - ``home_planet_count`` — of those, how many were already focal-owned
      at turn 0 (home planets). Subtract to get "available targets".
    - ``target_count`` — number of focal launches targeting a planet of
      this class within ``prefix_turns``.
    - ``first_capture_turn`` — earliest turn focal owned ANY non-home
      planet of this class within the window. ``None`` if never captured;
      caller substitutes ``prefix_turns`` when averaging across replays.
    - ``end_owned`` — 1 if focal owns ≥1 non-home planet of this class at
      the last observed turn of the window, else 0.
    """
    steps = flat_replay.get("steps") or []
    K = min(prefix_turns, len(steps))
    if K <= 0:
        return _empty_breakdown()

    turn0_planets = steps[0].get("planets", []) or []
    medians = compute_board_medians(turn0_planets)

    class_of: dict[int, str] = {}
    out = _empty_breakdown()
    home_ids: set[int] = set()
    for p in turn0_planets:
        pid = int(p[0])
        c = classify_planet(p, medians)
        class_of[pid] = c
        out[c]["n_planets"] += 1
        if int(p[1]) == player_id:
            out[c]["home_planet_count"] += 1
            home_ids.add(pid)

    actions_key = f"action_p{player_id}"
    last_owned_classes: set[str] = set()

    for i in range(K):
        step = steps[i]
        planets = step.get("planets", []) or []

        # Update non-home captures: which classes does focal own (excluding home).
        owned_classes_this_turn: set[str] = set()
        for p in planets:
            if int(p[1]) != player_id:
                continue
            pid = int(p[0])
            if pid in home_ids:
                continue
            cls = class_of.get(pid)
            if cls is None:
                continue
            owned_classes_this_turn.add(cls)
            if out[cls]["first_capture_turn"] is None:
                out[cls]["first_capture_turn"] = i
        last_owned_classes = owned_classes_this_turn

        # Tally launches by target class.
        by_id = {int(p[0]): p for p in planets}
        for action in step.get(actions_key, []) or []:
            if not action or len(action) < 3:
                continue
            try:
                from_pid = int(action[0])
                angle = float(action[1])
            except (TypeError, ValueError):
                continue
            src = by_id.get(from_pid)
            if src is None:
                continue
            tgt = _infer_target((float(src[2]), float(src[3])), angle, planets)
            if tgt is None:
                continue
            tcls = class_of.get(int(tgt[0]))
            if tcls is not None:
                out[tcls]["target_count"] += 1

    for c in ALL_CLASS_LABELS:
        out[c]["end_owned"] = 1 if c in last_owned_classes else 0

    return out


def aggregate_breakdowns(
    breakdowns: Iterable[dict[str, dict]], prefix_turns: int
) -> dict[str, dict]:
    """Average a list of per-replay breakdowns into per-class summary stats.

    Returns ``{class_label: {n_games, n_planets_per_game, home_per_game,
    target_count_per_game, target_intensity, first_capture_turn,
    end_owned_rate}}``. Empty classes still appear (zeroed) so downstream
    formatters can rely on the full 8-class key set.

    ``first_capture_turn`` substitutes ``prefix_turns`` for replays where
    the class was never captured.
    """
    breakdowns = list(breakdowns)
    n_games = len(breakdowns)
    total_targets_per_game = (
        sum(sum(b[c]["target_count"] for c in ALL_CLASS_LABELS) for b in breakdowns)
        / n_games
        if n_games else 0.0
    )
    summary: dict[str, dict] = {}
    for c in ALL_CLASS_LABELS:
        if n_games == 0:
            summary[c] = {
                "n_games": 0,
                "n_planets_per_game": 0.0,
                "home_per_game": 0.0,
                "target_count_per_game": 0.0,
                "target_intensity": 0.0,
                "target_share": 0.0,
                "first_capture_turn": float(prefix_turns),
                "end_owned_rate": 0.0,
            }
            continue
        n_planets = [b[c]["n_planets"] for b in breakdowns]
        home = [b[c]["home_planet_count"] for b in breakdowns]
        targets = [b[c]["target_count"] for b in breakdowns]
        captures = [
            b[c]["first_capture_turn"] if b[c]["first_capture_turn"] is not None
            else prefix_turns
            for b in breakdowns
        ]
        end_owned = [b[c]["end_owned"] for b in breakdowns]
        npg = sum(n_planets) / n_games
        targets_per_game = sum(targets) / n_games
        summary[c] = {
            "n_games": n_games,
            "n_planets_per_game": npg,
            "home_per_game": sum(home) / n_games,
            "target_count_per_game": targets_per_game,
            "target_intensity": (targets_per_game / npg) if npg > 0 else 0.0,
            "target_share": (targets_per_game / total_targets_per_game)
                if total_targets_per_game > 0 else 0.0,
            "first_capture_turn": sum(captures) / n_games,
            "end_owned_rate": sum(end_owned) / n_games,
        }
    return summary
