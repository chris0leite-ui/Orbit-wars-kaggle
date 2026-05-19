"""Per-geometry-class priority prior — closed-form multiplier on cheap_marginal_value.

Design: knowledge-base/concepts/per-class-priority-prior.md

priority(c, t) = exp(lambda_alpha * alpha(c) + lambda_gap * gap(c, t))

  alpha(c)  -- static top10-share minus our-share, averaged over 16 cells.
  gap(c, t) -- top10_share(c) - opp_share_in_flight(c, t).

lambda_alpha = lambda_gap = 0 collapses priority to 1.0 (no-op).

Tables baked from audit/2026-05-19-archetype-per-planet-class.json
(cells[].rows[].target_share_{top10,ours}, averaged across 16 cells).
"""
from __future__ import annotations

import math

from lib.per_planet_class import (
    ALL_CLASS_LABELS,
    classify_planet,
    compute_board_medians,
)


ALPHA_BY_CLASS: dict[str, float] = {
    "high_prod_rotating_inner": -0.0235,
    "high_prod_rotating_outer":  0.0000,
    "high_prod_static_inner":   -0.0468,
    "high_prod_static_outer":   -0.0490,
    "low_prod_rotating_inner":  +0.1001,
    "low_prod_rotating_outer":   0.0000,
    "low_prod_static_inner":    +0.0167,
    "low_prod_static_outer":    +0.0025,
}

TOP10_SHARE_BY_CLASS: dict[str, float] = {
    "high_prod_rotating_inner": 0.1879,
    "high_prod_rotating_outer": 0.0000,
    "high_prod_static_inner":   0.1292,
    "high_prod_static_outer":   0.3167,
    "low_prod_rotating_inner":  0.2132,
    "low_prod_rotating_outer":  0.0000,
    "low_prod_static_inner":    0.0419,
    "low_prod_static_outer":    0.1111,
}

assert set(ALPHA_BY_CLASS.keys()) == set(ALL_CLASS_LABELS)
assert set(TOP10_SHARE_BY_CLASS.keys()) == set(ALL_CLASS_LABELS)


def compute_class_of(raw_planets) -> dict[int, str]:
    """Map planet_id -> 8-class label, computed from turn-0-invariant fields.

    `raw_planets` is the observation's `planets` list (sequence of tuples /
    lists; index 0 is planet_id). The class is stable across all turns
    even for rotating planets, so this can be called at any step.
    """
    if not raw_planets:
        return {}
    medians = compute_board_medians(list(raw_planets))
    return {int(p[0]): classify_planet(p, medians) for p in raw_planets}


def compute_opp_share_in_flight(model, me: int,
                                class_of: dict[int, str]) -> dict[str, float]:
    """Tally enemy in-flight fleets per destination class, normalize to a share.

    Reads `model.ledger`, which is `{planet_id: [(eta, owner, ships), ...]}`
    -- already-inferred fleet destinations from `WorldModel.from_world`.
    Counts FLEETS (not ships) to match the audit's `target_count`
    convention (one launch = one tally regardless of fleet size).

    When no enemy fleets are in flight, returns `TOP10_SHARE_BY_CLASS` so
    that `gap = top10_share - opp_share = 0` and only the static alpha
    contributes. This is the turn-0 / opening fallback.
    """
    counts: dict[str, int] = {c: 0 for c in ALL_CLASS_LABELS}
    total = 0
    ledger = getattr(model, "ledger", None) or {}
    for pid, arrivals in ledger.items():
        cls = class_of.get(int(pid))
        if cls is None:
            continue
        for entry in arrivals:
            owner = int(entry[1])
            if owner == me or owner < 0:
                continue
            counts[cls] += 1
            total += 1

    if total == 0:
        return dict(TOP10_SHARE_BY_CLASS)
    return {c: counts[c] / total for c in ALL_CLASS_LABELS}


def priority_by_planet(class_of: dict[int, str],
                       opp_share: dict[str, float],
                       lambda_alpha: float,
                       lambda_gap: float) -> dict[int, float]:
    """Per-planet multiplier `{planet_id: exp(lambda_alpha * alpha(c) +
    lambda_gap * gap(c, t))}`.

    lambda_alpha = lambda_gap = 0 yields priority == 1.0 for every planet
    (ablation invariant).
    """
    out: dict[int, float] = {}
    for pid, cls in class_of.items():
        alpha = ALPHA_BY_CLASS.get(cls, 0.0)
        gap = TOP10_SHARE_BY_CLASS.get(cls, 0.0) - opp_share.get(cls, 0.0)
        out[int(pid)] = math.exp(lambda_alpha * alpha + lambda_gap * gap)
    return out
