"""Spearhead directional context — per-turn cache of "where is the opponent"
for friendly source planets, used by relay R-selection and the chooser's
target-alignment bonus.

Built once per turn in `agents/baseline/main.py` when at least one of
BASELINE_RELAY_SPEARHEAD or BASELINE_DIRECTIONAL_BONUS is set. Both
consumers (`relay_forward.emit_relay_forward`, `chooser_trajectory.
score_candidate_v4`) accept the context as an optional kwarg and
short-circuit to legacy behavior when it is None.

Per-source nearest-opp framing (not single opp_centroid) — addresses 4P
games where opponents on opposite sides of the board would average to a
meaningless centroid.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class SpearheadContext:
    nearest_opp_xy: dict[int, tuple[float, float]] = field(default_factory=dict)
    opp_centroid: tuple[float, float] | None = None  # telemetry only


def build_spearhead_context(planets, my_id: int) -> SpearheadContext:
    """Build per-turn directional context.

    `planets` is the obs.planets list. `my_id` is the focal seat. Returns
    a context populated for every friendly source planet; sources with no
    enemies in view get no entry (callers fall back to zero bonus).
    """
    opps = [p for p in planets if int(p.owner) >= 0 and int(p.owner) != int(my_id)]
    if not opps:
        return SpearheadContext()

    total_prod = 0.0
    cx = 0.0
    cy = 0.0
    for o in opps:
        w = float(o.production)
        total_prod += w
        cx += w * float(o.x)
        cy += w * float(o.y)
    centroid = (cx / total_prod, cy / total_prod) if total_prod > 0 else None

    nearest: dict[int, tuple[float, float]] = {}
    for p in planets:
        if int(p.owner) != int(my_id):
            continue
        best_d2 = float("inf")
        best_xy: tuple[float, float] | None = None
        px, py = float(p.x), float(p.y)
        for o in opps:
            ox, oy = float(o.x), float(o.y)
            d2 = (ox - px) * (ox - px) + (oy - py) * (oy - py)
            if d2 < best_d2:
                best_d2 = d2
                best_xy = (ox, oy)
        if best_xy is not None:
            nearest[int(p.id)] = best_xy

    return SpearheadContext(nearest_opp_xy=nearest, opp_centroid=centroid)


def cos_alignment(src_x: float, src_y: float,
                  tgt_x: float, tgt_y: float,
                  opp_x: float, opp_y: float) -> float:
    """Rectified cosine between (src -> tgt) and (src -> opp).

    Returns a value in [0.0, 1.0]: 1.0 when tgt is exactly along the
    opp direction from src, 0.0 when tgt is at >= 90 deg or behind src
    relative to opp. Rear targets get no penalty (just no bonus) — ETA
    or score already penalizes them, double-counting would distort.
    """
    vsx, vsy = tgt_x - src_x, tgt_y - src_y
    vox, voy = opp_x - src_x, opp_y - src_y
    nrm_s = math.hypot(vsx, vsy)
    nrm_o = math.hypot(vox, voy)
    if nrm_s <= 0.0 or nrm_o <= 0.0:
        return 0.0
    cos_th = (vsx * vox + vsy * voy) / (nrm_s * nrm_o)
    return max(0.0, cos_th)
