"""Geometry primitives for Orbit Wars.

Constants taken from `data/README.md` (Board Layout / Configuration). Hard-coded
rather than imported from `kaggle_environments` so the bundled single-file
submission can use this module without adding an env import line per call site.
"""

from __future__ import annotations

import math

# Board / sun geometry — match Configuration table in data/README.md.
BOARD_SIZE: float = 100.0
CENTER: float = 50.0           # both x and y; sun is at (CENTER, CENTER)
SUN_RADIUS: float = 10.0
ROTATION_RADIUS_LIMIT: float = 50.0  # planet rotates iff orbital_radius + planet_radius < this


Point = tuple[float, float]


def dist(a: Point, b: Point) -> float:
    """Euclidean distance between two 2D points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def sym_hypot(dx: float, dy: float) -> float:
    """Order-independent hypot — same bits for (dx, dy) and (dy, dx).

    Standard `math.hypot(a, b) = sqrt(a² + b²)` is mathematically
    symmetric in its arguments but NOT bit-exact under FP rounding:
    `a² + b²` and `b² + a²` can differ by 1 ULP because the addition
    is non-associative. Over thousands of mission-score comparisons,
    this 1-ULP noise turns near-ties into strict orderings, defeating
    σ-equivariant tie-breaks. `sym_hypot` canonicalises arguments to
    `hypot(min(|dx|,|dy|), max(|dx|,|dy|))` so σ-paired (src, target)
    pairs produce bit-equal distances.

    Ported from `origin/claude/game-theory-strategy-analysis-0oH4N`
    where the σ-equiv layer (this + planner _tb + score rounding) was
    the load-bearing change behind σ-equiv-v1 (μ=1041.4) and
    v7_minimax (μ=1063).
    """
    ax = abs(dx)
    ay = abs(dy)
    if ax > ay:
        ax, ay = ay, ax
    return math.hypot(ax, ay)


def point_to_segment_distance(p: Point, a: Point, b: Point) -> float:
    """Shortest distance from point `p` to segment a->b.

    Used to determine whether a fleet's straight-line path clips the sun
    (continuous collision check, per data/README.md::Fleet Movement).
    """
    ax, ay = a
    bx, by = b
    px, py = p
    dx, dy = bx - ax, by - ay
    seg_len2 = dx * dx + dy * dy
    if seg_len2 == 0.0:
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * dx + (py - ay) * dy) / seg_len2
    t = max(0.0, min(1.0, t))
    cx, cy = ax + t * dx, ay + t * dy
    return math.hypot(px - cx, py - cy)


def path_clears_sun(src: Point, dst: Point, safety: float = 0.0) -> bool:
    """True iff the segment src->dst stays at distance > SUN_RADIUS + safety
    from the sun. `safety` is a margin in board units (default 0 = exact rule).
    """
    return point_to_segment_distance((CENTER, CENTER), src, dst) > SUN_RADIUS + safety


def danger_3nn(
    target_xy: Point, target_id: int, planets: list, my_id: int
) -> int:
    """Signed allegiance count over the 3 planets nearest `target_xy`.

    Skips the target itself (matched by `target_id`). Returns an int in
    [-3, +3]: +1 per ally planet, -1 per enemy, 0 per neutral. Used as
    a stepwise spatial-danger feature for snipe / reinforce scoring
    (H17 / TID 699003). The discussion-reported result was that a
    count-based 3-NN hardcoded scoring beat a distance- and ship-weighted
    gradient form 16-0; this is the count-based form.

    `planets` is any iterable yielding objects with `.id`, `.x`, `.y`,
    and `.owner` attrs (e.g. our `lib.intent.Planet` view). Owner
    convention matches the env: -1 = neutral, otherwise = player id.
    """
    tx, ty = target_xy
    others = [p for p in planets if p.id != target_id]
    if not others:
        return 0
    others.sort(key=lambda p: math.hypot(p.x - tx, p.y - ty))
    score = 0
    for p in others[:3]:
        if p.owner == my_id:
            score += 1
        elif p.owner != -1:
            score -= 1
    return score
