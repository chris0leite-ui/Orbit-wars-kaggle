"""Mirror strategy primitives — the floor of the cannot-lose ladder.

Exploit Orbit Wars' D_4 board symmetry. Initial planet placement is
deterministically symmetric per seed: the env builds each group of 4
planets as the 4 rotations of one Q1 reference, and home planets in 2P
sit at the diagonal pair (base+0 ↔ base+3, i.e. 180° rotations of each
other). The same symmetry holds for every non-home planet group.

Public API:
    build_bijection(initial_planets)  -> dict[int, int]
        For each planet id, the id of its 180°-rotated counterpart.
    rotate_angle(theta)               -> float
        theta + π mod 2π.
    rotate_xy(x, y)                   -> (x', y')
        Reflect through the sun at (CENTER, CENTER).
    detect_num_players(planets)       -> int
        Count distinct non-neutral owners on the turn-0 board.
    diagonal_opponent(my_id, n)       -> int
        The opponent whose home is 180° from ours; in 2P that's `1-my_id`,
        in 4P it's `(my_id+2) % 4` (player j owns planet base+j per env).

Bijection construction is positional: at game start we walk
`initial_planets` and pair each planet id with the planet whose initial
(x, y) most closely matches (BOARD - x, BOARD - y). Tolerance 0.5 units
is well below planet spacing (PLANET_CLEARANCE = 1 unit in the env).
For orbiting planets this works because every member of an orbital
group rotates together with the same angular velocity, so two planets
that are mirror images at t=0 remain mirror images at every t.
"""

from __future__ import annotations

import math
from typing import Iterable

from lib.geometry import BOARD_SIZE, CENTER


def rotate_xy(x: float, y: float) -> tuple[float, float]:
    """180° rotation through (CENTER, CENTER)."""
    return (BOARD_SIZE - x, BOARD_SIZE - y)


def rotate_angle(theta: float) -> float:
    """θ → θ + π, normalised to [0, 2π)."""
    return (theta + math.pi) % (2 * math.pi)


def build_bijection(initial_planets, tol: float = 1.0) -> dict[int, int]:
    """Pair each planet id with its 180°-rotated counterpart by initial xy.

    `initial_planets` is the env-shipped list of [id, owner, x, y, r,
    ships, prod] tuples captured at t=0. Pairs are mutually exclusive
    (bijection); any planet without a match within `tol` is omitted.
    """
    bij: dict[int, int] = {}
    items = [(p[0], float(p[2]), float(p[3])) for p in initial_planets]
    for pid, x, y in items:
        rx, ry = rotate_xy(x, y)
        best_id, best_d2 = None, tol * tol
        for qid, qx, qy in items:
            if qid == pid:
                continue
            d2 = (qx - rx) ** 2 + (qy - ry) ** 2
            if d2 <= best_d2:
                best_d2 = d2
                best_id = qid
        if best_id is not None:
            bij[pid] = best_id
    # Trim to a true bijection: drop any entry whose partner doesn't
    # point back. (Should not happen with a clean symmetric board, but
    # cheap insurance.)
    return {a: b for a, b in bij.items() if bij.get(b) == a}


def detect_num_players(planets) -> int:
    """Count distinct non-neutral owners; reliable on turn 0."""
    owners = {p[1] for p in planets if p[1] != -1}
    return len(owners)


def diagonal_opponent(my_id: int, num_players: int) -> int:
    """Return the opponent across the 180° rotation axis from us.

    In 2P this is `1 - my_id`. In 4P, env assigns home `base+j` to
    player j; base+0 ↔ base+2? No — the symmetry analysis: positions
    in the group rotate by 90° each step, so base+0 and base+3 are
    diagonal (180°). Therefore in 4P the diagonal opponent of player 0
    is player 3, of player 1 is player 2.
    """
    if num_players == 2:
        return 1 - my_id
    if num_players == 4:
        # base+0 ↔ base+3, base+1 ↔ base+2 means player 0 ↔ 3, 1 ↔ 2.
        return {0: 3, 1: 2, 2: 1, 3: 0}[my_id]
    raise ValueError(f"unsupported num_players={num_players}")


def diff_new_fleets(curr_fleets, prev_ids: set[int]) -> list:
    """Fleets present this turn that weren't present last turn."""
    return [f for f in curr_fleets if f[0] not in prev_ids]


__all__ = [
    "rotate_xy",
    "rotate_angle",
    "build_bijection",
    "detect_num_players",
    "diagonal_opponent",
    "diff_new_fleets",
]
