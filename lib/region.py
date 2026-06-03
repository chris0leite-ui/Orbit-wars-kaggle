"""Orbital-parameter clustering of planets into stable regions + region value.

The unit of decision in the baseline agent is a single launch (source ->
target), scored independently. This module adds a *region* (chunk) view:
planets are bucketed by their (radius_band, angular_sector) in the CURRENT
orbital frame. Because every planet shares one global angular velocity
(`omega`), two planets in the same band+sector stay co-located across the
planning horizon -- so a region is STABLE without re-clustering per future
tick, and the clustering is DETERMINISTIC (no RNG, sorted iteration, integer
bucketing), which the agent bundle's parity test requires.

The region layer never emits a move on its own. It produces a value signal
and a "predictability" flag that `agents/baseline/main.py` uses to (a) bias
the existing per-launch candidates the K-step rollout chooser already
validates, and (b) aim the idle-mass "advance" pass at a frontier region.
This "feed the rollout, never replace it" shape is the design constraint
inherited from the falsified reach-frontier / analytical-slice tracks.

REUSES (no new physics, Rule 47):
  lib.geometry  -- CENTER, SUN_RADIUS, ROTATION_RADIUS_LIMIT
  lib.world_model.opp_contest_tick           (per-region contest tick)
  WorldModel.time_to_enemy_threat (method)   (per-planet hold time)

Cost: O(n) clustering; annotate calls time_to_enemy_threat / opp_contest_tick
once per planet (both bounded by enemy-planet count) -> O(n^2) worst case,
n~30 -> trivial inside the ~1s/turn budget. No re-clustering per future tick.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace

from lib.geometry import CENTER, SUN_RADIUS, ROTATION_RADIUS_LIMIT
from lib.world_model import opp_contest_tick

# Default bucketing. Overridable from main.py via env (kept OUT of this pure
# lib module so region.py has no env dependency and stays trivially testable).
DEFAULT_RADIUS_BANDS = 3       # inner / mid / outer ring
DEFAULT_ANGULAR_SECTORS = 6    # 60-degree wedges
HOLD_TIME_SATURATION = 120     # cap for time_to_enemy_threat == None (finite, board-safe)

_TWO_PI = 2.0 * math.pi


@dataclass(frozen=True)
class Region:
    key: tuple[int, int]                 # (radius_band, angular_sector) -- deterministic id
    planet_ids: tuple[int, ...]          # sorted -> deterministic
    centroid: tuple[float, float]        # mean (x, y) of members in the current frame
    value: float = 0.0                   # production-time-integral ranking signal
    contest_tick: int | None = None      # earliest enemy contest of the region (min over members)
    predictable: bool = False            # contest beyond the consolidation horizon (or None)
    n_mine: int = 0
    n_enemy: int = 0
    n_neutral: int = 0


def _radius_band(orb_r: float, n_bands: int) -> int:
    """Map orbital radius -> band index in [0, n_bands-1] (inner = 0).

    Orbiting planets live in (SUN_RADIUS, ROTATION_RADIUS_LIMIT); planets
    outside that clamp to the outer band. Deterministic for identical inputs.
    """
    lo, hi = SUN_RADIUS, ROTATION_RADIUS_LIMIT
    if orb_r <= lo:
        return 0
    if orb_r >= hi:
        return n_bands - 1
    frac = (orb_r - lo) / (hi - lo)
    return min(n_bands - 1, int(frac * n_bands))


def _angular_sector(angle: float, n_sectors: int) -> int:
    """Map a polar angle (atan2, [-pi, pi]) -> sector index in [0, n_sectors-1].

    Shift to [0, 2pi) first so the wrap point is deterministic.
    """
    a = angle % _TWO_PI
    return min(n_sectors - 1, int(a / _TWO_PI * n_sectors))


def _orbital_key(planet, n_bands: int, n_sectors: int) -> tuple[int, int]:
    dx, dy = float(planet[2]) - CENTER, float(planet[3]) - CENTER  # planet[2]=x, [3]=y
    orb_r = math.hypot(dx, dy)
    return (_radius_band(orb_r, n_bands), _angular_sector(math.atan2(dy, dx), n_sectors))


def cluster_regions(
    planets,
    *,
    n_bands: int = DEFAULT_RADIUS_BANDS,
    n_sectors: int = DEFAULT_ANGULAR_SECTORS,
) -> dict[tuple[int, int], Region]:
    """Bucket `planets` (Planet namedtuples) by (radius_band, angular_sector)
    of their current orbital position. Returns {key: Region}; only non-empty
    buckets are emitted. Deterministic: planets iterated in id order, members
    sorted by id.
    """
    buckets: dict[tuple[int, int], list] = {}
    for p in sorted(planets, key=lambda q: int(q.id)):
        buckets.setdefault(_orbital_key(p, n_bands, n_sectors), []).append(p)
    out: dict[tuple[int, int], Region] = {}
    for key, members in buckets.items():
        ids = tuple(sorted(int(p.id) for p in members))
        cx = sum(float(p.x) for p in members) / len(members)
        cy = sum(float(p.y) for p in members) / len(members)
        out[key] = Region(key=key, planet_ids=ids, centroid=(cx, cy))
    return out


def planet_to_region_key(
    planet,
    *,
    n_bands: int = DEFAULT_RADIUS_BANDS,
    n_sectors: int = DEFAULT_ANGULAR_SECTORS,
) -> tuple[int, int]:
    """O(1) reverse lookup: which region key a single Planet falls in.

    Uses IDENTICAL band/sector math to `cluster_regions` (shared helper).
    """
    return _orbital_key(planet, n_bands, n_sectors)


def region_value(region: Region, model, world, me: int, *, saturation: int = HOLD_TIME_SATURATION) -> float:
    """Production-time-integral ranking signal: sum_p production_p * hold_p.

    hold_p = WorldModel.time_to_enemy_threat(p) clamped to `saturation` when
    None (no reachable threat) so all-mine boards stay finite (no inf / NaN).
    This is a comparative ranking heuristic (it drives a bounded bias
    multiplier + frontier selection), not a precise win-integral.
    """
    total = 0.0
    for pid in region.planet_ids:
        p = world.planets_by_id.get(pid)
        if p is None:
            continue
        hold = model.time_to_enemy_threat(int(pid), int(me), world)
        hold_t = float(saturation if hold is None else min(int(hold), saturation))
        total += float(p.production) * hold_t
    return total


def region_contest(
    region: Region, model, world, me: int, *, consolidation_horizon: int,
) -> tuple[int | None, bool]:
    """Per-region nearest-source contest tick + predictability flag.

    contest = min over members of opp_contest_tick (the soonest the enemy
    could contest ANY member). This is the per-region nearest-source tick --
    NOT a worst-case over all enemy sources, which is the over-pessimism that
    falsified reach-frontier.

    predictable := contest is None OR contest > consolidation_horizon
    (the region's fate is settled beyond the horizon we plan to consolidate in).
    """
    ticks: list[int] = []
    for pid in region.planet_ids:
        t = opp_contest_tick(model, world, int(pid), int(me))
        if t is not None:
            ticks.append(int(t))
    contest = min(ticks) if ticks else None
    predictable = (contest is None) or (contest > int(consolidation_horizon))
    return contest, predictable


def annotate_regions(
    regions: dict[tuple[int, int], Region],
    model, world, me: int,
    *,
    consolidation_horizon: int,
    saturation: int = HOLD_TIME_SATURATION,
) -> dict[tuple[int, int], Region]:
    """One pass: fill value / contest_tick / predictable / ownership counts on
    every Region. Returns a NEW dict (frozen dataclasses -> replace)."""
    out: dict[tuple[int, int], Region] = {}
    for key, reg in regions.items():
        val = region_value(reg, model, world, me, saturation=saturation)
        contest, predictable = region_contest(
            reg, model, world, me, consolidation_horizon=consolidation_horizon,
        )
        n_mine = n_enemy = n_neu = 0
        for pid in reg.planet_ids:
            p = world.planets_by_id.get(pid)
            if p is None:
                continue
            o = int(p.owner)
            if o == int(me):
                n_mine += 1
            elif o < 0:
                n_neu += 1
            else:
                n_enemy += 1
        out[key] = replace(
            reg, value=val, contest_tick=contest, predictable=predictable,
            n_mine=n_mine, n_enemy=n_enemy, n_neutral=n_neu,
        )
    return out


def classify_regions(annotated: dict[tuple[int, int], Region], me: int) -> dict[tuple[int, int], str]:
    """Tag each region 'mine' | 'contested' | 'enemy' | 'neutral'.

    contested := has >=1 of mine AND (>=1 enemy OR a finite contest_tick).
    The ADVANCE frontier is the highest-value PREDICTABLE contested region.
    """
    tags: dict[tuple[int, int], str] = {}
    for key, reg in annotated.items():
        if reg.n_mine > 0 and (reg.n_enemy > 0 or reg.contest_tick is not None):
            tags[key] = "contested"
        elif reg.n_mine > 0:
            tags[key] = "mine"
        elif reg.n_enemy > 0:
            tags[key] = "enemy"
        else:
            tags[key] = "neutral"
    return tags


def frontier_region(
    annotated: dict[tuple[int, int], Region], tags: dict[tuple[int, int], str],
) -> Region | None:
    """The ADVANCE target: highest-value PREDICTABLE contested region.

    Deterministic tie-break (-value, key). Returns None if none qualify.
    """
    cand = [
        reg for key, reg in annotated.items()
        if tags.get(key) == "contested" and reg.predictable
    ]
    if not cand:
        return None
    cand.sort(key=lambda r: (-r.value, r.key))
    return cand[0]
