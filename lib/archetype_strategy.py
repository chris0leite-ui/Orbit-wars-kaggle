"""Per-archetype strategic expectations — machine-readable spec.

This module is the source of truth for "what does good play LOOK like
on archetype X?". It maps each of the 32 panel archetypes to a set of
behavioural-metric ranges drawn from ``lib.fingerprint.FEATURE_NAMES``
and the temporal features in ``scripts.extended_features``.

Consumed by:
- ``tests/test_archetype_strategies.py`` — pytest assertions
- ``scripts/archetype_report.py`` — per-agent divergence summary

The detailed strategic write-up that motivates each range lives at
``audit/2026-05-18-archetype-strategies.md``. The dict here mirrors
that document; if you change one, update the other.

Schema: ``EXPECTED_BEHAVIOR[archetype][metric_name] = (lo, hi)``,
where ``lo`` / ``hi`` are inclusive bounds; either may be ``None``
(unbounded). Metric names match either ``lib.fingerprint.FEATURE_NAMES``
or the temporal-features schema (``first_launch_step`` etc.).

Ranges are loose by design (~factor-2 tolerance) so per-seed RNG
variation doesn't false-alarm; what we want tests to catch is
SYSTEMATIC mismatch (e.g. "high_prod opening tempo not aggressive
enough" → first_launch_step > 10 instead of <= 5).
"""

from __future__ import annotations

from typing import Mapping

# ---------------------------------------------------------------------------
# Per-axis baseline expectations
# ---------------------------------------------------------------------------
#
# Each archetype's range = compose(prod_tier, rot_share, size_split).
# Composition rule: per-metric, take the INTERSECTION of axis ranges
# (tighter of the two bounds). If only one axis specifies a metric, that
# axis's bound wins.
#
# Metric units (from lib/fingerprint.py + scripts/extended_features.py):
#   first_launch_step       int   step of first non-empty action
#   early_launches          int   #launches in steps 0-30
#   mid_launches            int   #launches in steps 30-200
#   launches_per_turn       f64   mean over prefix
#   mean_fleet_size         f64   ships per launched fleet
#   p95_fleet_size          f64
#   multi_launch_turn_rate  f64   fraction of turns with >1 launch
#   mean_target_production  f64
#   targets_neutral_fraction f64
#   launch_angle_var        f64   variance of launch angles (radians^2)
#   sun_clip_launch_rate    f64   fraction of fleets routed through sun

_PROD_RULES: Mapping[str, dict[str, tuple[float | None, float | None]]] = {
    "low_prod": {
        # Scarce production → conservative. Few launches, larger fleets per
        # launch (justify each one), no rush. Top public opens slowly here.
        "first_launch_step": (None, 30),
        "early_launches": (None, 6),
        "launches_per_turn": (None, 1.5),
        "mean_fleet_size": (12.0, None),
        "targets_neutral_fraction": (0.40, None),
    },
    "med_low_prod": {
        # Tight resource competition — the v7_0 cell. Deeper lookahead
        # pays. Moderate tempo; don't burn ships on bounces.
        "first_launch_step": (None, 20),
        "early_launches": (1, 10),
        "launches_per_turn": (None, 2.5),
        "mean_fleet_size": (15.0, None),
    },
    "med_high_prod": {
        # Baseline's strong region — standard balanced play. Slightly
        # aggressive opening pays; consolidate mid-game.
        "first_launch_step": (None, 10),
        "early_launches": (3, None),
        "launches_per_turn": (0.8, None),
        "mean_fleet_size": (20.0, None),
    },
    "high_prod": {
        # Tempo dominates. H11 finding: top public agents fire from 90%+
        # planets by step 5; baseline fires from ~40%. This is THE biggest
        # known gap, so the strictest opening-tempo bounds live here.
        "first_launch_step": (None, 5),
        "early_launches": (6, None),
        "launches_per_turn": (1.2, None),
        "multi_launch_turn_rate": (0.20, None),
        "mean_fleet_size": (25.0, None),
    },
}

_ROT_RULES: Mapping[str, dict[str, tuple[float | None, float | None]]] = {
    "mostly_static": {
        # Targets are stable; orbit prediction barely matters. Launch
        # angles cluster around fixed bearings.
        "launch_angle_var": (None, 2.0),
    },
    "mixed_static": {
        # Some rotating targets; light orbit prediction.
        "launch_angle_var": (None, 2.5),
    },
    "mixed_rotating": {
        # Balanced mix — cognitive max. This is where baseline regresses;
        # high launch_angle_var is expected (you're firing at diverse
        # target classes).
        "launch_angle_var": (0.5, None),
    },
    "mostly_rotating": {
        # Most targets rotate; precise aim is critical.
        "launch_angle_var": (0.7, None),
    },
}

_SPLIT_RULES: Mapping[str, dict[str, tuple[float | None, float | None]]] = {
    "big_static": {
        # The high-prod prize planets are stationary — lock them down
        # early; expect mean_target_production to be biased high.
        "mean_target_production": (1.8, None),
    },
    "big_rotating": {
        # High-prod prize planets are moving; mean_target_production
        # bias same (you should still prefer high-prod targets).
        "mean_target_production": (1.5, None),
    },
}


def _compose_ranges(
    prod_key: str, rot_key: str, split_key: str
) -> dict[str, tuple[float | None, float | None]]:
    """Combine per-axis range dicts into one archetype range dict.

    Per-metric, take the INTERSECTION (tighter of two bounds).
    """
    combined: dict[str, tuple[float | None, float | None]] = {}
    for rules in (_PROD_RULES[prod_key], _ROT_RULES[rot_key], _SPLIT_RULES[split_key]):
        for name, (lo, hi) in rules.items():
            cur_lo, cur_hi = combined.get(name, (None, None))
            new_lo = lo if cur_lo is None else (lo if lo is not None and lo > cur_lo else cur_lo)
            new_hi = hi if cur_hi is None else (hi if hi is not None and hi < cur_hi else cur_hi)
            combined[name] = (new_lo, new_hi)
    return combined


PROD_KEYS = ("low_prod", "med_low_prod", "med_high_prod", "high_prod")
ROT_KEYS = ("mostly_static", "mixed_static", "mixed_rotating", "mostly_rotating")
SPLIT_KEYS = ("big_static", "big_rotating")

ARCHETYPES: tuple[str, ...] = tuple(
    f"{p}__{r}__{s}" for p in PROD_KEYS for r in ROT_KEYS for s in SPLIT_KEYS
)
assert len(ARCHETYPES) == 32

EXPECTED_BEHAVIOR: dict[str, dict[str, tuple[float | None, float | None]]] = {
    f"{p}__{r}__{s}": _compose_ranges(p, r, s)
    for p in PROD_KEYS
    for r in ROT_KEYS
    for s in SPLIT_KEYS
}

# ---------------------------------------------------------------------------
# Known regression archetypes (from the baseline-vs-v7_0 A/B, 2026-05-18)
# ---------------------------------------------------------------------------
# Tests xfail these cells — baseline systematically diverges from the spec
# on them. When archetype-aware logic is added and these flip to PASS,
# that's the signal we're closing the gap. Updated when new A/B data lands.
KNOWN_REGRESSIONS: frozenset[str] = frozenset({
    "low_prod__mixed_rotating__big_rotating",
    "med_high_prod__mostly_static__big_static",
    "med_low_prod__mixed_rotating__big_static",
    "med_low_prod__mixed_static__big_rotating",
    "med_low_prod__mixed_static__big_static",
})


def check(archetype: str, metrics: Mapping[str, float]) -> list[str]:
    """Return a list of human-readable violation strings for ``metrics``.

    Empty list = conforming. Each violation reads
    ``"first_launch_step=18.0 not in [None, 5]"``.
    """
    spec = EXPECTED_BEHAVIOR.get(archetype)
    if spec is None:
        return [f"unknown archetype: {archetype}"]
    violations = []
    for name, (lo, hi) in spec.items():
        if name not in metrics:
            continue  # metric not provided — skip silently
        v = metrics[name]
        if lo is not None and v < lo:
            violations.append(f"{name}={v:.3g} < {lo} (rule)")
        if hi is not None and v > hi:
            violations.append(f"{name}={v:.3g} > {hi} (rule)")
    return violations
