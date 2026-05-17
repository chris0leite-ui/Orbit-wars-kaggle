"""lib.metrics — pre-registration metric library.

Every pre-submit hypothesis under `audit/hypotheses/` declares a
behavioural metric (NOT just μ) that the new agent should change. This
module is the canonical implementation of those metrics so that:

1. The same metric is computed the same way at pre-registration time
   and post-submit measurement time.
2. The published baselines (v15, top-10, midpack) are pinned by unit
   tests — if the metric drifts, tests fail.
3. The pre-registration documents can name metrics by string
   (`waste_attack_fraction`) and `scripts/measure_hypothesis.py` can
   dispatch by string lookup.

Two input shapes:

- **Rollup metrics** take a dict like the output of
  `scripts.replay_mine.aggregate_across` — pre-aggregated bucket counts.
  Fast (O(1)); use whenever the answer can be computed from buckets
  alone.
- **Replay-walking metrics** take a list of `(replay_dict, team_name)`
  tuples. Used when per-turn or per-fleet detail is needed (e.g.
  first-launch step, multi-launch rate).

Baselines (documented):

- `V15_BASELINE` — `audit/replays/replay-mine-2026-05-17.json` (v15
  live champion, sub 52710995, 92 episodes, 9507 fleets).
- `TOP10_BASELINE` / `MIDPACK_BASELINE` — from
  `knowledge-base/concepts/top-performer-strategies.md` (50 top-10
  replays + 10 midpack), K=100-turn prefix.

Some metrics that the top-performer EDA names (`enemy_target_fraction`,
`mean_garrison_at_launch`, `comet_attempt_rate`) require ray-casting
each launch to find its intended target — NOT cheap to compute from
the replay JSON alone. Those are deferred to a follow-up commit; see
`# TODO(metrics-ray-cast)` markers.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence


# ---------------------------------------------------------------------------
# Documented baselines — keep in sync with audit/ + knowledge-base/
# ---------------------------------------------------------------------------


# v15 live champion (sub 52710995, 92 episodes). Source:
# audit/replays/replay-mine-2026-05-17.{md,json}. Values are fractions
# (not percentages) for uniformity with metric outputs.
V15_BASELINE: dict[str, float] = {
    "win_fraction":              0.428,   # 42.8%
    "defense_fraction":          0.328,   # 32.8%
    "waste_attack_fraction":     0.147,   # 14.7%
    "waste_comet_fraction":      0.001,   # ~0.1% (post swept-pair fix)
    "trajectory_waste_fraction": 0.090,   # 9.0%
    "inflight_fraction":         0.006,   # ~0.6%
}

# Top-10 (rank 1-10) and midpack (~rank 50-200) profiles, K=100-turn
# prefix. Source: knowledge-base/concepts/top-performer-strategies.md
# §4 "Common qualities across all top-10".
TOP10_BASELINE: dict[str, float] = {
    "first_launch_step":         4.1,
    "active_turn_fraction":      0.48,   # ≥1 launch / 500 turns
    "multi_launch_turn_rate":    0.48,   # ≥2 launches when active
    "sun_clip_rate":             0.062,
    "comet_kill_rate":           0.034,  # fraction of fleets killed by comets
    "enemy_target_fraction":     0.32,   # TODO(metrics-ray-cast)
    "mean_garrison_at_launch":   10.6,   # TODO(metrics-ray-cast)
}

MIDPACK_BASELINE: dict[str, float] = {
    "first_launch_step":         10.5,
    "active_turn_fraction":      0.38,
    "multi_launch_turn_rate":    0.38,
    "sun_clip_rate":             0.098,
    "comet_kill_rate":           0.134,
    "enemy_target_fraction":     0.14,   # TODO(metrics-ray-cast)
    "mean_garrison_at_launch":   22.0,   # TODO(metrics-ray-cast)
}


# Bucket-name constants — must match scripts.replay_mine.BUCKETS exactly.
_BUCKET_WIN = "win"
_BUCKET_DEFENSE = "defense"
_BUCKET_WASTE_ATTACK = "waste_attack"
_BUCKET_WASTE_COMET = "waste_comet"
_BUCKET_WASTE_TRAJ = "waste_trajectory"
_BUCKET_INFLIGHT = "inflight"


# ---------------------------------------------------------------------------
# Rollup metrics — `rollup` is the dict from replay_mine.aggregate_across
# ---------------------------------------------------------------------------


def _bucket_fraction(rollup: dict, bucket: str) -> float:
    """Generic helper: count(bucket) / total fleets."""
    n = int(rollup.get("n_fleets", 0))
    if not n:
        return 0.0
    return rollup.get("by_bucket", {}).get(bucket, 0) / n


def win_fraction(rollup: dict) -> float:
    """Fraction of our launches that captured the target. Higher is better.
    Baseline v15: 0.428."""
    return _bucket_fraction(rollup, _BUCKET_WIN)


def defense_fraction(rollup: dict) -> float:
    """Fraction of launches that reinforced our own planet. Higher is
    generally better (defensive value), but very high values can indicate
    timidity. Baseline v15: 0.328."""
    return _bucket_fraction(rollup, _BUCKET_DEFENSE)


def waste_attack_fraction(rollup: dict) -> float:
    """Fraction of launches that bounced (insufficient ships vs defenders,
    or arrived after the target was already ours, or lost to enemy
    reinforcement). LOWER is better. Baseline v15: 0.147."""
    return _bucket_fraction(rollup, _BUCKET_WASTE_ATTACK)


def waste_comet_fraction(rollup: dict) -> float:
    """Fraction of launches killed by a comet swept-pair collision.
    LOWER is better. Baseline v15 (post swept-pair classifier fix):
    ~0.001 — comet kills are rare; the older ~8.8% number was a
    mis-attribution of orbital planet hits."""
    return _bucket_fraction(rollup, _BUCKET_WASTE_COMET)


def trajectory_waste_fraction(rollup: dict) -> float:
    """Fraction of launches lost to sun crossing, out-of-bounds, or
    vanishing-in-space (mostly orbital planet sweeps). LOWER is better.
    Baseline v15: 0.090."""
    return _bucket_fraction(rollup, _BUCKET_WASTE_TRAJ)


def inflight_fraction(rollup: dict) -> float:
    """Fraction of launches still alive at episode end. Should be low —
    a high value means we're launching late or our fleets are too small
    to matter. Baseline v15: 0.006."""
    return _bucket_fraction(rollup, _BUCKET_INFLIGHT)


def sun_clip_rate(rollup: dict) -> float:
    """Fraction of launches that died via sun crossing. Finer-grained
    than `trajectory_waste_fraction` — reads `raw_outcomes['sun']`.
    Top-10: 0.062. Midpack: 0.098. v15 baseline: deferred — depends
    on raw_outcome breakdown across episodes."""
    n = int(rollup.get("n_fleets", 0))
    if not n:
        return 0.0
    return int(rollup.get("raw_outcomes", {}).get("sun", 0)) / n


def comet_kill_rate(rollup: dict) -> float:
    """Fraction of launches killed by a comet collision. This is the
    'cost' of comet-chase; not the same as `comet_attempt_rate` (which
    would be "fraction of launches AIMED at a comet" and requires
    ray-casting; see TODO(metrics-ray-cast)).
    Baseline v15 (post swept-pair fix): ~0.001."""
    return waste_comet_fraction(rollup)


# ---------------------------------------------------------------------------
# Replay-walking metrics — `replays_with_team` is iterable of
# (replay_dict, team_name) tuples
# ---------------------------------------------------------------------------


def _resolve_our_seat(replay: dict, team_name: str) -> int | None:
    """Index of our seat in the TeamNames list, or None if not present."""
    teams = replay.get("info", {}).get("TeamNames", []) or []
    for i, t in enumerate(teams):
        if t == team_name:
            return i
    return None


def _focal_actions(replay: dict, our_seat: int) -> Iterable[tuple[int, list]]:
    """Yield `(step_idx, action_list)` for every turn where we launched
    at least one fleet. `action_list` is `[[src_id, angle, ships], ...]`.
    """
    for t, step in enumerate(replay.get("steps", []) or []):
        if our_seat >= len(step):
            continue
        action = step[our_seat].get("action") or []
        if action:
            yield t, action


def first_launch_step(replays_with_team: Sequence[tuple[dict, str]],
                      *, default: int = 500) -> float:
    """Mean step of the first launch across episodes. LOWER is better.
    Top-10: 4.1. Midpack: 10.5.

    If an episode has no launch at all, contributes `default` (=500).
    """
    firsts: list[int] = []
    for replay, team_name in replays_with_team:
        seat = _resolve_our_seat(replay, team_name)
        if seat is None:
            continue
        first = default
        for t, _action in _focal_actions(replay, seat):
            first = t
            break
        firsts.append(first)
    if not firsts:
        return float(default)
    return sum(firsts) / len(firsts)


def active_turn_fraction(replays_with_team: Sequence[tuple[dict, str]]) -> float:
    """Fraction of OUR turns where we launched at least one fleet.
    HIGHER means more active. Top-10: 0.48. Midpack: 0.38.

    Total turns counts only seats where we're alive (status != DONE);
    a seat that's been eliminated at turn 200 still counts the 0..200
    turns it was alive but not the 201..500 it wasn't.
    """
    total = 0
    active = 0
    for replay, team_name in replays_with_team:
        seat = _resolve_our_seat(replay, team_name)
        if seat is None:
            continue
        for step in replay.get("steps", []) or []:
            if seat >= len(step):
                continue
            status = step[seat].get("status")
            if status == "DONE":
                # No more legal turns for this seat.
                break
            total += 1
            action = step[seat].get("action") or []
            if action:
                active += 1
    if not total:
        return 0.0
    return active / total


def multi_launch_turn_rate(replays_with_team: Sequence[tuple[dict, str]]) -> float:
    """Fraction of OUR-active turns (turns where we launched at all) on
    which we launched ≥2 fleets. HIGHER suggests gang-up / swarm
    behaviour. Top-10: 0.48. Midpack: 0.38.
    """
    active = 0
    multi = 0
    for replay, team_name in replays_with_team:
        seat = _resolve_our_seat(replay, team_name)
        if seat is None:
            continue
        for _t, action in _focal_actions(replay, seat):
            active += 1
            if len(action) >= 2:
                multi += 1
    if not active:
        return 0.0
    return multi / active


# ---------------------------------------------------------------------------
# Registry — for scripts/measure_hypothesis.py to dispatch by name
# ---------------------------------------------------------------------------


# Pure-rollup metrics: `f(rollup) -> float`. Cheap.
_ROLLUP_METRICS: dict[str, Any] = {
    "win_fraction":              win_fraction,
    "defense_fraction":          defense_fraction,
    "waste_attack_fraction":     waste_attack_fraction,
    "waste_comet_fraction":      waste_comet_fraction,
    "trajectory_waste_fraction": trajectory_waste_fraction,
    "inflight_fraction":         inflight_fraction,
    "sun_clip_rate":             sun_clip_rate,
    "comet_kill_rate":           comet_kill_rate,
}

# Replay-walking metrics: `f(replays_with_team) -> float`. Slower.
_REPLAY_METRICS: dict[str, Any] = {
    "first_launch_step":         first_launch_step,
    "active_turn_fraction":      active_turn_fraction,
    "multi_launch_turn_rate":    multi_launch_turn_rate,
}


def list_metrics() -> list[str]:
    """All registered metric names, sorted."""
    return sorted(set(_ROLLUP_METRICS) | set(_REPLAY_METRICS))


def is_rollup_metric(name: str) -> bool:
    return name in _ROLLUP_METRICS


def is_replay_metric(name: str) -> bool:
    return name in _REPLAY_METRICS


def get_metric(name: str):
    """Return the metric function by name. Raises KeyError if unknown."""
    if name in _ROLLUP_METRICS:
        return _ROLLUP_METRICS[name]
    if name in _REPLAY_METRICS:
        return _REPLAY_METRICS[name]
    raise KeyError(
        f"Unknown metric {name!r}. Known: {list_metrics()}"
    )


def baseline(name: str, source: str = "v15") -> float | None:
    """Look up the documented baseline for `name` from `source`.
    `source` is one of `v15`, `top10`, `midpack`. Returns None if the
    metric has no baseline from that source (e.g. v15 has no
    first_launch_step published).
    """
    if source == "v15":
        return V15_BASELINE.get(name)
    if source == "top10":
        return TOP10_BASELINE.get(name)
    if source == "midpack":
        return MIDPACK_BASELINE.get(name)
    raise KeyError(f"Unknown baseline source {source!r}")
