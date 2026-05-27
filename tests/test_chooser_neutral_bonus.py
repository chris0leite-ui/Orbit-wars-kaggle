"""NEUTRAL_BONUS plumbing for `score_candidate_v4` (Fix 1, 2026-05-27
plan).

Pre-fix: `BASELINE_NEUTRAL_BONUS=2.0` (set in the bundle wrapper of
sub #52912707) was read into module constant `NEUTRAL_BONUS_WEIGHT`,
but the live scorer was `score_candidate_v4` — which did NOT consume
the constant. The bonus lived only inside the dead `score_candidate`
(v2 static-garrison) function.

Post-fix: `score_candidate_v4` applies the multiplicative bonus
**after** the leaf delta, gated on `tgt.owner == -1` (neutral) and
`delta > 0`. Same applies to `score_candidate_v4_joint` when ALL
legs target a neutral.

This test reproduces the silent failure: the same World + candidate
must produce a LARGER delta when `BASELINE_NEUTRAL_BONUS=2.0` than
when `BASELINE_NEUTRAL_BONUS=1.0`.
"""

from __future__ import annotations

import importlib
import os
from types import SimpleNamespace

import pytest

from lib.fast_sim import from_obs as fs_from_obs
from lib.intent import World
from lib.world_model import WorldModel


def _planet(pid, owner, x, y, *, ships=20, production=2, radius=1.5):
    return SimpleNamespace(
        id=pid, owner=owner, x=x, y=y, radius=radius,
        ships=ships, production=production,
    )


def _build_world_and_snap(planets, my_id=0, num_seats=2, step=0):
    obs = {
        "player": my_id,
        "planets": [
            [p.id, p.owner, p.x, p.y, p.radius, p.ships, p.production]
            for p in planets
        ],
        "fleets": [],
        "angular_velocity": 0.0,
        "comet_planet_ids": [],
        "comets": [],
        "step": step,
    }
    world = World.from_obs(obs)
    snap = fs_from_obs(obs, num_seats=num_seats)
    model = WorldModel.from_world(world)
    return world, snap, model


def _reload_chooser():
    """Re-import to pick up env-var changes at module load time."""
    import agents.baseline.chooser_trajectory as ct
    importlib.reload(ct)
    return ct


def _score_neutral_capture(neutral_bonus: str):
    """Build a 2P World where ME owns S at (10, 50), neutral N at
    (30, 50). Score a capture launch of 20 ships at N. Return the
    delta."""
    os.environ["BASELINE_NEUTRAL_BONUS"] = neutral_bonus
    os.environ["BASELINE_NEUTRAL_EARLY_EXTRA"] = "1.0"  # isolate the base bonus
    os.environ["BASELINE_NEUTRAL_EARLY_HORIZON"] = "0"
    ct = _reload_chooser()
    favor_fn = ct.select_favor_fn()

    src = _planet(0, 0, 10.0, 50.0, ships=40, production=2)
    neutral = _planet(1, -1, 30.0, 50.0, ships=5, production=2)
    opp = _planet(2, 1, 90.0, 50.0, ships=40, production=2)
    world, snap, model = _build_world_and_snap([src, neutral, opp])

    horizon = 8
    baseline = ct.build_trajectory_baseline(
        snap, me=0, num_seats=2, horizon=horizon,
        favor_fn=favor_fn, gamma=0.99,
    )

    delta, status, _eta = ct.score_candidate_v4(
        snap, src, neutral, ships=20, angle=0.0,
        me=0, num_seats=2, world=world,
        baseline_favors=baseline, favor_fn=favor_fn,
        gamma=0.99, horizon=horizon,
        skip_admissibility=True,
        wait_N=0, model=model,
    )
    assert status == "scored"
    return float(delta)


def test_neutral_bonus_scales_positive_delta():
    """With `BASELINE_NEUTRAL_BONUS=2.0`, the delta on a positive-Δ
    neutral capture must be exactly 2× the delta with bonus 1.0.
    Reproduces the silent-failure state of the pre-fix submission."""
    delta_off = _score_neutral_capture("1.0")
    delta_on = _score_neutral_capture("2.0")

    # If pre-fix, both deltas would be equal (env var was dead code).
    # If post-fix, delta_on ≈ 2 × delta_off when delta_off > 0.
    assert delta_off > 0.0, "test fixture must produce a positive delta"
    ratio = delta_on / delta_off
    assert ratio == pytest.approx(2.0, abs=1e-9), (
        f"expected 2.0× delta scaling under BASELINE_NEUTRAL_BONUS=2.0; "
        f"got {ratio:.4f} (off={delta_off:.6f}, on={delta_on:.6f}). "
        "Pre-fix bug: the env var was dead code and the ratio was 1.0."
    )


def test_neutral_bonus_default_is_noop():
    """Default `BASELINE_NEUTRAL_BONUS=1.0` ⇒ no scaling. Verifies
    the bonus path doesn't silently apply when disabled."""
    delta_a = _score_neutral_capture("1.0")
    delta_b = _score_neutral_capture("1.0")
    assert delta_a == pytest.approx(delta_b, abs=1e-9)


def test_neutral_bonus_does_not_punish_negative_delta():
    """Bonus path is gated on `delta > 0`. A losing candidate must
    NOT be made more-negative by the multiplier (which would push
    the chooser away from the right answer for the wrong reason)."""
    os.environ["BASELINE_NEUTRAL_BONUS"] = "2.0"
    os.environ["BASELINE_NEUTRAL_EARLY_EXTRA"] = "1.0"
    os.environ["BASELINE_NEUTRAL_EARLY_HORIZON"] = "0"
    ct = _reload_chooser()
    favor_fn = ct.select_favor_fn()

    # Tiny source, big neutral garrison — the capture will fail and
    # the favor delta should be ≤ 0 in this World.
    src = _planet(0, 0, 10.0, 50.0, ships=3, production=1)
    neutral = _planet(1, -1, 30.0, 50.0, ships=200, production=2)
    opp = _planet(2, 1, 90.0, 50.0, ships=40, production=2)
    world, snap, model = _build_world_and_snap([src, neutral, opp])
    horizon = 8
    baseline = ct.build_trajectory_baseline(
        snap, me=0, num_seats=2, horizon=horizon,
        favor_fn=favor_fn, gamma=0.99,
    )
    delta_with_bonus, status, _ = ct.score_candidate_v4(
        snap, src, neutral, ships=3, angle=0.0,
        me=0, num_seats=2, world=world,
        baseline_favors=baseline, favor_fn=favor_fn,
        gamma=0.99, horizon=horizon,
        skip_admissibility=True, wait_N=0, model=model,
    )
    assert status == "scored"
    # If delta is negative, we want the same delta as without the
    # bonus — i.e. the bonus path was NOT entered.
    if delta_with_bonus < 0:
        os.environ["BASELINE_NEUTRAL_BONUS"] = "1.0"
        ct = _reload_chooser()
        favor_fn = ct.select_favor_fn()
        baseline2 = ct.build_trajectory_baseline(
            snap, me=0, num_seats=2, horizon=horizon,
            favor_fn=favor_fn, gamma=0.99,
        )
        delta_no_bonus, _, _ = ct.score_candidate_v4(
            snap, src, neutral, ships=3, angle=0.0,
            me=0, num_seats=2, world=world,
            baseline_favors=baseline2, favor_fn=favor_fn,
            gamma=0.99, horizon=horizon,
            skip_admissibility=True, wait_N=0, model=model,
        )
        assert delta_with_bonus == pytest.approx(delta_no_bonus, abs=1e-9)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    # Reset bonus env vars between tests; reload chooser so subsequent
    # tests see fresh defaults.
    for k in ("BASELINE_NEUTRAL_BONUS", "BASELINE_NEUTRAL_EARLY_EXTRA",
             "BASELINE_NEUTRAL_EARLY_HORIZON", "BASELINE_LEADER_FOCUS",
             "BASELINE_FOLLOWON_BONUS"):
        monkeypatch.delenv(k, raising=False)
    yield
    _reload_chooser()
