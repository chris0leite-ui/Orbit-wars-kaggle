"""Phase 4 tests for lib.trajectory_layer — SunFilter.

The load-bearing invariant: `SunFilter.is_safe(spec) == True` implies
the fleet provably does NOT die in the sun. Zero false negatives.

Pinned by:
- Hand-built geometric scenarios (sun ahead / sun behind / tangent /
  overshoot tail)
- Hypothesis property fuzz: random src/aim/world; whenever the filter
  says SAFE, the synthetic-fleet-via-fast_sim must NOT show
  outcome==`sun` over a long-enough rollout horizon.
"""

from __future__ import annotations

import math
import random
from typing import Any

import pytest

from hypothesis import HealthCheck, given, settings, strategies as st

pytestmark = pytest.mark.slow

from kaggle_environments import make

from lib.fast_sim import Snapshot, clone as fs_clone
from lib.fast_sim import from_obs as fs_from_obs
from lib.fast_sim import step as fs_step
from lib.trajectory_layer import (
    CENTER,
    SUN_RADIUS,
    LaunchSpec,
    SunFilter,
    SunVerdict,
    World,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _toy_obs(src_xy: tuple[float, float],
             *,
             src_radius: float = 2.0,
             src_ships: int = 50,
             angular_velocity: float = 0.0,
             ) -> dict:
    """Minimal obs with one source planet at (src_xy)."""
    return {
        "step": 0,
        "player": 0,
        "angular_velocity": angular_velocity,
        "planets": [
            [0, 0, src_xy[0], src_xy[1], src_radius, src_ships, 2],
        ],
        "initial_planets": [
            [0, 0, src_xy[0], src_xy[1], src_radius, src_ships, 2],
        ],
        "fleets": [],
        "comet_planet_ids": [],
        "comets": [],
        "next_fleet_id": 0,
    }


def _step_env_to_obs(seed: int, warmup: int, num_seats: int,
                     ) -> tuple[Any, int]:
    env = make("orbit_wars", configuration={"seed": seed}, debug=False)
    env.reset(num_agents=num_seats)
    rng = random.Random(seed * 11 + 3)
    for _ in range(warmup):
        obs0 = env.state[0].observation
        planets = (obs0["planets"] if isinstance(obs0, dict)
                   else obs0.planets)
        actions: list[list] = [[] for _ in range(num_seats)]
        for p in planets:
            owner = p[1]
            if 0 <= owner < num_seats and p[5] > 5 and rng.random() < 0.3:
                actions[owner].append([p[0], rng.uniform(0.0, 6.283),
                                       int(p[5] // 2)])
        env.step(actions)
    return env.state[0].observation, int(env.info.get("seed", seed))


def _committed_dies_in_sun(world: World,
                            spec: LaunchSpec,
                            *,
                            episode_seed: int,
                            num_seats: int,
                            max_steps: int = 50,
                            ) -> bool:
    """Apply the launch via fast_sim and step until the fleet either
    dies or arrives somewhere. Returns True iff the fleet's death is
    via SUN (point-to-segment hit on the sun disc).

    Identification: we track the fleet via its `from_planet_id`. The
    just-launched fleet is the one inbound from `spec.src_id` that
    wasn't in the obs pre-launch.
    """
    obs_before = world  # Phase 4 doesn't use this; we have the obs in caller
    raise NotImplementedError(
        "Use _commit_and_get_fleet_outcome instead — it has the obs.",
    )


def _commit_and_get_fleet_outcome(obs: Any,
                                    spec: LaunchSpec,
                                    *,
                                    episode_seed: int,
                                    num_seats: int,
                                    max_steps: int = 80,
                                    ) -> str:
    """Apply the launch to fast_sim and step until the synthetic fleet
    either dies (sun/oob/planet) or times out. Returns the outcome
    string: `"sun"`, `"oob"`, `"planet"`, `"alive"`, or `"never_existed"`.

    We track the fleet by its from_planet_id + matching angle/ships
    (Kaggle env doesn't surface a "died how" reason).
    """
    snap = fs_from_obs(obs, episode_seed=episode_seed, num_seats=num_seats)
    # Pre-existing fleet ids.
    pre_ids = {int(f[0]) for f in snap.obs["fleets"]}
    # Apply the launch.
    action = [[spec.src_id, spec.aim_angle, spec.ships]]
    other_actions = [[] for _ in range(num_seats - 1)]
    snap = fs_step(snap, [action] + other_actions, in_place=True)

    # Find the newly-launched fleet by id (NOT in pre_ids).
    new_fleets = [int(f[0]) for f in snap.obs["fleets"]
                  if int(f[0]) not in pre_ids]
    if not new_fleets:
        return "never_existed"
    target_fid = new_fleets[0]

    # Track this fleet position step-by-step. If it disappears within
    # `max_steps` AND the LAST recorded position was inside the sun's
    # safety zone (< SUN_RADIUS), classify as "sun". If last position
    # was outside the board, classify as "oob". Otherwise "planet".
    last_pos = None
    for f in snap.obs["fleets"]:
        if int(f[0]) == target_fid:
            last_pos = (float(f[2]), float(f[3]))
            break
    for _ in range(max_steps):
        snap = fs_step(snap, [[] for _ in range(num_seats)],
                        in_place=True)
        still_alive = False
        for f in snap.obs["fleets"]:
            if int(f[0]) == target_fid:
                last_pos = (float(f[2]), float(f[3]))
                still_alive = True
                break
        if not still_alive:
            # Fleet died this step. Classify via last_pos vs sun/board.
            if last_pos is None:
                return "never_existed"
            d_to_sun = math.hypot(last_pos[0] - CENTER, last_pos[1] - CENTER)
            if d_to_sun < SUN_RADIUS + 2.0:
                # Last seen close to the sun → killed by sun.
                # Use SUN_RADIUS + 2.0 as a wide identification band
                # (the fleet's position is the BEFORE-step position;
                # the death happened during the swept segment that
                # ended near or inside the sun).
                return "sun"
            if (last_pos[0] < 0.0 or last_pos[0] > 100.0
                    or last_pos[1] < 0.0 or last_pos[1] > 100.0):
                return "oob"
            return "planet"
    return "alive"


# ---------------------------------------------------------------------------
# Hand-built geometry — sun-ahead / sun-behind / overshoot tail
# ---------------------------------------------------------------------------


def test_aim_straight_at_sun_hits_sun():
    """Source at (20, 50) firing east (angle=0) goes through (50, 50)."""
    obs = _toy_obs((20.0, 50.0))
    world = World.from_obs(obs)
    sf = SunFilter(world)
    spec = LaunchSpec(src_id=0, aim_angle=0.0, ships=10, owner=0)
    assert sf.check(spec) == SunVerdict.HITS_SUN
    assert not sf.is_safe(spec)


def test_aim_perpendicular_to_sun_safe():
    """Source at (20, 50) firing north (angle=-pi/2 means up); path is
    along x=20, perpendicular distance to sun (50, 50) is 30. Safe."""
    obs = _toy_obs((20.0, 50.0))
    world = World.from_obs(obs)
    sf = SunFilter(world)
    spec = LaunchSpec(src_id=0, aim_angle=-math.pi / 2,
                       ships=10, owner=0)
    assert sf.check(spec) == SunVerdict.SAFE
    assert sf.is_safe(spec)


def test_aim_away_from_sun_safe():
    """Source at (20, 50) firing west (angle=pi) — sun is behind."""
    obs = _toy_obs((20.0, 50.0))
    world = World.from_obs(obs)
    sf = SunFilter(world)
    spec = LaunchSpec(src_id=0, aim_angle=math.pi, ships=10, owner=0)
    assert sf.check(spec) == SunVerdict.SAFE


def test_overshoot_tail_caught():
    """The crucial fix: source at (20, 50) firing east. Lead-aim might
    say "target at (35, 50)" (close, before the sun), but the fleet's
    INFINITE forward ray continues through (50, 50) and hits the sun.

    The closed-form check on the infinite ray catches this; the
    legacy `path_clears_sun` on the (src, target) segment would have
    passed it because the segment endpoint at (35, 50) is sun-clear.
    """
    obs = _toy_obs((20.0, 50.0))
    world = World.from_obs(obs)
    sf = SunFilter(world)
    spec = LaunchSpec(src_id=0, aim_angle=0.0, ships=10, owner=0)
    # Passing the lead-aim-style endpoint shouldn't matter — our
    # check is on the infinite ray.
    assert sf.check(spec, arrival_xy=(35.0, 50.0)) == SunVerdict.HITS_SUN


def test_tangent_inside_safety_margin_hits():
    """A path that just clips the sun's safety disc (SUN_RADIUS + 0.5)
    is rejected; just outside is accepted."""
    obs = _toy_obs((20.0, 50.0))
    world = World.from_obs(obs)
    sf = SunFilter(world)
    # The sun is at (50, 50). The fleet from (20, 50) at angle theta
    # passes the sun at perpendicular distance = sin(theta) * 30
    # (after projection onto direction). For theta such that sin(theta)
    # = 10.4 / 30 ≈ 0.347 → theta ≈ 0.354 rad: just at the boundary.
    boundary_theta = math.asin((SUN_RADIUS + 0.5) / 30.0)
    # Just inside the margin → HITS
    inside_theta = boundary_theta * 0.95
    spec_inside = LaunchSpec(src_id=0, aim_angle=inside_theta,
                              ships=10, owner=0)
    assert sf.check(spec_inside) == SunVerdict.HITS_SUN
    # Just outside the margin → SAFE
    outside_theta = boundary_theta * 1.10
    spec_outside = LaunchSpec(src_id=0, aim_angle=outside_theta,
                                ships=10, owner=0)
    assert sf.check(spec_outside) == SunVerdict.SAFE


def test_unknown_src_uncertain():
    """SunFilter on an unknown source returns UNCERTAIN; is_safe is False."""
    obs = _toy_obs((20.0, 50.0))
    world = World.from_obs(obs)
    sf = SunFilter(world)
    spec = LaunchSpec(src_id=999, aim_angle=0.0, ships=10, owner=0)
    assert sf.check(spec) == SunVerdict.UNCERTAIN
    assert not sf.is_safe(spec)


def test_custom_safety_margin():
    """`safety_margin=0.0` reduces conservatism; `safety_margin=5.0`
    increases it."""
    obs = _toy_obs((20.0, 50.0))
    world = World.from_obs(obs)
    # A path with perpendicular distance ≈ 10.3 to sun.
    theta = math.asin(10.3 / 30.0)
    spec = LaunchSpec(src_id=0, aim_angle=theta, ships=10, owner=0)

    sf_loose = SunFilter(world, safety_margin=0.0)
    assert sf_loose.check(spec) == SunVerdict.SAFE
    sf_strict = SunFilter(world, safety_margin=5.0)
    assert sf_strict.check(spec) == SunVerdict.HITS_SUN


# ---------------------------------------------------------------------------
# Hypothesis property test — THE Phase 4 gate
# ---------------------------------------------------------------------------


# Use a small CI profile (20 examples). Nightly can run with 200 via
# the `foundation_fuzz_nightly` profile already registered in
# tests/conftest_hypothesis.py.
@settings(
    max_examples=20,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow,
                            HealthCheck.function_scoped_fixture],
)
@given(
    # Sample from a region that gives the fleet ROOM to hit the sun
    # OR clearly miss; avoid degenerate spawn-on-sun.
    src_x=st.floats(min_value=12.0, max_value=88.0),
    src_y=st.floats(min_value=12.0, max_value=88.0),
    aim_angle=st.floats(min_value=0.0, max_value=2 * math.pi),
    ships=st.integers(min_value=1, max_value=99),
)
def test_safe_spec_never_dies_in_sun(src_x, src_y, aim_angle, ships):
    """Load-bearing invariant: if SunFilter says SAFE, fast_sim must
    NOT report the fleet dying in sun.

    Some SAFE specs will OOB or hit another planet; that's expected
    and not a SunFilter failure — we only assert "not sun".
    """
    # Don't generate spawn-on-sun cases (predictable HITS).
    if math.hypot(src_x - CENTER, src_y - CENTER) < SUN_RADIUS + 3.0:
        return  # Filter would correctly say HITS; not the assertion.

    obs = _toy_obs((src_x, src_y))
    world = World.from_obs(obs)
    sf = SunFilter(world)
    spec = LaunchSpec(src_id=0, aim_angle=aim_angle, ships=ships,
                       owner=0)
    if not sf.is_safe(spec):
        return  # The property only asserts about SAFE verdicts.

    outcome = _commit_and_get_fleet_outcome(
        obs, spec, episode_seed=12345, num_seats=2,
    )
    assert outcome != "sun", (
        f"SunFilter said SAFE but fast_sim says fleet died in sun: "
        f"src=({src_x}, {src_y}) angle={aim_angle} ships={ships}"
    )


# ---------------------------------------------------------------------------
# Integration with World — SunFilter on a real game state
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [7, 42, 100])
def test_sunfilter_on_real_world(seed: int):
    """For a real game state, SunFilter SAFE candidates must not die
    in sun under fast_sim. Picks our strongest planet, tries 8 evenly-
    spaced angles, asserts the SAFE ones don't sun-clip live.
    """
    obs, ep_seed = _step_env_to_obs(seed, warmup=30, num_seats=2)
    world = World.from_obs(obs, episode_seed=ep_seed)
    sf = SunFilter(world)

    owned = [p for p in world.planets
             if p.owner == 0 and p.ships >= 4 and not p.is_comet]
    if not owned:
        pytest.skip(f"seed={seed}: no usable source")
    src = max(owned, key=lambda p: p.ships)

    n_safe_checked = 0
    for k in range(8):
        angle = k * math.pi / 4
        spec = LaunchSpec(src_id=src.id, aim_angle=angle,
                           ships=3, owner=0)
        if not sf.is_safe(spec):
            continue
        outcome = _commit_and_get_fleet_outcome(
            obs, spec, episode_seed=ep_seed, num_seats=2,
        )
        assert outcome != "sun", (
            f"seed={seed} src={src.id} angle={angle}: "
            f"SunFilter SAFE but env killed by sun"
        )
        n_safe_checked += 1
    # At least one safe angle should exist on any reasonable board.
    assert n_safe_checked > 0, \
        f"seed={seed}: no SAFE angles among 8 — geometry too tight?"
