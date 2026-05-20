"""Replay-driven regression tests for `agents.baseline`.

R38 fix-verification harness: each test pins a specific failure or behavioural
fact observed in a real Kaggle ladder game. Subsequent phases of the
improvement plan (see /root/.claude/plans/so-you-have-multiple-tranquil-papert.md)
edit these tests as they fix the underlying failure mode — a passing test that
encodes the *current buggy behaviour* is the negative baseline; a passing
test that encodes the *fixed behaviour* is the positive regression gate.

Fixtures live in `tests/fixtures/replays/<scenario>.json` (committed) and were
sliced from large live-episode replays via `scripts/extract_replay_snapshot.py`
(replay JSONs themselves are gitignored — too big).

Scenarios on disk:
  - linrock_77150441_step44.json
      4P seat 3, step 44 of 169. Live opponent `linrock` has a 78-ship
      fleet 11.2 units from our home p11 (garrison ~24 ships). The live
      agent (sub 52827111) emits 45 ships outward from p13 instead of
      defending. Our current modular baseline emits nothing (wait_N>0
      reserve-without-emit bug at chooser.py:131-134). Either way home
      falls 3 turns later. Phase 1 + Phase 2 should change this.
  - claws_77164175_step226.json
      4P seat 3 (Claws perspective), step 226 of 500. Claws has just
      captured p31 (prod=1) with 506 ships landed; next live action is a
      506-ship relaunch from p31 onward (the relay pattern). Our current
      proposer does not produce own→own migration candidates. Phase 3
      should add this.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from agents.baseline import main as baseline_main


REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures" / "replays"


# Bound the chooser wallclock so the test suite stays under CI budget;
# matches the proposer/chooser parity envelope.
_TEST_WALLCLOCK_MS = "600"


def _load(name: str) -> dict:
    path = FIXTURE_DIR / name
    with path.open() as fh:
        return json.load(fh)


@pytest.fixture(autouse=True)
def _bound_wallclock(monkeypatch):
    monkeypatch.setenv("BASELINE_WALLCLOCK_MS", _TEST_WALLCLOCK_MS)


@pytest.fixture
def linrock_step44() -> dict:
    return _load("linrock_77150441_step44.json")


@pytest.fixture
def claws_step226() -> dict:
    return _load("claws_77164175_step226.json")


# ---------------------------------------------------------------------------
# linrock kill-chain — defensive failure mode (Phases 1, 2, 4 target this)
# ---------------------------------------------------------------------------


def test_linrock_step44_reproduces_under_emission(linrock_step44):
    """NEGATIVE BASELINE — pinning the current bug.

    On step 44 of the linrock loss, a 78-ship enemy fleet is 11.2 units from
    our home p11 (garrison ~24 ships, fleet ETA ≈ 2 turns). Our current
    chooser silently emits NOTHING this turn because the top-scoring
    candidate has `wait_N > 0` and reserves both src and tgt without
    actually launching (agents/baseline/chooser.py:131-134).

    After Phase 1 (wait_N>0 emit fix) this test MUST be updated — the
    chooser should emit a wait_N=0 alternate (anything that isn't the
    empty list is progress). Phase 2's garrison floor should additionally
    block outward bleeds from p11 itself.
    """
    obs = linrock_step44["obs"]
    cfg = linrock_step44["configuration"]
    action = baseline_main.agent(obs, cfg)
    assert action == [], (
        "Expected current chooser to silently under-emit (wait_N>0 reserve "
        "bug). Got a non-empty action — the bug may already be fixed; "
        "update this assertion to encode the post-fix expectation."
    )


def test_linrock_step44_home_under_threat(linrock_step44):
    """Sanity check on the fixture itself — the threat is real.

    Our home p11 sees an enemy fleet at distance ~11 units with size ≥60.
    If this stops being true, the fixture is stale and the defensive
    assertions above lose their meaning.
    """
    obs = linrock_step44["obs"]
    my_seat = linrock_step44["my_seat"]

    my_planets = [p for p in obs["planets"] if int(p[1]) == my_seat]
    assert my_planets, "expected at least one owned planet"
    p11 = next((p for p in my_planets if int(p[0]) == 11), None)
    assert p11 is not None, "fixture should still own home p11"

    import math

    px, py = float(p11[2]), float(p11[3])
    enemy_fleets = [f for f in obs["fleets"] if int(f[1]) != my_seat]
    inbound = [
        (int(f[0]), int(f[1]), int(f[6]),
         math.hypot(px - float(f[2]), py - float(f[3])))
        for f in enemy_fleets
    ]
    near_heavy = [t for t in inbound if t[2] >= 60 and t[3] < 20]
    assert near_heavy, (
        f"expected at least one ≥60-ship enemy fleet within 20u of p11; "
        f"saw {sorted(inbound, key=lambda t: t[3])[:3]}"
    )


# ---------------------------------------------------------------------------
# Claws relay pattern — offensive opportunity (Phase 3 targets this)
# ---------------------------------------------------------------------------


def test_claws_step226_relay_resource_present(claws_step226):
    """Sanity check on the fixture — p31 just got 506 ships.

    Phase 3 (migration_solver) will use this state to enumerate an
    own→own relay candidate from p31 to a richer Claws planet. This
    test asserts the resource is present so the Phase-3 assertion has
    a meaningful target.
    """
    obs = claws_step226["obs"]
    my_seat = claws_step226["my_seat"]

    p31 = next((p for p in obs["planets"] if int(p[0]) == 31), None)
    assert p31 is not None, "p31 must be in the snapshot"
    assert int(p31[1]) == my_seat, (
        f"p31 should be owned by seat {my_seat} (Claws); got owner={p31[1]}"
    )
    assert int(p31[5]) >= 500, (
        f"p31 should carry the just-landed relay stack (≥500 ships); "
        f"got {p31[5]}"
    )


def test_claws_step226_no_own_to_own_migration_today(claws_step226):
    """NEGATIVE BASELINE — pinning absence of relay candidates.

    The current proposer enumerates capture targets only (non-mine
    planets). It does NOT enumerate own→own ship repositioning. We can
    detect this indirectly: feed the snapshot to the proposer and check
    that no proposed candidate has `tgt.owner == my_seat`.

    Phase 3 (port migration_solver from origin/claude/strategy-framework-
    design-OyoYR-rebased) will change this. After Phase 3 this test
    MUST be updated to assert the OPPOSITE: that at least one own→own
    relay candidate is present in the prerank.
    """
    from kaggle_environments.envs.orbit_wars.orbit_wars import Planet, Fleet
    from lib.intent import World
    from lib.world_model import WorldModel
    from agents.baseline import proposer

    obs = claws_step226["obs"]
    my_seat = claws_step226["my_seat"]

    planets = [Planet(*p) for p in obs["planets"]]
    fleets = [Fleet(*f) for f in obs["fleets"]]
    my_planets = [p for p in planets if int(p.owner) == my_seat]
    other_planets = [p for p in planets if int(p.owner) != my_seat]
    assert my_planets and other_planets, "fixture should have mixed ownership"

    world = World.from_obs(obs)
    model = WorldModel.from_world(world)
    omega = float(obs.get("angular_velocity", 0.0))

    threatened_mine = [
        p for p in my_planets
        if model.time_to_enemy_threat(int(p.id), my_seat, world) is not None
    ]
    target_pool = other_planets + threatened_mine

    prerank = proposer.propose(
        my_planets, target_pool, world, model, my_seat, omega,
        baseline_len=proposer.MAX_HORIZON + 1,
    )

    own_to_own = [
        c for c in prerank if int(c[2].owner) == my_seat
        and int(c[2].id) != int(c[1].id)
        and int(c[2].id) not in {int(p.id) for p in threatened_mine}
    ]
    assert own_to_own == [], (
        "Expected NO own→own non-threat-relay candidates from the current "
        f"proposer; saw {len(own_to_own)}. Phase 3 should change this — "
        "update the assertion accordingly."
    )
