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
  - linrock_77150441_step12.json
      4P seat 3, step 12 of 169. Mid-opening; the top-scoring positive-Δ
      candidate is `wait_N=10` (deferred capture), and the chooser used
      to silently emit nothing because it reserved src+tgt without
      firing. Several positive-Δ wait_N=0 alternates from the same
      source were available the whole time. Phase 1 (wait_N>0 emit fix)
      flips this from `[]` to a real launch.
  - linrock_77150441_step44.json
      4P seat 3, step 44 of 169. Late kill-chain — a 78-ship enemy
      fleet is 11.2 units from our home p11 with the garrison too low
      to hold. The proposer produces NO positive-Δ candidates on the
      modular baseline (every offensive move makes the F1 ship-balance
      worse than idling). The agent correctly does nothing here; Phase
      2 (garrison floor) and Phase 4 (neighbour reinforce) are the
      structural fixes that prevent the state from reaching this point.
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
def linrock_step12() -> dict:
    return _load("linrock_77150441_step12.json")


@pytest.fixture
def linrock_step44() -> dict:
    return _load("linrock_77150441_step44.json")


@pytest.fixture
def claws_step226() -> dict:
    return _load("claws_77164175_step226.json")


# ---------------------------------------------------------------------------
# linrock kill-chain — defensive failure mode (Phases 1, 2, 4 target this)
# ---------------------------------------------------------------------------


def test_linrock_step12_phase1_emits_wait0_alternate(linrock_step12):
    """POSITIVE REGRESSION — locks the Phase 1 fix.

    At step 12 of the linrock loss, the top-scoring positive-Δ candidate
    is `wait_N = 10` (src=11, tgt=3). The pre-Phase-1 chooser reserved
    the src+tgt slot and emitted nothing this turn (the documented bug at
    chooser.py:131-134). Post-fix, the chooser skips the wait_N>0
    candidate without claiming the slot, and a positive-Δ wait_N=0
    alternate from the same source fires instead.

    Audit on the first 80 turns of this episode found 35 such turns
    where the bug bit (top scorer wait_N>0 with at least one positive-Δ
    wait_N=0 alternate available). This assertion guards against
    regression to the old reserve-without-emit logic.
    """
    obs = linrock_step12["obs"]
    cfg = linrock_step12["configuration"]
    action = baseline_main.agent(obs, cfg)
    assert action, (
        "Phase 1 fix requires the chooser to emit a wait_N=0 alternate "
        "rather than silently reserving and skipping. Got `[]` — has the "
        "reserve-without-emit logic been re-introduced at chooser.py?"
    )
    # Sanity: the emitted launch must be a valid (src, angle, ships)
    # triple referencing a planet we own.
    my_seat = linrock_step12["my_seat"]
    my_ids = {int(p[0]) for p in obs["planets"] if int(p[1]) == my_seat}
    for launch in action:
        assert int(launch[0]) in my_ids, (
            f"chooser emitted launch from non-owned planet: {launch}"
        )


def test_linrock_step44_garrison_floor_blocks_home_bleed(linrock_step44):
    """POSITIVE REGRESSION — locks the Phase 2 fix.

    At step 44, an enemy 78-ship fleet is 11.2 units from our home p11.
    The model.ledger places that fleet at p11 with eta ~3. The garrison
    floor `max_safe_launch_now(p11, ...)` must return 0 because:
        my garrison @ eta3 = 24 + 5*3 = 39
        enemy arrived @ eta3 = 78
        deficit = 78 + safety(2) − 39 = 41 → negative budget → clamped 0

    Pre-Phase-2 the proposer would let `enumerate_ship_counts` enumerate
    up to `src.ships=24` for p11. Post-Phase-2 the floor caps the budget
    at 0 so no fire-now candidate from p11 survives — preventing the
    home-bleed pattern.
    """
    from kaggle_environments.envs.orbit_wars.orbit_wars import Planet
    from lib.intent import World
    from lib.world_model import WorldModel
    from agents.baseline.proposer import max_safe_launch_now

    obs = linrock_step44["obs"]
    me = linrock_step44["my_seat"]
    world = World.from_obs(obs)
    model = WorldModel.from_world(world)
    p11 = next(Planet(*p) for p in obs["planets"] if int(p[0]) == 11)
    assert int(p11.owner) == me, "fixture should still own p11"
    floor = max_safe_launch_now(p11, world, model, me)
    assert floor == 0, (
        f"expected garrison floor 0 at p11 under 78-ship threat; got {floor}"
    )


def test_linrock_step44_no_positive_delta_so_correctly_idle(linrock_step44):
    """Pin a different failure mode — Phase 2/4 territory.

    By step 44 the kill chain is already in motion. Our proposer
    produces no positive-Δ candidates: every offensive launch worsens
    the F1 ship-balance (home is about to fall regardless), and the
    proposer enumerates no defensive/reinforcement candidates. So the
    chooser correctly does nothing here.

    This test pins the CURRENT outcome (empty action) and documents that
    Phase 1 does NOT affect this step. Phase 2 (garrison floor) and
    Phase 4 (neighbour reinforce) are the structural fixes that prevent
    the state from reaching this point — once they land we expect this
    test to remain `assert action == []` (the rescue has already
    happened earlier) OR to be replaced with an earlier-step counterpart.
    """
    obs = linrock_step44["obs"]
    cfg = linrock_step44["configuration"]
    action = baseline_main.agent(obs, cfg)
    assert action == [], (
        "Expected `[]` at step 44 because no candidate has positive Δ — "
        "if a launch is emitted here, the proposer's candidate pool has "
        "changed (Phase 2/3/4 work). Update or replace this assertion."
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
