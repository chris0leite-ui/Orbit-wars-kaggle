"""Replay-driven regression tests for `agents.baseline` (Phase 7 port).

Each test pins a specific behavioural fact observed in a real Kaggle
ladder game. Fixtures live in `tests/fixtures/replays/<scenario>.json`.

Scenarios:
  - claws_77164175_step223.json
      4P seat 3 (Claws perspective), step 223 of 500. Pre-launch state
      Claws sees before emitting [3, 1.005, 588] — the p3 → p31 leg of
      the relay chain to p23/p17/p7. p3 has ~784 ships, p31 (enemy,
      prod=1) has 43, p23 (enemy, prod=1) is one short hop further.
      Without the chain bonus, the prod=1 p31 looks low-value; with it,
      the downstream relay value shows up in cheap_delta.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agents.baseline import proposer
from kaggle_environments.envs.orbit_wars.orbit_wars import Planet, Fleet
from lib.intent import World
from lib.world_model import WorldModel


REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures" / "replays"


def _load(name: str) -> dict:
    with (FIXTURE_DIR / name).open() as fh:
        return json.load(fh)


@pytest.fixture
def claws_step223() -> dict:
    return _load("claws_77164175_step223.json")


def _build_world(obs: dict, me: int):
    planets = [Planet(*p) for p in obs["planets"]]
    fleets = [Fleet(*f) for f in obs.get("fleets", []) or []]
    my_planets = [p for p in planets if int(p.owner) == me]
    other_planets = [p for p in planets if int(p.owner) != me]
    world = World.from_obs(obs)
    model = WorldModel.from_world(world)
    omega = float(obs.get("angular_velocity", 0.0))
    threatened_mine = [
        p for p in my_planets
        if model.time_to_enemy_threat(int(p.id), me, world) is not None
    ]
    target_pool = other_planets + threatened_mine
    return my_planets, target_pool, world, model, omega


def test_claws_step223_chain_bonus_promotes_p3_p31(claws_step223, monkeypatch):
    """POSITIVE REGRESSION — Phase 7 chain-bonus on the Claws relay.

    Step 223 of ep 77164175 is the pre-launch state Claws sees before
    emitting [3, 1.005, 588] — the p3 → p31 leg of the relay chain.
    Without the bonus, capturing the prod=1 p31 looks low-value; with
    the bonus, the chain value to downstream targets (p23/p17/p7) is
    folded into the cheap_delta for the p3→p31 candidate.
    """
    obs = claws_step223["obs"]
    me = claws_step223["my_seat"]
    my_planets, target_pool, world, model, omega = _build_world(obs, me)

    monkeypatch.delenv("BASELINE_CHAIN_BONUS", raising=False)
    base = proposer.propose(
        my_planets, target_pool, world, model, me, omega,
        baseline_len=proposer.MAX_HORIZON + 1,
    )
    base_p3_p31 = [c for c in base if int(c[1].id) == 3 and int(c[2].id) == 31]
    assert base_p3_p31, (
        "expected a p3→p31 capture candidate to exist with chain bonus off"
    )
    base_cheap = max(c[0] for c in base_p3_p31)
    for c in base:
        assert len(c) == 9, "Phase 8 tuple shape is 9"
        assert c[8] is False, "chain bonus off should never set is_chain=True"

    monkeypatch.setenv("BASELINE_CHAIN_BONUS", "1")
    boosted = proposer.propose(
        my_planets, target_pool, world, model, me, omega,
        baseline_len=proposer.MAX_HORIZON + 1,
    )
    chain_p3_p31 = [
        c for c in boosted
        if int(c[1].id) == 3 and int(c[2].id) == 31 and c[8]
    ]
    assert chain_p3_p31, (
        f"expected at least one is_chain=True p3→p31 candidate with bonus on; "
        f"got {len(boosted)} candidates total, "
        f"{sum(1 for c in boosted if c[8])} chain-tagged"
    )
    chain_cheap = max(c[0] for c in chain_p3_p31)
    assert chain_cheap > base_cheap, (
        f"chain bonus must strictly raise p3→p31 cheap_delta; "
        f"base={base_cheap:.2f} chain={chain_cheap:.2f}"
    )
