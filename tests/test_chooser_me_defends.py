"""ME-defends rollout self-policy — territorial-undervaluation fix
(BASELINE_ME_DEFENDS, value-leaf axis 2026-06-03).

Root cause (audit/2026-06-01-live-replay-diagnosis.md): the champion hoards
ships instead of expanding. The value leaf is read after a short rollout in
which future-me sits IDLE while opponents react every tick, so a candidate
that expands while leaving an owned planet exposed is scored as if that planet
is simply lost — the territorial value of "I expand AND defend what I hold" is
invisible. The agent therefore prefers to sit on its ships.

Fix (Rule 40 modeling fix, already built, default OFF): when
BASELINE_ME_DEFENDS=1, future-me plays a purely-defensive reaction inside the
CANDIDATE rollout (never attacks), so a threatened owned planet that we could
in fact hold stays held at the leaf and its production accrues. The baseline
stays ME-idle (asymmetric on purpose, chooser_trajectory.py:582-590), so the
candidate is credited for the territory it defends.

Rule 38 reproduction: with DEFENDS OFF the expansion candidate is scored
WITHOUT crediting that future-me holds its threatened territory (under-valued);
with DEFENDS ON the same candidate scores strictly higher (the held planet's
value is restored). Plus a default-OFF byte-identity guard.

Fixture pattern mirrors tests/test_chooser_pv_eta.py.
"""

from __future__ import annotations

import importlib
import math
import os

import pytest

from lib.fast_sim import from_obs as fs_from_obs
from lib.intent import World
from lib.world_model import WorldModel

# Match the live champion value head (composite in 2P) so the reproduction
# exercises the same leaf the live agent uses.
os.environ.setdefault("BASELINE_VALUE_HEAD", "hybrid")


def _p(pid, owner, x, y, ships, prod, r=1.5):
    return [pid, owner, float(x), float(y), float(r), int(ships), int(prod)]


def _reload_chooser():
    import agents.baseline.chooser_trajectory as ct
    importlib.reload(ct)
    return ct


def _score_expansion(ct, *, launch_ships=50, horizon=18):
    """Score the expansion candidate (idle source -> fat neutral) in a state
    where a SEPARATE frontier planet has an uncovered enemy fleet inbound.

      id0 S  mine   (30,15) 80 ships  -> idle source, launches at N
      id1 R  mine   (55,20) 70 ships  -> reinforcer the defender can use
      id2 P  mine   (60,15)  5 ships  -> frontier planet, under threat
      id3 N  neutral(45,15)  5 ships  -> fat target (prod 9)
      id4 O  opp    (90,85) 40 ships  -> opp home (favor needs an opponent)
      fleet F owner1 (90,15) 50 ships -> inbound to P along -x (uncovered)
    """
    favor_fn = ct.select_favor_fn()
    planets = [
        _p(0, 0, 30, 15, 80, 2),
        _p(1, 0, 55, 20, 70, 2),
        _p(2, 0, 60, 15, 5, 2),
        _p(3, -1, 45, 15, 5, 9),
        _p(4, 1, 90, 85, 40, 2),
    ]
    ang_F = math.atan2(15 - 15, 60 - 90)  # from (90,15) toward P (60,15): -x
    fleets = [[200, 1, 90.0, 15.0, float(ang_F), 4, 50]]
    obs = {
        "player": 0, "planets": planets, "fleets": fleets,
        "angular_velocity": 0.0, "comet_planet_ids": [], "comets": [],
        "step": 20,
    }
    world = World.from_obs(obs)
    snap = fs_from_obs(obs, num_seats=2)
    model = WorldModel.from_world(world)

    src = next(p for p in world.planets_by_id.values() if int(p.id) == 0)
    tgt = next(p for p in world.planets_by_id.values() if int(p.id) == 3)
    ang = math.atan2(15 - 15, 45 - 30)  # S -> N, +x
    baseline = ct.build_trajectory_baseline(
        snap, me=0, num_seats=2, horizon=horizon, favor_fn=favor_fn, gamma=0.99)
    delta, status, eta = ct.score_candidate_v4(
        snap, src, tgt, ships=int(launch_ships), angle=float(ang),
        me=0, num_seats=2, world=world, baseline_favors=baseline,
        favor_fn=favor_fn, gamma=0.99, horizon=horizon,
        skip_admissibility=False, wait_N=0, eta_hint=0, model=model)
    assert status == "scored", f"expected scored, got {status}"
    return float(delta)


def test_defends_default_off():
    os.environ.pop("BASELINE_ME_DEFENDS", None)
    ct = _reload_chooser()
    assert ct._ME_DEFENDS_ENABLED is False, "DEFENDS must default OFF"


def test_defends_raises_expansion_score():
    """Rule 38: the expansion candidate is UNDER-valued with the passive-self
    leaf (DEFENDS off) and scores strictly higher once future-me defends its
    threatened territory (DEFENDS on)."""
    os.environ.pop("BASELINE_ME_DEFENDS", None)
    ct = _reload_chooser()
    delta_off = _score_expansion(ct)

    os.environ["BASELINE_ME_DEFENDS"] = "1"
    ct = _reload_chooser()
    assert ct._ME_DEFENDS_ENABLED is True
    delta_on = _score_expansion(ct)

    assert delta_on > delta_off + 1.0, (
        f"DEFENDS must raise the expansion candidate's score by the held "
        f"territory's value; off={delta_off:.3f} on={delta_on:.3f}"
    )


def test_defends_off_byte_identical():
    """Env unset vs '0' produce identical Δ (production path unchanged)."""
    os.environ.pop("BASELINE_ME_DEFENDS", None)
    ct = _reload_chooser()
    delta_unset = _score_expansion(ct)

    os.environ["BASELINE_ME_DEFENDS"] = "0"
    ct = _reload_chooser()
    assert ct._ME_DEFENDS_ENABLED is False
    delta_zero = _score_expansion(ct)

    assert delta_unset == delta_zero, (
        f"unset vs '0' must be byte-identical; "
        f"unset={delta_unset!r} zero={delta_zero!r}")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("BASELINE_ME_DEFENDS", raising=False)
    yield
    monkeypatch.delenv("BASELINE_ME_DEFENDS", raising=False)
    _reload_chooser()
