"""tablebase_hybrid agent: constructed scenarios + replay regression."""

from __future__ import annotations

import json
from pathlib import Path

from agents.tablebase_hybrid.main import agent as hybrid_agent
from agents.trajectory_roi.main import agent as roi_agent
from tests.scenarios.base import _obs, _planet


REPLAY_DIR = Path(__file__).parent.parent / "audit" / "live-episodes" / "52784853"


def test_free_capture_uses_solver_launch():
    # Tight geometry — solver picks LAUNCH at depth=14 within the
    # 50ms runtime budget.
    planets = [
        _planet(0, owner=0, x=10.0, y=50.0, ships=100, production=2),
        _planet(1, owner=-1, x=16.0, y=50.0, ships=5, production=1),
        _planet(2, owner=1, x=90.0, y=10.0, ships=10, production=1),
    ]
    obs = _obs(planets=planets, step=10, player=0)
    hybrid_emits = hybrid_agent(obs)
    assert hybrid_emits, (
        f"free capture: hybrid must launch; got {hybrid_emits}"
    )
    emits_from_p0 = [e for e in hybrid_emits if int(e[0]) == 0]
    assert emits_from_p0, (
        f"free capture: hybrid must launch from p0; got {hybrid_emits}"
    )
    # Solver's emit should be a launch heading east (toward p1 at +6,0).
    src, angle, ships = emits_from_p0[0]
    assert abs(angle) < 0.1, f"angle should be ~0 (east); got {angle}"
    assert ships >= 5


def test_bounce_source_idles():
    planets = [
        _planet(0, owner=0, x=10.0, y=50.0, ships=10, production=1),
        _planet(1, owner=1, x=30.0, y=50.0, ships=100, production=2),
        _planet(2, owner=-1, x=90.0, y=10.0, ships=10, production=1),
    ]
    obs = _obs(planets=planets, step=10, player=0)
    hybrid_emits = hybrid_agent(obs)
    emits_from_p0 = [e for e in hybrid_emits if int(e[0]) == 0]
    assert emits_from_p0 == [], (
        f"bounce: hybrid must IDLE from p0; got {hybrid_emits}"
    )


def test_no_clusters_is_passthrough():
    planets = []
    spacing = 25.0
    for i in range(6):
        x = 10.0 + (i % 3) * spacing
        y = 10.0 + (i // 3) * spacing
        owner = 0 if i == 0 else (1 if i == 5 else -1)
        planets.append(_planet(i, owner=owner, x=x, y=y,
                               ships=20, production=1))
    obs = _obs(planets=planets, step=10, player=0)
    roi_emits = roi_agent(obs)
    hybrid_emits = hybrid_agent(obs)
    assert hybrid_emits == roi_emits, (
        f"no-cluster turn must be a no-op. "
        f"roi={roi_emits} hybrid={hybrid_emits}"
    )


def test_mixed_clustered_and_freestanding_sources():
    # p0+p1 = bounce cluster (mine vs heavily-defended enemy).
    # p3 = freestanding mine planet far from the cluster + within
    # reach of p4 (neutral free capture). p2 = phantom-far enemy.
    planets = [
        _planet(0, owner=0, x=10.0, y=10.0, ships=10, production=1),
        _planet(1, owner=1, x=30.0, y=10.0, ships=100, production=2),
        _planet(2, owner=1, x=95.0, y=90.0, ships=10, production=1),
        _planet(3, owner=0, x=80.0, y=80.0, ships=100, production=2),
        _planet(4, owner=-1, x=85.0, y=80.0, ships=5, production=1),
    ]
    obs = _obs(planets=planets, step=10, player=0)
    hybrid_emits = hybrid_agent(obs)
    # p0 is in the bounce cluster → must IDLE.
    p0_emits = [e for e in hybrid_emits if int(e[0]) == 0]
    assert p0_emits == [], (
        f"clustered bounce source p0 must IDLE; got {hybrid_emits}"
    )
    # p3 is freestanding (cluster {0,1} doesn't include 3) → heuristic
    # should launch a free capture from p3 toward p4.
    p3_emits = [e for e in hybrid_emits if int(e[0]) == 3]
    assert p3_emits, (
        f"freestanding p3 free-capture should pass through; "
        f"got {hybrid_emits}"
    )


def test_replay_disagree_over_clustered_sources_idle():
    """episode-76990778-replay.json step=20 seat=0 cluster={0, 28, 34}:
    audit shows heuristic launches from {0, 28}; solver IDLEs.
    Hybrid must produce no emits from any of {0, 28, 34}."""
    replay_path = REPLAY_DIR / "episode-76990778-replay.json"
    if not replay_path.exists():
        return
    with replay_path.open() as f:
        replay = json.load(f)
    obs = dict(replay["steps"][20][0]["observation"])
    obs.setdefault("initial_planets",
                   [list(p) for p in obs.get("planets", [])])

    roi_emits = roi_agent(obs)
    hybrid_emits = hybrid_agent(obs)
    cluster_sources = {0, 28, 34}
    roi_from_cluster = [e for e in roi_emits
                        if int(e[0]) in cluster_sources]
    hybrid_from_cluster = [e for e in hybrid_emits
                           if int(e[0]) in cluster_sources]
    if not roi_from_cluster:
        return
    assert hybrid_from_cluster == [], (
        f"audit's DISAGREE-OVER cluster must IDLE in hybrid. "
        f"roi={roi_from_cluster} hybrid={hybrid_from_cluster}"
    )
