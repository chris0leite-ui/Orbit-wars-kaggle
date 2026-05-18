"""Phase A bundle oracles — black-box coordination properties.

Each oracle constructs a synthetic world, runs `agents.bundle.main.agent`,
and asserts an emit-output property. Helpers mirror the pattern in the
audit-workflow-performance-btjeK branch's `test_planner_oracles.py`
(_planet, _obs, _targets_of_emits) and the existing `test_bundle_agent.py`
monkeypatch knobs fixture.

Expected map (recorded as Phase A baseline):
  A1, A2, A6, A7, A8, A9 -> PASS
  A3, A5                 -> XFAIL strict
  A4, A10                -> XFAIL non-strict

XPASS on A3/A5 (strict) reports as failure: Phase B fix detected.
XPASS on A4/A10 (non-strict) is informational.
"""

from __future__ import annotations

import math

import pytest


# ---- knobs ----------------------------------------------------------------

_BUNDLE_DEFAULTS = {
    "BUNDLE_OPP_MODE": "event_driven",
    "BUNDLE_OWN_CANDS_PER_SOURCE": "5",
    "BUNDLE_TOTAL_MS": "200",
    "BUNDLE_MIRROR_MS": "60",
    "BUNDLE_OWN_LAUNCH_TURNS": "0",
    "BUNDLE_HORIZON": "15",
    "BUNDLE_OWN_MAX_DEPTH": "2",
    "BUNDLE_OWN_BEAM_WIDTH": "3",
    "BUNDLE_OPP_MAX_DEPTH": "1",
    "BUNDLE_OPP_CANDS_PER_SOURCE": "2",
}


@pytest.fixture
def bundle_knobs(monkeypatch):
    for k, v in _BUNDLE_DEFAULTS.items():
        monkeypatch.setenv(k, v)
    from agents.bundle.main import _LAST_BUNDLE
    _LAST_BUNDLE.clear()


# ---- helpers --------------------------------------------------------------

def _planet(pid, owner, x, y, ships=10, production=1, radius=1.5):
    return [int(pid), int(owner), float(x), float(y),
            float(radius), int(ships), int(production)]


def _obs(planets, fleets=None, step=0, player=0, angular_velocity=0.0):
    return {
        "player": int(player),
        "step": int(step),
        "planets": planets,
        "fleets": fleets or [],
        "comets": [],
        "comet_planet_ids": [],
        "angular_velocity": float(angular_velocity),
        "initial_planets": [list(p) for p in planets],
    }


def _emit(obs):
    from agents.bundle.main import agent
    return agent(obs, configuration=None)


def _targets_of_emits(obs, moves):
    """For each emit (src_id, angle, ships), raycast from source along angle
    and return the first planet-radius hit ID (skip if no hit).
    Assumes obs['angular_velocity']==0 — oracles do not rotate."""
    pmap = {p[0]: p for p in obs["planets"]}
    out = []
    for m in moves:
        src_id = int(m[0])
        angle = float(m[1])
        src = pmap.get(src_id)
        if src is None:
            continue
        sx, sy = src[2], src[3]
        dx, dy = math.cos(angle), math.sin(angle)
        best_id, best_d = None, float("inf")
        for p in obs["planets"]:
            if p[0] == src_id:
                continue
            ex, ey, pr = p[2] - sx, p[3] - sy, p[4]
            proj = ex * dx + ey * dy
            if proj < 0:
                continue
            perp_sq = ex * ex + ey * ey - proj * proj
            if perp_sq >= pr * pr:
                continue
            hit = proj - math.sqrt(pr * pr - perp_sq)
            if hit < best_d:
                best_d, best_id = hit, p[0]
        if best_id is not None:
            out.append(best_id)
    return out


# ---- A1 — capture credit sanity ------------------------------------------

def test_A1_capture_credit_sanity_100v5(bundle_knobs):
    """100-ship base vs 5-ship neutral. Path-integral evaluator must credit
    the trivial capture. If A1 fails, bug #15 analog: scoring is broken
    and every other oracle is suspect.

    Layout note: planets sit at y=25 so the source->target ray clears the
    sun at (50,50) r=10 with 25-unit perpendicular margin."""
    obs = _obs([
        _planet(0, 0, 30, 25, ships=100, production=1),
        _planet(1, -1, 70, 25, ships=5, production=1),
    ])
    assert 1 in _targets_of_emits(obs, _emit(obs))


# ---- A2 — path-integrated time-value -------------------------------------

def test_A2_path_integrated_time_value(bundle_knobs):
    """Near high-production neutral should out-rank far low-production
    neutral under horizon-15 path-integral scoring.

    P0->P1 distance ~22 (in-horizon at speed ~3.5); P0->P2 distance ~70
    is OUT of horizon-15 so P2 capture has no path-integral value at all.
    Both rays clear the sun."""
    obs = _obs([
        _planet(0, 0, 50, 20, ships=80, production=1),
        _planet(1, -1, 30, 30, ships=10, production=3),
        _planet(2, -1, 15, 80, ships=10, production=1),
    ])
    assert 1 in _targets_of_emits(obs, _emit(obs))


# ---- A3 — cleanup in dominance (bug #13 analog) --------------------------

@pytest.mark.xfail(strict=True,
                   reason="Cleanup-in-dominance: lite_greedy opp underplays "
                          "when opp is near-eliminated; Phase B dominance "
                          "detector + BundleSearch escalation target.")
def test_A3_cleanup_in_dominance_23p_vs_1p(bundle_knobs):
    """We own 23 planets ringed around the board, opp has 1. Agent should
    attack the opp planet to close out the game."""
    ring = []
    for i in range(23):
        theta = i * math.pi / 12.0
        r = 25.0 if (i % 2) == 0 else 35.0
        ring.append(_planet(i, 0, 50 + r * math.cos(theta),
                            50 + r * math.sin(theta),
                            ships=20, production=1))
    ring.append(_planet(23, 1, 85, 85, ships=15, production=2))
    ring.append(_planet(24, -1, 15, 85, ships=3, production=1))
    obs = _obs(ring)
    assert 23 in _targets_of_emits(obs, _emit(obs))


# ---- A4 — drain-frontier penalty -----------------------------------------

@pytest.mark.xfail(strict=False,
                   reason="Drain-frontier: event-driven opp may or may not "
                          "penalize source vulnerability under path-integral "
                          "scoring; outcome is genuinely uncertain.")
def test_A4_drain_frontier_penalty(bundle_knobs):
    """P0 is our only buffer against a large opp planet P3. Draining P0 to
    grab close neutral P2 leaves P0 exposed. Agent must NOT source from P0."""
    obs = _obs([
        _planet(0, 0, 25, 30, ships=30, production=1),  # frontier source
        _planet(1, 0, 12, 30, ships=10, production=1),  # safe rear
        _planet(2, -1, 45, 30, ships=8, production=1),  # tempting neutral
        _planet(3, 1, 35, 10, ships=80, production=2),  # opp threat from south
    ])
    srcs = [int(a[0]) for a in _emit(obs)]
    assert 0 not in srcs


# ---- A5 — reinforcement-aware launch (bug #14 me-half) -------------------

@pytest.mark.xfail(strict=True,
                   reason="Bug #14 me-half: without me-followup, bundle "
                          "treats source as drained-forever and rejects "
                          "profitable launches. Phase B target.")
def test_A5_reinforcement_aware_launch(bundle_knobs):
    """P0 has prod=3 — production will refill what we launch. Agent should
    commit to capturing neutral P2 once me-followup correctly models the
    refill, instead of holding back because P0 'looks empty' after launch."""
    obs = _obs([
        _planet(0, 0, 20, 50, ships=30, production=3),
        _planet(1, 0, 80, 50, ships=10, production=1),
        _planet(2, -1, 50, 80, ships=40, production=1),
    ])
    assert 2 in _targets_of_emits(obs, _emit(obs))


# ---- A6 — gang-up emergence (2x60 vs 1x100) ------------------------------

def test_A6_gangup_emergence(bundle_knobs, monkeypatch):
    """Combat parity: 60 vs (70 + travel-prod) loses; 60+60 wins.
    Beam search at depth=2 must discover the joint launch.

    Sources placed close to target (P0/P1 at x=60, P2 at x=85) so
    fleet travel (~35 units at log-speed ~3.3) arrives within
    horizon-15. Rays clear the sun by ~24 units."""
    monkeypatch.setenv("BUNDLE_TOTAL_MS", "400")
    obs = _obs([
        _planet(0, 0, 60, 25, ships=60, production=1),
        _planet(1, 0, 60, 75, ships=60, production=1),
        _planet(2, 1, 85, 50, ships=70, production=1),
    ])
    emits = _emit(obs)
    srcs = {int(a[0]) for a in emits}
    assert {0, 1} <= srcs, f"both sources must fire; got srcs={srcs}"


# ---- A7 — cross-turn delayed launch --------------------------------------

def test_A7_cross_turn_delayed_launch(bundle_knobs, monkeypatch):
    """P0 has 15 ships, prod=8; neutral P1 has 22 ships, prod=4.
    Launching now (15 vs 22+travel) loses; waiting 3 turns (15+24=39 vs
    22+travel) wins and the captured high-prod planet pays off the
    cost. Agent must commit a launch_turn>0 spec in the chosen bundle.

    Layout off y=50 to clear the sun. Horizon bumped to 20 so the
    post-capture production window is wide enough to make t=3 launch
    strictly positive-EV (otherwise it ties with idle and the agent
    rationally picks idle).

    Note: probes module-level `_LAST_BUNDLE` because emits are turn-0
    only and the cross-turn machinery isn't observable from the emit
    list alone. If Phase 7e refactors carry-over storage, this is the
    canary."""
    monkeypatch.setenv("BUNDLE_OWN_LAUNCH_TURNS", "0,3")
    monkeypatch.setenv("BUNDLE_TOTAL_MS", "400")
    monkeypatch.setenv("BUNDLE_HORIZON", "20")
    obs = _obs([
        _planet(0, 0, 40, 30, ships=15, production=8),
        _planet(1, -1, 60, 30, ships=22, production=4),
    ])
    _emit(obs)  # populate _LAST_BUNDLE[0]
    from agents.bundle.main import _LAST_BUNDLE
    b = _LAST_BUNDLE.get(0)
    assert b is not None and not b.is_empty, "agent must commit a bundle"
    assert any(s.launch_turn > 0 for s in b.launches), \
        "chosen bundle must contain a future-turn launch"


# ---- A8 — idle in stalemate (no-bleed) -----------------------------------

def test_A8_idle_in_stalemate(bundle_knobs):
    """Symmetric standoff: 20-ship attacker can't crack 20-ship defender
    (plus travel-time production). Distance 50 + log-speed for 20 ships
    (~2.4) means fleet doesn't even reach within horizon-15. Empty
    bundle is the floor; agent must not bleed ships into a loss."""
    obs = _obs([
        _planet(0, 0, 25, 30, ships=20, production=1),
        _planet(1, 1, 75, 30, ships=20, production=1),
    ])
    assert _emit(obs) == []


# ---- A9 — sun-blocked sanity ---------------------------------------------

def test_A9_sun_blocked_sanity(bundle_knobs):
    """P0 -> P1 is a horizontal line through the sun at (50,50) r=10.
    P2 sits above the sun and is reachable. SunFilter must reject the
    through-sun aim; agent must target P2 instead."""
    obs = _obs([
        _planet(0, 0, 25, 50, ships=100, production=1),
        _planet(1, -1, 75, 50, ships=5, production=1),   # blocked
        _planet(2, -1, 50, 88, ships=5, production=1),   # reachable
    ])
    targets = _targets_of_emits(obs, _emit(obs))
    assert 1 not in targets, "must reject through-sun target"
    assert 2 in targets, "must capture the only reachable neutral"


# ---- A10 — production-prioritized expansion ------------------------------

@pytest.mark.xfail(strict=False,
                   reason="Path-integral should prefer prod=4 over prod=1 "
                          "at equal distance; depends on weighting / tie-"
                          "break behavior, hence uncertain.")
def test_A10_production_prioritized_expansion(bundle_knobs):
    """Two equidistant neutrals; one prod=4, the other prod=1. Under
    horizon-15 path integral the high-production capture should
    dominate."""
    obs = _obs([
        _planet(0, 0, 50, 15, ships=60, production=1),
        _planet(1, -1, 30, 30, ships=8, production=4),
        _planet(2, -1, 70, 30, ships=8, production=1),
    ])
    assert 1 in _targets_of_emits(obs, _emit(obs))
