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
    """P0 has prod=4 — production will refill what we launch. Agent should
    commit to a first-wave attack on P2 (which alone loses, 14 ships vs
    15 neutral) because me-followup correctly models the second wave:
    by the time the lost first wave reaches P2 at ~t=5, P0 has
    refilled (1+5*4=21 ships) and lite_greedy at that event fires a
    second wave that captures the weakened P2. Without me-followup
    score is blind to the second wave, sees only the -14 ship loss, and
    rationally picks empty (idle).

    Layout tuned for the mechanic: P0 -> P2 distance 10 so both waves
    arrive within horizon=15. Original A5 used distance ~42, which put
    even the first wave's arrival outside horizon and made the
    mechanic structurally invisible regardless of me-followup."""
    obs = _obs([
        _planet(0, 0, 20, 30, ships=15, production=4),
        _planet(1, 0, 80, 30, ships=10, production=1),
        _planet(2, -1, 30, 30, ships=15, production=1),
    ])
    assert 2 in _targets_of_emits(obs, _emit(obs))


# ---- A6 — gang-up emergence (2x60 vs 1x100) ------------------------------

def test_A6_gangup_emergence(bundle_knobs, monkeypatch):
    """Combat parity: 60 vs (70 + travel-prod) loses; 60+60 wins.
    Beam search at depth=2 must discover the joint launch.

    Sources placed close to target (P0/P1 at x=60, P2 at x=85) so
    fleet travel (~35 units at log-speed ~3.3) arrives within
    horizon-15. Rays clear the sun by ~24 units.

    Budget bumped to 800ms (preemptive for Phase B me-followup,
    which adds ~3x per-score-call cost when BUNDLE_ME_FOLLOWUP=lite)."""
    monkeypatch.setenv("BUNDLE_TOTAL_MS", "800")
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


# ---- Phase E Phase 1 — joint coordination oracles ------------------------

@pytest.fixture
def joint_bonus_on(monkeypatch, bundle_knobs):
    """Enable joint coordination bonus + frontier seeding. Inherits the
    Phase A defaults from `bundle_knobs`; only flips the joint knobs."""
    monkeypatch.setenv("BUNDLE_JOINT_BONUS", "0.5")
    monkeypatch.setenv("BUNDLE_JOINT_SEEDS", "10")


def test_J1_joint_preferred_over_far_solo(joint_bonus_on):
    """Two close 30-ship sources P0 + P1 face a 50-defender enemy target P2.
    Neither solo captures (30 < 50+1). Joint P0+P1=60 > 50 → captures.
    A distant 60-ship source P3 (far enough to be lower-EV under horizon-15
    path integral) would solo-capture but arrives too late.

    With joint bonus on, the search must emit BOTH P0 and P1 launches
    at P2, not the far solo from P3.

    Layout: P0+P1 are at y=25 (lower-left/lower-right of target), target
    P2 at y=35 ship-near to both; P3 far at y=75 above the sun."""
    obs = _obs([
        _planet(0, 0, 30, 25, ships=30, production=1),
        _planet(1, 0, 70, 25, ships=30, production=1),
        _planet(2, 1, 50, 35, ships=50, production=2),    # enemy target
        _planet(3, 0, 50, 75, ships=60, production=1),    # far solo source
    ])
    moves = _emit(obs)
    # Sources that launched (any angle):
    src_ids = {int(m[0]) for m in moves}
    # Targets actually hit by emitted moves:
    tgt_ids = set(_targets_of_emits(obs, moves))
    # JOINT property: both close sources fire at P2.
    assert 0 in src_ids, f"P0 must launch as joint partner; emitted {moves}"
    assert 1 in src_ids, f"P1 must launch as joint partner; emitted {moves}"
    assert 2 in tgt_ids, f"joint must aim at P2; emitted {moves} → {tgt_ids}"


# ---- Phase E Phase 2 — bounce-penalty oracles ----------------------------

@pytest.fixture
def bounce_penalty_on(monkeypatch, bundle_knobs):
    """Enable bounce penalty. Inherits the Phase A defaults from
    `bundle_knobs`; only flips the bounce weight knob."""
    monkeypatch.setenv("BUNDLE_BOUNCE_WEIGHT", "0.5")


def test_B1_solo_bounce_not_emitted(bounce_penalty_on):
    """Single-source bounce-only state: our P0 (20 ships) is the ONLY
    launch source; enemy P1 has 50 defenders. The only target P0 can
    productively reach is P1, but 20 <= 50 means the launch bounces.

    With bounce_weight=0.5, the chooser must NOT emit the bouncing
    launch — empty bundle must win.

    Layout: P0 our 20-ship source at (30, 25); P1 enemy 50-defender
    target at (50, 35). No buffer / second source / neutral.
    Avoids any 'sacrifice ships to deny opp' incentive — opp's only
    planet is P1, not adjacent to anything of ours.
    """
    obs = _obs([
        _planet(0, 0, 30, 25, ships=20, production=1),    # our source
        _planet(1, 1, 50, 35, ships=50, production=1),    # over-defended enemy
    ])
    moves = _emit(obs)
    # Bounce penalty must reject the 20-ship launch at P1.
    # Empty bundle is acceptable; any other safe target is acceptable
    # too (there are none in this layout, so empty is the only option).
    hits = set(_targets_of_emits(obs, moves))
    assert 1 not in hits, (
        f"bounce penalty must reject solo 20sh launch at 50-defender P1; "
        f"emitted {moves} → hits {hits}"
    )


def test_B2_joint_not_penalized(bounce_penalty_on, monkeypatch):
    """With bounce penalty AND joint bonus both on: two 30-ship sources
    at a 50-defender target form a successful joint (30+30 > 50). Each
    leg ALONE bounces (30 <= 50), so naive per-launch bounce penalty
    would penalize the joint to oblivion. The joint-aware exemption
    must prevent that — bundle must still emit the joint.

    This tests the Phase 2 ↔ Phase 1 interaction (B2 from the plan)."""
    monkeypatch.setenv("BUNDLE_JOINT_BONUS", "0.5")
    monkeypatch.setenv("BUNDLE_JOINT_SEEDS", "10")
    obs = _obs([
        _planet(0, 0, 30, 25, ships=30, production=1),    # source A
        _planet(1, 0, 70, 25, ships=30, production=1),    # source B
        _planet(2, 1, 50, 35, ships=50, production=2),    # joint target
    ])
    moves = _emit(obs)
    src_ids = {int(m[0]) for m in moves}
    tgt_ids = set(_targets_of_emits(obs, moves))
    # Both P0 and P1 must fire at P2 (the joint property).
    # With joint exemption working, bounce penalty doesn't kill it.
    assert 0 in src_ids and 1 in src_ids, (
        f"joint must survive bounce penalty when joint_bonus on; "
        f"emitted {moves} from sources {src_ids}"
    )
    assert 2 in tgt_ids, (
        f"joint must aim at P2; emitted {moves} → {tgt_ids}"
    )


def test_J2_no_joint_when_pair_insufficient(joint_bonus_on):
    """Two sources P0 (20 ships) + P1 (30 ships) face a 60-defender target P2.
    Joint sum = 50 < 60 → CANNOT capture even together.

    Bundle must NOT emit the joint (it would just bounce both fleets).
    Either fires solo at a different target (P3 = 5-ship neutral), or
    holds. The KEY assertion: P2 is NOT both-hit by a P0+P1 joint.
    """
    obs = _obs([
        _planet(0, 0, 30, 25, ships=20, production=1),
        _planet(1, 0, 70, 25, ships=30, production=1),
        _planet(2, 1, 50, 35, ships=60, production=2),   # enemy too strong
        _planet(3, -1, 50, 18, ships=5, production=1),    # cheap neutral
    ])
    moves = _emit(obs)
    # Which launches HIT P2?
    p2_hitters = set()
    pmap = {p[0]: p for p in obs["planets"]}
    for m in moves:
        src_id = int(m[0])
        # Reuse the helper: returns first-hit planet for THIS move.
        hits = _targets_of_emits(obs, [m])
        if hits and hits[0] == 2:
            p2_hitters.add(src_id)
    # Forbid the joint at P2: at most ONE source allowed to fire at it
    # (a solo bounce is bad but not the joint property we're testing).
    assert len(p2_hitters) < 2, (
        f"Joint at P2 must not emit when sum 20+30=50 < defenders 60; "
        f"emitted {moves}; P2 hitters={p2_hitters}"
    )
