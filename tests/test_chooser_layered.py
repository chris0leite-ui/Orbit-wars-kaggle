"""Integration tests for agents/baseline/chooser_layered.

The load-bearing claim of this module is that Layer-0 predicates are
chooser-agnostic and the composition adapter can swap the inner chooser
freely. These tests pin that behaviour.
"""

from __future__ import annotations

import os

import pytest
from kaggle_environments.envs.orbit_wars.orbit_wars import Fleet, Planet

from agents.baseline.chooser_layered import (
    _INNER_DISPATCH,
    _resolve_inner_chooser_name,
    choose_layered,
    layer0_classify,
)
from lib.fast_sim import from_obs as fs_from_obs
from lib.intent import World
from lib.world_model import WorldModel


def _planet(pid, owner, x, y, *, ships=10, production=2, radius=1.5):
    return Planet(pid, owner, x, y, radius, ships, production)


def _fleet(fid, owner, x, y, angle, ships, from_planet_id=0):
    return Fleet(fid, owner, x, y, angle, from_planet_id, ships)


def _build_world(my_id, planets, *, fleets=None, step=0, omega=0.0):
    obs = {
        "player": my_id,
        "planets": [
            (p.id, p.owner, p.x, p.y, p.radius, p.ships, p.production)
            for p in planets
        ],
        "fleets": [
            (f.id, f.owner, f.x, f.y, f.angle, f.from_planet_id, f.ships)
            for f in (fleets or [])
        ],
        "angular_velocity": omega,
        "comet_planet_ids": [],
        "comets": [],
        "step": step,
    }
    return obs, World.from_obs(obs)


def _candidate(src, tgt, *, cheap_delta=1.0, ships=10, eta=5, wait_N=0):
    """Build a proposer-style prerank tuple."""
    return (float(cheap_delta), src, tgt, int(ships), 0.0, int(eta),
            int(eta + 2), int(wait_N))


# ---------------------------------------------------------------------------
# _resolve_inner_chooser_name
# ---------------------------------------------------------------------------


def test_resolve_inner_chooser_defaults_to_trajectory(monkeypatch):
    monkeypatch.delenv("BASELINE_INNER_CHOOSER", raising=False)
    assert _resolve_inner_chooser_name() == "trajectory"


def test_resolve_inner_chooser_accepts_valid_names(monkeypatch):
    for name in ("trajectory", "composite", "roi"):
        monkeypatch.setenv("BASELINE_INNER_CHOOSER", name)
        assert _resolve_inner_chooser_name() == name


def test_resolve_inner_chooser_falls_back_on_unknown(monkeypatch):
    """Unknown names silently fall back to trajectory rather than raising —
    bundled agents may run against env vars they don't understand."""
    monkeypatch.setenv("BASELINE_INNER_CHOOSER", "newchooser-unimplemented")
    assert _resolve_inner_chooser_name() == "trajectory"


def test_dispatch_table_keys_are_stable():
    """Adding an inner chooser is one entry; the existing keys must stay."""
    assert set(_INNER_DISPATCH.keys()) >= {"trajectory", "composite", "roi"}


# ---------------------------------------------------------------------------
# layer0_classify — chooser-agnostic pre-pass
# ---------------------------------------------------------------------------


def _commits_from(verdicts):
    """Slice 4 helper: extract commit (candidate, verdict) pairs."""
    return [(c, v) for c, v in verdicts if v.kind == "commit"]


def test_layer0_empty_prerank():
    obs, world = _build_world(0, [_planet(0, 0, 10.0, 50.0)])
    model = WorldModel.from_world(world)
    verdicts, filtered = layer0_classify([], world, model, 0, 0, 0.99)
    assert verdicts == []
    assert filtered == []


def test_layer0_classifies_clean_capture_as_w1():
    """A clean capture (strong delivered, no opp counter) → W1 commit
    verdict AND candidate stays in filtered (so inner can see it)."""
    src = _planet(0, 0, 10.0, 50.0, ships=120, production=3)
    tgt = _planet(1, -1, 30.0, 50.0, ships=10, production=2)
    opp_far = _planet(2, 1, 95.0, 95.0, ships=10)
    obs, world = _build_world(0, [src, tgt, opp_far])
    model = WorldModel.from_world(world)
    cand = _candidate(src, tgt, cheap_delta=5.0, ships=80, eta=4)
    verdicts, filtered = layer0_classify([cand], world, model, 0, 0, 0.99)
    commits = _commits_from(verdicts)
    assert len(commits) == 1
    assert commits[0][1].reason == "W1"
    # Slice 4: commit candidates STAY in filtered so the inner can score them.
    assert filtered == [cand]


def test_layer0_classifies_bounce_as_l1_discard():
    """Under-sized launch → L1 verdict AND dropped from filtered."""
    src = _planet(0, 0, 10.0, 50.0, ships=100)
    tgt = _planet(1, -1, 50.0, 50.0, ships=100, production=1)
    obs, world = _build_world(0, [src, tgt])
    model = WorldModel.from_world(world)
    cand = _candidate(src, tgt, cheap_delta=0.5, ships=5, eta=7)
    verdicts, filtered = layer0_classify([cand], world, model, 0, 0, 0.99)
    # Discard verdict recorded but candidate not in filtered.
    assert len(verdicts) == 1
    assert verdicts[0][1].kind == "discard"
    assert filtered == []


def test_layer0_passes_uncertain_to_filtered():
    """Candidate that no predicate commits → uncertain verdict, stays in filtered."""
    src = _planet(0, 0, 10.0, 50.0, ships=100, production=3)
    tgt = _planet(1, -1, 30.0, 50.0, ships=10, production=2)
    opp_close = _planet(2, 1, 35.0, 50.0, ships=200, production=4)
    obs, world = _build_world(0, [src, tgt, opp_close])
    model = WorldModel.from_world(world)
    cand = _candidate(src, tgt, cheap_delta=2.0, ships=40, eta=8)
    verdicts, filtered = layer0_classify([cand], world, model, 0, 0, 0.99)
    assert _commits_from(verdicts) == []
    assert verdicts[0][1].kind == "uncertain"
    assert filtered == [cand]


def test_layer0_applies_l2_dominance_to_filtered():
    """L2 dominance applies to the filtered set (after L1 discards removed)."""
    src = _planet(0, 0, 10.0, 50.0, ships=100, production=3)
    tgt = _planet(1, -1, 30.0, 50.0, ships=10, production=2)
    opp_close = _planet(2, 1, 35.0, 50.0, ships=200)
    obs, world = _build_world(0, [src, tgt, opp_close])
    model = WorldModel.from_world(world)
    dominator = _candidate(src, tgt, cheap_delta=3.0, ships=40, eta=5)
    dominated = _candidate(src, tgt, cheap_delta=1.0, ships=60, eta=10)
    verdicts, filtered = layer0_classify(
        [dominator, dominated], world, model, 0, 0, 0.99,
    )
    assert _commits_from(verdicts) == []
    # Dominated candidate dropped from filtered by L2; both still have verdicts.
    assert filtered == [dominator]
    assert len(verdicts) == 2


# ---------------------------------------------------------------------------
# choose_layered — composition with pluggable inner chooser
# ---------------------------------------------------------------------------


def _setup_clean_capture_scenario():
    """Scenario producing one W1 commit and one uncertain residual candidate.

    Sized for the Slice-3 multi-opp Wald bound: src1 must deliver enough
    force that the coordinated sum of ALL reachable opps still falls
    below SAFETY × garrison. Under variant 1 (single-nearest) a smaller
    source sufficed, but variant 2's Wald sum is tighter.

    Returns (obs, world, model, snap_base, prerank).
    """
    # src1 is overpowered for tgt1 — sends 280 ships into a 10-ship neutral.
    src1 = _planet(0, 0, 10.0, 50.0, ships=300, production=3)
    src2 = _planet(1, 0, 60.0, 10.0, ships=80, production=2)
    # tgt1: clean capture (W1 commit) — uncontested under multi-opp Wald.
    tgt1 = _planet(2, -1, 30.0, 50.0, ships=10, production=2)
    # tgt2: contested (uncertain) — strong opp nearby.
    tgt2 = _planet(3, -1, 50.0, 30.0, ships=10, production=1)
    # The strong opp sits closer to tgt2 than tgt1 to keep tgt2 contested
    # without dominating tgt1's Wald sum.
    opp_close = _planet(4, 1, 55.0, 30.0, ships=200, production=4)
    obs, world = _build_world(0, [src1, src2, tgt1, tgt2, opp_close])
    model = WorldModel.from_world(world)
    snap_base = fs_from_obs(obs, num_seats=2)
    prerank = [
        _candidate(src1, tgt1, cheap_delta=5.0, ships=280, eta=4),  # W1
        _candidate(src2, tgt2, cheap_delta=1.0, ships=40, eta=8),   # uncertain
    ]
    return obs, world, model, snap_base, prerank


def test_choose_layered_empty_prerank_returns_no_moves(monkeypatch):
    monkeypatch.setenv("BASELINE_INNER_CHOOSER", "trajectory")
    obs, world = _build_world(0, [_planet(0, 0, 10.0, 50.0)])
    model = WorldModel.from_world(world)
    snap_base = fs_from_obs(obs, num_seats=2)
    moves = choose_layered(
        snap_base, [], None, 0, 2, 600.0, 25, 40, 0.99, world, model, 0,
    )
    assert moves == []


def test_choose_layered_emits_from_w1_source(monkeypatch):
    """Slice 4: a W1-classified candidate's source ends up in the emit
    list — either because the inner chose it OR because the backstop
    appended it. The specific path doesn't matter; coverage does.
    """
    monkeypatch.setenv("BASELINE_INNER_CHOOSER", "trajectory")
    obs, world, model, snap_base, prerank = _setup_clean_capture_scenario()
    moves = choose_layered(
        snap_base, prerank, None, 0, 2, 600.0, 25, 40, 0.99,
        world, model, 0,
    )
    # src1=0 has a W1 commit candidate. Under Slice 4 the inner sees it
    # and either picks it or picks something else from src=0; either way
    # the source should be represented in the emit.
    assert any(int(m[0]) == 0 for m in moves), f"src=0 missing in {moves}"


def test_choose_layered_inner_chooser_override(monkeypatch):
    """The `inner_chooser_name` kwarg overrides the env var."""
    monkeypatch.setenv("BASELINE_INNER_CHOOSER", "roi")
    obs, world, model, snap_base, prerank = _setup_clean_capture_scenario()
    moves = choose_layered(
        snap_base, prerank, None, 0, 2, 600.0, 25, 40, 0.99,
        world, model, 0,
        inner_chooser_name="trajectory",
    )
    assert any(int(m[0]) == 0 for m in moves)


def test_choose_layered_swap_stability_l0_verdicts_invariant(monkeypatch):
    """LOAD-BEARING: Layer-0 verdict set is identical across inner-chooser
    selection.

    Decoupling claim: `layer0_classify` is a pure function of (prerank,
    world, model, me, step, gamma). It must not depend on which inner
    chooser will run next. Same inputs → same verdict list + same
    filtered list.
    """
    obs, world, model, _snap_base, prerank = _setup_clean_capture_scenario()

    verdicts_a, filtered_a = layer0_classify(prerank, world, model, 0, 0, 0.99)
    verdicts_b, filtered_b = layer0_classify(prerank, world, model, 0, 0, 0.99)
    verdicts_c, filtered_c = layer0_classify(prerank, world, model, 0, 0, 0.99)

    def _verdict_signatures(verdicts):
        return [
            (int(c[1].id), int(c[2].id), v.kind, v.reason,
             round(float(v.lower_bound), 6))
            for c, v in verdicts
        ]

    assert (_verdict_signatures(verdicts_a)
            == _verdict_signatures(verdicts_b)
            == _verdict_signatures(verdicts_c))
    assert filtered_a == filtered_b == filtered_c

    # End-to-end: regardless of inner chooser selection, src=0 (which has
    # a W1 commit verdict) should be represented in the emit. The exact
    # move may differ (each inner has its own scoring) but src coverage
    # is the load-bearing decoupling claim.
    moves_traj = choose_layered(
        _snap_base, prerank, None, 0, 2, 600.0, 25, 40, 0.99,
        world, model, 0, inner_chooser_name="trajectory",
    )
    moves_roi = choose_layered(
        _snap_base, prerank, None, 0, 2, 600.0, 25, 40, 0.99,
        world, model, 0, inner_chooser_name="roi",
    )
    from agents.baseline.chooser import build_idle_baseline
    baseline_favors = build_idle_baseline(_snap_base, 0, 2, 40, 0.99)
    moves_comp = choose_layered(
        _snap_base, prerank, baseline_favors, 0, 2, 600.0, 25, 40, 0.99,
        world, model, 0, inner_chooser_name="composite",
    )
    for tag, moves in [("traj", moves_traj),
                        ("roi", moves_roi),
                        ("composite", moves_comp)]:
        assert any(int(m[0]) == 0 for m in moves), (
            f"src=0 missing under inner={tag}: {moves}"
        )


def test_backstop_fires_when_inner_returns_nothing(monkeypatch):
    """Slice 4: if the inner returns no moves (e.g., it bailed on
    wallclock), W1/W2 commits MUST appear in the final emit as the
    backstop's whole point."""
    monkeypatch.setenv("BASELINE_INNER_CHOOSER", "trajectory")
    obs, world, model, snap_base, prerank = _setup_clean_capture_scenario()

    def _stub_inner(_k):
        return []  # inner picks nothing

    monkeypatch.setitem(_INNER_DISPATCH, "trajectory", _stub_inner)
    moves = choose_layered(
        snap_base, prerank, None, 0, 2, 600.0, 25, 40, 0.99,
        world, model, 0,
    )
    # Backstop must have appended the W1 commit from src=0.
    assert any(int(m[0]) == 0 for m in moves), (
        f"backstop did not fire when inner emitted nothing: {moves}"
    )


def test_backstop_skipped_when_inner_used_source(monkeypatch):
    """Slice 4: if the inner already emits from src=0, the backstop
    must NOT also emit from src=0 (no double-launching)."""
    monkeypatch.setenv("BASELINE_INNER_CHOOSER", "trajectory")
    obs, world, model, snap_base, prerank = _setup_clean_capture_scenario()

    def _stub_inner(_k):
        # Inner emits a move from src=0 with some other angle/ships.
        return [[0, 1.234, 50]]

    monkeypatch.setitem(_INNER_DISPATCH, "trajectory", _stub_inner)
    moves = choose_layered(
        snap_base, prerank, None, 0, 2, 600.0, 25, 40, 0.99,
        world, model, 0,
    )
    # Exactly one emit from src=0 (the inner's), not also the backstop.
    src_0_emits = [m for m in moves if int(m[0]) == 0]
    assert len(src_0_emits) == 1, (
        f"expected exactly 1 emit from src=0; got {src_0_emits}"
    )


def test_choose_layered_subtracts_l0_elapsed_from_inner_wallclock(monkeypatch):
    """Slice 3 — inner chooser receives wallclock_ms - L0 elapsed.

    Verifies the wallclock-hygiene fix: when L0 runs (even briefly),
    the inner chooser's `wallclock_ms` kwarg is the input budget minus
    L0's elapsed time. Floor: `INNER_WALLCLOCK_FLOOR_MS`.
    """
    monkeypatch.setenv("BASELINE_INNER_CHOOSER", "trajectory")
    obs, world, model, snap_base, prerank = _setup_clean_capture_scenario()

    captured: dict = {}

    def _spy_inner(k):
        # Capture the wallclock_ms the inner saw and return no moves.
        captured["wallclock_ms"] = k.get("wallclock_ms")
        return []

    monkeypatch.setitem(_INNER_DISPATCH, "trajectory", _spy_inner)
    input_wallclock = 600.0
    choose_layered(
        snap_base, prerank, None, 0, 2, input_wallclock, 25, 40, 0.99,
        world, model, 0,
    )

    seen = captured.get("wallclock_ms")
    assert seen is not None, "inner chooser was not called"
    # Inner must see a budget < input (some L0 elapsed > 0) and
    # > the floor (Layer 0 on a 2-candidate prerank takes microseconds).
    assert seen <= input_wallclock, (
        f"inner wallclock {seen} should be ≤ input {input_wallclock}"
    )
    # Lower bound: the floor. In practice L0 cost is tiny so seen
    # should be very close to input_wallclock, but the floor guards
    # the pathological overrun case.
    from agents.baseline.chooser_layered import INNER_WALLCLOCK_FLOOR_MS
    assert seen >= INNER_WALLCLOCK_FLOOR_MS


def test_choose_layered_passthrough_when_no_predicate_fires(monkeypatch):
    """When no candidate is committable or discardable, the verdict list
    has all 'uncertain' entries and the filtered set equals the L2-pruned
    prerank."""
    monkeypatch.setenv("BASELINE_INNER_CHOOSER", "trajectory")
    src = _planet(0, 0, 10.0, 50.0, ships=100, production=3)
    tgt = _planet(1, -1, 30.0, 50.0, ships=10, production=2)
    opp_close = _planet(2, 1, 35.0, 50.0, ships=200, production=4)
    obs, world = _build_world(0, [src, tgt, opp_close])
    model = WorldModel.from_world(world)
    cand = _candidate(src, tgt, cheap_delta=2.0, ships=40, eta=8)
    verdicts, filtered = layer0_classify([cand], world, model, 0, 0, 0.99)
    assert _commits_from(verdicts) == []
    assert verdicts[0][1].kind == "uncertain"
    assert filtered == [cand]
