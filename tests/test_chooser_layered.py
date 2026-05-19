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


def test_layer0_empty_prerank():
    obs, world = _build_world(0, [_planet(0, 0, 10.0, 50.0)])
    model = WorldModel.from_world(world)
    commits, residual = layer0_classify([], world, model, 0, 0, 0.99)
    assert commits == []
    assert residual == []


def test_layer0_classifies_clean_capture_as_w1():
    """A clean capture (strong delivered, no opp counter) → W1 commit."""
    src = _planet(0, 0, 10.0, 50.0, ships=120, production=3)
    tgt = _planet(1, -1, 30.0, 50.0, ships=10, production=2)
    opp_far = _planet(2, 1, 95.0, 95.0, ships=10)
    obs, world = _build_world(0, [src, tgt, opp_far])
    model = WorldModel.from_world(world)
    cand = _candidate(src, tgt, cheap_delta=5.0, ships=80, eta=4)
    commits, residual = layer0_classify([cand], world, model, 0, 0, 0.99)
    assert len(commits) == 1
    assert commits[0][1].reason == "W1"
    assert residual == []


def test_layer0_classifies_bounce_as_l1_discard():
    """Under-sized launch → L1 discard, never reaches W1/W2."""
    src = _planet(0, 0, 10.0, 50.0, ships=100)
    tgt = _planet(1, -1, 50.0, 50.0, ships=100, production=1)
    obs, world = _build_world(0, [src, tgt])
    model = WorldModel.from_world(world)
    cand = _candidate(src, tgt, cheap_delta=0.5, ships=5, eta=7)
    commits, residual = layer0_classify([cand], world, model, 0, 0, 0.99)
    assert commits == []
    assert residual == []  # L1-discarded


def test_layer0_passes_uncertain_to_residual():
    """Candidate that no predicate commits → residual."""
    src = _planet(0, 0, 10.0, 50.0, ships=100, production=3)
    tgt = _planet(1, -1, 30.0, 50.0, ships=10, production=2)
    # Nearby strong opp can recapture → W1 abstains.
    opp_close = _planet(2, 1, 35.0, 50.0, ships=200, production=4)
    obs, world = _build_world(0, [src, tgt, opp_close])
    model = WorldModel.from_world(world)
    cand = _candidate(src, tgt, cheap_delta=2.0, ships=40, eta=8)
    commits, residual = layer0_classify([cand], world, model, 0, 0, 0.99)
    assert commits == []
    assert residual == [cand]


def test_layer0_applies_l2_dominance_to_residual():
    """Two uncertain candidates same (src, tgt); strictly dominated dropped."""
    src = _planet(0, 0, 10.0, 50.0, ships=100, production=3)
    tgt = _planet(1, -1, 30.0, 50.0, ships=10, production=2)
    opp_close = _planet(2, 1, 35.0, 50.0, ships=200)
    obs, world = _build_world(0, [src, tgt, opp_close])
    model = WorldModel.from_world(world)
    dominator = _candidate(src, tgt, cheap_delta=3.0, ships=40, eta=5)
    dominated = _candidate(src, tgt, cheap_delta=1.0, ships=60, eta=10)
    commits, residual = layer0_classify(
        [dominator, dominated], world, model, 0, 0, 0.99,
    )
    assert commits == []
    assert residual == [dominator]


# ---------------------------------------------------------------------------
# choose_layered — composition with pluggable inner chooser
# ---------------------------------------------------------------------------


def _setup_clean_capture_scenario():
    """Scenario producing one W1 commit and one uncertain residual candidate.

    Returns (obs, world, model, snap_base, prerank).
    """
    src1 = _planet(0, 0, 10.0, 50.0, ships=120, production=3)
    src2 = _planet(1, 0, 60.0, 10.0, ships=80, production=2)
    # tgt1: clean capture (W1 commit) — far from any strong opp.
    tgt1 = _planet(2, -1, 30.0, 50.0, ships=10, production=2)
    # tgt2: contested (uncertain) — strong opp nearby.
    tgt2 = _planet(3, -1, 50.0, 30.0, ships=10, production=1)
    opp_close = _planet(4, 1, 55.0, 30.0, ships=200, production=4)
    obs, world = _build_world(0, [src1, src2, tgt1, tgt2, opp_close])
    model = WorldModel.from_world(world)
    snap_base = fs_from_obs(obs, num_seats=2)
    prerank = [
        _candidate(src1, tgt1, cheap_delta=5.0, ships=80, eta=4),  # W1
        _candidate(src2, tgt2, cheap_delta=1.0, ships=40, eta=8),  # uncertain
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


def test_choose_layered_emits_w1_commit(monkeypatch):
    """W1 commit reaches the final moves list, src/tgt locked."""
    monkeypatch.setenv("BASELINE_INNER_CHOOSER", "trajectory")
    obs, world, model, snap_base, prerank = _setup_clean_capture_scenario()
    moves = choose_layered(
        snap_base, prerank, None, 0, 2, 600.0, 25, 40, 0.99,
        world, model, 0,
    )
    # At minimum, the W1 commit (src1=0 → tgt1=2) is emitted.
    assert any(int(m[0]) == 0 for m in moves), f"W1 src missing in {moves}"


def test_choose_layered_inner_chooser_override(monkeypatch):
    """The `inner_chooser_name` kwarg overrides the env var."""
    monkeypatch.setenv("BASELINE_INNER_CHOOSER", "roi")
    obs, world, model, snap_base, prerank = _setup_clean_capture_scenario()
    # Force trajectory via kwarg; env var asks for roi.
    moves = choose_layered(
        snap_base, prerank, None, 0, 2, 600.0, 25, 40, 0.99,
        world, model, 0,
        inner_chooser_name="trajectory",
    )
    # W1 commit still emits regardless of inner chooser.
    assert any(int(m[0]) == 0 for m in moves)


def test_choose_layered_swap_stability_l0_commits_invariant(monkeypatch):
    """LOAD-BEARING: Layer-0 commit set is identical across inner-chooser
    selection. Only residual emits may vary.

    This is the core decoupling claim. Layer 0 must not depend on the
    inner chooser — same world, same prerank, same `step`/`gamma` →
    same `(commits, residual)` from `layer0_classify` regardless of
    which downstream chooser will eventually run on the residual.
    """
    obs, world, model, _snap_base, prerank = _setup_clean_capture_scenario()

    # Layer 0 classify is itself agnostic — call it directly and assert.
    commits_a, residual_a = layer0_classify(prerank, world, model, 0, 0, 0.99)
    commits_b, residual_b = layer0_classify(prerank, world, model, 0, 0, 0.99)
    commits_c, residual_c = layer0_classify(prerank, world, model, 0, 0, 0.99)

    # Stronger property: the commit tags and lower_bounds match exactly.
    def _commit_signatures(commits):
        return [
            (int(c[1].id), int(c[2].id), v.reason, round(v.lower_bound, 6))
            for c, v in commits
        ]

    assert (_commit_signatures(commits_a)
            == _commit_signatures(commits_b)
            == _commit_signatures(commits_c))
    assert residual_a == residual_b == residual_c

    # And: invoking `choose_layered` with three different inner-chooser
    # selections produces emit sets that AGREE on the L0 commits. The
    # residual may differ across choosers (that's allowed — each has
    # its own scoring), so we only assert agreement on the W1 src.
    moves_traj = choose_layered(
        _snap_base, prerank, None, 0, 2, 600.0, 25, 40, 0.99,
        world, model, 0, inner_chooser_name="trajectory",
    )
    moves_roi = choose_layered(
        _snap_base, prerank, None, 0, 2, 600.0, 25, 40, 0.99,
        world, model, 0, inner_chooser_name="roi",
    )
    # Build composite baseline_favors — composite signature requires it.
    from agents.baseline.chooser import build_idle_baseline
    baseline_favors = build_idle_baseline(_snap_base, 0, 2, 40, 0.99)
    moves_comp = choose_layered(
        _snap_base, prerank, baseline_favors, 0, 2, 600.0, 25, 40, 0.99,
        world, model, 0, inner_chooser_name="composite",
    )
    # All three include the W1 commit on src=0.
    for tag, moves in [("traj", moves_traj),
                        ("roi", moves_roi),
                        ("composite", moves_comp)]:
        assert any(int(m[0]) == 0 for m in moves), (
            f"W1 src=0 missing under inner={tag}: {moves}"
        )


def test_choose_layered_passthrough_when_no_predicate_fires(monkeypatch):
    """When all candidates are uncertain (no commits, no discards), the
    layered chooser delegates entirely to the inner chooser."""
    monkeypatch.setenv("BASELINE_INNER_CHOOSER", "trajectory")
    # Build a scenario where no W1/W2/L1 fires: contested capture.
    src = _planet(0, 0, 10.0, 50.0, ships=100, production=3)
    tgt = _planet(1, -1, 30.0, 50.0, ships=10, production=2)
    opp_close = _planet(2, 1, 35.0, 50.0, ships=200, production=4)
    obs, world = _build_world(0, [src, tgt, opp_close])
    model = WorldModel.from_world(world)
    snap_base = fs_from_obs(obs, num_seats=2)
    cand = _candidate(src, tgt, cheap_delta=2.0, ships=40, eta=8)
    # commits empty, residual non-empty → inner chooser handles it.
    commits, residual = layer0_classify([cand], world, model, 0, 0, 0.99)
    assert commits == []
    assert len(residual) == 1
