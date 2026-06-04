"""Unit tests for the opp-projection registry + translator in ``producer_plus``.

Covers Step 3 of ``state/MIGRATION_PLAN.md``: the bit-identical default,
the registry lookup, and the tuple-to-bucket-delta translation.
"""
from __future__ import annotations

import importlib.util
import os
import sys

import pytest
import torch


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRODUCER_PLUS_DIR = os.path.join(REPO_ROOT, "agents", "producer_plus")
PRODUCER_DIR = os.path.join(REPO_ROOT, "agents", "producer")


@pytest.fixture(scope="module")
def opp_projector():
    for p in (PRODUCER_DIR, PRODUCER_PLUS_DIR, REPO_ROOT):
        if p not in sys.path:
            sys.path.insert(0, p)
    spec = importlib.util.spec_from_file_location(
        "producer_plus_opp_projector_test",
        os.path.join(PRODUCER_PLUS_DIR, "opp_projector.py"),
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["producer_plus_opp_projector_test"] = module
    spec.loader.exec_module(module)
    return module


def test_none_projector_returns_empty(opp_projector):
    obs = {"player": 0, "planets": [], "fleets": [], "step": 0}
    assert opp_projector._none_projector(obs, my_id=0, num_seats=2, horizon=8) == []


def test_get_projector_default_is_none(opp_projector):
    assert opp_projector.get_projector(None) is opp_projector._none_projector
    assert opp_projector.get_projector("none") is opp_projector._none_projector


def test_get_projector_unknown_falls_back_to_none(opp_projector):
    """Misconfigured env var must never raise — degrade to no projection."""
    assert opp_projector.get_projector("garbage_name") is opp_projector._none_projector
    assert opp_projector.get_projector("") is opp_projector._none_projector


def test_get_projector_lite_greedy_registered(opp_projector):
    assert opp_projector.get_projector("lite_greedy") is opp_projector._lite_greedy_projector
    assert opp_projector.get_projector("LITE_GREEDY") is opp_projector._lite_greedy_projector
    assert opp_projector.get_projector("  lite_greedy  ") is opp_projector._lite_greedy_projector


def test_lite_greedy_projector_swallows_errors(opp_projector):
    """Malformed obs must return [] rather than crash the agent."""
    assert opp_projector._lite_greedy_projector(
        None, my_id=0, num_seats=2, horizon=8,
    ) == []
    assert opp_projector._lite_greedy_projector(
        {"player": 0}, my_id=0, num_seats=2, horizon=8,
    ) == []


def _planet_ids(*ids: int) -> torch.Tensor:
    return torch.tensor(list(ids), dtype=torch.long)


def test_translator_empty_inputs(opp_projector):
    delta = opp_projector.arrivals_tuples_to_buckets_delta(
        [], _planet_ids(5, 7, 9), A=2, H=8,
        device=torch.device("cpu"), dtype=torch.float32,
    )
    assert delta.shape == (3, 8, 2)
    assert torch.equal(delta, torch.zeros(3, 8, 2))


def test_translator_basic_mapping(opp_projector):
    # tgt_id=7 lives at slot index 1; eta=3 → bucket index 2.
    delta = opp_projector.arrivals_tuples_to_buckets_delta(
        [(7, 3, 1, 12.0)], _planet_ids(5, 7, 9), A=2, H=8,
        device=torch.device("cpu"), dtype=torch.float32,
    )
    assert delta[1, 2, 1].item() == pytest.approx(12.0)
    # Everything else must remain zero.
    delta[1, 2, 1] = 0.0
    assert torch.equal(delta, torch.zeros(3, 8, 2))


def test_translator_skips_out_of_window(opp_projector):
    """eta > H is outside the scoring forecast — must drop, not overflow."""
    delta = opp_projector.arrivals_tuples_to_buckets_delta(
        [(7, 99, 1, 12.0), (7, 0, 1, 12.0), (7, -1, 1, 12.0)],
        _planet_ids(5, 7, 9), A=2, H=8,
        device=torch.device("cpu"), dtype=torch.float32,
    )
    assert torch.equal(delta, torch.zeros(3, 8, 2))


def test_translator_skips_unknown_planet(opp_projector):
    delta = opp_projector.arrivals_tuples_to_buckets_delta(
        [(42, 3, 1, 12.0)], _planet_ids(5, 7, 9), A=2, H=8,
        device=torch.device("cpu"), dtype=torch.float32,
    )
    assert torch.equal(delta, torch.zeros(3, 8, 2))


def test_translator_skips_bad_owner(opp_projector):
    delta = opp_projector.arrivals_tuples_to_buckets_delta(
        [(7, 3, -1, 12.0), (7, 3, 5, 12.0)],
        _planet_ids(5, 7, 9), A=2, H=8,
        device=torch.device("cpu"), dtype=torch.float32,
    )
    assert torch.equal(delta, torch.zeros(3, 8, 2))


def test_translator_skips_nonpositive_ships(opp_projector):
    delta = opp_projector.arrivals_tuples_to_buckets_delta(
        [(7, 3, 1, 0.0), (7, 3, 1, -5.0)],
        _planet_ids(5, 7, 9), A=2, H=8,
        device=torch.device("cpu"), dtype=torch.float32,
    )
    assert torch.equal(delta, torch.zeros(3, 8, 2))


def test_translator_sums_duplicates(opp_projector):
    delta = opp_projector.arrivals_tuples_to_buckets_delta(
        [(7, 3, 1, 4.0), (7, 3, 1, 6.0), (7, 3, 1, 2.0)],
        _planet_ids(5, 7, 9), A=2, H=8,
        device=torch.device("cpu"), dtype=torch.float32,
    )
    assert delta[1, 2, 1].item() == pytest.approx(12.0)


def test_translator_h_zero(opp_projector):
    delta = opp_projector.arrivals_tuples_to_buckets_delta(
        [(7, 1, 1, 12.0)], _planet_ids(5, 7, 9), A=2, H=0,
        device=torch.device("cpu"), dtype=torch.float32,
    )
    assert delta.shape == (3, 0, 2)


def test_translator_handles_invalid_planet_id_in_obs(opp_projector):
    """planet_ids may contain ``-1`` padding slots — those must be ignored."""
    delta = opp_projector.arrivals_tuples_to_buckets_delta(
        [(7, 3, 1, 12.0)], _planet_ids(5, -1, 7, 9, -1), A=2, H=8,
        device=torch.device("cpu"), dtype=torch.float32,
    )
    # tgt_id=7 → slot index 2 (skipping the -1 padding row, but the dict still uses raw slot indices).
    assert delta[2, 2, 1].item() == pytest.approx(12.0)


# ---------------------------------------------------------------------------
# Integration: wrap-and-restore actually mutates fleet_buckets at scoring
# time AND restores them after run_turn returns.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def producer_plus_main():
    """Load ``producer_plus/main.py`` under a unique sys.modules name.

    Mirrors the loader pattern in ``test_producer_plus_adaptive_k.py``.
    """
    import importlib.util
    for p in (PRODUCER_DIR, PRODUCER_PLUS_DIR, REPO_ROOT):
        if p not in sys.path:
            sys.path.insert(0, p)
    spec = importlib.util.spec_from_file_location(
        "producer_plus_main_integration_test",
        os.path.join(PRODUCER_PLUS_DIR, "main.py"),
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["producer_plus_main_integration_test"] = module
    spec.loader.exec_module(module)
    return module


def test_wrap_and_restore_integration(monkeypatch, producer_plus_main):
    """Augmentation must be visible at scoring time AND undone after.

    Spies on ``movement.garrison_status`` to snapshot ``fleet_buckets`` at
    the moment scoring reads it. After ``run_turn`` returns, the
    persisted movement's bucket cell that we augmented must equal the
    pre-augmentation value, and the touched planet's
    ``garrison_dirty_from`` must be at-or-below the horizon so the next
    read rebuilds rather than serves stale-augmented cache.

    NOTE: must mutate ``sys.modules['opp_projector']`` — that is the
    module instance ``producer_plus_main`` actually imports from. The
    ``opp_projector`` test fixture loads a SEPARATE copy under a
    different sys.modules name; mutating it would have no effect here.
    """
    # producer_plus_main fixture has already loaded opp_projector under
    # the canonical name during its `from opp_projector import ...`.
    opp_module = sys.modules.get("opp_projector")
    assert opp_module is not None, (
        "producer_plus_main must have imported opp_projector by name"
    )

    from kaggle_environments import make

    env = make("orbit_wars", configuration={"seed": 7}, debug=False)
    env.reset()
    raw_obs = dict(env.state[0].observation)
    assert raw_obs.get("step") == 0
    assert raw_obs.get("player") == 0
    planets = raw_obs.get("planets") or []
    assert len(planets) >= 2

    # planets entries are flat lists: [planet_id, owner, x, y, ..., ships, prod].
    # Pick a planet OWNED BY THE OPPONENT (player 1) so the projection
    # looks plausible — though wrap-and-restore is the only invariant
    # under test here. Fall back to planet 0 if no opp planet exists yet.
    opp_planet_id = None
    for pl in planets:
        if int(pl[1]) == 1:
            opp_planet_id = int(pl[0])
            break
    if opp_planet_id is None:
        opp_planet_id = int(planets[0][0])

    arrivals = [(opp_planet_id, 3, 1, 12.0)]

    def probe_projector(obs, *, my_id, num_seats, horizon):
        return arrivals

    opp_module.PROJECTORS["__test_probe__"] = probe_projector
    monkeypatch.setenv("PRODUCER_PLUS_OPP_PROJECTOR", "__test_probe__")
    monkeypatch.delenv("PRODUCER_PLUS_ADAPTIVE_K", raising=False)

    # Use a fresh runtime so the test does not depend on global state.
    runtime = producer_plus_main.ProducerLiteRuntime()

    # Drive ONE turn manually so we can observe the movement object the
    # runtime created. ``agent`` would create its own _RUNTIME-scoped one.
    from orbit_lite.adapter import single_obs_to_tensor  # noqa: F401
    import torch as _torch

    runtime.memory.raw_obs = raw_obs
    captured = {}

    def make_spy(real_method, movement_ref):
        def spy(*args, **kwargs):
            captured.setdefault("at_score", []).append(
                movement_ref["m"].fleet_buckets.detach().clone()
            )
            return real_method(*args, **kwargs)
        return spy

    # First call to build movement state; we patch after the runtime
    # has created it, since the spy needs the movement reference.
    obs_tensors = producer_plus_main.single_obs_to_tensor(raw_obs, player_id=0)
    # Set player count before run_turn to avoid the runtime probe path.
    runtime.memory.cached_player_count = (
        producer_plus_main.largest_initial_player_count(obs_tensors)
    )

    # Pre-build movement via a vanilla run_turn pass (env-OFF) so we can
    # snapshot original buckets cleanly. Use the no-op projector for this
    # baseline pass.
    monkeypatch.setenv("PRODUCER_PLUS_OPP_PROJECTOR", "none")
    with _torch.no_grad():
        runtime.tensor_action(obs_tensors)
    movement = runtime.memory.movement
    assert movement is not None
    pre_buckets = movement.fleet_buckets.detach().clone()

    # Now re-run with the probe projector active and capture at scoring.
    monkeypatch.setenv("PRODUCER_PLUS_OPP_PROJECTOR", "__test_probe__")
    movement_ref = {"m": movement}
    real_garrison_status = movement.garrison_status
    movement.garrison_status = make_spy(real_garrison_status, movement_ref)

    runtime.memory.raw_obs = raw_obs
    with _torch.no_grad():
        runtime.tensor_action(obs_tensors)

    # Spy must have fired at least once.
    assert "at_score" in captured and len(captured["at_score"]) >= 1

    # Find the slot for opp_planet_id in movement.planet_ids.
    ids = movement.planet_ids.detach().to("cpu").tolist()
    slot = ids.index(opp_planet_id)

    # At scoring time, the projected arrival must be present in the
    # augmented buckets (eta=3 → bucket index 2, owner=1, ships=12).
    at_score = captured["at_score"][0]
    augmented_cell = at_score[slot, 2, 1].item()
    pre_cell = pre_buckets[slot, 2, 1].item()
    assert augmented_cell == pytest.approx(pre_cell + 12.0), (
        f"expected augmented bucket = original + 12.0, "
        f"got augmented={augmented_cell} original={pre_cell}"
    )

    # After run_turn returns, fleet_buckets must be restored to original.
    # apply_private_planned_launches MAY have mutated launch sources or
    # targets, so we check the SPECIFIC augmented cell only: it must
    # equal the pre-augmentation value (the 12.0 was undone).
    post_cell = movement.fleet_buckets[slot, 2, 1].item()
    assert post_cell == pytest.approx(pre_cell), (
        f"restore failed: post-run cell={post_cell} expected={pre_cell}"
    )

    # touched_slots' garrison_dirty_from must be at-or-below the horizon
    # so the next garrison_status read rebuilds from restored buckets
    # rather than serving the stale-augmented cache.
    H = int(movement.movement_horizon)
    assert movement.garrison_dirty_from is not None
    dirty_at_slot = int(movement.garrison_dirty_from[slot].item())
    assert dirty_at_slot <= H, (
        f"post-restore invalidation missing: dirty_from[{slot}]={dirty_at_slot} > H={H} "
        f"→ next garrison_status would return stale-augmented data"
    )

    # Cleanup: remove the probe from the registry so other tests in the
    # session aren't affected.
    opp_module.PROJECTORS.pop("__test_probe__", None)
