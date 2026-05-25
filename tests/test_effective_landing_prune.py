"""Layer Z — effective-landing prune oracle tests.

Verifies that BASELINE_EFFECTIVE_LANDING_PRUNE=1 rejects launches whose
`ships - prod_target * eta` falls below the safety margin, and that the
default-off path is a strict no-op against the current baseline.
"""
import importlib
import os
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _reload_proposer_with_env(env: dict):
    """Reload `agents.baseline.proposer` with overridden env vars so the
    module-level CONSTS pick up the test values."""
    saved = {k: os.environ.get(k) for k in env}
    try:
        for k, v in env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        import agents.baseline.proposer as p
        importlib.reload(p)
        return p
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _mock_world_model(prod_target: float = 3.0, current_step: int = 50):
    """Minimal world + model stubs that `cheap_marginal_value` reads."""
    world = SimpleNamespace(step=current_step)

    class _Model:
        def owner_at(self, _planet_id, _arrival_step):
            return -1  # neutral -> capture branch (ships > pred_ships)

        def ships_at(self, _planet_id, _arrival_step):
            return 0.0  # empty target -> any ships > 0 captures

        def time_to_enemy_threat(self, *_a, **_k):
            return None

    return world, _Model()


def _tgt(production: float = 3.0):
    return SimpleNamespace(id=1, production=production, owner=-1)


def test_default_off_is_noop():
    """With env unset, the cheap_marginal_value formula is the pre-Layer-Z
    formula. A 2-ship long-haul capture vs prod=3 eta=25 still returns
    the positive capture value (no Layer-Z reject)."""
    p = _reload_proposer_with_env({
        "BASELINE_EFFECTIVE_LANDING_PRUNE": None,
        "BASELINE_EFFECTIVE_LANDING_MARGIN": None,
    })
    assert p.EFFECTIVE_LANDING_PRUNE_ENABLED is False
    world, model = _mock_world_model(prod_target=3.0)
    val = p.cheap_marginal_value(
        src=SimpleNamespace(),
        tgt=_tgt(production=3.0),
        ships=2,
        eta=25,
        world=world,
        model=model,
        me=0,
        wait_N=0,
    )
    # Positive: 0.05 * 3 * pv > 0; no reject.
    assert val > 0.0, f"expected positive capture value, got {val}"


def test_prune_rejects_long_haul_small_fleet():
    """With prune enabled, 2-ship vs prod=3 eta=25:
    effective_landing = 2 - 3*25 = -73 → below SAFETY_MARGIN=1.0 → reject."""
    p = _reload_proposer_with_env({
        "BASELINE_EFFECTIVE_LANDING_PRUNE": "1",
        "BASELINE_EFFECTIVE_LANDING_MARGIN": "1.0",
    })
    assert p.EFFECTIVE_LANDING_PRUNE_ENABLED is True
    world, model = _mock_world_model(prod_target=3.0)
    val = p.cheap_marginal_value(
        src=SimpleNamespace(),
        tgt=_tgt(production=3.0),
        ships=2,
        eta=25,
        world=world,
        model=model,
        me=0,
        wait_N=0,
    )
    # CHEAP_REJECT_THRESHOLD = -10.0 → returns -11.0 (rejects via gate at line 943).
    assert val <= p.CHEAP_REJECT_THRESHOLD, f"expected reject, got {val}"


def test_prune_accepts_short_haul_small_fleet_vs_neutral():
    """With prune enabled, 2-ship vs prod=0 eta=4:
    effective_landing = 2 - 0*4 = 2 → above margin → accept."""
    p = _reload_proposer_with_env({
        "BASELINE_EFFECTIVE_LANDING_PRUNE": "1",
        "BASELINE_EFFECTIVE_LANDING_MARGIN": "1.0",
    })
    world, model = _mock_world_model(prod_target=0.0)
    val = p.cheap_marginal_value(
        src=SimpleNamespace(),
        tgt=_tgt(production=0.0),
        ships=2,
        eta=4,
        world=world,
        model=model,
        me=0,
        wait_N=0,
    )
    assert val > p.CHEAP_REJECT_THRESHOLD, f"expected accept, got {val}"


def test_prune_accepts_large_fleet_long_haul():
    """With prune enabled, 100-ship vs prod=3 eta=25:
    effective_landing = 100 - 75 = 25 → well above margin → accept."""
    p = _reload_proposer_with_env({
        "BASELINE_EFFECTIVE_LANDING_PRUNE": "1",
        "BASELINE_EFFECTIVE_LANDING_MARGIN": "1.0",
    })
    world, model = _mock_world_model(prod_target=3.0)
    val = p.cheap_marginal_value(
        src=SimpleNamespace(),
        tgt=_tgt(production=3.0),
        ships=100,
        eta=25,
        world=world,
        model=model,
        me=0,
        wait_N=0,
    )
    assert val > p.CHEAP_REJECT_THRESHOLD, f"expected accept, got {val}"


def test_margin_tunable():
    """SAFETY_MARGIN=10 makes the prune stricter: a 5-ship vs prod=1 eta=2
    has effective_landing = 5 - 2 = 3 → below 10 → reject. The same launch
    at margin=1 would accept."""
    p_strict = _reload_proposer_with_env({
        "BASELINE_EFFECTIVE_LANDING_PRUNE": "1",
        "BASELINE_EFFECTIVE_LANDING_MARGIN": "10.0",
    })
    world, model = _mock_world_model(prod_target=1.0)
    val = p_strict.cheap_marginal_value(
        src=SimpleNamespace(),
        tgt=_tgt(production=1.0),
        ships=5,
        eta=2,
        world=world,
        model=model,
        me=0,
        wait_N=0,
    )
    assert val <= p_strict.CHEAP_REJECT_THRESHOLD, f"strict margin should reject, got {val}"

    p_lax = _reload_proposer_with_env({
        "BASELINE_EFFECTIVE_LANDING_PRUNE": "1",
        "BASELINE_EFFECTIVE_LANDING_MARGIN": "1.0",
    })
    val = p_lax.cheap_marginal_value(
        src=SimpleNamespace(),
        tgt=_tgt(production=1.0),
        ships=5,
        eta=2,
        world=world,
        model=model,
        me=0,
        wait_N=0,
    )
    assert val > p_lax.CHEAP_REJECT_THRESHOLD, f"lax margin should accept, got {val}"


def test_prune_does_not_affect_bounce_branch():
    """Bounce candidates (ships < pred_ships) already get a strong negative
    via -0.5 * ships. The Z gate is upstream of the bounce path, so it
    doesn't change bounce behavior."""
    p = _reload_proposer_with_env({
        "BASELINE_EFFECTIVE_LANDING_PRUNE": "1",
        "BASELINE_EFFECTIVE_LANDING_MARGIN": "1.0",
    })
    world = SimpleNamespace(step=50)

    class _Model:
        def owner_at(self, _planet_id, _arrival_step):
            return -1
        def ships_at(self, _planet_id, _arrival_step):
            return 100.0  # heavy defenders -> ships < pred_ships -> bounce
        def time_to_enemy_threat(self, *_a, **_k):
            return None

    val = p.cheap_marginal_value(
        src=SimpleNamespace(),
        tgt=_tgt(production=3.0),
        ships=10,
        eta=5,
        world=world,
        model=_Model(),
        me=0,
        wait_N=0,
    )
    assert val == -0.5 * 10.0, f"expected bounce penalty -5.0, got {val}"


# ---------------------------------------------------------------------------
# Layer Z v2 tests (2026-05-25) — pred_ships subtraction.
# ---------------------------------------------------------------------------


def _mock_world_model_defended(pred_ships: float, current_step: int = 50):
    """World+model where the target has a non-zero predicted garrison
    (defended cohort). pred_owner is neutral so the capture branch runs."""
    world = SimpleNamespace(step=current_step)

    class _Model:
        def owner_at(self, _planet_id, _arrival_step):
            return -1  # neutral -> capture branch (ships > pred_ships)

        def ships_at(self, _planet_id, _arrival_step):
            return float(pred_ships)

        def time_to_enemy_threat(self, *_a, **_k):
            return None

    return world, _Model()


def test_v2_subtracts_pred_ships():
    """v2 — defended target (pred_ships=15), ships=20, prod=2, eta=5.
    v1 would have computed `20 - 2*5 = 10` (accept). v2 computes
    `(20 - 15) - 2*5 = -5` (reject). This is the core v2 regression test:
    v1 was giving defended targets a hidden ship-count credit."""
    p = _reload_proposer_with_env({
        "BASELINE_EFFECTIVE_LANDING_PRUNE": "1",
        "BASELINE_EFFECTIVE_LANDING_MARGIN": "1.0",
    })
    world, model = _mock_world_model_defended(pred_ships=15.0)
    val = p.cheap_marginal_value(
        src=SimpleNamespace(),
        tgt=_tgt(production=2.0),
        ships=20,
        eta=5,
        world=world,
        model=model,
        me=0,
        wait_N=0,
    )
    assert val <= p.CHEAP_REJECT_THRESHOLD, (
        f"v2 must reject defended-target headroom-shortfall, got {val}"
    )


def test_v2_neutral_target_matches_v1_regression():
    """When pred_ships=0 (neutral), v2's `(ships - pred_ships) - prod·eta`
    collapses to `ships - prod·eta` — bit-identical to v1. This is the
    explicit guard ensuring v2 doesn't regress on the cohort that v1
    already handled correctly."""
    p = _reload_proposer_with_env({
        "BASELINE_EFFECTIVE_LANDING_PRUNE": "1",
        "BASELINE_EFFECTIVE_LANDING_MARGIN": "1.0",
    })
    world, model = _mock_world_model(prod_target=3.0)  # ships_at=0
    # Same case as test_prune_rejects_long_haul_small_fleet: ships=2, prod=3,
    # eta=25. v1 effective_landing = 2 - 75 = -73. v2 same. Reject.
    val_reject = p.cheap_marginal_value(
        src=SimpleNamespace(),
        tgt=_tgt(production=3.0),
        ships=2,
        eta=25,
        world=world,
        model=model,
        me=0,
        wait_N=0,
    )
    assert val_reject <= p.CHEAP_REJECT_THRESHOLD
    # Same case as test_prune_accepts_large_fleet_long_haul: ships=100,
    # prod=3, eta=25. v1 effective_landing = 100 - 75 = 25. v2 same. Accept.
    val_accept = p.cheap_marginal_value(
        src=SimpleNamespace(),
        tgt=_tgt(production=3.0),
        ships=100,
        eta=25,
        world=world,
        model=model,
        me=0,
        wait_N=0,
    )
    assert val_accept > p.CHEAP_REJECT_THRESHOLD


def test_v2_default_on_via_buildup_planner():
    """Importing `agents.buildup_planner.main` sets the env default so
    the v2 prune fires for that agent without an A/B harness explicitly
    setting BASELINE_EFFECTIVE_LANDING_PRUNE. Setdefault means an
    explicit `=0` from a test environment still overrides."""
    # Clear any prior value (incl. setdefault from previous test imports).
    saved = os.environ.pop("BASELINE_EFFECTIVE_LANDING_PRUNE", None)
    try:
        # Force a re-import of the agent's main module.
        for mod in ("agents.buildup_planner.main", "agents.buildup_planner"):
            sys.modules.pop(mod, None)
        import agents.buildup_planner.main  # noqa: F401
        assert os.environ.get("BASELINE_EFFECTIVE_LANDING_PRUNE") == "1", (
            "buildup_planner main.py must setdefault Layer Z v2 ON"
        )
    finally:
        if saved is None:
            os.environ.pop("BASELINE_EFFECTIVE_LANDING_PRUNE", None)
        else:
            os.environ["BASELINE_EFFECTIVE_LANDING_PRUNE"] = saved
