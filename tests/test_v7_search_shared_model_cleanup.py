"""Exception-safety regression for `_shared_world_model` attach/detach.

`lib/v7_search.score_candidate` attaches a `_shared_world_model`
attribute to the per-seat observation Struct during the K-step rollout
so mirror policies can skip an expensive WorldModel rebuild
(`lib/opp_model.py:76, 112`). The old implementation cleaned up with a
bare `del` after the policy call — if the policy raised, the attribute
leaked onto the snapshot's observation. That leak is invisible inside
`score_candidate` (the clone is discarded on raise) but breaks the
bundler parity gate, which calls source and bundle back-to-back on the
same input obs and compares actions.

The fix wraps attach/detach in a context manager
(`_bind_shared_world_model`); this test pins the invariant.
"""

from __future__ import annotations

import pytest

from lib import v7_search


class _Obs:
    """Stand-in for `kaggle_environments.utils.Struct` — supports
    attribute set / get / del like the rollout's observation Struct."""


def _make_obs_pair():
    return _Obs(), _Obs()


def test_attaches_when_model_provided():
    a, b = _make_obs_pair()
    model = object()
    with v7_search._bind_shared_world_model((a, b), model):
        assert a._shared_world_model is model
        assert b._shared_world_model is model
    assert not hasattr(a, "_shared_world_model")
    assert not hasattr(b, "_shared_world_model")


def test_detach_runs_even_when_body_raises():
    a, b = _make_obs_pair()
    model = object()
    with pytest.raises(RuntimeError, match="boom"):
        with v7_search._bind_shared_world_model((a, b), model):
            assert a._shared_world_model is model
            raise RuntimeError("boom")
    assert not hasattr(a, "_shared_world_model")
    assert not hasattr(b, "_shared_world_model")


def test_no_attach_when_model_is_none():
    a, b = _make_obs_pair()
    with v7_search._bind_shared_world_model((a, b), None):
        assert not hasattr(a, "_shared_world_model")
        assert not hasattr(b, "_shared_world_model")
    assert not hasattr(a, "_shared_world_model")
    assert not hasattr(b, "_shared_world_model")


def test_empty_obs_list_is_a_noop():
    model = object()
    with v7_search._bind_shared_world_model((), model):
        pass  # nothing to assert; just must not raise


def test_detach_tolerates_external_deletion_inside_body():
    """If the policy itself removed the attribute (unlikely but defensive),
    the context manager's exit must not raise an AttributeError."""
    a, b = _make_obs_pair()
    model = object()
    with v7_search._bind_shared_world_model((a, b), model):
        del a._shared_world_model
    assert not hasattr(a, "_shared_world_model")
    assert not hasattr(b, "_shared_world_model")


# ---------------------------------------------------------------------------
# _effective_wallclock_ms — parity-test budget override
# ---------------------------------------------------------------------------


def test_effective_wallclock_returns_arg_when_env_unset(monkeypatch):
    monkeypatch.delenv("ORBIT_WARS_PARITY_WALLCLOCK_MS", raising=False)
    assert v7_search._effective_wallclock_ms(700.0) == 700.0


def test_effective_wallclock_honors_env_override(monkeypatch):
    monkeypatch.setenv("ORBIT_WARS_PARITY_WALLCLOCK_MS", "60000")
    assert v7_search._effective_wallclock_ms(700.0) == 60000.0


def test_effective_wallclock_falls_back_on_invalid_env(monkeypatch):
    monkeypatch.setenv("ORBIT_WARS_PARITY_WALLCLOCK_MS", "not-a-number")
    # Invalid value must not crash the agent — fall back to the caller's.
    assert v7_search._effective_wallclock_ms(700.0) == 700.0


def test_effective_wallclock_treats_empty_string_as_unset(monkeypatch):
    monkeypatch.setenv("ORBIT_WARS_PARITY_WALLCLOCK_MS", "")
    assert v7_search._effective_wallclock_ms(700.0) == 700.0
