"""Smoke tests for scripts/tournament.py (D.1 fixture).

Scope is intentionally narrow: schema correctness, reward-level
reproducibility (per `audit/friction.md::env-not-fully-seed-deterministic`,
ship counts are NOT stable across runs but rewards are), and the
helpers (Wilson CI, p95). Strategy correctness is not tested here —
that's measured by tournament outcomes, not unit assertions.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
sys.path.insert(0, str(SCRIPTS))

# Import the module under test by file path so the test suite doesn't
# require a `scripts/__init__.py` package layout. The module must be
# registered in sys.modules before exec_module so @dataclass can resolve
# `cls.__module__` while building each class.
spec = importlib.util.spec_from_file_location("tournament", SCRIPTS / "tournament.py")
tournament = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
sys.modules["tournament"] = tournament
spec.loader.exec_module(tournament)  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# Pure helpers — fast, no env runs
# ---------------------------------------------------------------------------


def test_wilson_ci_bounds_a_known_winrate():
    lo, hi = tournament._wilson_ci(7, 10)
    assert 0.0 <= lo <= 0.7 <= hi <= 1.0
    assert lo < hi


def test_wilson_ci_zero_n_returns_zeroes():
    assert tournament._wilson_ci(0, 0) == (0.0, 0.0)


def test_wilson_ci_unanimous_loss_or_win_brackets_endpoints():
    lo_w, hi_w = tournament._wilson_ci(10, 10)
    lo_l, hi_l = tournament._wilson_ci(0, 10)
    assert hi_w == pytest.approx(1.0, abs=1e-6) or hi_w < 1.0
    assert lo_l == pytest.approx(0.0, abs=1e-6) or lo_l > 0.0
    assert hi_l < 1.0
    assert lo_w > 0.0


def test_p95_short_list_falls_back_to_max():
    assert tournament._p95([1.0, 2.0, 3.0]) == 3.0
    assert tournament._p95([]) == 0.0


def test_classify_outcome_signs():
    assert tournament._classify([1, -1]) == "p0_win"
    assert tournament._classify([-1, 1]) == "p1_win"
    assert tournament._classify([0, 0]) == "draw"
    assert tournament._classify([1, 1]) == "draw"  # observed simultaneous tie at step limit


# ---------------------------------------------------------------------------
# Tournament smoke — small panels, narrow seed bag
# ---------------------------------------------------------------------------


def test_random_self_play_returns_well_formed_result():
    """Schema check: matrix[name][name] has the required PairStat fields."""
    result = tournament.run_tournament(
        agents={"random": "random"},
        seeds=[42],
        include_self_play=True,
    )
    assert "random" in result.matrix
    assert "random" in result.matrix["random"]
    stat = result.matrix["random"]["random"]
    assert stat.n == 1
    assert stat.p0_wins + stat.p1_wins + stat.draws == stat.n
    assert 0.0 <= stat.wilson_lo <= stat.wilson_hi <= 1.0
    assert len(stat.games) == 1
    g = stat.games[0]
    assert g.seed == 42
    assert len(g.rewards) == 2
    assert g.n_steps > 0


def test_two_agent_grid_populates_all_four_cells_when_self_play_on():
    """Cross-product completeness: include_self_play=True yields a×a, a×b, b×a, b×b."""
    result = tournament.run_tournament(
        agents={"random": "random", "noop": _noop_agent},
        seeds=[42],
        include_self_play=True,
    )
    assert set(result.matrix.keys()) == {"random", "noop"}
    for a in ("random", "noop"):
        for b in ("random", "noop"):
            assert b in result.matrix[a], f"missing cell {a} -> {b}"
            assert result.matrix[a][b].n == 1


def test_two_agent_grid_skips_self_when_self_play_off():
    result = tournament.run_tournament(
        agents={"random": "random", "noop": _noop_agent},
        seeds=[42],
        include_self_play=False,
    )
    assert "random" not in result.matrix["random"]
    assert "noop" not in result.matrix["noop"]
    assert "noop" in result.matrix["random"]
    assert "random" in result.matrix["noop"]


def test_reward_stability_across_runs():
    """Per friction note `env-not-fully-seed-deterministic`: rewards are stable
    across re-runs at the same seed for **deterministic** agents (shipped
    baseline) — even though ship counts and step counts are not. The `random`
    builtin uses Python's process-global `random` module, so back-to-back
    runs of `random vs random` are NOT stable across calls and would make a
    bad fixture for the stability gate.
    """
    REPO_ROOT = Path(__file__).resolve().parents[1]
    baseline = str(REPO_ROOT / "opponents" / "v3_snipe_frozen.py")
    panel = {"baseline": baseline}
    r1 = tournament.run_tournament(agents=panel, seeds=[42], include_self_play=True)
    r2 = tournament.run_tournament(agents=panel, seeds=[42], include_self_play=True)
    g1 = r1.matrix["baseline"]["baseline"].games[0]
    g2 = r2.matrix["baseline"]["baseline"].games[0]
    assert g1.rewards == g2.rewards


def test_to_json_dict_is_serialisable():
    """Persistence path must produce JSON-clean output."""
    import json as _json

    result = tournament.run_tournament(
        agents={"random": "random"},
        seeds=[42],
        include_self_play=True,
    )
    payload = result.to_json_dict()
    s = _json.dumps(payload)  # raises if any non-serialisable field leaks in
    assert '"matrix"' in s
    assert '"random"' in s


def test_persists_json_when_out_dir_given(tmp_path):
    out = tmp_path / "tourneys"
    result = tournament.run_tournament(
        agents={"random": "random"},
        seeds=[42],
        include_self_play=True,
        out_dir=out,
    )
    files = list(out.iterdir())
    assert len(files) == 1
    written = files[0].read_text()
    assert result.timestamp_utc.replace(":", "").replace("-", "") in files[0].name
    assert '"matrix"' in written


def test_loaded_baseline_beats_random_both_sides():
    """Regression for the co_argcount class of bug: when `_load_agent` returns
    a file-loaded callable wrapped by `_timed`, the wrapper signature must match
    what kaggle_environments' arity-trim expects, otherwise the inner agent gets
    called with zero args and silently no-ops. Day-1 finding pins shipped
    baseline > random on the published seeds; if this test ever fails with
    baseline going 0/N, suspect a wrapper signature regression first.
    """
    baseline = str(REPO / "opponents" / "v3_snipe_frozen.py")
    result = tournament.run_tournament(
        agents={"random": "random", "baseline": baseline},
        seeds=[42, 1],
        include_self_play=False,
    )
    bvr = result.matrix["baseline"]["random"]
    rvb = result.matrix["random"]["baseline"]
    assert bvr.p0_wins == bvr.n, f"baseline as P0 should win all; got {bvr.p0_wins}/{bvr.n}"
    assert rvb.p1_wins == rvb.n, f"baseline as P1 should win all; got {rvb.p1_wins}/{rvb.n}"
    # And the timer must have observed nonzero turn durations (else _timed is
    # silently bypassed and we'd miss future budget regressions too).
    assert bvr.p0_p95_turn_ms > 0.0, "expected per-turn timing to be populated for baseline-as-P0"


# ---------------------------------------------------------------------------
# Parallel runner — workers>1 reaches the same reward-level outcome as
# workers=1 on deterministic agents. Per friction note
# `env-not-fully-seed-deterministic`, ship counts and step counts are not
# stable across re-runs even of identical agents, so we compare only the
# reward signal — that IS stable for deterministic agents.
# ---------------------------------------------------------------------------


def test_parallel_runner_matches_sequential_for_deterministic_agent():
    REPO_ROOT = Path(__file__).resolve().parents[1]
    baseline = str(REPO_ROOT / "opponents" / "v3_snipe_frozen.py")
    panel = {"baseline": baseline}
    seeds = [42, 1, 7, 13]
    seq = tournament.run_tournament(
        agents=panel, seeds=seeds, include_self_play=True, workers=1,
    )
    par = tournament.run_tournament(
        agents=panel, seeds=seeds, include_self_play=True, workers=2,
    )
    seq_stat = seq.matrix["baseline"]["baseline"]
    par_stat = par.matrix["baseline"]["baseline"]
    assert seq_stat.n == par_stat.n
    assert seq_stat.p0_wins == par_stat.p0_wins
    assert seq_stat.p1_wins == par_stat.p1_wins
    assert seq_stat.draws == par_stat.draws
    seq_rewards = sorted((g.seed, tuple(g.rewards)) for g in seq_stat.games)
    par_rewards = sorted((g.seed, tuple(g.rewards)) for g in par_stat.games)
    assert seq_rewards == par_rewards


def test_parallel_runner_rejects_callable_agent():
    """workers>1 needs picklable string specs; callables would fail mid-run."""
    with pytest.raises(ValueError, match="workers>1 requires string"):
        tournament.run_tournament(
            agents={"noop": _noop_agent},
            seeds=[42],
            include_self_play=True,
            workers=2,
        )


# ---------------------------------------------------------------------------
# Test fixture: minimal callable agent
# ---------------------------------------------------------------------------


def _noop_agent(obs, config=None):
    """Empty-action agent — lets the env play out without launches."""
    return []
