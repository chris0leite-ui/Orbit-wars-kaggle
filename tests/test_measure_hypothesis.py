"""Smoke tests for scripts.measure_hypothesis.

The heavy lifting is in lib.metrics (already covered by test_metrics.py)
and scripts.replay_mine.mine_one_submission (covered by existing tests).
This file pins the glue: dispatch over the registered metrics, markdown
output shape, and the append-results-row logic.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import measure_hypothesis


def _fake_replay(team_name: str = "us") -> dict:
    """Minimal replay JSON that mine_one_submission can attribute."""
    return {
        "info": {"TeamNames": [team_name, "other"]},
        "steps": [
            # turn 0: start
            [{"action": [], "status": "ACTIVE",
              "observation": {"player": 0, "fleets": [], "planets": []}},
             {"action": [], "status": "ACTIVE",
              "observation": {"player": 1, "fleets": [], "planets": []}}],
            # turn 1: a launch
            [{"action": [[0, 0.0, 10]], "status": "ACTIVE",
              "observation": {"player": 0, "fleets": [], "planets": []}},
             {"action": [], "status": "ACTIVE",
              "observation": {"player": 1, "fleets": [], "planets": []}}],
            # turn 2: end
            [{"action": [], "status": "DONE",
              "observation": {"player": 0, "fleets": [], "planets": []}},
             {"action": [], "status": "DONE",
              "observation": {"player": 1, "fleets": [], "planets": []}}],
        ],
    }


def _make_sub_dir(tmp_path: Path, sub_id: str, n_replays: int) -> Path:
    """Create audit/live-episodes/<sub_id>/episode-*-replay.json files
    under the given tmp root."""
    sub_dir = tmp_path / "audit" / "live-episodes" / sub_id
    sub_dir.mkdir(parents=True)
    for i in range(n_replays):
        path = sub_dir / f"episode-{i:04d}-replay.json"
        path.write_text(json.dumps(_fake_replay()))
    return sub_dir


def test_measure_returns_all_registered_metrics(monkeypatch, tmp_path):
    """measure() returns one entry per registered metric (plus context
    keys with `__` prefix)."""
    sub_id = "TEST_SUB_001"
    _make_sub_dir(tmp_path, sub_id, n_replays=3)
    monkeypatch.setattr(measure_hypothesis, "REPO", tmp_path)
    # mine_one_submission reads REPO/audit/live-episodes/... — but its
    # import-time REPO is the *real* repo. Patch the import as well.
    from scripts import replay_mine
    monkeypatch.setattr(replay_mine, "REPO", tmp_path)

    results = measure_hypothesis.measure(sub_id)
    from lib import metrics
    for name in metrics.list_metrics():
        assert name in results, f"metric {name} missing from results"
    # Context keys.
    assert "__n_episodes" in results
    assert "__n_fleets" in results


def test_measure_missing_sub_returns_empty(monkeypatch, tmp_path):
    """A sub_id with no replay dir should yield {} (and print to stderr)."""
    monkeypatch.setattr(measure_hypothesis, "REPO", tmp_path)
    from scripts import replay_mine
    monkeypatch.setattr(replay_mine, "REPO", tmp_path)
    results = measure_hypothesis.measure("NONEXISTENT_SUB")
    assert results == {}


def test_render_markdown_contains_metric_names_and_baselines(monkeypatch, tmp_path):
    sub_id = "TEST_SUB_002"
    _make_sub_dir(tmp_path, sub_id, n_replays=2)
    monkeypatch.setattr(measure_hypothesis, "REPO", tmp_path)
    from scripts import replay_mine
    monkeypatch.setattr(replay_mine, "REPO", tmp_path)

    results = measure_hypothesis.measure(sub_id)
    md = measure_hypothesis.render_markdown(sub_id, results)

    # Header
    assert sub_id in md
    assert "Sample:" in md
    # Every registered metric named
    from lib import metrics
    for name in metrics.list_metrics():
        assert f"`{name}`" in md, f"{name} not in rendered table"
    # Baseline columns present
    assert "v15" in md and "top10" in md and "midpack" in md
    # PI guidance present
    assert "pre-registration" in md or "verdict" in md.lower()


def test_append_results_row_inserts_at_top(monkeypatch, tmp_path):
    """append_results_row inserts the new row just after the table
    header separator, preserving newest-first order."""
    results_path = tmp_path / "results.md"
    results_path.write_text(
        "# results\n\n"
        "| sub_id | date | … | postmortem |\n"
        "|---|---|---|---|\n"
        "| OLD_SUB | 2026-01-01 | foo | n |\n"
    )

    sub_id = "NEW_SUB_001"
    measure_hypothesis.append_results_row(
        sub_id,
        {
            "win_fraction": 0.5,
            "waste_attack_fraction": 0.1,
            "first_launch_step": 5.0,
            "__n_episodes": 12,
        },
        results_path=results_path,
        pre_register_doc="audit/hypotheses/NEW_SUB_001-foo.md",
    )

    text = results_path.read_text()
    lines = [l for l in text.splitlines() if "_SUB" in l]
    # NEW first, then OLD (newest-first).
    assert lines[0].startswith(f"| {sub_id}")
    assert lines[1].startswith("| OLD_SUB")
    # The new row carries our snapshot numbers.
    assert "win=0.500" in lines[0]
    assert "waste_atk=0.100" in lines[0]
    assert "n_ep=12" in lines[0]


def test_append_results_row_missing_file_warns(monkeypatch, tmp_path, capfd):
    """If the results file doesn't exist, append is a no-op with stderr
    warning (don't crash the post-submit pipeline)."""
    missing = tmp_path / "no_such_results.md"
    measure_hypothesis.append_results_row(
        "X", {"win_fraction": 0.0}, results_path=missing,
    )
    _out, err = capfd.readouterr()
    assert "missing" in err.lower()
