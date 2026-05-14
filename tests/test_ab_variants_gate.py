"""Unit tests for the multi-anchor gate helpers in scripts/ab_variants.py.

Only exercises pure functions: `_wilson_ci`, `_per_anchor_summarise`,
`_anchor_gate`. Constructs a fake `tournament.TournamentResult` with
hand-set `PairStat` rows so we don't need to run real games.

This complements the existing strategy-grading loop documented in
`/root/.claude/plans/taking-the-role-of-buzzing-rossum.md` (idea G).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(SCRIPTS))

# Load tournament first so `ab_variants`'s `import scripts.tournament`
# resolves the same module object.
_t_spec = importlib.util.spec_from_file_location(
    "tournament", SCRIPTS / "tournament.py"
)
tournament = importlib.util.module_from_spec(_t_spec)  # type: ignore[arg-type]
sys.modules["tournament"] = tournament
_t_spec.loader.exec_module(tournament)  # type: ignore[union-attr]

# Also expose under `scripts.tournament` for `import scripts.tournament`.
import scripts as _scripts_pkg  # noqa: E402
sys.modules["scripts.tournament"] = tournament
setattr(_scripts_pkg, "tournament", tournament)

import scripts.ab_variants as ab_variants  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers: build a fake TournamentResult with a controlled win matrix.
# ---------------------------------------------------------------------------


def _stat(p0_wins: int, p1_wins: int, draws: int = 0) -> tournament.PairStat:
    return tournament.PairStat(
        p0_name="x",
        p1_name="y",
        n=p0_wins + p1_wins + draws,
        p0_wins=p0_wins,
        p1_wins=p1_wins,
        draws=draws,
    )


def _result(matrix: dict[str, dict[str, tournament.PairStat]]) -> tournament.TournamentResult:
    return tournament.TournamentResult(
        timestamp_utc="test",
        agents={k: f"<{k}>" for k in matrix.keys()},
        seeds=[1],
        include_self_play=False,
        matrix=matrix,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_per_anchor_pools_both_seats_correctly():
    # candidate=A, anchor=B.
    # A-as-P0 vs B-as-P1: A wins 8, B wins 2 (matrix[A][B] = 8/2/0).
    # A-as-P1 vs B-as-P0: A wins 6, B wins 4 (matrix[B][A] = 4/6/0
    #   from B's perspective; so candidate A's wins come from p1_wins=6).
    matrix = {
        "A": {"B": _stat(p0_wins=8, p1_wins=2, draws=0)},
        "B": {"A": _stat(p0_wins=4, p1_wins=6, draws=0)},
    }
    per = ab_variants._per_anchor_summarise(_result(matrix), "A", ["B"])
    row = per["B"]
    assert row["wins"] == 14  # 8 as P0 + 6 as P1
    assert row["losses"] == 6  # 2 as P0 (opponent's p1_wins) + 4 as P1 (opp's p0_wins)
    assert row["draws"] == 0
    assert row["n"] == 20
    assert row["winrate"] == 0.7


def test_per_anchor_handles_missing_seat_cells():
    # Only one direction populated (e.g., bundle ran out of compute).
    matrix = {
        "A": {"B": _stat(p0_wins=10, p1_wins=0, draws=0)},
        "B": {},
    }
    per = ab_variants._per_anchor_summarise(_result(matrix), "A", ["B"])
    row = per["B"]
    assert row["wins"] == 10
    assert row["losses"] == 0
    assert row["n"] == 10


def test_per_anchor_multiple_anchors_independent():
    # A is strong vs B (90%) but weak vs C (30%) — the classic
    # non-transitive trap idea G is designed to catch.
    matrix = {
        "A": {
            "B": _stat(p0_wins=9, p1_wins=1, draws=0),
            "C": _stat(p0_wins=3, p1_wins=7, draws=0),
        },
        "B": {"A": _stat(p0_wins=1, p1_wins=9, draws=0)},  # A wins 9 as P1
        "C": {"A": _stat(p0_wins=7, p1_wins=3, draws=0)},  # A wins 3 as P1
    }
    per = ab_variants._per_anchor_summarise(_result(matrix), "A", ["B", "C"])
    assert per["B"]["winrate"] == 0.9  # 18 / 20
    assert per["C"]["winrate"] == 0.3  # 6 / 20


def test_gate_passes_when_all_anchors_meet_threshold():
    per = {
        "B": {"wilson_lo": 0.60, "wilson_hi": 0.80, "winrate": 0.70, "wins": 14, "losses": 6, "draws": 0, "n": 20},
        "C": {"wilson_lo": 0.56, "wilson_hi": 0.78, "winrate": 0.67, "wins": 13, "losses": 7, "draws": 0, "n": 20},
    }
    verdict = ab_variants._anchor_gate(per, threshold=0.55)
    assert verdict["pass"] is True
    assert verdict["failing_anchors"] == []
    assert set(verdict["passing_anchors"]) == {"B", "C"}


def test_gate_fails_on_any_single_anchor_regression():
    # B passes, C fails. Overall verdict is FAIL — the entire point of
    # this gate vs the existing pooled `_summarise`.
    per = {
        "B": {"wilson_lo": 0.70, "wilson_hi": 0.85, "winrate": 0.78, "wins": 16, "losses": 4, "draws": 0, "n": 20},
        "C": {"wilson_lo": 0.40, "wilson_hi": 0.62, "winrate": 0.50, "wins": 10, "losses": 10, "draws": 0, "n": 20},
    }
    verdict = ab_variants._anchor_gate(per, threshold=0.55)
    assert verdict["pass"] is False
    assert verdict["failing_anchors"] == ["C"]
    assert verdict["passing_anchors"] == ["B"]


def test_gate_pass_when_anchor_set_is_empty():
    # Edge case: candidate declared but no anchors (single-variant run).
    verdict = ab_variants._anchor_gate({}, threshold=0.55)
    assert verdict["pass"] is True
    assert verdict["failing_anchors"] == []


def test_pooled_summary_can_mask_a_per_anchor_regression():
    # End-to-end check that the new gate catches what the existing
    # `_summarise` would miss: A beats B 90-10 and loses to C 30-70.
    # Pooled winrate: 24 / 40 = 60% (Wilson-lo ~45-47%); but vs C
    # alone Wilson-lo is well below 0.55. The pooled view gives a
    # ~borderline number; the per-anchor view rejects.
    matrix = {
        "A": {
            "B": _stat(p0_wins=9, p1_wins=1),
            "C": _stat(p0_wins=3, p1_wins=7),
        },
        "B": {"A": _stat(p0_wins=1, p1_wins=9)},
        "C": {"A": _stat(p0_wins=7, p1_wins=3)},
    }
    res = _result(matrix)
    pooled = ab_variants._summarise(res, ["A", "B", "C"])
    assert pooled["A"]["winrate"] == 0.6  # 24/40

    per = ab_variants._per_anchor_summarise(res, "A", ["B", "C"])
    verdict = ab_variants._anchor_gate(per, threshold=0.55)
    assert verdict["pass"] is False
    assert verdict["failing_anchors"] == ["C"]
