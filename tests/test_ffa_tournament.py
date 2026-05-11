"""Tests for scripts/ffa_tournament.py — 4P FFA primitive."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# Load via the same shim ffa_panel uses so the module name is stable.
_spec = importlib.util.spec_from_file_location(
    "ffa_tournament", REPO / "scripts" / "ffa_tournament.py"
)
ffa_tournament = importlib.util.module_from_spec(_spec)
sys.modules["ffa_tournament"] = ffa_tournament
_spec.loader.exec_module(ffa_tournament)

from scripts._agent_paths import resolve_agent_path


# ---------------------------------------------------------------------------
# _seat_assignment — pure helper
# ---------------------------------------------------------------------------


def test_seat_assignment_places_focal_in_chosen_seat():
    bg_specs = ["b0", "b1", "b2"]
    bg_names = ["b0", "b1", "b2"]
    specs, names = ffa_tournament._seat_assignment(
        focal_name="F",
        background_names=bg_names,
        focal_seat=2,
        focal_spec="F",
        background_specs=bg_specs,
    )
    assert names[2] == "F"
    assert specs[2] == "F"
    # Backgrounds fill the other seats in order (skip focal seat).
    assert names == ["b0", "b1", "F", "b2"]


def test_seat_assignment_each_seat_gets_focal_exactly_once_across_rotations():
    bg_specs = ["b0", "b1", "b2"]
    seen_focal_seats = []
    for focal_seat in range(4):
        _, names = ffa_tournament._seat_assignment(
            focal_name="F", background_names=bg_specs,
            focal_seat=focal_seat, focal_spec="F", background_specs=bg_specs,
        )
        seen_focal_seats.append(names.index("F"))
    assert seen_focal_seats == [0, 1, 2, 3]


# ---------------------------------------------------------------------------
# FFAGameRecord — first_place logic
# ---------------------------------------------------------------------------


def test_first_place_when_focal_has_max_reward():
    rec = ffa_tournament.FFAGameRecord(
        seed=1, focal_seat=0, seat_names=["F", "a", "b", "c"],
        rewards=[1, -1, -1, -1], statuses=["DONE"] * 4, n_steps=100,
    )
    assert rec.focal_first_place() is True


def test_not_first_place_when_focal_loses():
    rec = ffa_tournament.FFAGameRecord(
        seed=1, focal_seat=1, seat_names=["a", "F", "b", "c"],
        rewards=[1, -1, -1, -1], statuses=["DONE"] * 4, n_steps=100,
    )
    assert rec.focal_first_place() is False


def test_first_place_on_tied_max():
    # Live env allows multi-way ties — every seat at max-score gets +1.
    rec = ffa_tournament.FFAGameRecord(
        seed=1, focal_seat=2, seat_names=["a", "b", "F", "c"],
        rewards=[1, -1, 1, -1], statuses=["DONE"] * 4, n_steps=500,
    )
    assert rec.focal_first_place() is True


# ---------------------------------------------------------------------------
# run_ffa_tournament — end-to-end smoke
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_run_ffa_tournament_smoke_4p_one_seed():
    """One game with 4 agents — sanity-check schema + first-place semantics."""
    focal_path = resolve_agent_path("roi")
    bg_paths = [
        resolve_agent_path("weakest"),
        resolve_agent_path("enemy_first"),
        resolve_agent_path("baseline"),
    ]
    res = ffa_tournament.run_ffa_tournament(
        focal=focal_path,
        background=bg_paths,
        focal_name="roi",
        background_names=["weakest", "enemy_first", "baseline"],
        seeds=[42],
        players=4,
        rotate_seats=False,   # 1 game per seed
        workers=1,
    )
    assert res.n_games == 1
    g = res.games[0]
    assert len(g.rewards) == 4
    assert len(g.statuses) == 4
    assert g.focal_seat == 0
    # Exactly one of the focal/non-focal seats is at the max reward.
    assert max(g.rewards) in {1, 0, -1}
    # focal_turn_ms is populated.
    assert len(g.focal_turn_ms) > 0


@pytest.mark.slow
def test_rotate_seats_produces_4_games_per_seed():
    focal_path = resolve_agent_path("roi")
    bg_paths = [resolve_agent_path(n) for n in
                ["weakest", "enemy_first", "baseline"]]
    res = ffa_tournament.run_ffa_tournament(
        focal=focal_path, background=bg_paths,
        focal_name="roi",
        background_names=["weakest", "enemy_first", "baseline"],
        seeds=[42], players=4, rotate_seats=True, workers=1,
    )
    assert res.n_games == 4
    # Every seat 0..3 is occupied by focal exactly once.
    assert sorted(g.focal_seat for g in res.games) == [0, 1, 2, 3]


def test_background_size_validation():
    """players-1 background entries are required."""
    with pytest.raises(ValueError, match="background must have"):
        ffa_tournament.run_ffa_tournament(
            focal="random", background=["random"],   # only 1, need 3
            seeds=[1], players=4, rotate_seats=False, workers=1,
        )
