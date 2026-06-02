"""Oracle tests for the exact-oracle REFINER (Rule 49, augment-not-replace).

Mechanism under test (agents/baseline/chooser_refine.choose_refine):
  - Degrades to the champion EXACTLY when nothing is added (we append, never
    rebuild) — the no-regression guarantee.
  - TEAMWORK-ADD: a coalition the champion (independent scoring) can't see is
    appended when its EXACT marginal joint value is positive and it doesn't
    conflict with the champion's locks.
  - DROP-ONE (opt-in): a champion launch whose removal raises the exact joint
    value is stripped from the emit.

Reuses the world/prerank builders from tests/test_chooser_greedy.
"""

from __future__ import annotations

import agents.baseline.chooser_refine as cr
from agents.baseline.chooser_refine import choose_refine
from agents.baseline.chooser_trajectory import choose_trajectory
from tests.test_chooser_greedy import GAMMA, HORIZON, _prerank, _redundant_scenario


def _empty_coalitions(*a, **k):
    return iter(())


# ---------------------------------------------------------------------------
# (a) No atoms ⇒ refine output is the champion's output verbatim.
# ---------------------------------------------------------------------------

def test_refine_no_atoms_equals_champion(monkeypatch):
    monkeypatch.setattr(cr, "generate_sync_coalitions", _empty_coalitions)
    snap, world, model, S1, S2, T = _redundant_scenario()
    prerank = _prerank(world, (S1, T, 20), (S2, T, 20))

    champ = choose_trajectory(snap, prerank, None, 0, 2, 2000.0, 5, HORIZON,
                              GAMMA, world, model)
    snap2, world2, model2, S1b, S2b, Tb = _redundant_scenario()
    prerank2 = _prerank(world2, (S1b, Tb, 20), (S2b, Tb, 20))
    refined = choose_refine(snap2, prerank2, None, 0, 2, 2000.0, 5, HORIZON,
                            GAMMA, world2, model2)

    assert refined[0] == champ[0]   # moves identical
    assert refined[1] == champ[1]   # commits identical


# ---------------------------------------------------------------------------
# (b) A positive-marginal coalition the champion can't reach is appended.
#     Oracle stubbed so any set containing a waiting leg gains; the atom uses
#     a source the champion did not lock.
# ---------------------------------------------------------------------------

def test_refine_appends_positive_coalition(monkeypatch):
    snap, world, model, S1, S2, T = _redundant_scenario()
    prerank = _prerank(world, (S1, T, 20))            # champion fires S1→T

    champ_moves, champ_commits = choose_trajectory(
        snap, prerank, None, 0, 2, 2000.0, 5, HORIZON, GAMMA, world, model)
    champ_src = {int(m[0]) for m in champ_moves}
    by_id = {int(p.id): p for p in world.planets_by_id.values()}
    opp = by_id[3]
    # Coalition atom on a target the champion DIDN'T lock (the opp planet),
    # from sources the champion didn't use: fire-now leg (S2) + waiting leg
    # (T as a stand-in src; geometry is irrelevant — the scorer is stubbed).
    assert int(S2.id) not in champ_src
    atom = [
        (S2, opp, 12, 0.5, 0),            # fire-now leg → moves
        (T, opp, 8, 0.5, 3),              # waiting leg → sync commit
    ]

    def _one_atom(*a, **k):
        yield (atom, 5)

    def _fake_delta(snap_base, launches, *a, **k):
        # Any set with a waiting leg (the coalition) scores higher.
        return (5.0 if any(int(L[4]) > 0 for L in launches) else 1.0), "scored"

    monkeypatch.setattr(cr, "generate_sync_coalitions", _one_atom)
    monkeypatch.setattr(cr, "_delta", _fake_delta)

    moves, commits = choose_refine(snap, prerank, None, 0, 2, 2000.0, 5,
                                   HORIZON, GAMMA, world, model)

    # Champion's move survives, plus the atom's fire-now leg; the waiting leg
    # is surfaced as a sync_joint commit.
    assert [int(S2.id), 0.5, 12] in moves
    assert any(c.get("sync_joint") and c["src_id"] == int(T.id) for c in commits)
    assert champ_moves[0] in moves       # champion launch not removed


# ---------------------------------------------------------------------------
# (c) DROP-ONE strips a champion launch the oracle says is pure waste.
# ---------------------------------------------------------------------------

def test_refine_drops_exact_waste(monkeypatch):
    monkeypatch.setenv("BASELINE_REFINE_DROP", "1")
    monkeypatch.setattr(cr, "generate_sync_coalitions", _empty_coalitions)
    snap, world, model, S1, S2, T = _redundant_scenario()
    prerank = _prerank(world, (S1, T, 20))
    champ_moves, _ = choose_trajectory(snap, prerank, None, 0, 2, 2000.0, 5,
                                       HORIZON, GAMMA, world, model)
    assert champ_moves, "champion should fire at least one move"

    # Oracle: the full set scores LOWER than the empty/without set ⇒ the lone
    # champion launch is "waste" and must be dropped.
    def _fake_delta(snap_base, launches, *a, **k):
        return (0.0 if launches else 10.0), "scored"

    monkeypatch.setattr(cr, "_delta", _fake_delta)
    moves, commits = choose_refine(snap, prerank, None, 0, 2, 2000.0, 5,
                                   HORIZON, GAMMA, world, model)
    assert moves == []      # the wasteful launch was stripped
