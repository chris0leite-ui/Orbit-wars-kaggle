"""Observation-grounded scenarios — Phase 1b: DI1 + G1.

DI1: distant-idleness. Encodes PI's "the planets that are far off
simply cannot see what to do" diagnosis of failure mode (e), the
universal aggression-deficit (BPJKs d=+1.26σ launches_per_turn,
d=-0.82σ mean_garrison_at_launch).

G1: garrison-counter. Encodes failure mode (c) — 13.9% of all live
fleets bounce on enemy garrison (`audit/2026-05-19-replay-mine-pre-
roi.md`).

Both scenarios FAIL on the current submitted agent (`agents.baseline.
main`) by construction. The scenarios stay green only once the ROI
rebuild ships the architectural commitments PI articulated: every
planet gets a richer candidate menu (forward-redeploy + long-range
commit + multi-source bundle + direct strike), and the scorer
penalises post-launch source exposure.
"""

from __future__ import annotations

import math

import pytest

from tests.scenarios.base import (
    Scenario, ValidationResult, _obs, _planet, _targets_of_emits,
    all_scenarios, register,
)


# ---- DI1 — distant-idleness ----------------------------------------------


@register
class DI1_DistantIdleness(Scenario):
    """A far-rear planet sitting on 200 ships must either redeploy
    forward or commit a large-enough direct strike. Sitting idle for
    K=10 turns is FAIL.

    Layout (all y != 50 to clear the sun at (50,50) r=10):
      P0  us    (30, 30) ships=20  prod=1  — near-front, can't solo P2
      P1  us    (15, 80) ships=200 prod=1  — far-rear, the IDLE planet
      P2  enemy (70, 30) ships=50  prod=1  — attackable but needs >50

    P0 alone undersizes (20 < 51 + prod·travel). A correct agent must
    use P1 for: (i) own→own redeploy to P0, or (ii) long-range commit
    to P2 with ≥100 ships.

    Source: BPJKs aggression-deficit data (off-branch on origin/main)
    + PI's "can't see what to do" diagnosis articulated 2026-05-19 AM.
    """

    name: str = "DI1"
    rationale: str = (
        "Distant idle planets must produce redeploy/long-commit "
        "candidates; sitting on >100 ships for 10 turns is the bug."
    )
    source: str = (
        "audit/2026-05-19-replay-mine-pre-roi.md + BPJKs "
        "audit/2026-05-18-archetype-action-audit-gap-vs-even.md "
        "(off-branch on origin/main)"
    )
    flavour: str = "multi-turn"
    rollout_K: int = 10

    P0 = 0  # our near-front
    P1 = 1  # our far-rear (the test subject)
    P2 = 2  # enemy near-front
    LARGE_COMMIT_MIN_SHIPS = 100  # minimum size for option (ii)

    def setup(self) -> dict:
        return _obs([
            _planet(self.P0, owner=0, x=30, y=30, ships=20,  production=1),
            _planet(self.P1, owner=0, x=15, y=80, ships=200, production=1),
            _planet(self.P2, owner=1, x=70, y=30, ships=50,  production=1),
        ])

    def validate(self, emit_log: list[list],
                 world_log: list[dict]) -> ValidationResult:
        # Result-oriented test. The bug isn't "P1 emitted nothing" —
        # baseline pays lip service with token launches (18 ships from a
        # 200-ship planet) and still hoards. The real failure is that
        # P1's ships sit unused for the whole K-turn window. We test the
        # CONSEQUENCE: by turn K, did P1 actually deploy enough of its
        # hoard that the front is meaningfully reinforced?
        #
        # Two PASS paths:
        #   (i)  we hold P2 (we won the front-line trade), OR
        #   (ii) P1 has dropped below 100 ships AND (P0 garrison or our
        #        in-flight committed > 80) — i.e. the resource moved.
        # FAIL  = P1 still sitting on ≥100 ships at end with no progress.

        # Sum ships launched from P1 across all turns.
        ships_launched_from_p1 = sum(
            int(e[2]) for emits in emit_log
            for e in emits if int(e[0]) == self.P1
        )
        # Find P1's final remaining ships via fast_sim trace on the
        # final world. world_log[-1] is the obs *at the start of* the
        # last turn we drove; we want the state AFTER all K turns. We
        # walk the world_log to get the most up-to-date P1 ship count
        # that's available, conservatively.
        final_obs = world_log[-1]
        p1_final = next((p for p in final_obs["planets"] if p[0] == self.P1), None)
        p0_final = next((p for p in final_obs["planets"] if p[0] == self.P0), None)
        p2_final = next((p for p in final_obs["planets"] if p[0] == self.P2), None)
        p1_ships = int(p1_final[5]) if p1_final else 0
        p0_ships = int(p0_final[5]) if p0_final else 0
        p2_owner = int(p2_final[1]) if p2_final else 1

        if p2_owner == 0:
            return ValidationResult(
                True,
                f"We hold P2 at turn {len(world_log)} — front-line won "
                f"(P1 sent {ships_launched_from_p1} ships, ends with {p1_ships})",
            )
        if p1_ships < 100 and (p0_ships > 80 or ships_launched_from_p1 > 80):
            return ValidationResult(
                True,
                f"P1 deployed its hoard: launched {ships_launched_from_p1} "
                f"ships, ends with {p1_ships}; P0 ships={p0_ships}",
            )
        return ValidationResult(
            False,
            f"Distant-idleness FAIL: P1 still has {p1_ships} ships after "
            f"{len(world_log)} turns; launched only {ships_launched_from_p1} "
            f"ships total; P0 ships={p0_ships}; P2 still enemy",
        )

    def self_check(self) -> ValidationResult:
        base = super().self_check()
        if not base.passed:
            return base
        # Sanity: both "valid resolutions" are actually reachable from P1.
        # Forward-redeploy P1 → P0:
        obs = self.setup()
        p0 = obs["planets"][self.P0]
        p1 = obs["planets"][self.P1]
        p2 = obs["planets"][self.P2]
        # Build a single-emit raycast probe for each candidate.
        ang_p1_p0 = math.atan2(p0[3] - p1[3], p0[2] - p1[2])
        hit_p0 = _targets_of_emits(obs, [[self.P1, ang_p1_p0, 50]])
        if not hit_p0 or hit_p0[0] != self.P0:
            return ValidationResult(
                False, f"P1→P0 redeploy raycast didn't hit P0; hit={hit_p0}",
            )
        ang_p1_p2 = math.atan2(p2[3] - p1[3], p2[2] - p1[2])
        hit_p2 = _targets_of_emits(obs, [[self.P1, ang_p1_p2, 120]])
        if not hit_p2 or hit_p2[0] != self.P2:
            return ValidationResult(
                False, f"P1→P2 long-commit raycast didn't hit P2; hit={hit_p2}",
            )
        return ValidationResult(True, "DI1 self-check ok")


# ---- G1 — garrison-counter ----------------------------------------------


@register
class G1_GarrisonCounter(Scenario):
    """Attack on a near-front ENEMY whose defenders accrete during
    flight time. Bare-minimum launches bounce on a defender count that
    grew past our committed ship count by arrival. Plus the source
    drains, exposing us to opp counter-strikes.

    Layout (off y=50 to clear the sun):
      P0  us     (30, 30) ships=80  prod=1
      P1  enemy  (55, 30) ships=25  prod=2  — bait: 80 >> 25, looks easy
      P2  enemy  (75, 30) ships=80  prod=2  — counter-launch threat

    Why this bites: ETA P0→P1 at fleet_speed(40)≈3.0 = ~8 turns. P1's
    prod=2 means defenders rise from 25 to ~41 by arrival. A 40-ship
    launch arrives vs 41 defenders → bounce. P2 reacts and counter-
    launches to recapture if we slimly succeed. Correct behaviour:
    EITHER skip, OR over-margin (>55 ships, ideally >65 to handle
    counter).

    Source: `audit/2026-05-19-replay-mine-pre-roi.md` — 13.9% live
    `bounced_enemy` prevalence (7913 / 56842 fleets across 5 subs).
    """

    name: str = "G1"
    rationale: str = (
        "Attack on near-front enemy with mid-flight production AND "
        "adjacent enemy counter-launch: only correct moves are skip "
        "or commit large-enough to capture-then-hold. Result-oriented."
    )
    source: str = "audit/2026-05-19-replay-mine-pre-roi.md (13.9% bounced_enemy)"
    flavour: str = "multi-turn"
    rollout_K: int = 14  # enough for our launch + arrival + opp counter

    P0 = 0
    P1 = 1
    P2 = 2

    def setup(self) -> dict:
        # P2 placed close enough to P1 that lite_greedy_policy will pick
        # P1 as its high-ROI target once it flips to us. Distance P2→P1
        # is 12 units so the counter ETA fits inside K=14.
        return _obs([
            _planet(self.P0, owner=0, x=30, y=30, ships=80, production=1),
            _planet(self.P1, owner=1, x=58, y=30, ships=25, production=2),
            _planet(self.P2, owner=1, x=70, y=30, ships=100, production=2),
        ])

    def validate(self, emit_log: list[list],
                 world_log: list[dict]) -> ValidationResult:
        # Did we ever launch at P1?
        launched_at_p1 = False
        first_launch_size = 0
        first_launch_turn = -1
        for t, emits in enumerate(emit_log):
            obs_t = world_log[t]
            for emit in emits:
                if int(emit[0]) != self.P0:
                    continue
                hits = _targets_of_emits(obs_t, [emit])
                if hits and hits[0] == self.P1:
                    if not launched_at_p1:
                        first_launch_size = int(emit[2])
                        first_launch_turn = t
                    launched_at_p1 = True

        # Check end state.
        final = world_log[-1]
        p0_final = next((p for p in final["planets"] if p[0] == self.P0), None)
        p1_final = next((p for p in final["planets"] if p[0] == self.P1), None)
        p0_ships = int(p0_final[5]) if p0_final else 0
        p1_owner = int(p1_final[1]) if p1_final else 1
        p1_ships = int(p1_final[5]) if p1_final else 0

        # PASS path A: agent skipped the trap entirely. The bounce-bug
        # is "launched and lost"; skipping is correct conservative play
        # regardless of what else happens (e.g. opp may overrun P0 — not
        # the agent's fault, not what this scenario tests).
        if not launched_at_p1:
            return ValidationResult(
                True,
                f"Skipped the P0→P1 trap (P0 ends with {p0_ships}, "
                f"P2 may or may not have overrun us — not under test)",
            )
        # PASS path B: agent launched AND we hold P1 with non-trivial
        # defenders at turn K (survived the counter).
        if p1_owner == 0 and p1_ships >= 10:
            return ValidationResult(
                True,
                f"Launched {first_launch_size} ships at P1 (turn "
                f"{first_launch_turn}) and held it through K={len(emit_log)}; "
                f"P1 garrison={p1_ships}",
            )
        return ValidationResult(
            False,
            f"Garrison-counter FAIL: launched {first_launch_size} ships at "
            f"P1 (turn {first_launch_turn}); end state P1 owner={p1_owner} "
            f"ships={p1_ships}; P0 ships={p0_ships}",
        )

    def self_check(self) -> ValidationResult:
        base = super().self_check()
        if not base.passed:
            return base
        # Sanity: P0 → P1 IS physically reachable (otherwise the test is
        # vacuous — the agent can't even commit the prohibited launch).
        obs = self.setup()
        p0 = obs["planets"][self.P0]
        p1 = obs["planets"][self.P1]
        ang = math.atan2(p1[3] - p0[3], p1[2] - p0[2])
        hit = _targets_of_emits(obs, [[self.P0, ang, 35]])
        if not hit or hit[0] != self.P1:
            return ValidationResult(
                False, f"P0→P1 raycast didn't hit P1; hit={hit}",
            )
        return ValidationResult(True, "G1 self-check ok")


# ---- pytest tests -------------------------------------------------------
#
# These tests run the scenarios against the CURRENT submitted agent
# (`agents.baseline.main`). By construction both must FAIL — the
# scenarios encode known live-data failure modes. XFAIL(strict=True) so
# an unexpected PASS surfaces as a test failure: it means the ROI work
# (or some other change) actually fixed the bug, and we should remove
# the xfail.


@pytest.mark.xfail(strict=True, reason=(
    "DI1: distant-idleness encodes a known failure of the current "
    "submitted agent. PASS here means the ROI rebuild has shipped the "
    "redeploy + long-commit primitives — remove xfail when that lands."
))
def test_DI1_distant_idleness_against_baseline():
    sc = DI1_DistantIdleness()
    result = sc.run("agents.baseline.main")
    assert result.passed, result.explanation


def test_G1_garrison_counter_against_baseline():
    """Honest scope note (Phase 1b limitation): baseline's chooser
    correctly skips the bait in this 3-planet synthetic shape — the
    live 13.9% bounce-rate is an interaction effect at 20+-planet
    scale that doesn't reproduce in a small constructed board (where
    the trap is fully visible to baseline's heuristics).

    G1 stays in the suite as a regression check: it should keep
    passing as we modify the agent — if a future change makes the
    agent BOUNCE on this visible trap, the scenario will catch it.
    Phase 1c should add a G1-realistic variant either extracted from
    a replay or built on a larger board.
    """
    sc = G1_GarrisonCounter()
    result = sc.run("agents.baseline.main")
    assert result.passed, result.explanation


def test_self_check_DI1():
    """Self-check is a precondition: the layout must be physically
    valid (round-trip, reachable launches for the resolutions)."""
    sc = DI1_DistantIdleness()
    chk = sc.self_check()
    assert chk.passed, chk.explanation


def test_self_check_G1():
    sc = G1_GarrisonCounter()
    chk = sc.self_check()
    assert chk.passed, chk.explanation


def test_registry_contains_phase_1b_scenarios():
    names = {sc.name for sc in all_scenarios()}
    assert {"DI1", "G1"} <= names, names
