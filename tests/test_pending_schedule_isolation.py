"""Bug #1 — pending_schedule game-id collision / cross-game state leak.

The OLD design kept a module-level `_PENDING` dict keyed by
`(my_id, game_id)`. When `obs_d['episode_seed']` was absent (the
kaggle case), the game_id fell back to `hash(initial_planets) % 2^31`.
In a tournament harness that runs multiple games sequentially in one
Python process, that fallback could collide on similar configurations
and let scheduled fires from one game decant inside the next — exactly
the "ships missing targets" symptom observed live.

Fix: `PendingSchedule.begin_turn(fingerprint)` resets the instance
when the fingerprint changes. `commit_persistent` calls this at the
top of every turn with a fingerprint derived from episode_seed (or
the initial planet layout). The module-level singleton stays for
backward compat with existing callers, but the new fingerprint-reset
loop means cross-game state can no longer leak even when the dict is
shared.

Pin tests (Rule 38) — pre-fix these fail; post-fix they pass.
"""

from __future__ import annotations


def _make_fire(*, src_id=0, tgt_id=1, fire_step=10, ships=5):
    from lib.pipeline.pending_schedule import ScheduledFire
    return ScheduledFire(
        src_id=src_id, tgt_id=tgt_id, ships=ships, angle=0.0,
        fire_step=fire_step, committed_at_step=0, wait_N_original=1,
    )


def test_pending_schedule_instance_resets_on_fingerprint_change():
    """A single PendingSchedule instance, seeing two games via
    begin_turn(fingerprint), must wipe state between them."""
    from lib.pipeline.pending_schedule import PendingSchedule

    ps = PendingSchedule()
    ps.begin_turn(fingerprint=("game_A_layout",))
    ps.commit(my_id=0, new_fires=[_make_fire(fire_step=10)])
    assert len(ps.get_pending(0)) == 1, "fire from game A must be retained"

    # Same fingerprint mid-game → no reset.
    ps.begin_turn(fingerprint=("game_A_layout",))
    assert len(ps.get_pending(0)) == 1, (
        "same fingerprint must NOT trigger reset"
    )

    # Different fingerprint → new game.
    ps.begin_turn(fingerprint=("game_B_layout",))
    assert len(ps.get_pending(0)) == 0, (
        "fingerprint change MUST wipe state to prevent cross-game leak"
    )


def test_two_separate_instances_are_isolated():
    """Two PendingSchedule instances do not share state."""
    from lib.pipeline.pending_schedule import PendingSchedule

    a = PendingSchedule()
    b = PendingSchedule()
    a.commit(my_id=0, new_fires=[_make_fire()])
    assert len(a.get_pending(0)) == 1
    assert len(b.get_pending(0)) == 0, (
        "separate PendingSchedule instances must not share state"
    )


def test_commit_persistent_isolates_state_across_back_to_back_games():
    """End-to-end: drive `commit_persistent` against two consecutive
    games (different initial planets) in the SAME process. After the
    second game's first turn, no pending fire from the first game
    should remain.
    """
    from lib.intent import Planet, World
    from lib.joint_solver.columns import Column
    from lib.pipeline.commit_persistent import commit_persistent
    from lib.pipeline.pending_schedule import get_default_pending
    from lib.pipeline.types import DecisionResult, TurnContext
    from lib.world_model import WorldModel

    pending = get_default_pending()
    pending.reset()

    def _make_ctx(*, initial_planets, step_now):
        planets = [
            Planet(id=0, owner=0, x=20.0, y=50.0, radius=5.0, ships=20,
                   production=2),
            Planet(id=1, owner=1, x=80.0, y=50.0, radius=5.0, ships=10,
                   production=2),
        ]
        obs = {
            "player": 0,
            "initial_planets": initial_planets,
            "step": step_now,
        }
        world = World(my_id=0, planets_by_id={p.id: p for p in planets},
                      omega=0.0, comet_ids=frozenset(), step=step_now,
                      obs_raw=obs)
        model = WorldModel(ledger={}, timelines={}, horizon=100)
        return TurnContext(
            obs_d=obs, configuration=None, me=0, num_seats=2,
            step_now=step_now, omega=0.0, planets=planets, fleets=[],
            my_planets=[planets[0]], other_planets=[planets[1]],
            world=world, model=model,
        )

    # Game A — turn 1. The decision schedules one wait_N=5 fire.
    ctx_a = _make_ctx(initial_planets=[[0, 0, 20.0, 50.0, 5.0, 20, 2]],
                      step_now=1)
    fired = [Column(
        column_id=0, src_id=0, tgt_id=1, ships=5, wait_N=5, angle=0.0,
        eta=3, owner=0, value=1.0,
    )]
    decision_a = DecisionResult(
        moves=[], fired_columns=fired, objective=0.0, status="ok",
    )
    res_a = commit_persistent(decision_a, ctx_a)
    assert res_a.persisted_state["n_new_pending"] == 1
    # Confirm pending state was populated.
    assert len(pending.get_pending(0)) == 1, (
        "game A's commit_persistent should have stored 1 fire"
    )

    # Game B — turn 0. DIFFERENT initial_planets → fingerprint changes.
    # Before the fix, game B's first turn would have inherited game A's
    # pending fire when their fingerprint hashes collided.
    ctx_b = _make_ctx(initial_planets=[[0, 0, 30.0, 60.0, 5.0, 25, 3]],
                      step_now=0)
    decision_b = DecisionResult(
        moves=[], fired_columns=[], objective=0.0, status="ok",
    )
    res_b = commit_persistent(decision_b, ctx_b)

    assert len(pending.get_pending(0)) == 0, (
        f"game B should start with empty pending after fingerprint "
        f"change reset; got {pending.get_pending(0)}"
    )
    assert res_b.persisted_state["n_decanted_due"] == 0
