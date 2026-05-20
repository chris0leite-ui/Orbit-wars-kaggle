"""Stage 7 — persistent commit (closes wait_N evaporation).

Alternative to `commit_stateless`. At each turn:

  1. Prune stale pending fires (source lost / not owned).
  2. Decant pending fires whose `fire_step == step_now` — these become
     wait_N=0 emissions this turn.
  3. Add the LP's own wait_N=0 emissions (decision.moves).
  4. Commit the LP's wait_N>0 fired_columns to the pending list for
     future decant.
  5. De-duplicate per-source: if both a decanted fire and an LP fire
     come from the same source, accept both (the LP saw the source's
     current ship count and computed feasibility; the decant uses the
     fixed ship count committed earlier — over-spend risk is on the
     env, which will validate at action-application time).

Game ID is taken from `ctx.obs_d['episode_seed']` if present, else
falls back to a deterministic hash of the initial planet config so
parallel test games don't collide.
"""

from __future__ import annotations

from lib.pipeline.pending_schedule import (
    ScheduledFire,
    commit,
    decant_due,
    prune_past,
    prune_stale,
)
from lib.pipeline.types import CommittedMoves, DecisionResult, TurnContext
from lib.trajectory import predict_fleet_fate


def _game_id(ctx: TurnContext) -> int:
    """Derive a stable game identifier from the observation."""
    obs_d = ctx.obs_d
    # Prefer explicit episode_seed if the env exposes it.
    eid = obs_d.get("episode_seed")
    if eid is not None:
        return int(eid)
    # Fallback: hash of the initial planet config (stable across turns).
    init = obs_d.get("initial_planets") or []
    return abs(hash(tuple(tuple(p) for p in init))) % (2 ** 31)


def commit_persistent(decision: DecisionResult, ctx: TurnContext) -> CommittedMoves:
    """Persistent-schedule Stage-7 implementation."""
    my_id = int(ctx.me)
    game_id = _game_id(ctx)
    step_now = int(ctx.step_now)

    # 1. Prune stale.
    prune_stale(my_id, game_id, ctx.world)
    prune_past(my_id, game_id, step_now)

    # 2. Decant pending fires due this turn.
    decanted = decant_due(my_id, game_id, step_now)
    # 2a. Revalidate each decanted fire against the CURRENT world. The
    # angle/ships were committed at fire_step − wait_N with the world's
    # then-state; by now planets have orbited and other fleets/comets
    # may have moved. predict_fleet_fate(... wait_N=0) tells us whether
    # the stored angle still lands on the intended target. If not, drop
    # the fire — closes the seed-1 "no_target_resolved" regression
    # observed in Phase C validation (5 of 67 emissions misfired).
    decanted_moves: list = []
    n_decant_revalidate_dropped = 0
    for f in decanted:
        src = ctx.world.planets_by_id.get(int(f.src_id))
        tgt = ctx.world.planets_by_id.get(int(f.tgt_id))
        if src is None or tgt is None:
            n_decant_revalidate_dropped += 1
            continue
        # Source ownership: may have flipped since commit; prune_stale
        # already handled this, but double-check here for safety.
        if int(src.owner) != my_id:
            n_decant_revalidate_dropped += 1
            continue
        # Source ship-feasibility against the LIVE garrison.
        if int(src.ships) < int(f.ships):
            n_decant_revalidate_dropped += 1
            continue
        try:
            fate = predict_fleet_fate(
                src, tgt, float(f.angle), int(f.ships), ctx.world, wait_N=0,
            )
        except Exception:
            n_decant_revalidate_dropped += 1
            continue
        if fate is None or fate.outcome != "target":
            n_decant_revalidate_dropped += 1
            continue
        if int(fate.hit_planet_id) != int(f.tgt_id):
            # The trajectory hits some other planet now; not our intent.
            n_decant_revalidate_dropped += 1
            continue
        decanted_moves.append([int(f.src_id), float(f.angle), int(f.ships)])

    # 3. LP's own wait_N=0 emissions this turn.
    lp_moves = list(decision.moves or [])

    # 4. Commit new wait_N>0 fires for future decant.
    new_pending: list[ScheduledFire] = []
    for col in decision.fired_columns:
        w = int(col.wait_N)
        if w <= 0:
            continue
        new_pending.append(ScheduledFire(
            src_id=int(col.src_id),
            tgt_id=int(col.tgt_id),
            ships=int(col.ships),
            angle=float(col.angle),
            fire_step=step_now + w,
            committed_at_step=step_now,
            wait_N_original=w,
        ))
    if new_pending:
        commit(my_id, game_id, new_pending)

    # 5. Concatenate decant + LP. No de-dup; env will validate.
    all_moves = decanted_moves + lp_moves

    return CommittedMoves(
        moves=all_moves,
        persisted_state={
            "n_decanted_due": len(decanted),
            "n_decanted_emitted": len(decanted_moves),
            "n_decant_revalidate_dropped": n_decant_revalidate_dropped,
            "n_new_pending": len(new_pending),
            "game_id": game_id,
        },
    )
