"""Stage 7 — persistent commit (closes wait_N evaporation).

Alternative to `commit_stateless`. At each turn:

  0. (Bug #1) Tell the pending-schedule container what game we're in;
     it resets state if the fingerprint changes (new game in the same
     Python process — the tournament-harness scenario).
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
"""

from __future__ import annotations

import math

from lib.pipeline.pending_schedule import (
    ScheduledFire,
    commit,
    decant_due,
    prune_past,
    prune_stale,
    get_default_pending,
)
from lib.pipeline.types import CommittedMoves, DecisionResult, TurnContext
from lib.trajectory import predict_fleet_fate
from lib.world_model import comet_remaining_lifetime


# Comet drain (post-LP override). The LP rejects comet-as-source
# launches whose targets can't be captured with the comet's ships, so
# captured comets often idle until their path expires and their ships
# are forfeit. When remaining lifetime is small AND we still hold the
# comet AND there's a reachable own planet, force-drain the remaining
# ships (the alternative is losing them entirely).
COMET_DRAIN_LIFETIME_THRESHOLD = 15  # turns
COMET_DRAIN_MIN_SHIPS = 5
COMET_DRAIN_RESERVE = 1  # leave 1 ship behind (env keeps the drain feasible
                          # against any other inbound on the comet itself)


def _emit_comet_drains(
    ctx: TurnContext, fired_columns: list, decanted_moves: list,
) -> list[list]:
    """Post-LP override: drain my captured comets whose path is about
    to end. The LP would otherwise leave the ships on the comet to be
    forfeit at expiry.

    Strategy: for each comet I own with remaining_lifetime ≤
    COMET_DRAIN_LIFETIME_THRESHOLD, compute the ships not already
    committed by LP wait_N=0 launches or decanted moves from this
    comet, then send them all (minus a reserve) to the nearest own
    non-comet planet. Trajectory validated with predict_fleet_fate.

    Returns a list of `[src_id, angle, ships]` moves to append to the
    turn's emission.
    """
    me = int(ctx.me)
    world = ctx.world
    comet_ids = world.comet_ids
    if not comet_ids:
        return []
    # Own non-comet planets are the drain destinations.
    own_non_comet = [
        p for p in ctx.my_planets
        if int(p.id) not in comet_ids
    ]
    if not own_non_comet:
        return []

    # Ships committed-from-each-comet this turn (LP wait_N=0 fires + decants).
    committed_by_src: dict[int, int] = {}
    for col in (fired_columns or []):
        if int(col.wait_N) != 0:
            continue
        if int(col.owner) != me:
            continue
        committed_by_src[int(col.src_id)] = (
            committed_by_src.get(int(col.src_id), 0) + int(col.ships)
        )
    for mv in (decanted_moves or []):
        committed_by_src[int(mv[0])] = (
            committed_by_src.get(int(mv[0]), 0) + int(mv[2])
        )

    drain_moves: list[list] = []
    for p in ctx.my_planets:
        if int(p.id) not in comet_ids:
            continue
        life = comet_remaining_lifetime(int(p.id), world)
        if life is None or life > COMET_DRAIN_LIFETIME_THRESHOLD:
            continue
        committed = committed_by_src.get(int(p.id), 0)
        available = int(p.ships) - int(committed) - COMET_DRAIN_RESERVE
        if available < COMET_DRAIN_MIN_SHIPS:
            continue
        # Rank own non-comet planets by distance and try each. A comet
        # is moving fast (comet_speed ~ 4 units/step); the trajectory
        # toward the nearest planet may immediately collide with the
        # comet itself or with another planet en route. Try the K
        # closest candidates and pick the first viable one.
        ranked = sorted(
            own_non_comet,
            key=lambda t: (
                (float(p.x) - float(t.x)) ** 2
                + (float(p.y) - float(t.y)) ** 2
            ),
        )
        emitted = False
        for tgt in ranked[:6]:
            angle = math.atan2(
                float(tgt.y) - float(p.y),
                float(tgt.x) - float(p.x),
            )
            try:
                fate = predict_fleet_fate(
                    p, tgt, float(angle), int(available), world, wait_N=0,
                )
            except Exception:
                continue
            if fate is None or fate.outcome != "target":
                continue
            if int(fate.hit_planet_id) != int(tgt.id):
                continue
            drain_moves.append([int(p.id), float(angle), int(available)])
            emitted = True
            break
        if not emitted:
            continue

    return drain_moves


def _game_fingerprint(ctx: TurnContext):
    """Derive a stable per-game fingerprint from the observation.

    Used by `PendingSchedule.begin_turn` to detect new games and reset
    state. `episode_seed` is the strongest signal; the initial planet
    layout is a stable fallback (it is identical across all turns of a
    game and differs between games with different seeds).
    """
    obs_d = ctx.obs_d
    eid = obs_d.get("episode_seed")
    if eid is not None:
        return ("eid", int(eid))
    init = obs_d.get("initial_planets") or []
    return ("init", tuple(tuple(p) for p in init))


def commit_persistent(decision: DecisionResult, ctx: TurnContext) -> CommittedMoves:
    """Persistent-schedule Stage-7 implementation."""
    my_id = int(ctx.me)
    step_now = int(ctx.step_now)

    # 0. Game-boundary detection (Bug #1): the default pending-schedule
    # singleton is shared across all games in this Python process. If
    # this turn belongs to a new game (fingerprint changed), wipe.
    pending = get_default_pending()
    pending.begin_turn(_game_fingerprint(ctx))

    # game_id retained for telemetry only (NOT used for state keying).
    game_id_for_telemetry = abs(hash(_game_fingerprint(ctx))) % (2 ** 31)

    # 1. Prune stale.
    prune_stale(my_id, game_id_for_telemetry, ctx.world)
    prune_past(my_id, game_id_for_telemetry, step_now)

    # 2. Decant pending fires due this turn.
    decanted = decant_due(my_id, game_id_for_telemetry, step_now)
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
        commit(my_id, game_id_for_telemetry, new_pending)

    # 5. Concatenate decant + LP. No de-dup; env will validate.
    all_moves = decanted_moves + lp_moves

    # 6. Comet-drain override (post-LP). The LP doesn't model comet-
    # expiry forfeiture; without this, ships idle on captured comets
    # until the path ends and get lost. See `_emit_comet_drains`.
    drain_moves = _emit_comet_drains(
        ctx, decision.fired_columns, decanted_moves,
    )
    all_moves = all_moves + drain_moves

    return CommittedMoves(
        moves=all_moves,
        persisted_state={
            "n_decanted_due": len(decanted),
            "n_decanted_emitted": len(decanted_moves),
            "n_decant_revalidate_dropped": n_decant_revalidate_dropped,
            "n_new_pending": len(new_pending),
            "n_comet_drains": len(drain_moves),
            "game_id": game_id_for_telemetry,
        },
    )
