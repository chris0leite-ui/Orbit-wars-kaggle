"""Tiny scripted JAX opponents for in-training progress evals.

greedy_policy: every turn, each owned planet with a decent garrison
sends 50% at the best-ROI target it can plausibly beat (nearest
neutral/enemy with projected defense < committed ships). Pure-JAX,
same (tgt, frac) action format as the learned policy.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from lib.game.jax.jax_types import GameState, MAX_PLANETS
from rl.features import FRACS, N_FRACS


def greedy_policy(state: GameState, tables, seat):
    """Returns (tgt (P,), frac (P,), src_mask (P,)) for one seat."""
    alive = state.planets_alive
    mine = (state.planets_owner == seat) & alive
    ships = state.planets_ships.astype(jnp.float32)

    aim = tables["aim"]
    eta = aim["eta"]
    ok = aim["valid"] & ~aim["sun_hit"] & ~jnp.eye(MAX_PLANETS, dtype=bool)

    # Send 50% of garrison; commit = what we send.
    commit = jnp.floor(0.5 * ships)  # (P,) per source
    # Defense proj: target ships + prod*eta for owned targets.
    owned_t = (state.planets_owner >= 0) & alive
    def_proj = (ships[None, :]
                + jnp.where(owned_t, state.planets_prod, 0)[None, :]
                .astype(jnp.float32) * eta)
    not_mine_t = alive & (state.planets_owner != seat)
    beatable = (def_proj < commit[:, None] * 0.9) & not_mine_t[None, :] & ok

    # Score: production per eta (ROI-ish); pick argmax per source.
    prod_t = jnp.maximum(state.planets_prod.astype(jnp.float32), 0.5)
    score = prod_t[None, :] / (eta + 5.0)
    score = jnp.where(beatable, score, -1e9)
    best = jnp.argmax(score, axis=1)                       # (P,)
    has_target = jnp.max(score, axis=1) > -1e8

    launch = mine & (ships >= 12.0) & has_target
    tgt = jnp.where(launch, best, MAX_PLANETS)             # hold otherwise
    frac = jnp.full((MAX_PLANETS,), 1, dtype=jnp.int32)    # 50%
    src_mask = mine & (state.planets_ships >= 1)
    return tgt, frac, src_mask


def idle_policy(state: GameState, tables, seat):
    """Never launches."""
    P = MAX_PLANETS
    tgt = jnp.full((P,), P, dtype=jnp.int32)
    frac = jnp.zeros((P,), dtype=jnp.int32)
    mine = (state.planets_owner == seat) & state.planets_alive
    return tgt, frac, mine & (state.planets_ships >= 1)


def producer_lite_policy(state: GameState, tables, seat):
    """Defense-aware expander with reinforcement — a stronger league
    anchor in the spirit of the Producer engine.

    Per owned planet, targets are valued by production/(eta+4) with a
    capture-feasibility gate: committed ships must exceed the target's
    projected defense (garrison + production growth + net inbound
    support) by 15%. When ahead on material it also prices enemy
    planets; planets with no beatable target and a fat garrison relay
    50% to the most threatened own planet (keeps ships at the front).
    """
    alive = state.planets_alive
    mine = (state.planets_owner == seat) & alive
    not_mine = alive & (state.planets_owner != seat)
    enemy = not_mine & (state.planets_owner >= 0)
    ships = state.planets_ships.astype(jnp.float32)

    aim = tables["aim"]
    eta = aim["eta"]
    ok = aim["valid"] & ~aim["sun_hit"] & ~jnp.eye(MAX_PLANETS, dtype=bool)

    commit = jnp.floor(0.6 * ships)
    owned_t = (state.planets_owner >= 0) & alive
    prod_f = state.planets_prod.astype(jnp.float32)
    def_proj = ships[None, :] + jnp.where(owned_t, prod_f, 0.0)[None, :] * eta

    # Net inbound: enemy-of-mine reinforcements strengthen the target.
    arrive = tables["arrive"]  # (P, A, B)
    seat_ids = jnp.arange(4)
    enemy_seats = (seat_ids != seat) & (seat_ids < state.num_agents)
    inb_enemy = jnp.sum(
        jnp.where(enemy_seats[None, :, None], arrive, 0.0), axis=(1, 2))
    inb_mine = jnp.sum(arrive[:, seat, :], axis=-1)
    def_total = def_proj + (inb_enemy - inb_mine)[None, :]

    beatable = (commit[:, None] > def_total * 1.15) & not_mine[None, :] & ok

    # Material posture: when ahead, enemy planets gain value.
    my_mat = jnp.sum(jnp.where(mine, ships, 0.0))
    en_mat = jnp.sum(jnp.where(enemy, ships, 0.0))
    ahead = my_mat > en_mat * 1.1
    val = jnp.maximum(prod_f, 0.5)[None, :] / (eta + 4.0)
    val = jnp.where(enemy[None, :] & ~ahead, val * 0.35, val)
    score = jnp.where(beatable, val, -1e9)
    best = jnp.argmax(score, axis=1)
    has_target = jnp.max(score, axis=1) > -1e8

    # Reinforcement fallback: most-threatened own planet (highest enemy
    # inbound), if any; else hold.
    threat = jnp.where(mine, inb_enemy, -1.0)
    front = jnp.argmax(threat)
    has_front = jnp.max(threat) > 0.0
    relay_ok = ok[:, front] & mine & (ships >= 30.0) & has_front

    launch = mine & (ships >= 10.0) & has_target
    tgt = jnp.where(
        launch, best,
        jnp.where(relay_ok, front, MAX_PLANETS))
    # fraction idx: 2 (75%) for captures, 1 (50%) for relays
    frac = jnp.where(launch, 2, 1).astype(jnp.int32)
    src_mask = mine & (state.planets_ships >= 1)
    return tgt, frac, src_mask


def rusher_policy(state: GameState, tables, seat):
    """All-in early pressure: every owned planet with >= 15 ships fires
    75% at the weakest reachable enemy planet (any defense), preferring
    close ones. Punishes passive openings; teaches wave defense."""
    alive = state.planets_alive
    mine = (state.planets_owner == seat) & alive
    enemy = alive & (state.planets_owner != seat) & (state.planets_owner >= 0)
    neutral = alive & (state.planets_owner == -1)
    ships = state.planets_ships.astype(jnp.float32)

    aim = tables["aim"]
    eta = aim["eta"]
    ok = aim["valid"] & ~aim["sun_hit"] & ~jnp.eye(MAX_PLANETS, dtype=bool)

    # Primary: enemy targets, weighted to weak + close. Early (first 40
    # steps) take cheap neutrals instead to build production.
    early = state.step < 40
    tgt_set = jnp.where(early, neutral, enemy)
    weakness = 1.0 / (ships[None, :] + 10.0)
    score = jnp.where(ok & tgt_set[None, :], weakness / (eta + 2.0), -1e9)
    best = jnp.argmax(score, axis=1)
    has_target = jnp.max(score, axis=1) > -1e8

    launch = mine & (ships >= 15.0) & has_target
    tgt = jnp.where(launch, best, MAX_PLANETS)
    frac = jnp.full((MAX_PLANETS,), 2, dtype=jnp.int32)  # 75%
    src_mask = mine & (state.planets_ships >= 1)
    return tgt, frac, src_mask


# Anchor set for league sampling. producer_lite is excluded: head2head
# showed it LOSES to greedy 2/24 (too passive: 1.15x defense margin +
# self-relay loops) — a weaker anchor would only inject noise. rusher
# beats greedy 21/24 and supplies the sustained-pressure style the
# producer-family losses exposed.
SCRIPTED_POLICIES = {
    "greedy": greedy_policy,
    "rusher": rusher_policy,
}
