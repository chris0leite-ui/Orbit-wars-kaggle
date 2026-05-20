"""Differential leaf chooser — closed-form leaf evaluation, no fast_sim.

Plan reference: /root/.claude/plans/take-the-lens-of-magical-shore.md §13.

Replaces `chooser_trajectory.choose_trajectory`'s per-candidate
`fast_sim` rollout with a closed-form WorldModel-projection leaf
state. Same per-candidate signature, same emit shape; different
substrate.

Per candidate:
  1. Augment `model.ledger` with our hypothetical arrival.
  2. Re-run `simulate_planet_timeline` ONLY on the target planet.
  3. Read leaf state at horizon = wait_N + eta + SETTLE.
  4. Compute favor at the leaf via `value.favor`'s formula projected
     onto the (planets, in-flight fleets, our launched fleet) state.
  5. Score = leaf_favor(with action) - leaf_favor(idle baseline).
  6. Greedy emit by Δ desc (1 per source, 1 per target).

No opp policy. No fast_sim. No leaf-favor World rebuild. Cost per
candidate: O(planets × horizon) for the target timeline re-run +
O(planets + fleets) for the favor projection = ~0.5-2 ms typical,
vs trajectory's 5-20 ms.

Opt-in via `BASELINE_CHOOSER=differential` (top-level) or
`BASELINE_INNER_CHOOSER=differential` under the layered chooser.

PI directive: this addresses the diagnosis that Slice 6 made worse
(stacking commits on top of inner = noise). Slice 8 REPLACES the
inner's noisy substrate (`lite_greedy` rollout) with a deterministic
one — same decision authority, cleaner information.
"""

from __future__ import annotations

import math

# Single-line imports below — bundler constraint (see proposer.py:71-76).
# Module-level only; in-function imports break the bundle's indent.
from lib.scoring import pv_horizon
from lib.world_model import simulate_planet_timeline


EPISODE_STEPS: int = 500
DEFAULT_GAMMA: float = 0.99
SETTLE_TURNS: int = 3
# Mirror the 4P-A2 constants from agents.baseline.value so projected
# favor matches the runtime favor function exactly.
ELIMINATION_BONUS: float = 55.0
WEAK_ENEMY_THRESHOLD: float = 110.0
WEAKEST_ENEMY_MULT_4P: float = 1.5
ELIMINATION_GATE_RATIO: float = 0.9
STRENGTH_PROD_WEIGHT: float = 15.0


def _projected_state_at(world, model, me: int, horizon: int, action=None):
    """Return `(ships_by_owner, prod_by_owner)` projected at tick `horizon`.

    `action`, if provided, is `(src, tgt, ships, wait_N, eta)`. The
    target planet's timeline is re-simulated with our arrival
    appended; the source planet's ship count is decremented; if our
    fleet is still in flight at `horizon`, its ships are counted in
    our in-flight total.

    Closed-form: uses `model.timelines` for all planets except the
    target (which gets re-simulated). In-flight fleets are read from
    `model.ledger` — entries with `arr_eta > horizon` are still in
    flight at the leaf.
    """
    ships_by_owner: dict[int, float] = {}
    prod_by_owner: dict[int, float] = {}

    augmented_target_timeline = None
    src_id = -1
    src_launched_ships = 0
    wait_N = 0
    arrival = 0

    if action is not None:
        src, tgt, ships, wait_N, eta = action
        src_id = int(src.id)
        src_launched_ships = int(ships)
        arrival = int(wait_N) + int(eta)
        base_arrivals = list(model.ledger.get(int(tgt.id), []))
        augmented = base_arrivals + [(arrival, int(me), int(ships))]
        tgt_planet = world.planets_by_id[int(tgt.id)]
        augmented_target_timeline = simulate_planet_timeline(
            tgt_planet, augmented, horizon=horizon,
        )

    target_id = int(action[1].id) if action is not None else -1

    for p in world.planets_by_id.values():
        pid = int(p.id)
        if pid == target_id and augmented_target_timeline is not None:
            tl = augmented_target_timeline
        else:
            tl = model.timelines.get(pid)
        if tl is None:
            continue

        owner_at = tl["owner_at"]
        ships_at = tl["ships_at"]
        # Clamp horizon read to timeline horizon (model built with default 250).
        read_h = min(int(horizon), int(tl["horizon"]))
        owner_h = int(owner_at.get(read_h, int(p.owner)))
        ships_h = float(ships_at.get(read_h, float(p.ships)))

        # Source adjustment: subtract the ships we launched. Only if
        # we actually launched by `horizon` (H >= wait_N) AND source
        # still belongs to us in the projection (sanity guard against
        # source captured pre-launch — rare).
        if (action is not None
                and pid == src_id
                and int(horizon) >= int(wait_N)
                and owner_h == int(me)):
            ships_h = max(0.0, ships_h - float(src_launched_ships))

        if owner_h < 0:
            continue
        ships_by_owner[owner_h] = ships_by_owner.get(owner_h, 0.0) + ships_h
        prod_by_owner[owner_h] = prod_by_owner.get(owner_h, 0.0) + float(p.production)

    # In-flight fleets at `horizon`: ledger entries whose arrival is
    # strictly AFTER `horizon` are still in flight at the leaf.
    for planet_id, arrivals in model.ledger.items():
        for (arr_eta, arr_owner, arr_ships) in arrivals:
            if int(arr_eta) > int(horizon) and int(arr_owner) >= 0:
                # Skip our augmented arrival — it's accounted for
                # separately below to avoid double-counting (we built
                # `augmented` only for the target's timeline; the
                # ledger we're iterating here is the ORIGINAL, so our
                # arrival isn't yet here).
                pass  # nothing to skip; original ledger doesn't contain our action
                ships_by_owner[int(arr_owner)] = (
                    ships_by_owner.get(int(arr_owner), 0.0) + float(arr_ships)
                )

    # Our launched fleet: in flight iff wait_N <= H < arrival.
    if (action is not None
            and int(wait_N) <= int(horizon) < int(arrival)):
        ships_by_owner[int(me)] = (
            ships_by_owner.get(int(me), 0.0) + float(src_launched_ships)
        )

    return ships_by_owner, prod_by_owner


def _favor_from_state(ships_by_owner, prod_by_owner, me: int,
                     num_seats: int, leaf_step: int, gamma: float) -> float:
    """Compute favor from projected (ships, prod) dicts.

    Mirrors `agents.baseline.value.favor`'s 2P/4P split exactly so
    Δ-favor between idle and with-action branches is on the same
    scale the trajectory chooser would have used.
    """
    my_ships = float(ships_by_owner.get(int(me), 0.0))
    my_prod = float(prod_by_owner.get(int(me), 0.0))

    opps = sorted(
        o for o in (set(ships_by_owner) | set(prod_by_owner))
        if o != int(me) and o >= 0
    )

    elim_bonus = 0.0
    if int(num_seats) <= 2 or len(opps) < 2:
        opp_ships = max(
            (ships_by_owner.get(o, 0.0) for o in opps), default=0.0,
        )
        opp_prod = max(
            (prod_by_owner.get(o, 0.0) for o in opps), default=0.0,
        )
    else:
        opp_strengths = {
            o: ships_by_owner.get(o, 0.0)
               + prod_by_owner.get(o, 0.0) * STRENGTH_PROD_WEIGHT
            for o in opps
        }
        weakest = min(opps, key=lambda o: opp_strengths[o])
        weakest_str = opp_strengths[weakest]
        opp_ships = sum(
            ships_by_owner.get(o, 0.0)
            * (WEAKEST_ENEMY_MULT_4P if o == weakest else 1.0)
            for o in opps
        )
        opp_prod = sum(
            prod_by_owner.get(o, 0.0)
            * (WEAKEST_ENEMY_MULT_4P if o == weakest else 1.0)
            for o in opps
        )
        my_strength = my_ships + my_prod * STRENGTH_PROD_WEIGHT
        if (weakest_str <= WEAK_ENEMY_THRESHOLD
                and my_strength >= ELIMINATION_GATE_RATIO * weakest_str):
            elim_bonus = ELIMINATION_BONUS

    pv = pv_horizon(int(leaf_step), 0, gamma=float(gamma),
                    t_total=EPISODE_STEPS)
    return (my_ships - opp_ships) + (my_prod - opp_prod) * float(pv) + elim_bonus


def score_candidate_differential(c, world, model, me: int, num_seats: int,
                                  gamma: float = DEFAULT_GAMMA,
                                  settle: int = SETTLE_TURNS) -> float:
    """Closed-form Δ-favor for one candidate.

    Returns `leaf_favor(with action) - leaf_favor(idle baseline)`.
    No fast_sim, no opp policy.
    """
    cheap_delta, src, tgt, ships, angle, eta, horizon_hint, wait_N = c
    arrival = int(wait_N) + int(eta)
    H = arrival + int(settle)

    step = int(getattr(world, "step", 0) or 0)
    leaf_step = step + H

    action = (src, tgt, ships, wait_N, eta)
    ships_with, prod_with = _projected_state_at(
        world, model, int(me), H, action=action,
    )
    favor_with = _favor_from_state(
        ships_with, prod_with, int(me), int(num_seats), leaf_step, gamma,
    )

    ships_idle, prod_idle = _projected_state_at(
        world, model, int(me), H, action=None,
    )
    favor_idle = _favor_from_state(
        ships_idle, prod_idle, int(me), int(num_seats), leaf_step, gamma,
    )

    return favor_with - favor_idle


def choose_differential(snap_base, prerank, baseline_favors,
                        me: int, num_seats: int, wallclock_ms: float,
                        min_horizon: int, max_horizon: int, gamma: float,
                        world, model) -> list:
    """Drop-in replacement for `chooser_trajectory.choose_trajectory`.

    Signature-compatible with the trajectory chooser for use under
    `chooser_layered`'s `_INNER_DISPATCH` mapping. Unused kwargs
    (`snap_base`, `baseline_favors`, `min_horizon`, `max_horizon`,
    `wallclock_ms`) are accepted for parity but ignored — the
    differential scorer has no rollout to budget and no idle
    baseline to subtract (built into the per-candidate Δ).
    """
    if not prerank:
        return []

    # Slice 8c: filter wait_N > 0 before scoring. The differential's
    # closed-form Δ-favor rewards longer wait plans (more production
    # accrues at the leaf) but the chooser's emit logic only fires
    # wait_N == 0. Without this filter, high-Δ wait plans dominate
    # the per-source ranking, lock the source, emit nothing — and
    # the source idles forever. Treating differential as fire-now
    # only: wait plans are handled implicitly by the proposer
    # re-emitting fire-now candidates once the source accumulates
    # enough ships naturally.
    # See audit/2026-05-20-slice8-differential-validation.md for the
    # diagnosis (single-game introspect showed avg 0.75 emits/turn
    # under wait-N pollution; trajectory baseline at ~1.5/turn).
    prerank = [c for c in prerank if int(c[7]) == 0]
    if not prerank:
        return []

    scored: list = []
    for c in prerank:
        cheap_delta, src, tgt, ships, angle, eta, horizon_hint, wait_N = c

        # Slice 9: migration candidate detection. Migration moves are
        # own→own (`tgt.owner == me`) AND don't fire under enemy
        # threat at the target (defensive reinforces have inbound
        # threat — those would be W2's territory). The differential's
        # Δ-favor projection would correctly return 0 for own→own
        # moves (no ownership / production change at the leaf). Instead,
        # use `cheap_delta` (which carries the migration solver's
        # closed-form ΔCapture-EV value) directly as the score.
        is_migration = False
        if int(tgt.owner) == int(me):
            threat_eta = None
            try:
                threat_eta = model.time_to_enemy_threat(
                    int(tgt.id), int(me), world,
                )
            except Exception:
                threat_eta = None
            if threat_eta is None:
                is_migration = True

        if is_migration:
            # The solver pre-computed value lives in cheap_delta.
            # Skip Δ-favor projection (which would be 0 anyway).
            delta = float(cheap_delta)
        else:
            try:
                delta = score_candidate_differential(
                    c, world, model, int(me), int(num_seats), float(gamma),
                )
            except Exception:
                continue
        if delta > 0.0:
            scored.append((delta, src, tgt, ships, angle, wait_N))

    if not scored:
        return []

    scored.sort(key=lambda c: -c[0])

    used_srcs: set = set()
    used_tgts: set = set()
    moves: list = []
    for _delta, src, tgt, ships, angle, wait_N in scored:
        sid, tid = int(src.id), int(tgt.id)
        if sid in used_srcs or tid in used_tgts:
            continue
        used_srcs.add(sid)
        used_tgts.add(tid)
        if int(wait_N) == 0:
            moves.append([sid, float(angle), int(ships)])
        # wait_N>0: reserve src+tgt, emit nothing this turn (matches
        # the trajectory chooser's wait-band semantics).
    return moves
