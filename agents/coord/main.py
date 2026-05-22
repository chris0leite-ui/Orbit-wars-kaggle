"""coord — multi-source bundle coordinator (Day 1 scaffold).

Replaces minimal's per-source-then-joint-pair flow with bundle-as-primitive
+ Lagrangian-priced clearing. Day 1 deliverable: data types, attack-bundle
enumeration, unit tests. The agent entry is a placeholder until later days
wire cheap-filter / Tier-2 / clearing / emit.

Design reference: `/root/.claude/plans/eventual-skipping-breeze.md`.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace
from enum import Enum
from itertools import combinations

from agents.minimal.main import (
    EPISODE_STEPS,
    GAMMA,
    MIN_FLEET_SIZE,
    MAX_HORIZON,
    SIM_SETTLE_TURNS,
    CHEAP_REJECT_THRESHOLD,
    NUM_TARGETS_PER_SOURCE,
    aim_and_eta,
    cheap_marginal_value,
    enumerate_ship_counts,
    favor_hybrid,
    nearest_k,
    wait_then_fire_variants,
    _source_survives_launch,
    _target_holdable_after_capture,
    _target_cost_parity_ok,
)
from lib.scoring import pv_horizon
from lib.trajectory import predict_fleet_fate
from lib.world_model import WAVE_LOOKAHEAD, comet_remaining_lifetime


NEAREST_SOURCES_PER_TARGET = 5
MAX_BUNDLE_SIZE = 3
ARRIVAL_WINDOW_SLACK = 2
DEFEND_LOOKAHEAD = 30

# Cheap-filter — opportunity cost penalty per committed ship (small;
# the leaf-Δ from favor_hybrid dominates the score, this just breaks
# ties against bundles that needlessly drain large garrisons for low gain).
CHEAP_OPPORTUNITY_COST = 0.01
# Top-K admitted to Tier-2. Calibrated from Day 4 probe — K=50 gave
# 89% attack rank-1 retention; K=75 widens to ~97% by admitting bundles
# whose cheap_score is within ~20% of the top-50 boundary. Tier-2 cost
# 75 × ~5ms = 375ms, still under the 600ms agent budget when paired
# with the existing safe_deadline pre-bail watchdog.
CHEAP_FILTER_TOP_K = 75


class BundleKind(Enum):
    ATTACK = "attack"
    DEFEND = "defend"
    RECAPTURE = "recap"


@dataclass(frozen=True)
class Leg:
    src_id: int
    ships: int
    angle: float
    wait_N: int
    eta: int

    @property
    def arrival_step(self) -> int:
        return self.wait_N + self.eta


@dataclass(frozen=True)
class Bundle:
    target_id: int
    arrival_step: int
    legs: tuple[Leg, ...]
    kind: BundleKind
    cheap_score: float = 0.0
    tier2_score: float = 0.0


def _admissible_fire_now(src, tgt, angle: float, ships: int, world) -> bool:
    """Fire-now physics admissibility — must hit target, comet not expired
    before arrival. Mirrors the same filters minimal's propose() applies at
    L797-L804.
    """
    fate = predict_fleet_fate(src, tgt, float(angle), int(ships), world)
    if fate.outcome != "target":
        return False
    if int(tgt.id) in world.comet_ids:
        life = comet_remaining_lifetime(int(tgt.id), world)
        if life is None or life <= int(fate.step):
            return False
    return True


def _legs_for_pair(src, tgt, world, model, me: int, omega: float,
                   baseline_len: int) -> list[Leg]:
    """Build all candidate legs for one (src, tgt) — ship-count variants
    + wait-grid variants, with per-leg filters applied. Reuses minimal's
    helpers verbatim.
    """
    legs: list[Leg] = []

    # Fire-now ship-count variants.
    for ships in enumerate_ship_counts(src, tgt, model, omega, me, world):
        if ships < MIN_FLEET_SIZE or ships > int(src.ships):
            continue
        angle, eta = aim_and_eta(src, tgt, ships, omega, world=world)
        horizon = max(eta + SIM_SETTLE_TURNS, 1)
        if horizon >= baseline_len:
            continue
        cheap = cheap_marginal_value(
            src, tgt, ships, eta, world, model, me, wait_N=0,
        )
        if cheap <= CHEAP_REJECT_THRESHOLD:
            continue
        if not _admissible_fire_now(src, tgt, angle, ships, world):
            continue
        if not _source_survives_launch(src, int(ships), 0, world, model, me):
            continue
        if not _target_holdable_after_capture(
            src, tgt, int(ships), 0, int(eta), world, model, me,
        ):
            continue
        if not _target_cost_parity_ok(
            src, tgt, int(ships), 0, int(eta), world, model, me,
        ):
            continue
        legs.append(Leg(
            src_id=int(src.id), ships=int(ships),
            angle=float(angle), wait_N=0, eta=int(eta),
        ))

    # Wait-grid variants. Skip admissibility filter on wait_N>0 (geometry
    # shifts at launch time; fast_sim's collision resolution catches it
    # downstream — same convention as minimal's propose L794-L795).
    for w_ships, w_wait, w_angle, w_eta in wait_then_fire_variants(
        src, tgt, model, omega, me, world=world,
    ):
        w_horizon = max(w_wait + w_eta + SIM_SETTLE_TURNS, 1)
        if w_horizon >= baseline_len:
            continue
        w_cheap = cheap_marginal_value(
            src, tgt, w_ships, w_eta, world, model, me, wait_N=w_wait,
        )
        if w_cheap <= CHEAP_REJECT_THRESHOLD:
            continue
        if not _source_survives_launch(
            src, int(w_ships), int(w_wait), world, model, me,
        ):
            continue
        if not _target_holdable_after_capture(
            src, tgt, int(w_ships), int(w_wait), int(w_eta),
            world, model, me,
        ):
            continue
        if not _target_cost_parity_ok(
            src, tgt, int(w_ships), int(w_wait), int(w_eta),
            world, model, me,
        ):
            continue
        legs.append(Leg(
            src_id=int(src.id), ships=int(w_ships),
            angle=float(w_angle), wait_N=int(w_wait), eta=int(w_eta),
        ))

    return legs


def _cluster_arrival_windows(legs: list[Leg],
                             slack: int = ARRIVAL_WINDOW_SLACK) -> list[list[Leg]]:
    """For each leg as anchor, return the set of legs whose arrival_step is
    in [anchor.arrival_step, anchor.arrival_step + slack].

    Over-enumerates (one window per anchor) so no admissible bundle is
    missed; dedup happens at the bundle-emission step via frozenset of leg
    identities.
    """
    if not legs:
        return []
    legs_sorted = sorted(legs, key=lambda L: L.arrival_step)
    windows: list[list[Leg]] = []
    for i, anchor in enumerate(legs_sorted):
        upper = anchor.arrival_step + slack
        window: list[Leg] = []
        for j in range(i, len(legs_sorted)):
            if legs_sorted[j].arrival_step > upper:
                break
            window.append(legs_sorted[j])
        if window:
            windows.append(window)
    return windows


def _emit_subsets(window: list[Leg], max_size: int) -> list[tuple[Leg, ...]]:
    """All subsets of `window` of size 1..max_size with NO source repeats."""
    out: list[tuple[Leg, ...]] = []
    for r in range(1, max_size + 1):
        for subset in combinations(window, r):
            seen_srcs = set()
            ok = True
            for leg in subset:
                if leg.src_id in seen_srcs:
                    ok = False
                    break
                seen_srcs.add(leg.src_id)
            if ok:
                out.append(subset)
    return out


def enumerate_attack_bundles(my_planets, target_pool, world, model,
                             me: int, omega: float,
                             baseline_len: int = MAX_HORIZON + 1
                             ) -> list[Bundle]:
    """Enumerate multi-source ATTACK bundles for each non-own target.

    For each target × arrival-window × subset-of-nearest-N-reachable-sources
    (size 1..MAX_BUNDLE_SIZE):

    1. Build candidate legs via minimal's per-source-per-target machinery
       (ship-count + wait-grid + per-leg filters).
    2. Cluster legs by arrival_step into windows of width
       ARRIVAL_WINDOW_SLACK.
    3. Per window, enumerate subsets where no source repeats.
    4. Emit one Bundle per unique subset (frozenset-of-legs dedup).

    cheap_score and tier2_score are zeroed; populated by later passes.
    """
    legs_by_target: dict[int, list[Leg]] = defaultdict(list)
    for tgt in target_pool:
        if int(tgt.owner) == me:
            continue  # ATTACK targets non-own only
        # Limit to nearest-N reachable sources per target.
        sources = nearest_k(my_planets, tgt, NEAREST_SOURCES_PER_TARGET)
        for src in sources:
            if int(src.ships) < MIN_FLEET_SIZE:
                continue
            if int(src.id) == int(tgt.id):
                continue
            legs_by_target[int(tgt.id)].extend(
                _legs_for_pair(src, tgt, world, model, me, omega, baseline_len),
            )

    seen: set[tuple[int, frozenset]] = set()
    bundles: list[Bundle] = []
    for tgt_id, legs in legs_by_target.items():
        if not legs:
            continue
        windows = _cluster_arrival_windows(legs, slack=ARRIVAL_WINDOW_SLACK)
        for window in windows:
            for subset in _emit_subsets(window, MAX_BUNDLE_SIZE):
                key_legs = frozenset(
                    (L.src_id, L.ships, L.wait_N, L.eta) for L in subset
                )
                key = (tgt_id, key_legs)
                if key in seen:
                    continue
                seen.add(key)
                arrival_step = max(L.arrival_step for L in subset)
                bundles.append(Bundle(
                    target_id=tgt_id,
                    arrival_step=arrival_step,
                    legs=tuple(subset),
                    kind=BundleKind.ATTACK,
                ))
    return bundles


def enumerate_defend_bundles(my_planets, world, model, me: int, omega: float,
                             baseline_len: int = MAX_HORIZON + 1
                             ) -> list[Bundle]:
    """Enumerate multi-source DEFEND bundles for each own planet under threat.

    `time_to_enemy_threat` (not `incoming_enemy_eta`) captures BOTH in-flight
    enemy fleets AND potential launches from nearby enemy planets — matches
    the threat surface that minimal's `threatened_mine` filter uses
    (agents/minimal/main.py L1353-1356). `incoming_enemy_eta` alone would
    miss preemptive defense.

    For each threatened own planet:
    1. Find peer source planets via nearest-N.
    2. Build candidate legs via Day-1 `_legs_for_pair` — reuses every
       per-leg filter including `_source_survives_launch` (so we don't
       drain a peer that's itself threatened).
    3. Hard-filter legs whose arrival_step >= enemy_eta (too slow).
    4. Enumerate subsets size 1..MAX_BUNDLE_SIZE with no source repeats.

    NO arrival-window clustering — same-owner reinforcements arriving on
    DIFFERENT turns add to garrison additively. The only constraint is
    `arrival_step < enemy_eta`, enforced per-leg.

    cheap_score and tier2_score are zeroed; populated by later passes.
    """
    bundles: list[Bundle] = []
    seen: set[tuple[int, frozenset]] = set()
    for own in my_planets:
        enemy_eta = model.time_to_enemy_threat(int(own.id), me, world)
        if enemy_eta is None or enemy_eta > DEFEND_LOOKAHEAD:
            continue
        peers = nearest_k(my_planets, own, NEAREST_SOURCES_PER_TARGET)
        all_legs: list[Leg] = []
        for peer in peers:
            if int(peer.id) == int(own.id):
                continue
            if int(peer.ships) < MIN_FLEET_SIZE:
                continue
            pair_legs = _legs_for_pair(peer, own, world, model, me, omega,
                                       baseline_len)
            for leg in pair_legs:
                if leg.arrival_step >= enemy_eta:
                    continue
                all_legs.append(leg)
        if not all_legs:
            continue
        for subset in _emit_subsets(all_legs, MAX_BUNDLE_SIZE):
            key_legs = frozenset(
                (L.src_id, L.ships, L.wait_N, L.eta) for L in subset
            )
            key = (int(own.id), key_legs)
            if key in seen:
                continue
            seen.add(key)
            arrival_step = max(L.arrival_step for L in subset)
            bundles.append(Bundle(
                target_id=int(own.id),
                arrival_step=arrival_step,
                legs=tuple(subset),
                kind=BundleKind.DEFEND,
            ))
    return bundles


def enumerate_recapture_bundles(my_planets, target_pool, world, model,
                                me: int, omega: float,
                                baseline_len: int = MAX_HORIZON + 1
                                ) -> list[Bundle]:
    """v1 stub — recapture deferred to v2.

    "Recently lost" detection requires inter-turn state (the previous-turn
    owner of each planet), which the stateless agent doesn't track.
    `WorldModel.owner_at` returns FUTURE predictions, not past history.

    The high-value cheap targets that recapture would prioritise —
    currently enemy/neutral planets with low garrison and high production
    — are ALREADY enumerated by `enumerate_attack_bundles`. The
    kind=RECAPTURE label would be metadata-only; bundle composition would
    not change.

    PI-approved deferral 2026-05-22. Revisit in v2 if recapture-priority
    scoring turns out to add value, at which point a small module-level
    dict cache keyed by planet_id storing previous-turn-owner can be
    added with bounded blast radius.
    """
    return []


# ---------------------------------------------------------------------------
# Cheap-filter — synthesised-obs Δ-favor without rollout.
# ---------------------------------------------------------------------------

def _resolve_target_post_bundle(bundle: "Bundle", target_planet,
                                 pred_owner, pred_ships, me: int
                                 ) -> tuple[int, int]:
    """Apply combat-rule-1 between the bundle and the predicted target
    state at bundle.arrival_step. Returns (post_owner, post_ships).

    For ATTACK / RECAPTURE: if bundle force > predicted defender,
    capture; else defender absorbs and we lose the bundle's ships.

    For DEFEND: my reinforcement ADDS to garrison if predicted owner
    at arrival is still me. If we'd already lost the planet by arrival,
    treat as recapture (combat rule 1 vs the new owner's garrison).
    """
    bundle_force = sum(L.ships for L in bundle.legs)
    pred_owner_val = (
        pred_owner if pred_owner is not None else int(target_planet.owner)
    )
    pred_ships_val = int(
        pred_ships if pred_ships is not None else target_planet.ships
    )

    if bundle.kind == BundleKind.DEFEND:
        if pred_owner_val == me:
            return me, pred_ships_val + bundle_force
        # Already lost by arrival_step (rare given DEFEND deadline filter,
        # but possible if threat ETA is very tight) — handle as recapture.
        if bundle_force > pred_ships_val:
            return me, bundle_force - pred_ships_val
        return pred_owner_val, pred_ships_val - bundle_force

    # ATTACK / RECAPTURE
    if pred_owner_val == me:
        return me, pred_ships_val + bundle_force
    if bundle_force > pred_ships_val:
        return me, bundle_force - pred_ships_val
    return pred_owner_val, pred_ships_val - bundle_force


def _cheap_synth_step(bundle: "Bundle", world, model, me: int) -> int:
    """Choose the step at which to synthesise for the cheap-filter.

    For ATTACK / RECAPTURE: bundle.arrival_step (post-capture state).

    For DEFEND: max(arrival_step, enemy_eta + 1). The enemy's hit at
    `enemy_eta` resolves combat; synthesising at `enemy_eta + 1` lets
    `model.owner_at` show the post-combat ownership (likely enemy if no
    reinforcement) so `_resolve_target_post_bundle` correctly computes
    "did my reinforcement flip the outcome?". Synthesising at
    `arrival_step` instead misses this — at arrival_step the planet is
    still ours and the bundle is just garrison-padding, giving DEFEND
    an under-priced score relative to ATTACK.
    """
    if bundle.kind == BundleKind.DEFEND:
        enemy_eta = model.time_to_enemy_threat(
            int(bundle.target_id), me, world,
        )
        if enemy_eta is not None and enemy_eta >= 0:
            return max(int(bundle.arrival_step), int(enemy_eta) + 1)
    return int(bundle.arrival_step)


def _synthesise_obs_at_step(world, model, me: int, step_offset: int,
                             target_id: int | None = None,
                             target_override: tuple[int, int] | None = None,
                             source_subtractions: dict[int, int] | None = None,
                             ) -> dict:
    """Build a synthesised obs dict at `step_offset` from now.

    With both `target_override` and `source_subtractions` None, this is
    the IDLE projection — what the world would look like at step_offset
    if nobody acted.

    With them set, it reflects a bundle's effect: target planet uses
    `target_override` (owner, ships); each source planet has its idle
    projection reduced by `source_subtractions[src_id]` committed ships.
    """
    if source_subtractions is None:
        source_subtractions = {}

    planets_synth: list[tuple] = []
    for pid, p in world.planets_by_id.items():
        pred_owner = model.owner_at(pid, step_offset)
        pred_ships = model.ships_at(pid, step_offset)
        if pid == target_id and target_override is not None:
            owner = int(target_override[0])
            ships = max(0, int(target_override[1]))
        elif pid in source_subtractions:
            base_owner = (
                int(pred_owner) if pred_owner is not None else int(p.owner)
            )
            base_ships = (
                int(pred_ships) if pred_ships is not None else int(p.ships)
            )
            owner = base_owner
            ships = max(0, base_ships - source_subtractions[pid])
        else:
            owner = (
                int(pred_owner) if pred_owner is not None else int(p.owner)
            )
            ships = max(
                0, int(pred_ships) if pred_ships is not None else int(p.ships),
            )
        planets_synth.append((
            int(pid), owner, float(p.x), float(p.y), float(p.radius),
            int(ships), int(p.production),
        ))

    return {
        "player": int(me),
        "step": int(world.step) + step_offset,
        "planets": planets_synth,
        "fleets": [],
        "angular_velocity": float(getattr(world, "omega", 0.0)),
        "comet_planet_ids": list(getattr(world, "comet_ids", [])),
    }


def _synthesise_post_arrival_obs(bundle: "Bundle", world, model,
                                  me: int,
                                  step_offset: int | None = None) -> dict:
    """Synthesised obs reflecting the bundle's effect at the cheap-filter
    synthesis step (default: `_cheap_synth_step(bundle, ...)`).

    Used by `_bundle_cheap_delta`; thin wrapper over
    `_synthesise_obs_at_step` that applies the bundle's combat resolution
    on the target and subtracts committed ships from sources.
    """
    if step_offset is None:
        step_offset = _cheap_synth_step(bundle, world, model, me)

    sources_committed: dict[int, int] = defaultdict(int)
    for leg in bundle.legs:
        sources_committed[int(leg.src_id)] += int(leg.ships)

    target_p = world.planets_by_id.get(int(bundle.target_id))
    if target_p is None:
        # Defensive: synthesise idle if target missing.
        return _synthesise_obs_at_step(world, model, me, step_offset)
    pred_owner = model.owner_at(int(bundle.target_id), step_offset)
    pred_ships = model.ships_at(int(bundle.target_id), step_offset)
    post_owner, post_ships = _resolve_target_post_bundle(
        bundle, target_p, pred_owner, pred_ships, me,
    )

    return _synthesise_obs_at_step(
        world, model, me, step_offset,
        target_id=int(bundle.target_id),
        target_override=(int(post_owner), int(post_ships)),
        source_subtractions=dict(sources_committed),
    )


def _production_balance_pv(planets: list, me: int, step: int,
                            horizon: int = MAX_HORIZON) -> float:
    """PV-discounted (my_prod − opp_prod) × pv_horizon over the rollout
    look-ahead. Mirrors `composite_capture_value`'s gated PV term so the
    cheap filter sees the future value of producing planets WITHOUT
    relying on the rollout (Tier-2 captures it via K-step sim; the
    cheap filter is rollout-free).

    The default composite head gates this term behind
    COMPOSITE_PRODUCTION_PV=1 (off in production). The cheap filter adds
    it explicitly because without it, capturing equal-production planets
    yields favor_hybrid Δ ≈ 0, indistinguishable from defender-erosion
    attacks.
    """
    my_prod = 0.0
    opp_prod = 0.0
    for p in planets:
        owner = int(p[1])
        prod = float(p[6])
        if owner == me:
            my_prod += prod
        elif owner >= 0:
            opp_prod += prod
    pv = pv_horizon(int(step), 0, gamma=GAMMA, t_total=EPISODE_STEPS)
    # Truncate the PV to the rollout horizon — Tier-2 sees ~MAX_HORIZON
    # turns of production; the cheap filter should approximate the same
    # truncated value, not the full 99-turn pv. This keeps cheap and
    # Tier-2 scores in similar magnitude ranges.
    truncated_pv = min(pv, float(horizon))
    return (my_prod - opp_prod) * truncated_pv


def _estimate_threat_strength(target_id: int, enemy_eta: int,
                               world, model, me: int) -> float:
    """Estimate the enemy ship force impacting `target_id` at `enemy_eta`,
    accounting for BOTH in-flight enemy fleets AND potential launches
    from nearby enemy planets. Mirrors minimal's `capture_size` own-target
    branch (agents/minimal/main.py L242-263) so the cheap-filter values
    defense by the same threat surface that triggered enumeration.
    """
    enemy_inflight = sum(
        sh
        for (eta_arr, owner, sh) in model.ledger.get(int(target_id), [])
        if owner != me and eta_arr <= int(enemy_eta) + WAVE_LOOKAHEAD
    )
    enemy_potential = 0.0
    if enemy_inflight <= 0:
        best_enemy_ships = 0.0
        best_enemy_prod = 0.0
        for p in world.planets_by_id.values():
            if int(p.owner) < 0 or int(p.owner) == me:
                continue
            if int(p.ships) > best_enemy_ships:
                best_enemy_ships = float(p.ships)
                best_enemy_prod = float(p.production)
        enemy_potential = best_enemy_ships + best_enemy_prod * float(enemy_eta)
    return float(max(enemy_inflight, enemy_potential))


def _defend_cheap_delta(bundle: "Bundle", world, model, me: int) -> float:
    """Cheap Δ for DEFEND bundles. Uses minimal's threat-strength
    formulation (in-flight + potential enemy launch) rather than the
    synthesised-obs approach used for ATTACK.

    Rationale: `WorldModel` only simulates in-flight fleets. For
    preemptive defense (no enemy fleet airborne yet, just an adjacent
    enemy planet), the model's idle projection still shows me owning
    the planet at enemy_eta + 1, so the synthesised-obs approach
    under-prices the bundle to ~0. The explicit threat-strength
    calculation matches what triggered the bundle's enumeration in the
    first place (`time_to_enemy_threat`) and credits the bundle with
    PV-discounted production preservation IF it covers the shortfall.

    Value model (for the cheap-filter only — Tier-2 does the precise
    leaf-Δ via a real rollout):
      - shortfall = enemy_strength − (target.ships + target.prod × eta)
      - if bundle covers shortfall: + target.production × truncated PV
      - if partial coverage: scaled fraction of full save value
      - minus epsilon × total ships committed
    """
    target_id = int(bundle.target_id)
    target_p = world.planets_by_id.get(target_id)
    if target_p is None:
        return -float("inf")

    enemy_eta = model.time_to_enemy_threat(target_id, me, world)
    if enemy_eta is None:
        return 0.0

    enemy_strength = _estimate_threat_strength(
        target_id, enemy_eta, world, model, me,
    )
    my_garrison_at_eta = (
        float(target_p.ships) + float(target_p.production) * float(enemy_eta)
    )
    shortfall = max(0.0, enemy_strength - my_garrison_at_eta + 1.0)

    bundle_force = float(sum(L.ships for L in bundle.legs))
    coverage = (
        1.0 if shortfall <= 0
        else min(1.0, bundle_force / shortfall)
    )

    pv_full = pv_horizon(
        int(world.step), int(enemy_eta),
        gamma=GAMMA, t_total=EPISODE_STEPS,
    )
    truncated_pv = min(pv_full, float(MAX_HORIZON))
    save_value = coverage * float(target_p.production) * truncated_pv

    return save_value - CHEAP_OPPORTUNITY_COST * bundle_force


def _attack_cheap_delta(bundle: "Bundle", world, model, me: int,
                        num_seats: int) -> float:
    """Cheap Δ for ATTACK / RECAPTURE bundles via synthesised-obs.

    Δ = (favor_with_bundle + prod_pv_with_bundle)
      − (favor_idle      + prod_pv_idle)
      − CHEAP_OPPORTUNITY_COST × total ships committed

    Both synthesis points use the same step (`bundle.arrival_step`) so
    time-passing effects cancel; the residual is the bundle's effect.
    Production-PV is added explicitly because favor_hybrid's PV term is
    gated off in production.
    """
    step_offset = _cheap_synth_step(bundle, world, model, me)

    idle_obs = _synthesise_obs_at_step(world, model, me, step_offset)
    idle_favor = favor_hybrid(idle_obs, me, num_seats, gamma=GAMMA)
    idle_pv = _production_balance_pv(
        idle_obs["planets"], me, int(idle_obs["step"]),
    )

    bundle_obs = _synthesise_post_arrival_obs(
        bundle, world, model, me, step_offset=step_offset,
    )
    bundle_favor = favor_hybrid(bundle_obs, me, num_seats, gamma=GAMMA)
    bundle_pv = _production_balance_pv(
        bundle_obs["planets"], me, int(bundle_obs["step"]),
    )

    total_ships = sum(L.ships for L in bundle.legs)
    return (
        (bundle_favor + bundle_pv) - (idle_favor + idle_pv)
        - CHEAP_OPPORTUNITY_COST * float(total_ships)
    )


def _bundle_cheap_delta(bundle: "Bundle", world, model, me: int,
                        num_seats: int,
                        current_favor: float | None = None) -> float:
    """Cheap Δ-favor dispatcher (rollout-free). DEFEND bundles use the
    explicit threat-strength formulation; ATTACK / RECAPTURE use the
    synthesised-obs formulation. The `current_favor` parameter is kept
    for backward compatibility with earlier signatures; both paths
    compare against idle-at-same-step internally.
    """
    if bundle.kind == BundleKind.DEFEND:
        return _defend_cheap_delta(bundle, world, model, me)
    return _attack_cheap_delta(bundle, world, model, me, num_seats)


def cheap_filter_bundles(bundles: list[Bundle], world, model, me: int,
                          num_seats: int, K: int = CHEAP_FILTER_TOP_K
                          ) -> list[Bundle]:
    """Score every bundle by `_bundle_cheap_delta`, return top-K by score.

    Returns bundles with `cheap_score` field populated. tier2_score
    remains zero until the Day-5 Tier-2 pass.
    """
    if not bundles:
        return []
    scored: list[Bundle] = []
    for b in bundles:
        score = _bundle_cheap_delta(b, world, model, me, num_seats)
        scored.append(replace(b, cheap_score=score))
    scored.sort(key=lambda b: -b.cheap_score)
    return scored[:K]


def agent(obs, configuration=None):
    """Placeholder — Day 7 wires the full bundle pipeline."""
    raise NotImplementedError(
        "agents.coord: bundle pipeline not yet wired; see plan day 7"
    )
