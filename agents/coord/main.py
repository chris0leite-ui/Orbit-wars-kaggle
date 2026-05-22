"""coord — multi-source bundle coordinator (Day 1 scaffold).

Replaces minimal's per-source-then-joint-pair flow with bundle-as-primitive
+ Lagrangian-priced clearing. Day 1 deliverable: data types, attack-bundle
enumeration, unit tests. The agent entry is a placeholder until later days
wire cheap-filter / Tier-2 / clearing / emit.

Design reference: `/root/.claude/plans/eventual-skipping-breeze.md`.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from itertools import combinations

from agents.minimal.main import (
    MIN_FLEET_SIZE,
    MAX_HORIZON,
    SIM_SETTLE_TURNS,
    CHEAP_REJECT_THRESHOLD,
    NUM_TARGETS_PER_SOURCE,
    aim_and_eta,
    cheap_marginal_value,
    enumerate_ship_counts,
    nearest_k,
    wait_then_fire_variants,
    _source_survives_launch,
    _target_holdable_after_capture,
    _target_cost_parity_ok,
)
from lib.trajectory import predict_fleet_fate
from lib.world_model import comet_remaining_lifetime


NEAREST_SOURCES_PER_TARGET = 5
MAX_BUNDLE_SIZE = 3
ARRIVAL_WINDOW_SLACK = 2
DEFEND_LOOKAHEAD = 30


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


def agent(obs, configuration=None):
    """Placeholder — Day 7 wires the full bundle pipeline."""
    raise NotImplementedError(
        "agents.coord: bundle pipeline not yet wired; see plan day 7"
    )
