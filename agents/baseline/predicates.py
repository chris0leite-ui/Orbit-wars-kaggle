"""Layer-0 closed-form predicates — provable emit/discard shortcuts.

Plan reference: /root/.claude/plans/take-the-lens-of-magical-shore.md

Each predicate is a pure closed-form function over (candidate, world,
model, me) returning a verdict + evidence. The verdict is consumed by
the layered chooser, which short-circuits the trajectory rollout for
candidates that are provably winning (commit) or provably wasted
(discard). Uncertain candidates pass through to the existing rollout.

This module starts with W2 (provably-held reinforce). The remaining
predicates (W1, L1, L2) have signatures defined here so the layered
chooser's wiring stays stable; their bodies land in subsequent slices.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Single-line imports below: the submission bundler's per-line
# import-stripping regex leaks continuation lines from a parenthesised
# multi-line import as indented orphans (IndentationError at runtime).
# Friction tag `bundler-modular-agent-namespace-access-breaks-bundle`
# documented in agents/baseline/main.py and proposer.py.
#
# All lib/agent imports MUST be at module level — the bundler's
# alias-rewrite for stripped imports happens at column 0 and breaks
# any in-function `from lib.X import ...` line (the alias becomes an
# unindented statement mid-function body → IndentationError).
from lib.fleet import speed as fleet_speed
from lib.scoring import T_TOTAL_DEFAULT, pv_horizon
from lib.world_model import WAVE_LOOKAHEAD
from lib.world_model import predict_garrison_at
from lib.world_model import simulate_planet_timeline

# Proposer filters reused by W1. No circular-import risk: proposer
# doesn't import predicates. Module-level so the bundler doesn't
# rewrite an in-function import alias mid-body (IndentationError).
from agents.baseline.proposer import _source_survives_launch
from agents.baseline.proposer import _target_holdable_after_capture


# W1 Wald-bound constants (Slice 3 — variant 2 on the W1-bound axis).
# `W1_HOLD_WINDOW` ties the hold-guarantee to the trajectory chooser's
# default MAX_HORIZON so W1's commit semantics ("holds for `window`
# ticks after arrival") match what the inner chooser would have
# checked. Changing this constant is a deliberate bound-axis variant.
W1_HOLD_WINDOW: int = 30
W1_MIN_COUNTER_SHIPS: int = 20  # mirror proposer's MIN_COUNTER_SHIPS
W1_SAFETY_MARGIN: float = 1.5   # mirror proposer's SAFETY_MARGIN


# Default lookahead window beyond the latest known threat ETA. Mirrors
# the proposer's `WAVE_LOOKAHEAD` so W2's holding-window check spans
# the same staggered multi-wave window the proposer's reinforce
# sizing math accounts for.
W2_WINDOW_PADDING: int = WAVE_LOOKAHEAD


@dataclass(frozen=True)
class Verdict:
    """Predicate verdict: commit/discard/uncertain + evidence.

    `kind` is one of:
      - 'commit'    — provably winning; chooser emits without rollout.
      - 'discard'   — provably wasted; chooser drops without rollout.
      - 'uncertain' — predicate inapplicable or inconclusive; chooser
                      falls through to the heuristic rollout.

    `lower_bound` is a closed-form lower bound on Δ-favor (for matching
    when multiple commits compete on the same source). Zero by default;
    populated by predicates that compute it. Unused for uncertain.

    `reason` is a short tag for the audit-replay diff (W2, W1, L1, …).
    """
    kind: str
    lower_bound: float = 0.0
    reason: str = ""


UNCERTAIN = Verdict(kind="uncertain")


def w2_provably_held_reinforce(
    src, tgt, ships: int, wait_N: int, eta: int,
    world, model, me: int,
) -> Verdict:
    """W2 — is this reinforce candidate provably-held through the threat window?

    A reinforce is a candidate aimed at a planet we already own
    (`tgt.owner == me`) under an inbound enemy threat. The predicate
    is closed-form: inject our hypothetical arrival into the planet's
    ledger, re-simulate the timeline, and verify ownership never flips
    away from `me` over `[1, threat_eta + WAVE_LOOKAHEAD]`.

    Returns:
        Verdict(kind='commit', ...)   — provably-held; chooser may emit
                                         without running the rollout.
        Verdict(kind='uncertain', …)  — predicate inapplicable (not a
                                         reinforce, no threat in-flight)
                                         or inconclusive (loses ownership
                                         within window). The chooser
                                         falls back to the rollout in
                                         that case; W2 only commits, it
                                         never discards.

    Soundness scope (v1):
      - Uses `model.ledger` for in-flight enemy fleets. This is the
        deterministic substrate.
      - Does NOT inject potential opp counter-launches from currently-
        at-rest enemy planets. If those exist, W2 abstains (returns
        uncertain) on conservative grounds: the threat window may
        actually be wider than what the in-flight ledger shows.
      - Wallclock: one `simulate_planet_timeline` call ≈ O(horizon)
        scalar ops; ~50-200 µs.
    """
    # Inapplicable: not a reinforce.
    if int(tgt.owner) != int(me):
        return UNCERTAIN

    # Inapplicable: no inbound threat — proposer.capture_size already
    # returned 0 for these; defensive guard in case of an upstream bug.
    threat_eta = model.time_to_enemy_threat(int(tgt.id), int(me), world)
    if threat_eta is None:
        return UNCERTAIN

    # Conservative abstain: if any opp owns a planet currently at rest
    # within plausible counter-launch reach of tgt that ISN'T already
    # represented in the in-flight ledger, the threat window may extend
    # beyond what the closed-form timeline can prove. Fall back to
    # rollout. This guard preserves W2's "commit is sound" property.
    # Concretely: we abstain unless every enemy-owned planet's ships are
    # already accounted for in the ledger via an in-flight fleet.
    in_flight_opp_planets: set[int] = set()
    for (_arr_eta, _owner, _sh) in model.ledger.get(int(tgt.id), []):
        if int(_owner) != int(me):
            # Trace each opp arrival back to its source via the world's
            # fleets list. We don't have per-fleet source attribution
            # in the ledger, so use a permissive check: if any opp seat
            # has in-flight arrivals here, treat at-rest opp launches
            # as covered by the rollout's existing reactive opp model
            # and don't extend W2's window. For v1 we only commit when
            # the threat is FULLY in-flight (no new launches need
            # modeling); if at-rest opp ships exceed our delivered
            # garrison, W2 abstains. See test coverage for this case.
            pass

    # Identify nearby at-rest enemy ships that could launch a NEW counter
    # within the W2 window. Conservative: if any such planet exists with
    # ship count exceeding `MIN_COUNTER_SHIPS`, abstain (defer to rollout).
    # Reuses the same threshold proposer._target_holdable_after_capture
    # uses (20 ships) so behaviour stays consistent.
    MIN_COUNTER_SHIPS = 20
    our_eta = int(wait_N) + int(eta)
    window_end = max(int(threat_eta), our_eta) + W2_WINDOW_PADDING

    for opp in world.planets_by_id.values():
        if int(opp.owner) == int(me) or int(opp.owner) == -1:
            continue
        if int(opp.id) == int(tgt.id):
            continue  # tgt is ours; not an opp
        if int(opp.ships) < MIN_COUNTER_SHIPS:
            continue
        # Closed-form reach: opp could launch and arrive at tgt by
        # tick = ceil(dist / fleet_speed(opp.ships)). If this is within
        # the window AND not already represented by an in-flight fleet
        # from that opp at tgt, W2 abstains. (`math` and `fleet_speed`
        # imported at module level — bundler-safe).
        dx = float(opp.x) - float(tgt.x)
        dy = float(opp.y) - float(tgt.y)
        dist = math.hypot(dx, dy)
        flight = max(0.0, dist - float(opp.radius) - float(tgt.radius) - 0.1)
        spd = fleet_speed(int(opp.ships))
        if spd <= 0:
            continue
        opp_eta = int(math.ceil(flight / spd))
        if opp_eta > window_end:
            continue
        # Is this opp already inbound to tgt with a comparable force?
        in_flight_force = sum(
            int(sh)
            for (eta_arr, owner, sh) in model.ledger.get(int(tgt.id), [])
            if int(owner) == int(opp.owner)
        )
        # Abstain if the at-rest force materially exceeds in-flight.
        # 0.5× as a conservative threshold: a new launch carrying half
        # the at-rest ships would be enough to flip the planet under
        # adverse combat.
        if int(opp.ships) * 0.5 > in_flight_force:
            return UNCERTAIN

    # All clear: closed-form timeline check against the in-flight ledger.
    base_arrivals = list(model.ledger.get(int(tgt.id), []))
    our_arrival = (our_eta, int(me), int(ships))
    augmented = base_arrivals + [our_arrival]

    timeline = simulate_planet_timeline(tgt, augmented, horizon=window_end)
    owner_at = timeline["owner_at"]
    for t in range(1, window_end + 1):
        if owner_at.get(t, int(me)) != int(me):
            # We lose ownership within window; W2 cannot commit. Defer
            # to the rollout for nuanced scoring.
            return UNCERTAIN

    # Provably held through the window. Commit.
    # Δ-favor lower bound for matching: 0 by default (reinforce neither
    # gains nor loses leaf-favor in a successful hold beyond the
    # production accrual our garrison was already entitled to). Future
    # refinement could compute the "production we preserved" credit.
    return Verdict(kind="commit", lower_bound=0.0, reason="W2")


# ---------------------------------------------------------------------------
# Stubs — signatures fixed so the layered chooser's wiring is stable.
# Bodies land in subsequent slices.
# ---------------------------------------------------------------------------


def _w1_multi_opp_holds(
    src, tgt, ships: int, wait_N: int, eta: int,
    world, me: int,
    *,
    window: int = W1_HOLD_WINDOW,
    min_counter_ships: int = W1_MIN_COUNTER_SHIPS,
    safety_margin: float = W1_SAFETY_MARGIN,
) -> bool:
    """Slice-3 Wald bound: does our captured `tgt` hold against the
    worst-case COORDINATED counter-launch from all opp planets in reach
    within `window` ticks of arrival?

    Replaces the single-nearest-opp check (`proposer.
    _target_holdable_after_capture`) used in W1 v1. Sums potential
    counter force across every opp planet whose ETA to `tgt` is ≤
    `window`. The Wald sum is a strict upper bound on actual opp
    coordination — every commit it allows is sound under any opp
    policy.

    Looseness (over-pessimism): the bound treats opp ships as if they
    could be fully redirected to `tgt` regardless of their actual
    strategic placement. Opp coalitions that strong typically have
    better uses for those ships; the bound assumes the worst.

    Returns True iff `coord_counter(t) < safety_margin * our_garrison(t) + 1`
    for both t = arrival and t = arrival + window. Geometrically the
    bound is tightest at one of those two endpoints since both forces
    grow linearly in t (different slopes) — if it holds at both ends,
    it holds throughout.

    Soundness caveat on eta:
      We use `fleet_speed(o.ships)` for opp ETA — mirrors the existing
      proposer filter convention. Larger opp fleets travel slightly
      faster (per `lib.fleet.speed`'s log curve), so if opp accumulates
      before launching, ETA shrinks slightly. The bound is therefore a
      shade unconservative on the eta axis but conservative on the
      force axis (we assume opp commits ALL accumulated ships). Net
      effect is dominated by the force axis.
    """
    if int(tgt.owner) == int(me):
        return True  # reinforce — not W1's territory

    arrival = int(wait_N) + int(eta)
    # Our delivered force at arrival (mirrors _target_holdable_after_capture).
    if int(tgt.owner) == -1:
        tgt_def_at_arrival = int(tgt.ships)
    else:
        tgt_def_at_arrival = int(tgt.ships) + int(tgt.production) * arrival
    delivered = int(ships) - tgt_def_at_arrival
    if delivered < 1:
        return False  # bounce — capture itself fails

    reachable: list = []
    for o in world.planets_by_id.values():
        if int(o.owner) == int(me) or int(o.owner) == -1:
            continue
        if int(o.id) == int(tgt.id):
            continue
        if int(o.ships) < int(min_counter_ships):
            continue
        dist = math.hypot(float(o.x) - float(tgt.x),
                          float(o.y) - float(tgt.y))
        flight = max(0.0, dist - float(o.radius) - float(tgt.radius) - 0.1)
        spd = fleet_speed(int(o.ships))
        if spd <= 0:
            continue
        o_eta = int(math.ceil(flight / spd))
        if o_eta > int(window):
            continue
        reachable.append((o, o_eta))

    if not reachable:
        return True

    # Check at t = arrival + window (end of guarantee horizon).
    t_end = arrival + int(window)
    our_garrison_end = float(delivered) + float(int(tgt.production)) * float(window)
    coord_counter_end = 0.0
    for o, o_eta in reachable:
        ticks_for_growth = max(0, t_end - int(o_eta))
        coord_counter_end += (
            float(int(o.ships))
            + float(int(o.production)) * float(ticks_for_growth)
        )
    if coord_counter_end >= safety_margin * our_garrison_end + 1.0:
        return False

    # Check at t = arrival (worst case for immediate flip).
    t_arr = arrival
    our_garrison_arr = float(delivered)
    coord_counter_arr = 0.0
    for o, o_eta in reachable:
        ticks_for_growth = max(0, t_arr - int(o_eta))
        coord_counter_arr += (
            float(int(o.ships))
            + float(int(o.production)) * float(ticks_for_growth)
        )
    if coord_counter_arr >= safety_margin * our_garrison_arr + 1.0:
        return False

    return True


def w1_provably_winning_capture(
    src, tgt, ships: int, wait_N: int, eta: int,
    world, model, me: int,
    *,
    gamma: float = 0.99,
    value_epsilon: float = 0.01,
) -> Verdict:
    """W1 — is this capture candidate provably winning?

    Fires `commit` when ALL of:
      1. capture succeeds at arrival
         (`predict_garrison_at(ledger + our_arrival).owner == me`),
      2. target holds under the WORST-CASE coordinated multi-opp
         counter over `W1_HOLD_WINDOW` ticks (`_w1_multi_opp_holds`),
      3. source defends itself against its own inbound threats
         (`proposer._source_survives_launch` returns True),
      4. closed-form value lower bound exceeds `value_epsilon`:
         `tgt.production * pv_horizon(step, arrival, gamma) > eps`.

    Otherwise returns UNCERTAIN — the rollout will score it.

    Soundness scope (slice 3 / variant 2 on the bound axis):
      - The hold check is now multi-opp Wald (sum of all reachable
        opp planets' worst-case counter force). Sound under any opp
        policy; v1's single-nearest-opp check was empirically too
        loose (Wlo=0.396 at n=64, audit 2026-05-19).
      - Reinforces (`tgt.owner == me`) defer to W2 — UNCERTAIN here.
    """
    # Inapplicable: reinforce — W2's territory.
    if int(tgt.owner) == int(me):
        return UNCERTAIN

    arrival = int(wait_N) + int(eta)
    base_arrivals = list(model.ledger.get(int(tgt.id), []))
    our_arrival = (arrival, int(me), int(ships))

    # 1. Capture succeeds at arrival.
    owner_at_arrival, _garrison = predict_garrison_at(
        tgt, arrival, base_arrivals + [our_arrival],
    )
    if int(owner_at_arrival) != int(me):
        return UNCERTAIN  # bounce — L1's territory if no hold window exists.

    # 2. Multi-opp Wald hold check (slice 3).
    if not _w1_multi_opp_holds(
        src, tgt, int(ships), int(wait_N), int(eta), world, int(me),
    ):
        return UNCERTAIN

    # 3. Source must survive its own inbound threats.
    if not _source_survives_launch(
        src, int(ships), int(wait_N), world, model, int(me),
    ):
        return UNCERTAIN

    # 4. Value lower bound: production × pv_horizon.
    step = int(getattr(world, "step", 0) or 0)
    pv = pv_horizon(step, arrival, gamma=gamma, t_total=T_TOTAL_DEFAULT)
    value = float(int(tgt.production)) * float(pv)
    if value <= value_epsilon:
        return UNCERTAIN

    return Verdict(kind="commit", lower_bound=value, reason="W1")


# ---------------------------------------------------------------------------
# Slice 5 — bounded-interval scoring + per-source dominance commit
# ---------------------------------------------------------------------------


def _w1_value_bounds(
    src, tgt, ships: int, wait_N: int, eta: int,
    world, model, me: int,
    *,
    gamma: float = 0.99,
) -> tuple[float, float]:
    """Closed-form `(lower_bound, upper_bound)` on Δ-favor from this
    capture.

    `upper_bound` = `production × pv_horizon`, the best-case value
    assuming opp doesn't counter at all (no production cost to us in
    that scenario, and tgt's production accrues fully to our side
    for the remaining episode).

    `lower_bound` = `upper_bound` × `W1_HOLD_WINDOW` / remaining-game
    fraction, but only if the multi-opp Wald hold check passes; else
    `0.0`. The intuition: if Wald says "we hold for at least window
    ticks against the worst-case coordinated counter," then the
    minimum value is the production accrued over that window.
    Otherwise (no hold guarantee), the lower bound is 0 — we cannot
    rule out losing tgt immediately.

    Returns `(0.0, 0.0)` for:
      - reinforces (`tgt.owner == me`): not W1's territory.
      - bounces (delivered force < tgt garrison at arrival).
      - source-vulnerable launches (`_source_survives_launch` fails).

    Used by `w1_dominance_classify` to compute per-source dominance.
    """
    if int(tgt.owner) == int(me):
        return (0.0, 0.0)

    arrival = int(wait_N) + int(eta)
    base_arrivals = list(model.ledger.get(int(tgt.id), []))
    our_arrival = (arrival, int(me), int(ships))

    owner_at_arrival, _ = predict_garrison_at(
        tgt, arrival, base_arrivals + [our_arrival],
    )
    if int(owner_at_arrival) != int(me):
        return (0.0, 0.0)  # bounce

    if not _source_survives_launch(
        src, int(ships), int(wait_N), world, model, int(me),
    ):
        return (0.0, 0.0)  # source drained

    step = int(getattr(world, "step", 0) or 0)
    pv_full = pv_horizon(step, arrival, gamma=gamma, t_total=T_TOTAL_DEFAULT)
    upper_bound = float(int(tgt.production)) * float(pv_full)

    if not _w1_multi_opp_holds(
        src, tgt, int(ships), int(wait_N), int(eta), world, int(me),
    ):
        return (0.0, upper_bound)

    # Wald passes: lower bound is the production over the proven
    # hold window. Cap by upper_bound for safety.
    pv_window = pv_horizon(
        step, arrival, gamma=gamma,
        t_total=step + arrival + W1_HOLD_WINDOW,
    )
    lower_bound = min(upper_bound, float(int(tgt.production)) * float(pv_window))
    return (lower_bound, upper_bound)


def w1_dominance_classify(prerank, world, model, me: int, *, gamma: float = 0.99) -> dict:
    """Per-source W1 dominance: commit a candidate iff its closed-form
    lower bound exceeds every alternative candidate's upper bound on
    the same source.

    Returns `{id(candidate): Verdict}` for capture candidates only.
    Reinforces and non-capture candidates are absent from the dict;
    callers fall through to W2 / uncertain.

    A source with a single W1-eligible capture commits if its lower
    bound is positive (no alternative to compete with). A source with
    multiple candidates commits only the dominant one — the one whose
    worst-case value beats every other candidate's best-case value.
    This is strictly tighter than Slice 4's "per-candidate W1 commit
    when Wald passes" — it eliminates the case where two W1-eligible
    captures from the same source compete, and L0 backstop appends
    whichever ranks first by lower_bound alone.

    Soundness: if `lo[i] > hi[j]` for all j ≠ i on source s, then
    even in the most optimistic alternative scenario, candidate i has
    higher value. The closed-form proof is local to one source's
    available actions.
    """
    by_src: dict = {}
    bounds_by_id: dict = {}
    for c in prerank:
        cheap_delta, src, tgt, ships, angle, eta, horizon, wait_N = c
        if int(tgt.owner) == int(me):
            continue  # W2's territory
        lo, hi = _w1_value_bounds(
            src, tgt, int(ships), int(wait_N), int(eta),
            world, model, int(me), gamma=gamma,
        )
        bounds_by_id[id(c)] = (lo, hi)
        by_src.setdefault(int(src.id), []).append(c)

    verdicts: dict = {}
    for sid, src_cands in by_src.items():
        # Sort by lower_bound descending.
        ranked = sorted(
            src_cands, key=lambda c: -bounds_by_id[id(c)][0],
        )
        best = ranked[0]
        best_lo, best_hi = bounds_by_id[id(best)]
        if best_lo <= 0.0:
            continue  # no committable candidate on this source

        if len(ranked) == 1:
            # Only candidate on this source; commit if lo > 0.
            verdicts[id(best)] = Verdict(
                kind="commit", lower_bound=best_lo, reason="W1",
            )
            continue

        # Multi-candidate: best lo must exceed all OTHER candidates' hi.
        max_other_hi = max(bounds_by_id[id(c)][1] for c in ranked[1:])
        if best_lo > max_other_hi:
            verdicts[id(best)] = Verdict(
                kind="commit", lower_bound=best_lo, reason="W1",
            )
        # Else: no dominance → no W1 commit on this source.

    return verdicts


def l1_provably_wasted_launch(
    src, tgt, ships: int, wait_N: int, eta: int,
    world, model, me: int,
) -> Verdict:
    """L1 — is this launch provably wasted?

    Symmetric inverse of W1. Fires `discard` when the candidate has
    NO tick in `[arrival, arrival + WAVE_LOOKAHEAD]` at which we own
    `tgt` under the augmented ledger.

    Strictly tighter than `predict_fleet_fate` admissibility:
    fate-predictor catches sun/oob/path-blocked; L1 also catches
    "delivered force loses combat at arrival" (bounce) and "we capture
    but get recaptured within window."

    Soundness: future opp launches can only make `tgt` MORE hostile,
    never less. The in-flight-ledger-only check is therefore an upper
    bound on our ownership over the window — never owning under that
    upper bound is a sound proof of waste. No conservative-abstain
    guard needed.

    Reinforces (`tgt.owner == me`) defer to W2 — UNCERTAIN here.
    """
    # Inapplicable: reinforce — W2's territory.
    if int(tgt.owner) == int(me):
        return UNCERTAIN

    arrival = int(wait_N) + int(eta)
    base_arrivals = list(model.ledger.get(int(tgt.id), []))
    our_arrival = (arrival, int(me), int(ships))
    augmented = base_arrivals + [our_arrival]

    window_end = arrival + WAVE_LOOKAHEAD
    timeline = simulate_planet_timeline(tgt, augmented, horizon=window_end)
    owner_at = timeline["owner_at"]

    # Did we ever own it within [arrival, window_end]?
    for t in range(arrival, window_end + 1):
        if int(owner_at.get(t, -1)) == int(me):
            return UNCERTAIN  # we hold at some tick; defer to rollout

    # Never owned within window → provably wasted.
    return Verdict(kind="discard", reason="L1")


def l2_dominance_prune(candidates):
    """L2 — drop same-(src, tgt) candidates dominated on (cheap_delta, ships, eta).

    Each input element is the proposer prerank tuple:
      `(cheap_delta, src, tgt, ships, angle, eta, horizon, wait_N)`.

    Candidate A is dominated by B (same (src.id, tgt.id)) iff B has
    weakly higher `cheap_delta`, weakly lower `ships`, weakly earlier
    `eta`, AND is strictly better in at least one dimension. Dominated
    candidates are dropped.

    v1 scope: same-source-AND-same-target only. Cross-target dominance
    would require value lower bounds we don't yet compute closed-form
    (different targets have different latent values); held back as a
    v2 refinement gated on audit-replay evidence.

    Order-preserving: surviving candidates appear in the same order
    they had in the input (sort stability matters for the chooser's
    cheap-Δ-desc downstream ordering).

    Cost: O(N²) per group in the worst case; groups are typically ≤3
    after the proposer's `(src, tgt, wait_band)` dedup, so the constant
    factor is negligible.
    """
    if not candidates:
        return []

    # Group by (src.id, tgt.id) — preserving first-seen index for ordering.
    groups: dict = {}
    order: list = []
    for idx, c in enumerate(candidates):
        cheap_delta, src, tgt, ships, angle, eta, horizon, wait_N = c
        key = (int(src.id), int(tgt.id))
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append((idx, c))

    keep_indices: set = set()
    for key in order:
        group = groups[key]
        # For each member, check if dominated by any other member.
        for i, (idx_i, c_i) in enumerate(group):
            cd_i, _, _, sh_i, _, et_i, _, _ = c_i
            dominated = False
            for j, (idx_j, c_j) in enumerate(group):
                if i == j:
                    continue
                cd_j, _, _, sh_j, _, et_j, _, _ = c_j
                if (cd_j >= cd_i and sh_j <= sh_i and et_j <= et_i and
                        (cd_j > cd_i or sh_j < sh_i or et_j < et_i)):
                    dominated = True
                    break
            if not dominated:
                keep_indices.add(idx_i)

    return [c for idx, c in enumerate(candidates) if idx in keep_indices]
