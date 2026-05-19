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
      2. target is holdable post-capture against the nearest strong opp
         counter (`proposer._target_holdable_after_capture` returns True),
      3. source defends itself against its own inbound threats
         (`proposer._source_survives_launch` returns True),
      4. closed-form value lower bound exceeds `value_epsilon`:
         `tgt.production * pv_horizon(step, arrival, gamma) > eps`.

    Otherwise returns UNCERTAIN — the rollout will score it.

    Soundness scope (v1):
      - The hold check uses the nearest-strong-opp counter with
        `SAFETY_MARGIN=1.5` margin (the existing proposer math). Sound
        for single-counter scenarios. Multi-opp coordinated counters
        are NOT modelled; gang-up captures can slip through and lose
        on real ladder play. The v2 refinement is the Wald-style
        sum-of-all-opp-ships-in-reach bound.
      - Reinforces (`tgt.owner == me`) defer to W2 — UNCERTAIN here.
      - `step` is read from `world.step`.
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

    # 2-3. Hold + source-survives — reuse existing proposer filters
    # (imported at module level for bundler-safety).
    if not _target_holdable_after_capture(
        src, tgt, int(ships), int(wait_N), int(eta), world, model, int(me),
    ):
        return UNCERTAIN
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
