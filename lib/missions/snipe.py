"""Snipe mission builder — capture enemy / neutral planets via cost-aware ROI.

For every (our-planet, non-our-planet) pair, produce one Mission candidate.
**2026-05-11 ROI upgrade**: the score now trades off VALUE against COST
in ships (and travel time), addressing the gap the doc flagged
(`docs/strategies/simple-roi.md` "Where ROI can lose" lines 64-69):

    value = production × max(1, 500 - step - eta)
    score = priority × value / (ships_to_send + distance + 1)

Additive (not multiplicative) cost in the denominator: pure value/cost
over-corrects toward 1-ship 1-prod targets, which is a different bug.
Keeping distance in the denominator preserves the travel-time discount.

**2026-05-11 PM games-analysis upgrade**: two multiplicative priority
modifiers address weaknesses surfaced in
`audit/2026-05-11-v3-snipe-games-analysis.md`:

1. **Neutral / comet bonus** (NEUTRAL_BONUS, COMET_BONUS). 78.6% of
   comet-steps in v3_snipe's live replays sat neutral; we captured
   only 4.9%. Score function under-priced low-production unclaimed
   targets even though they're essentially free (no garrison growth
   while neutral, no opponent claim required).
2. **4P spoiler** (LEADER_MULTIPLIER). When we're ranked 3rd or 4th
   in ship-totals, attack the leader's planets preferentially. In
   v3_snipe wins, median 58% of our 4P captures came from the leader;
   in losses, 45%. v3 has no explicit leader detection.

Filter: drop pairs where the WorldModel predicts the target will already
be ours with surplus garrison at our fleet's arrival step.
"""

from __future__ import annotations

import math

from lib.fleet import speed as fleet_speed
from lib.geometry import danger_3nn
from lib.intent import World
from lib.mission import Mission
from lib.scoring import DANGER3_KAPPA, MIN_DANGER3_MULT, PV_GAMMA, expected_hold, pv_horizon
from lib.world_model import WorldModel, comet_remaining_lifetime

# sym_hypot was imported here for the σ-equiv layer (cherry-picked
# from origin/claude/game-theory-strategy-analysis-0oH4N). REVERTED for
# v9 (2026-05-12) — v7.6 bisect found σ-equiv regresses v7_0 by ~54pp.
# Restoring math.hypot for src↔target distance below.

# Total game length in steps (Configuration table, data/README.md).
EPISODE_STEPS = 500

# Priority multipliers (calibrated from games analysis).
# NEUTRAL_BONUS and COMET_BONUS were attempted at 1.5 / 1.3 but regressed in
# 32-seed 2P A/B (28.1% Wilson [18.6%, 40.1%]); they tipped the scorer toward
# easy neutrals when contested enemy planets were the binding constraint.
# Disabled (= 1.0) pending a more selective heuristic (opening-only, or
# distance-conditioned). See audit/2026-05-11-v3-snipe-games-analysis.md.
NEUTRAL_BONUS = 1.0
COMET_BONUS = 1.0
# LEADER_MULTIPLIER only fires when our_rank >= 2 (4P/larger games where we
# are below 2nd place). 2P games are unaffected. Pending 4P FFA validation.
LEADER_MULTIPLIER = 1.5
# NON_LEADER_MULTIPLIER — H20 / [E] / TID 697397: Gemini's Day-2 4P
# kingmaker logic boosts the leader (×1.5, above) AND down-weights
# non-leader opponents (×0.8). Default 1.0 = no down-weighting (current
# behaviour). A/B candidate: 0.8. Like LEADER_MULTIPLIER, only fires
# when our_rank >= 2 — 2P unaffected.
NON_LEADER_MULTIPLIER = 1.0

# H10 (2026-05-14): enemy-target multiplier. Top-10 replay analysis
# (knowledge-base/concepts/top-performer-strategies.md §H10) finds
# enemy-target picks at 32% vs midpack 14% — a ×2.3 gap. Multiplying
# the snipe priority by ENEMY_MULTIPLIER when `t.owner ≠ ourselves
# AND t.owner ≠ -1` shifts the scorer toward enemy captures over
# neutral expansion. Default 1.0 (identity) so v7_0 / v3.5.1 baselines
# stay unchanged; v7_7 sets it to 1.3 inside `agent(obs)` per the
# safe-monkey-patch friction pattern (set per-agent-call, never
# persists across processes; see `module-mutation-patching-has-
# worker-reuse-race`, 2026-05-12).
ENEMY_MULTIPLIER = 1.0

# Airtime penalty (v3.5, 2026-05-11): ships in flight are committed-cost.
# A fleet en route can't defend its home planet, can't be redirected, and
# may bounce if the world-state has shifted. Phase-0 idle-source decomposition
# (audit/2026-05-11-idle-breakdown-v3-snipe-phase0.md) showed ~96% of all
# idle classifications come from `intent.ships > src.ships` in validate +
# arrival_size, and the worst offenders are LONG-eta targets where
# arrival_size's `target.ships + production * eta + 1` over-estimates the
# source garrison. Penalising airtime in the score formula shifts target
# selection toward closer (lower-eta) captures, reducing both opportunity
# cost AND the dominant mechanism-drop bucket.
#
# Coefficient interpretation: adds `AIRTIME_PENALTY_WEIGHT * eta` to the
# denominator. eta is bounded in [1, ~30] for the 100x100 board, so at
# weight=1.0 the penalty caps at ~30 vs typical denominators of 50-150 — a
# moderate soft penalty.
#
# **v3.5 A/B verdict (audit/2026-05-11-v3.5-airtime-and-endgame-burn.md):**
# - AIRTIME=1.0 regresses heavily vs v3.4 baseline (43.8% Wilson at 32-seed).
# - AIRTIME=0.5 looked like +4.7pp lift at 32-seed but converged to 52.3%
#   Wilson=[43.7, 60.8] at 64-seed — statistically indistinguishable from
#   baseline.
# - Default reverted to 0.0 (identity). Constant kept for future research
#   (e.g., phase-decay variant, src-conditional variant, multiplicative form).
AIRTIME_PENALTY_WEIGHT = 0.0

# Endgame burn (v3.5, Exp 1): in the final ~30 turns of a game, neutrals
# matter more than enemy captures because (a) neutrals don't grow ships
# (no arrival_size bump → reliably launchable), (b) we have little time
# left to extract production value from contested captures. Boost neutral
# target priority by ENDGAME_NEUTRAL_BONUS once step >= ENDGAME_STEP.
#
# **v3.5 A/B verdict:** as part of the airtime+endgame composite at 64-seed,
# the lift was indistinguishable from baseline. Standalone (eg_only, no
# airtime) saw 40 draws / 64 games = stalemate. Default reverted to 1.0
# (identity). Constant kept for future research (e.g., size-conditional
# burn, neutrals-near-source-only).
ENDGAME_STEP = 470
ENDGAME_NEUTRAL_BONUS = 1.0


# Drop comet chasing entirely (H15 / H18). The top-10 capture-rate
# fingerprint is 3.4 % vs midpack 13.4 %; emanuellcs's public spoofing
# agent formalises a break-even filter. Additionally, our [C] audit on
# 2026-05-13 confirmed that `lib/trajectory.predict_fleet_fate` treats
# comets as STATIC (the `is_orbiting` gate excludes comets — they
# follow path indices, not orbital math), so the fleet's aim assumes
# the comet stays put for the entire flight. By the time the fleet
# arrives, the comet has moved `eta * cometSpeed ≈ eta * 4` units
# along its path. Most comet captures silently fail this way.
# DROP_COMET_TARGETS = 1 filters comet targets out of the proposer
# entirely. Default 0 = current behaviour.
DROP_COMET_TARGETS = 0

# Affordability filter (v3.5+): when True, propose a Mission only if the
# source planet can fund the base capture (target.ships + 1) ALONE. Phase-0
# idle-trace showed ~45% of all idle classifications are
# MECHANISM_DROP:validate, which fires on `intent.ships > src.ships`.
# Filtering at proposal time lets the source's runner-up affordable target
# win settle_plan's per-source greedy instead of being silently dropped
# downstream. Drawback: blocks gang-up (multiple sources contributing to
# one target) — but gang-up doesn't actually work today (each intent is
# independently sized by arrival_size), so the filter is a near-pure
# improvement to idle rate. Default OFF (= 0) until validated by A/B.
# Stored as int so scripts/ab_variants.py can patch it (its regex requires
# a numeric literal).
PROPOSER_AFFORDABILITY_FILTER = 0


# HAV (Hold-Aware Value) — see plan file 2026-05-14 section.
# USE_HAV=1 caps each target's PV horizon by the time-to-enemy-threat
# at that target. Targets in enemy territory get shortened hold, often
# dropping value to zero (proposer skips). Default 0 = identity (PV
# at full remaining-game horizon).
USE_HAV = 0

# Tiered Mission emission. When USE_HOLDING_TIER=1, additionally emit
# a "holding" Mission per (src, target) sized to absorb expected
# enemy counter-attack within HOLD_WINDOW turns. When
# USE_OPERATIONAL_TIER=1, additionally emit an "operational" Mission
# sized to also project a follow-on capture from the captured target
# to its cheapest reachable nearby unowned planet within
# FOLLOWON_RADIUS. settle_plan picks the highest-scoring tier per
# source. Defaults 0 = no extra tiers, current behaviour.
USE_HOLDING_TIER = 0
USE_OPERATIONAL_TIER = 0

# Tier constants — see plan file for derivation. All tunable via
# ab_variants. The defaults are conservative starting points; expect
# to sweep them after a binary on/off PASS.
SOURCE_DEFENSE_RESERVE = 8     # never strand the source below this
OP_RESERVE = 5                 # extra ships in operational fleet
MIN_FOLLOWON_HOLD = 10         # don't propose op-tier if followon is shaky
FOOTHOLD_DISCOUNT = 0.5        # follow-on value weighted at half-credit
HOLD_WINDOW = 10               # look this many turns past arrival for counter-attack
FOLLOWON_RADIUS = 40.0         # max distance from target → followon
HAV_MIN_HOLD = 5               # floor on HAV expected-hold (turns).
                               # `time_to_enemy_threat` is conservative
                               # (any nearby enemy "could" launch);
                               # the floor prevents over-pruning of
                               # contested-but-still-valuable targets.


def _max_enemy_arrival_within(
    ledger_entries: list, my_id: int, eta_lo: int, eta_hi: int,
) -> int:
    """Sum of enemy ship counts in the ledger arriving within
    `[eta_lo, eta_hi]` (inclusive). Used by the holding tier to
    estimate the counter-attack the post-capture garrison must
    survive."""
    if not ledger_entries:
        return 0
    total = 0
    for f_eta, f_owner, f_ships in ledger_entries:
        if f_owner == my_id:
            continue
        if eta_lo <= f_eta <= eta_hi:
            total += int(f_ships)
    return total


def _followon_hold_estimate(
    followon, target, world: World, model: WorldModel, my_id: int, f_eta: int,
) -> int:
    """Estimate how many turns we'd hold `followon` after capturing it
    from `target` (the about-to-be-captured forward base).

    Like `expected_hold` but explicitly EXCLUDES `target` from the
    enemy threat set, because we're about to flip target to our side.
    """
    step_now = int(world.step)
    remaining = max(0, EPISODE_STEPS - step_now - f_eta)
    if remaining == 0:
        return 0

    # In-flight enemy fleets toward followon — keep as-is.
    best: int | None = model.incoming_enemy_eta(followon.id, my_id)

    # Potential launches from each enemy planet EXCEPT the target.
    for p in world.planets_by_id.values():
        if p.id == followon.id or p.id == target.id:
            continue
        if p.owner == my_id or p.owner == -1:
            continue
        if p.ships <= 0:
            continue
        dx = followon.x - p.x
        dy = followon.y - p.y
        d = math.hypot(dx, dy)
        v = fleet_speed(int(p.ships))
        if v <= 0:
            continue
        eta = int(math.ceil(d / v))
        if best is None or eta < best:
            best = eta

    if best is None:
        return remaining
    hold = max(0, int(best) - int(f_eta))
    return min(remaining, hold)


def _best_followon(target, world: World, model: WorldModel, my_id: int,
                   radius: float):
    """Find the cheapest reachable nearby unowned planet from `target`,
    returning `(followon_planet, capture_cost, eta_from_target,
    expected_hold)` or `None` if no follow-on qualifies.

    Used by the operational tier: the captured `target` becomes a
    forward base; the follow-on is the next move from there. Only
    considers planets that are NOT ours, NOT comets, within `radius`
    units of `target`, and predicted to be holdable for at least
    `MIN_FOLLOWON_HOLD` turns after the follow-on arrives (computed
    AS IF target were already ours).
    """
    candidates = []
    for n in world.planets_by_id.values():
        if n.id == target.id:
            continue
        if n.owner == my_id:
            continue
        if n.id in world.comet_ids:
            continue
        dx = n.x - target.x
        dy = n.y - target.y
        d = math.hypot(dx, dy)
        if d > radius:
            continue
        cost = max(1, int(n.ships) + 1)
        v = fleet_speed(cost)
        if v <= 0:
            continue
        f_eta = int(math.ceil(d / v))
        eh = _followon_hold_estimate(n, target, world, model, my_id, f_eta)
        if eh < MIN_FOLLOWON_HOLD:
            continue
        candidates.append((n, cost, f_eta, eh))
    if not candidates:
        return None
    # Pick the highest production/cost ratio — best per-ship payoff.
    candidates.sort(key=lambda x: x[0].production / max(1, x[1]), reverse=True)
    return candidates[0]


def _player_totals(world: World) -> dict[int, float]:
    """Aggregate ships across planets + in-flight fleets for each player.

    Used by the 4P spoiler logic to identify the current leader.
    """
    totals: dict[int, float] = {}
    for p in world.planets_by_id.values():
        if p.owner == -1:
            continue
        totals[p.owner] = totals.get(p.owner, 0) + p.ships
    raw = world.obs_raw
    fleets_raw = (
        raw.get("fleets", []) if isinstance(raw, dict) else getattr(raw, "fleets", [])
    )
    for f in fleets_raw:
        # Fleet schema: [id, owner, x, y, angle, from_planet_id, ships].
        owner = f[1]
        ships = f[6]
        if owner == -1:
            continue
        totals[owner] = totals.get(owner, 0) + ships
    return totals


def _leader_pid(world: World) -> tuple[int | None, int | None]:
    """Return (leader_pid, our_rank) for 4P spoiler scoring.

    Rank is 0-indexed (0 = leader). If we're alone or only-vs-one
    other player, returns (None, None) — no spoiler applies in 2P.
    """
    totals = _player_totals(world)
    if len(totals) < 3:
        return None, None  # 2P or solo — no spoiler
    ordered = sorted(totals.items(), key=lambda kv: -kv[1])
    leader_pid = ordered[0][0]
    our_rank = None
    for i, (pid, _ships) in enumerate(ordered):
        if pid == world.my_id:
            our_rank = i
            break
    return leader_pid, our_rank


# Aggressive sizing (added 2026-05-12 for v3.5.1):
# Top-10 fingerprint analysis (knowledge-base/concepts/top-performer-strategies.md)
# shows mean fleet 38 vs midpack 29 (+33%) and mean garrison-at-launch 11
# vs midpack 22 (half). Translating: top-10 sends a higher FRACTION of
# source garrison per launch. When `aggressive=True` and the source has
# more than AGGRESSIVE_MIN_GARRISON ships, base_ships is set to
# `min(src.ships * AGGRESSIVE_FRACTION, src.ships - AGGRESSIVE_RESERVE)`
# capped above by target_min — so we always send at least what's needed
# to capture, and at most a fixed fraction of garrison.
#
# Parameter sweep (audit/tournaments/sizing-sweep-20260512T044157Z.json):
# 0.7 dominates 0.6 / 0.8 / 0.9 in both vs-baseline winrate and
# head-to-head. 32-seed 2P A/B vs v3_snipe: 68.8% Wilson lo 56.6% [PASS].
# 8-seed × 4-seat 4P FFA vs weak background: 96.9% (vs v3_snipe baseline
# 93.8% in same panel).
AGGRESSIVE_FRACTION = 0.7
AGGRESSIVE_RESERVE = 5
AGGRESSIVE_MIN_GARRISON = 12


def propose_snipe_missions(
    world: World,
    model: WorldModel,
    aggressive: bool = False,
) -> list[Mission]:
    """Build one snipe Mission per (our source, non-our target) pair.

    `aggressive=False` (default) uses the v3.4 minimum-viable formula
    `max(1, t.ships + 1)` — preserves the parity-gated v3_snipe bundle.
    `aggressive=True` uses the top-10-aligned sizing formula. v3.5.1
    is the first agent to pass aggressive=True.
    """
    if not world.planets_by_id:
        return []
    my_planets = [
        p for p in world.planets_by_id.values() if p.owner == world.my_id
    ]
    if not my_planets:
        return []
    targets = [
        p for p in world.planets_by_id.values() if p.owner != world.my_id
    ]
    if not targets:
        return []

    step_now = int(world.step)
    leader_pid, our_rank = _leader_pid(world)
    spoiler_on = leader_pid is not None and our_rank is not None and our_rank >= 2

    missions: list[Mission] = []
    for src in my_planets:
        for t in targets:
            if DROP_COMET_TARGETS and t.id in world.comet_ids:
                # H15/H18 — drop comet targets entirely (see flag docstring).
                continue
            d = math.hypot(t.x - src.x, t.y - src.y)
            target_min = max(1, int(t.ships) + 1)
            if aggressive and src.ships > AGGRESSIVE_MIN_GARRISON:
                fraction_size = max(1, int(src.ships * AGGRESSIVE_FRACTION))
                cap = max(1, int(src.ships) - AGGRESSIVE_RESERVE)
                base_ships = max(target_min, min(fraction_size, cap))
            else:
                base_ships = target_min
            if PROPOSER_AFFORDABILITY_FILTER and base_ships > src.ships:
                # Source can't fund this capture alone; let its smaller
                # affordable runner-up win settle_plan's per-source greedy.
                # OFF by default (regressed in 64-seed A/B); kept for
                # future ablation. See main's optimize-ship-strategy-tDPXx.
                continue
            v = fleet_speed(base_ships)
            eta = int(math.ceil(d / max(v, 1e-6))) if v > 0 else 0
            pred_owner = model.owner_at(t.id, eta)
            pred_ships = model.ships_at(t.id, eta) or 0.0
            if pred_owner == world.my_id and pred_ships >= base_ships:
                # Target will be ours with surplus garrison; redundant.
                continue
            # Comet-lifetime correction: comets leave the board at
            # `len(path) - path_index` steps from now; capping time_to_hold
            # by remaining lifetime stops us scoring "long-run yield" on a
            # comet that's about to depart. `pv_horizon` with PV_GAMMA=1.0
            # is identity to the prior linear `max(0, lifetime − eta)`
            # form; PV_GAMMA<1.0 discounts future production geometrically
            # (TID 699003).
            is_comet = t.id in world.comet_ids
            if is_comet:
                rem = comet_remaining_lifetime(t.id, world)
                if (rem or 0) <= eta:
                    # H15 (main 2026-05-13 cb02fd9): comet leaves the
                    # board before our fleet arrives — don't emit a
                    # Mission. Lets the source's runner-up win the
                    # per-source slot in settle_plan instead of
                    # consuming it with a degenerate score≈0 candidate.
                    continue
                # PV horizon over the remaining-lifetime budget; identity
                # to `max(0, rem - eta)` at PV_GAMMA=1.0.
                time_to_hold = max(0.0, pv_horizon(0, eta, PV_GAMMA, rem or 0))
            else:
                if USE_HAV:
                    # HAV-1: cap PV horizon by predicted hold window
                    # (time-to-enemy-threat at target). Soft floor at
                    # HAV_MIN_HOLD turns rather than dropping the
                    # Mission — `time_to_enemy_threat` is over-pessimistic
                    # for centrally-located targets (it assumes enemy
                    # will dedicate full garrison to this target which
                    # isn't realistic). Floor lets settle_plan pick a
                    # contested target if it's still the best option.
                    eh = expected_hold(t.id, eta, world, model, EPISODE_STEPS)
                    eh = max(HAV_MIN_HOLD, eh)
                    time_to_hold = max(1.0, pv_horizon(0, 0, PV_GAMMA, eh))
                else:
                    time_to_hold = max(
                        1.0, pv_horizon(step_now, eta, PV_GAMMA, EPISODE_STEPS)
                    )
            value = t.production * time_to_hold

            # Cost-aware ROI baseline + priority modifiers.
            priority = 1.0
            if t.owner == -1:
                # Unclaimed: no garrison growth during flight, no opponent
                # competition. Bonus reflects the easier capture.
                priority *= COMET_BONUS if is_comet else NEUTRAL_BONUS
                if step_now >= ENDGAME_STEP:
                    # Late-game burn: neutrals stay launchable (no
                    # production growth → no arrival_size bump), so prefer
                    # them over high-growth enemy captures we likely can't
                    # afford in the remaining turn budget.
                    priority *= ENDGAME_NEUTRAL_BONUS
            if spoiler_on:
                if t.owner == leader_pid:
                    priority *= LEADER_MULTIPLIER
                elif t.owner != -1 and t.owner != world.my_id:
                    # Non-leader other player's planet — Gemini-style 4P
                    # kingmaker down-weight (H20). Neutrals + our own
                    # planets are unaffected.
                    priority *= NON_LEADER_MULTIPLIER
            # H10 enemy-target multiplier (default 1.0 = no change).
            # Applies whenever the target is owned by an enemy (not us,
            # not neutral). v7_7 sets ENEMY_MULTIPLIER=1.3 to bias
            # toward enemy snipes. Composes multiplicatively with the
            # H20 NON_LEADER_MULTIPLIER above when both fire.
            if t.owner >= 0 and t.owner != world.my_id:
                priority *= ENEMY_MULTIPLIER
            # Cost-aware ROI denominator (legacy) + optional airtime term.
            # - base_ships + d + 1: original v3.4 form. Wave-1b's
            #   `0.5 × base_ships` rebalance was NEUTRAL at 50% in phys-only
            #   A/B (audit/2026-05-12-v3.5-stack-results.md); reverted on
            #   merge to preserve main's parity invariants. v3.5.1's
            #   value driver was the AGGRESSIVE_FRACTION ship sizing, not
            #   this denominator.
            # - AIRTIME_PENALTY_WEIGHT × eta: optional discount for far
            #   targets. Default weight=0 (identity).
            score = priority * value / (
                base_ships + d + AIRTIME_PENALTY_WEIGHT * eta + 1.0
            )
            # 3-NN allegiance danger map (H17). At κ=0 (default) the
            # multiplier is exactly 1.0 — no effect on existing tests.
            if DANGER3_KAPPA != 0.0:
                d3 = danger_3nn(
                    (t.x, t.y), t.id,
                    list(world.planets_by_id.values()),
                    world.my_id,
                )
                score *= max(MIN_DANGER3_MULT, 1.0 + DANGER3_KAPPA * d3)

            missions.append(Mission(
                mission_class="snipe",
                src_id=src.id,
                target_id=t.id,
                ships=base_ships,
                score=score,
                eta=eta,
            ))

            # --------- Tier 2: Holding (2026-05-14 plan, HAV-2) ----------
            # Size the fleet to absorb the strongest expected enemy
            # arrival within HOLD_WINDOW turns of our capture. Skip
            # comets (their threat model is dominated by lifetime, not
            # counter-attack).
            S_hold: int | None = None
            if USE_HOLDING_TIER and not is_comet:
                counter = _max_enemy_arrival_within(
                    model.ledger.get(t.id, []),
                    my_id=world.my_id,
                    eta_lo=eta + 1, eta_hi=eta + HOLD_WINDOW,
                )
                if counter > 0:
                    prod_during = int(t.production) * HOLD_WINDOW
                    deficit = counter - prod_during
                    if deficit > 0:
                        S_hold = base_ships + int(deficit) + 1
                if S_hold is not None and S_hold > base_ships:
                    if int(src.ships) - S_hold >= SOURCE_DEFENSE_RESERVE:
                        src_threat = model.incoming_enemy_eta(src.id, world.my_id)
                        if src_threat is None or src_threat > eta:
                            # Holding tier permanently denies the
                            # planet to enemy → use full remaining-game
                            # horizon (not the HAV-capped one).
                            hold_t = max(
                                1.0,
                                pv_horizon(step_now, eta, PV_GAMMA, EPISODE_STEPS),
                            )
                            hold_value = t.production * hold_t
                            hold_score = priority * hold_value / (
                                S_hold + d + AIRTIME_PENALTY_WEIGHT * eta + 1.0
                            )
                            if DANGER3_KAPPA != 0.0:
                                hold_score *= max(
                                    MIN_DANGER3_MULT,
                                    1.0 + DANGER3_KAPPA * danger_3nn(
                                        (t.x, t.y), t.id,
                                        list(world.planets_by_id.values()),
                                        world.my_id,
                                    ),
                                )
                            missions.append(Mission(
                                mission_class="snipe",
                                src_id=src.id,
                                target_id=t.id,
                                ships=S_hold,
                                score=hold_score,
                                eta=eta,
                                note="hold",
                            ))
                    else:
                        S_hold = None  # source can't afford the holding tier

            # --------- Tier 3: Operational / foothold (HAV-3) -----------
            if USE_OPERATIONAL_TIER and not is_comet:
                foothold = _best_followon(
                    t, world, model, world.my_id, FOLLOWON_RADIUS,
                )
                if foothold is not None:
                    f_target, f_cost, f_eta_from_t, f_hold = foothold
                    base_for_op = S_hold if S_hold is not None else base_ships
                    S_op = base_for_op + f_cost + OP_RESERVE
                    if S_op > base_ships and int(src.ships) - S_op >= SOURCE_DEFENSE_RESERVE:
                        src_threat = model.incoming_enemy_eta(src.id, world.my_id)
                        if src_threat is None or src_threat > eta:
                            # Capture value: full hold (we're holding +
                            # projecting from this base).
                            op_t = max(
                                1.0,
                                pv_horizon(step_now, eta, PV_GAMMA, EPISODE_STEPS),
                            )
                            op_value = t.production * op_t
                            # Foothold value: discounted follow-on PV.
                            f_pv = pv_horizon(0, 0, PV_GAMMA, f_hold)
                            op_value += FOOTHOLD_DISCOUNT * f_target.production * f_pv
                            op_score = priority * op_value / (
                                S_op + d + AIRTIME_PENALTY_WEIGHT * eta + 1.0
                            )
                            if DANGER3_KAPPA != 0.0:
                                op_score *= max(
                                    MIN_DANGER3_MULT,
                                    1.0 + DANGER3_KAPPA * danger_3nn(
                                        (t.x, t.y), t.id,
                                        list(world.planets_by_id.values()),
                                        world.my_id,
                                    ),
                                )
                            missions.append(Mission(
                                mission_class="snipe",
                                src_id=src.id,
                                target_id=t.id,
                                ships=S_op,
                                score=op_score,
                                eta=eta,
                                note=f"op→{f_target.id}",
                            ))
    return missions
