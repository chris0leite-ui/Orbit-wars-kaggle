"""Hold-time + per-candidate reward for the reach-frontier chooser.

`hold_time(p)` is the doctrine's central quantity: the duration we expect
to extract production-time integral from planet p before opp threatens.

For a planet we capture: `hold = ρ_opp(p) - ρ_me(p)`.
For a planet we own: `hold = ρ_opp(p) - t_now`.

Hold is capped at the remaining-game horizon (a planet can't pay out more
than (T - now) of integral regardless of opp reach).

Reward: `R(s, p, k) = p̃·hold - λ_loss·losses - λ_risk·risk` per
doctrine §4. λ_risk priced at 50 (heavy: fleets dying is a §8.4 critical
failure). λ_loss at 1 (price one lost ship at one ship).
"""

from __future__ import annotations


# Risk: heavy by design — a fleet dying mid-flight is doctrine §8.4
# critical failure (sun/OOB/comet expiry).
LAMBDA_RISK_DEFAULT: float = 50.0

# Loss: ships consumed in capture combat. Doctrine §4's "price one lost
# ship at one ship" baseline (λ_loss = 1.0) was empirically too
# conservative — the formula compares integral gain (p̃·hold) against a
# one-shot loss but doesn't credit the post-capture garrison that pays
# out OVER THE WHOLE remaining-game window. With λ_loss = 1.0 the
# chooser refused most early-game captures; we recalibrated down to 0.1
# to bias toward fires that capture even at modest local cost. Per
# design §12 Q1, knob-tuned; locked at the v1 default and flagged as a
# v1.1 calibration axis if v1 clears the eval gate.
LAMBDA_LOSS_DEFAULT: float = 0.1

# Minimum hold-time floor for FEASIBLE capture candidates (those that
# survived the `target_owner_at_arrival != me` and `k > expected_garrison`
# pre-filter in `assignment._columns_from_reach`). Origin:
# `audit/2026-05-27-rf-v1-root-cause.md` Bug 1 — `WorldModel.time_to_enemy_threat`
# returns a worst-case opp reach that systematically dominates our ρ_me
# mid-game, collapsing `max(0, ρ_opp − ρ_me)` to 0 for every reachable
# target. The chooser then prices every launch at `−λ_loss · expected_garrison`
# (negative) and the diagonal noop column at cost 0 wins every Hungarian
# row, producing 87 % silence and a 0/20 loss to baseline.
#
# Justification: feasibility-filtered candidates ARE captures we can
# physically make against the predicted garrison-at-arrival. Even when
# `time_to_enemy_threat` says opp COULD reach the planet 8 ticks from
# now, in practice opp has competing demands (2P with many targets, 4P
# with three opps competing). The 30-tick floor models the empirical
# "opp recapture rarely happens inside one orbital period" prior. Tuned
# in B5 iteration; flagged for v1.1 calibration if B5 clears.
MIN_HOLD_FLOOR_DEFAULT: float = 30.0


def compute_hold_times(
    world,
    me: int,
    my_reach: dict,                  # (src_id, tgt_id) -> [ReachEntry]
    opp_reach: dict[int, float],     # tgt_id -> ρ_opp
    step_now: int,
    *,
    game_horizon: int = 500,
    min_hold_floor: float = MIN_HOLD_FLOOR_DEFAULT,
) -> dict[int, float]:
    """Per-target hold_time, capped at remaining game ticks.

    For capture targets that pass the upstream feasibility filter,
    hold is floored at `min_hold_floor` (see comment on the constant
    above — fixes the Bug 1 silent-87% spiral identified in
    `audit/2026-05-27-rf-v1-root-cause.md`). Defensive holds (mine
    planets) are NOT floored; their hold is the time until opp
    threatens, and 0 means "already under threat, no positive value
    from holding."

    Returns `target_id -> hold_time` (float). 0.0 for targets we can't
    reach. The map is keyed on every planet we observe (not just
    reachable ones) so the chooser can also compute "defend reward"
    for our own sources.
    """
    me_id = int(me)
    remaining = float(max(0, int(game_horizon) - int(step_now)))
    floor = float(min_hold_floor)

    # Pre-index my_reach by target_id for the cheapest entry per target.
    cheapest_me: dict[int, float] = {}
    for (src_id, tgt_id), entries in my_reach.items():
        if not entries:
            continue
        first = entries[0]
        cur = cheapest_me.get(int(tgt_id))
        if cur is None or first.cost_tick < cur:
            cheapest_me[int(tgt_id)] = float(first.cost_tick)

    out: dict[int, float] = {}
    for tgt_id, p in world.planets_by_id.items():
        tid = int(tgt_id)
        rho_opp = opp_reach.get(tid, float("inf"))
        if int(p.owner) == me_id:
            # Defensive (own planet): no floor. 0 = "under immediate threat."
            if rho_opp == float("inf"):
                hold = remaining
            else:
                hold = max(0.0, float(rho_opp))
        else:
            # Capture: floor at `min_hold_floor` for reachable candidates.
            # The feasibility filter in `_columns_from_reach` ensures we
            # only reward candidates that physically can flip the
            # planet, so the floor is safe — every floored candidate is
            # a real capture with at least min_hold_floor ticks of
            # production integral before opp's worst-case recapture.
            rho_me = cheapest_me.get(tid)
            if rho_me is None:
                # Unreachable — no candidate, no need to floor.
                hold = 0.0
            elif rho_opp == float("inf"):
                hold = max(floor, remaining - float(rho_me))
            else:
                hold = max(floor, float(rho_opp) - float(rho_me))
        out[tid] = min(hold, remaining)
    return out


def per_candidate_reward(
    entry,
    target,
    hold_time: float,
    *,
    lambda_risk: float = LAMBDA_RISK_DEFAULT,
    lambda_loss: float = LAMBDA_LOSS_DEFAULT,
    risk: float = 0.0,
) -> float:
    """Doctrine reward: R = p̃·hold − λ_loss·losses − λ_risk·risk."""
    production = float(getattr(target, "production", 0.0))
    losses = max(0.0, float(entry.expected_garrison))
    return (
        production * float(hold_time)
        - float(lambda_loss) * losses
        - float(lambda_risk) * float(risk)
    )
