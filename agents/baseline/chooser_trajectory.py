"""Trajectory-first chooser — drop-in alternative to `chooser.choose`.

Replaces the K-step fast_sim rollout + composite-leaf-value approach
with deterministic trajectory analysis + single-tick combat prediction.

Pipeline per turn:
  1. Iterate proposer's prerank candidates in cheap-Δ order.
  2. For each candidate `(src, tgt, ships, angle, eta, wait_N)`:
     a. `predict_fleet_fate(src, tgt, angle, ships, world)` — drop
        on `sun` / `oob` / `timeout` / `planet` (path-blocked by a
        different planet) / `comet_collision` (predicted-hit comet).
     b. For surviving "target" outcomes that ARE comets, drop if the
        comet's remaining lifetime ≤ ETA (expires before arrival).
     c. `predict_garrison_at(tgt, eta, ledger[tgt.id] + [our arrival])`
        — single-tick combat result.
     d. Score:
          captured (was-enemy → now-us)  → production × time_remaining,
                                            capped at comet life.
          reinforced (already-ours)      → skip (no extra credit;
                                            threat reinforcement is
                                            handled at proposer.propose).
          bounced (still-not-us)         → -ships (waste penalty).
  3. Sort surviving by score desc; greedy non-dogpile dedup by
     (src_id, tgt_id); emit `wait_N==0` winners only (`wait_N>0`
     reserves src+tgt, emits nothing this turn).

No K-step rollout, no leaf value-function approximation, no fast_sim
state cloning. Cost is O(candidates × (trajectory_steps + eta + arrivals)).

PI critique 2026-05-17: "we should be thinking in fleet trajectories";
"sun-deaths should be 0% with proper trajectory analysis". See
`knowledge-base/concepts/trajectory-first-architecture.md`.
"""

from __future__ import annotations

import math

from lib.trajectory import predict_fleet_fate
from lib.world_model import comet_remaining_lifetime, predict_garrison_at


EPISODE_STEPS_TOTAL: int = 500
WASTE_WEIGHT: float = 0.5
CAPTURE_REWARD_WEIGHT: float = 0.05

# Multi-launch budget (Step A, 2026-05-17 v2): the v1 chooser hard-
# capped at 1 launch per source per turn. v15 routinely emits
# parallel launches (3-5 sources × 1-2 fleets each); the 1/source
# limit was a load-bearing reason v1 lost 0/32 vs v15. v2 tracks
# remaining ships per source and emits multiple launches until the
# source's ships fall below the next candidate's requirement.
#
# MIN_SOURCE_RESERVE = 0: v15-line baseline does NOT hold a reserve;
# proposer's MIN_FLEET_SIZE filter ensures we never emit absurdly
# small launches. Holding even 2 ships back blocks the very common
# early-game case where a 10-ship home wants to send all 10 to a
# 5-ship neutral. Adopt v15's "spend it all if it captures" stance.
MIN_SOURCE_RESERVE: int = 0

# Opp 1-turn lookahead (Step C, 2026-05-17 v2): predict each opp
# source's likely best 1 launch; inject into ledger before scoring.
# Knobs first-pass; tune if A/B is borderline.
OPP_NEAREST_K: int = 4
OPP_SHIP_FRACTION: float = 0.8
OPP_MIN_SHIPS: int = 4


def score_candidate(src, tgt, ships: int, angle: float, eta_hint: int,
                    me: int, world, ledger: dict,
                    ) -> tuple[float, str, int | None]:
    """Score a single candidate launch.

    Returns `(score, status, fate_step)`:
        `status` ∈ {'captured', 'reinforced', 'bounced', 'sun', 'oob',
                    'timeout', 'comet_collision', 'comet_expired',
                    'path_blocked'}
        `fate_step` = the tick of the resolving event (None if dropped pre-flight).
    """
    fate = predict_fleet_fate(src, tgt, angle, ships, world)

    if fate.outcome == "sun":
        return (float("-inf"), "sun", fate.step)
    if fate.outcome == "oob":
        return (float("-inf"), "oob", fate.step)
    if fate.outcome == "timeout":
        return (float("-inf"), "timeout", fate.step)
    if fate.outcome == "planet":
        # Hit a non-target planet first. Could be a comet collision
        # (engine treats comets as planets) — distinguish via comet_ids.
        if fate.hit_planet_id in world.comet_ids:
            return (float("-inf"), "comet_collision", fate.step)
        return (float("-inf"), "path_blocked", fate.step)

    # outcome == "target": fleet reaches the intended planet at fate.step.
    eta = int(fate.step)

    # Comet-expired guard: if the target IS a comet and runs out of
    # path at/before our arrival, the planet won't exist for capture.
    if int(tgt.id) in world.comet_ids:
        life = comet_remaining_lifetime(int(tgt.id), world)
        if life is None or life <= eta:
            return (float("-inf"), "comet_expired", eta)

    # Sparse single-tick combat prediction. Include our hypothetical
    # arrival in the ledger so resolve_arrivals handles same-tick
    # combat correctly with any other fleets due that tick.
    base_arrivals = list(ledger.get(int(tgt.id), []))
    our_arrival = (eta, int(me), int(ships))
    pred_owner, _pred_garrison = predict_garrison_at(
        tgt, eta, base_arrivals + [our_arrival],
    )

    if pred_owner != me:
        # We didn't end up holding the planet — bounce / under-sized.
        return (-WASTE_WEIGHT * ships, "bounced", eta)

    # We hold the planet at eta. Was it ours before our arrival?
    # If the planet was already me (with no enemy interference), this
    # is reinforcement — no extra credit. Otherwise it's a capture.
    if int(tgt.owner) == me:
        # Check whether anything would flip it away from us between now
        # and eta-1 (in which case our arrival is a recapture).
        pred_owner_without_us, _ = predict_garrison_at(
            tgt, eta, base_arrivals,
        )
        if pred_owner_without_us == me:
            # Still ours without us — pure reinforcement.
            return (0.0, "reinforced", eta)
        # We recaptured a planet that would otherwise have been lost.
        # Credit the recapture like a fresh capture (production × time).
        time_remaining = max(0, EPISODE_STEPS_TOTAL - int(world.step) - eta)
        held = time_remaining
        if int(tgt.id) in world.comet_ids:
            life = comet_remaining_lifetime(int(tgt.id), world)
            if life is not None:
                held = min(held, max(0, life - eta))
        return (CAPTURE_REWARD_WEIGHT * float(tgt.production) * float(held),
                "captured", eta)

    # Fresh capture (planet was not ours).
    time_remaining = max(0, EPISODE_STEPS_TOTAL - int(world.step) - eta)
    held = time_remaining
    if int(tgt.id) in world.comet_ids:
        life = comet_remaining_lifetime(int(tgt.id), world)
        if life is not None:
            held = min(held, max(0, life - eta))
    return (CAPTURE_REWARD_WEIGHT * float(tgt.production) * float(held),
            "captured", eta)


def predict_opp_responses(world, me: int, num_seats: int,
                          ) -> list[tuple[int, int, int, int]]:
    """1-turn opp lookahead: project each enemy source's likely best
    launch into a list of (target_pid, eta, opp_owner, ships) tuples
    that can be merged into our arrival ledger.

    Heuristic per opp source: scan its `OPP_NEAREST_K` nearest non-opp
    targets; pick the first one whose straight-line trajectory is
    admissible (no sun, no oob, no comet collision). Project a fleet
    of `OPP_SHIP_FRACTION × src.ships` ships.

    Closes Gap 2 of v1's 0/32 failure: composite-head A/B's K-step
    rollout simulates opp counter-launches via lib.opp_model.
    lite_greedy_policy; v1 trajectory chooser had no opp model so
    every candidate was scored as if the opp would play idle. With
    this projection, our score_candidate sees the ledger with the
    enemy's likely counter-fleet already accounted for.

    First-pass heuristic; not a full opp model. May:
      - Overestimate opp competence (assumes they pick optimal target).
      - Underestimate launches (only 1 per source).
      - Miss multi-target threats (e.g. gang-ups).
    All tolerable for a first-cut A/B; refine if results are promising.
    """
    opp_arrivals: list[tuple[int, int, int, int]] = []
    all_planets = list(world.planets_by_id.values())
    for opp_id in range(num_seats):
        if opp_id == me:
            continue
        opp_planets = [p for p in all_planets if int(p.owner) == opp_id]
        for opp_src in opp_planets:
            if int(opp_src.ships) < OPP_MIN_SHIPS:
                continue
            # Nearest non-opp targets.
            others = sorted(
                ((math.hypot(p.x - opp_src.x, p.y - opp_src.y), p)
                 for p in all_planets
                 if int(p.owner) != opp_id and int(p.id) != int(opp_src.id)),
                key=lambda d_p: d_p[0],
            )
            ships = max(1, int(int(opp_src.ships) * OPP_SHIP_FRACTION))
            for _d, opp_tgt in others[:OPP_NEAREST_K]:
                angle = math.atan2(opp_tgt.y - opp_src.y,
                                   opp_tgt.x - opp_src.x)
                fate = predict_fleet_fate(opp_src, opp_tgt, angle,
                                          ships, world)
                if fate.outcome == "target":
                    opp_arrivals.append(
                        (int(opp_tgt.id), int(fate.step), opp_id, ships),
                    )
                    break  # 1 projection per opp source is enough
    return opp_arrivals


def merge_ledgers(base: dict, projected: list[tuple[int, int, int, int]],
                  ) -> dict:
    """Add projected (target_pid, eta, owner, ships) tuples into a copy
    of `base` (per-planet list of (eta, owner, ships))."""
    out = {pid: list(v) for pid, v in base.items()}
    for tgt_pid, eta, owner, ships in projected:
        out.setdefault(tgt_pid, []).append((eta, owner, ships))
    return out


def choose_trajectory(snap_base, prerank, baseline_favors,
                      me: int, num_seats: int, wallclock_ms: float,
                      min_horizon: int, max_horizon: int, gamma: float,
                      world, model) -> list[list]:
    """Drop-in alternative to `chooser.choose`.

    The `snap_base` / `baseline_favors` / `min_horizon` / `max_horizon`
    / `gamma` args are unused (kept for signature parity with
    `chooser.choose` so the dispatcher in `main.py` is a simple swap).
    The trajectory chooser doesn't roll out and doesn't need an idle
    baseline.

    v2 (2026-05-17 PM):
    - 1-turn opp lookahead: predict_opp_responses projects each opp
      source's best launch; ledger merged BEFORE scoring (every
      predict_garrison_at sees the pessimistic state).
    - Multi-launch budget: drops "1 launch per source" dedup; tracks
      ship sub-budget per source.
    """
    if not prerank:
        return []

    # Step C — pessimistic ledger including opp counter-launches.
    opp_projected = predict_opp_responses(world, me, num_seats)
    ledger = merge_ledgers(model.ledger, opp_projected)

    scored: list[tuple] = []
    for cheap_delta, src, tgt, ships, angle, eta_hint, _, wait_N in prerank:
        # First-cut: only fire-now candidates. wait_N>0 candidates
        # reserve src+tgt without emitting; since we typically have
        # 1-3 source planets, a single wait_N>0 reservation blocks
        # every other launch from that source. Defer wait-then-fire
        # to a follow-up (it needs time-discounted scoring to
        # compete with fire-now properly).
        if int(wait_N) != 0:
            continue
        score, status, _ = score_candidate(
            src, tgt, int(ships), float(angle), int(eta_hint),
            me, world, ledger,
        )
        # 'captured' covers both fresh captures AND defense-counterfactual
        # recaptures (score_candidate scores them identically via the
        # tgt.owner==me + pred_owner_without_us!=me branch).
        if status in ("captured",) and score > 0.0:
            scored.append((score, src, tgt, ships, angle, wait_N))

    if not scored:
        return []

    scored.sort(key=lambda c: -c[0])

    # Step A — multi-launch ship-budget dedup. Drops the v1 "1 launch
    # per source" rule; each source can emit multiple fleets until its
    # ship budget falls below MIN_SOURCE_RESERVE. Tgt dedup stays
    # (don't dogpile a single target).
    src_budget: dict[int, int] = {}
    for p in world.planets_by_id.values():
        if int(p.owner) == me:
            src_budget[int(p.id)] = max(0, int(p.ships) - MIN_SOURCE_RESERVE)
    used_tgts: set[int] = set()
    moves: list[list] = []
    for _score, src, tgt, ships, angle, wait_N in scored:
        sid, tid = int(src.id), int(tgt.id)
        if src_budget.get(sid, 0) < int(ships):
            continue  # source out of ships
        if tid in used_tgts:
            continue  # don't dogpile this target
        src_budget[sid] -= int(ships)
        used_tgts.add(tid)
        if int(wait_N) == 0:
            moves.append([sid, float(angle), int(ships)])
        # wait_N > 0 already filtered above; this is a no-op safety.
    return moves
