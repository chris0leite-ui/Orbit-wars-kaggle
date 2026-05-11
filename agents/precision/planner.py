"""Global joint planner — v2.

ROI-greedy over single shots + 2-source waves, with robust min-max scoring
against TWO enemy hypotheses (greedy-theirs and worst-for-us). The plan that
maximizes its WORST-case score across the two enemy projections wins.

The frozen v1 planner remains available as `plan_turn_v1` for benchmarking.
"""
from __future__ import annotations

import dataclasses
import time
from agents.precision import bundling, enemy_model, intercept, prediction, scoring


# ---- v1 (frozen baseline for head-to-head) -------------------------------

def plan_turn_v1(
    world: dict,
    deadline: float,
    max_shots: int = 12,
    horizon_steps: int = 200,
) -> list[intercept.Shot]:
    """The greedy planner committed in iteration 1 (frozen for benchmarking)."""
    me = world["player"]
    menu_deadline = deadline - 0.65 * max(deadline - time.perf_counter(), 0)
    menu = intercept.build_shot_menu(world, deadline=menu_deadline)
    if not menu:
        return []

    candidates: list[intercept.Shot] = []
    for shots in menu.values():
        candidates.extend(shots)
    if not candidates:
        return []

    src_remaining: dict[int, int] = {}
    for p in world["planets"]:
        if p.owner == me:
            src_remaining[p.id] = p.ships

    chosen: list[intercept.Shot] = []
    base_score = prediction.plan_score(world, chosen, horizon_steps=horizon_steps)

    for _ in range(max_shots):
        if time.perf_counter() >= deadline:
            break
        best_shot = None
        best_score = base_score
        for shot in candidates:
            if time.perf_counter() >= deadline:
                break
            if shot in chosen:
                continue
            avail = src_remaining.get(shot.src_id, 0)
            if shot.ship_count > avail:
                continue
            trial = chosen + [shot]
            score = prediction.plan_score(world, trial, horizon_steps=horizon_steps)
            if score > best_score:
                best_score = score
                best_shot = shot
        if best_shot is None:
            break
        chosen.append(best_shot)
        src_remaining[best_shot.src_id] -= best_shot.ship_count
        base_score = best_score

    return chosen


# ---- v2: ROI + waves + robust min-max ------------------------------------

class _Candidate:
    """Wraps either a single Shot or a Wave for uniform handling."""
    __slots__ = ("shots", "tgt_id", "ship_cost_by_src", "total_ships", "arrival_step", "roi")

    def __init__(self, shots: tuple[intercept.Shot, ...], tgt_id: int,
                 arrival_step: int, roi: float):
        self.shots = shots
        self.tgt_id = tgt_id
        self.arrival_step = arrival_step
        self.roi = roi
        cost: dict[int, int] = {}
        for s in shots:
            cost[s.src_id] = cost.get(s.src_id, 0) + s.ship_count
        self.ship_cost_by_src = cost
        self.total_ships = sum(cost.values())

    def fits(self, src_remaining: dict[int, int]) -> bool:
        for src, c in self.ship_cost_by_src.items():
            if src_remaining.get(src, 0) < c:
                return False
        return True

    def emit(self) -> list[intercept.Shot]:
        return list(self.shots)


WAVE_MARGIN = 1.10  # wave ROI must beat best single-source ROI by this factor

# Strike-window timing on enemy captures: schedule shots to land just after a
# projected enemy capture, when the post-capture garrison is at its weakest.
STRIKE_WINDOW_ALPHA = 0.7   # confidence damping: enemy may not actually launch
STRIKE_WINDOW_DELTAS = (1, 2, 3, 5)   # ticks after projected enemy arrival
STRIKE_WINDOW_MAX_HORIZON = 100   # cap arrival_step at obs_step + this


def plan_turn(
    world: dict,
    deadline: float,
    max_picks: int = 15,
    horizon_steps: int = 200,
    enable_waves: bool = False,
) -> list[intercept.Shot]:
    """ROI-greedy planner with optional wave bundling and robust enemy modeling.

    Wave gating (when enabled): emit a 2-source wave only when no single-source
    shot captures the target. Single shots are preferred whenever they suffice.

    `enable_waves` defaults to False: even with the strict no-single-shot gate,
    empirically waves commit ships from two sources prematurely and reduce later
    flexibility against v1. Capability remains for follow-up tuning (e.g., once
    multi-turn commitment continuation is added).
    """
    me = world["player"]
    obs_step = world["step"]
    by_id = world["planet_by_id"]

    # 1. Project enemy actions under both hypotheses (modest cost; bounded).
    end_step = obs_step + horizon_steps  # cap the "lifetime production" value at the horizon
    enemy_greedy = enemy_model.project_enemy_actions_greedy(world, k_shots_per_player=1, end_step=end_step)
    enemy_worst = enemy_model.project_enemy_actions_worst_for_us(world, k_shots_per_player=1, end_step=end_step)

    # 2. Defense reserve from the worst-case projection.
    reserve = scoring.defense_reserve_table(world, enemy_worst, horizon=WAVE_DEFENSE_HORIZON)

    # 3. Build candidate pool: single shots + (gated) waves.
    candidates: list[_Candidate] = []
    best_single_roi_per_target: dict[int, float] = {}

    # 3a. Single shots — reuse build_shot_menu with reserve.
    def reserve_fn(p):
        return reserve.get(p.id, 0)
    menu_deadline = time.perf_counter() + 0.25 * max(deadline - time.perf_counter(), 0)
    menu = intercept.build_shot_menu(world, defense_reserve_fn=reserve_fn, deadline=menu_deadline)
    for (src_id, tgt_id), shots in menu.items():
        tgt = by_id.get(tgt_id)
        if tgt is None:
            continue
        for shot in shots:
            roi_g = scoring.shot_roi(shot, tgt, world, end_step=end_step, enemy_arrivals=enemy_greedy)
            roi_w = scoring.shot_roi(shot, tgt, world, end_step=end_step, enemy_arrivals=enemy_worst)
            roi = min(roi_g, roi_w)
            if roi <= 0:
                continue
            candidates.append(_Candidate(
                shots=(shot,), tgt_id=tgt_id,
                arrival_step=obs_step + shot.eta, roi=roi,
            ))
            if roi > best_single_roi_per_target.get(tgt_id, 0.0):
                best_single_roi_per_target[tgt_id] = roi

    # 3a-bis. Strike-window candidates — shots timed to land just AFTER a
    # projected enemy capture, when the target's defender is at its absolute
    # weakest. The post-capture garrison projection is already handled by
    # scoring._defender_at; we just need to propose the candidate shots that
    # arrive in that window.
    my_planets_owned = [p for p in world["planets"] if p.owner == me]
    # Dedup projected arrivals by (planet, step) so greedy + worst overlaps
    # don't duplicate work.
    sw_seen: set[tuple[int, int]] = set()
    sw_cache = intercept.SweepCache(world["omega"], obs_step)
    for arr in list(enemy_greedy) + list(enemy_worst):
        if (arr.planet_id, arr.step) in sw_seen:
            continue
        sw_seen.add((arr.planet_id, arr.step))
        target = by_id.get(arr.planet_id)
        if target is None:
            continue
        # Target shouldn't be one of OUR planets (we don't strike-window our own).
        if target.owner == me:
            continue
        for delta in STRIKE_WINDOW_DELTAS:
            arrival_step = arr.step + delta
            if arrival_step <= obs_step or arrival_step > obs_step + STRIKE_WINDOW_MAX_HORIZON:
                continue
            for src in my_planets_owned:
                if src.id == target.id:
                    continue
                # Respect defense reserve: find_shot_for_arrival sees src.ships
                # but we must subtract the reserve manually since find_shot_for_arrival
                # doesn't take a reserve. Build a synthetic source with reduced ships.
                avail = src.ships - reserve.get(src.id, 0)
                if avail < 1:
                    continue
                src_view = dataclasses.replace(src, ships=avail)
                shot = intercept.find_shot_for_arrival(
                    src_view, target, arrival_step, world, cache=sw_cache,
                )
                if shot is None:
                    continue
                # Score under both enemy hypotheses (defender prediction uses
                # the same enemy_arrivals to keep the post-capture state).
                roi_g = scoring.shot_roi(shot, target, world,
                                          end_step=end_step, enemy_arrivals=enemy_greedy)
                roi_w = scoring.shot_roi(shot, target, world,
                                          end_step=end_step, enemy_arrivals=enemy_worst)
                roi = min(roi_g, roi_w)
                if roi <= 0:
                    continue
                damped = STRIKE_WINDOW_ALPHA * roi
                candidates.append(_Candidate(
                    shots=(shot,), tgt_id=target.id,
                    arrival_step=arrival_step, roi=damped,
                ))
                if damped > best_single_roi_per_target.get(target.id, 0.0):
                    best_single_roi_per_target[target.id] = damped

    # 3b. Waves — gated: emit only if no single-shot captures OR wave beats best
    # single-source ROI for that target by WAVE_MARGIN.
    if enable_waves:
        waves_deadline = time.perf_counter() + 0.25 * max(deadline - time.perf_counter(), 0)
        waves = bundling.candidate_waves(
            world, defense_reserve=reserve, extra_arrivals=enemy_worst,
            end_step=end_step, max_sources=2, deadline=waves_deadline,
        )
        for w in waves:
            tgt = by_id.get(w.target_id)
            if tgt is None:
                continue
            roi_g = scoring.wave_roi(list(w.shots), tgt, world, end_step=end_step, enemy_arrivals=enemy_greedy)
            roi_w = scoring.wave_roi(list(w.shots), tgt, world, end_step=end_step, enemy_arrivals=enemy_worst)
            roi = min(roi_g, roi_w)
            if roi <= 0:
                continue
            best_single = best_single_roi_per_target.get(w.target_id, 0.0)
            # Strict gate: only emit a wave when NO single-source shot captures
            # this target (best_single == 0). Single shots are preferred whenever
            # they suffice — bundling is for targets out of reach to any one source.
            if best_single > 0:
                continue
            candidates.append(_Candidate(
                shots=w.shots, tgt_id=w.target_id,
                arrival_step=w.arrival_step, roi=roi,
            ))

    # 3c. Deduplicate candidates by (sorted shot signature). Strike-window may
    # produce a shot identical to one already in build_shot_menu (same src,
    # tgt, eta, ship_count) — keep the higher-ROI one.
    def _cand_key(c: _Candidate) -> tuple:
        return tuple(sorted((s.src_id, s.tgt_id, s.eta, s.ship_count) for s in c.shots))
    deduped: dict[tuple, _Candidate] = {}
    for c in candidates:
        k = _cand_key(c)
        if k not in deduped or c.roi > deduped[k].roi:
            deduped[k] = c
    candidates = list(deduped.values())

    if not candidates:
        return []

    # 4. Greedy add by ROBUST plan-score improvement.
    src_remaining: dict[int, int] = {}
    for p in world["planets"]:
        if p.owner == me:
            src_remaining[p.id] = p.ships - reserve.get(p.id, 0)

    chosen_shots: list[intercept.Shot] = []
    chosen_targets: set[int] = set()  # don't double-commit on one target

    def robust_score(plan: list[intercept.Shot]) -> float:
        sg = prediction.plan_score(world, plan, horizon_steps=horizon_steps,
                                    extra_arrivals=enemy_greedy)
        sw = prediction.plan_score(world, plan, horizon_steps=horizon_steps,
                                    extra_arrivals=enemy_worst)
        # Weighted: prefer greedy-realistic (70%) but penalize plans that
        # collapse under worst-case (30%). Avoids over-pessimism vs benign
        # opponents while still being robust against aggressive ones.
        return 0.7 * sg + 0.3 * sw

    base = robust_score(chosen_shots)

    # Sort candidates by ROI to give the greedy a strong initial order.
    candidates.sort(key=lambda c: c.roi, reverse=True)

    # Track which Candidate produced each shot so we can swap chosen-out later.
    chosen_cands: list[_Candidate] = []

    picked = 0
    while picked < max_picks and time.perf_counter() < deadline:
        best_cand = None
        best_score = base
        for cand in candidates:
            if cand.tgt_id in chosen_targets:
                continue
            if not cand.fits(src_remaining):
                continue
            if time.perf_counter() >= deadline:
                break
            trial = chosen_shots + cand.emit()
            score = robust_score(trial)
            if score > best_score:
                best_score = score
                best_cand = cand
        if best_cand is None:
            break
        chosen_shots.extend(best_cand.emit())
        chosen_cands.append(best_cand)
        chosen_targets.add(best_cand.tgt_id)
        for src, c in best_cand.ship_cost_by_src.items():
            src_remaining[src] -= c
        base = best_score
        picked += 1

    # 5. One-pass swap-improvement: for each chosen candidate, try replacing it
    # with any unchosen one. Accept the swap if robust_score improves and the
    # deadline allows. Cheap insurance against early-greedy lock-in.
    swap_improved = True
    swap_passes = 0
    while swap_improved and swap_passes < 1 and time.perf_counter() < deadline:
        swap_improved = False
        swap_passes += 1
        for i in range(len(chosen_cands)):
            if time.perf_counter() >= deadline:
                break
            cur = chosen_cands[i]
            best_alt = None
            best_alt_score = base
            # Reconstruct ship budget without `cur`.
            budget_without = dict(src_remaining)
            for src, c in cur.ship_cost_by_src.items():
                budget_without[src] = budget_without.get(src, 0) + c
            chosen_without = [c for j, c in enumerate(chosen_cands) if j != i]
            shots_without = [s for c in chosen_without for s in c.emit()]
            targets_without = {c.tgt_id for c in chosen_without}
            for alt in candidates:
                if alt is cur:
                    continue
                if alt.tgt_id in targets_without:
                    continue
                if not alt.fits(budget_without):
                    continue
                if time.perf_counter() >= deadline:
                    break
                trial = shots_without + alt.emit()
                score = robust_score(trial)
                if score > best_alt_score:
                    best_alt_score = score
                    best_alt = alt
            if best_alt is not None:
                # Commit the swap.
                chosen_cands[i] = best_alt
                chosen_shots = [s for c in chosen_cands for s in c.emit()]
                chosen_targets = {c.tgt_id for c in chosen_cands}
                # Rebuild remaining budget from scratch (cheaper than diff).
                src_remaining = {p.id: p.ships - reserve.get(p.id, 0)
                                 for p in world["planets"] if p.owner == me}
                for c in chosen_cands:
                    for src, ships in c.ship_cost_by_src.items():
                        src_remaining[src] -= ships
                base = best_alt_score
                swap_improved = True

    return chosen_shots


WAVE_DEFENSE_HORIZON = 30


def emit_actions(plan: list[intercept.Shot]) -> list[list]:
    return [[shot.src_id, shot.angle, shot.ship_count] for shot in plan]
