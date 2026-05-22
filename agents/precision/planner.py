"""Global joint planner — v2.

ROI-greedy over single shots + 2-source waves, with robust min-max scoring
against TWO enemy hypotheses (greedy-theirs and worst-for-us). The plan that
maximizes its WORST-case score across the two enemy projections wins.

The frozen v1 planner remains available as `plan_turn_v1` for benchmarking.
"""
from __future__ import annotations

import dataclasses
import time
from agents.precision import bundling, enemy_model, fast_sim, intercept, prediction, scoring


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
    base_score = fast_sim.plan_score(world, chosen, horizon_steps=horizon_steps)

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
            score = fast_sim.plan_score(world, trial, horizon_steps=horizon_steps)
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
    """Wraps either a single Shot or a Wave for uniform handling.

    `extra_threats` is the post-commitment enemy projection that the planner
    attaches to wave candidates (iter 6 — closes the v2_frozen regression).
    Single-shot candidates leave it as `()`. When the greedy/swap scores a
    plan including this candidate, these arrivals are added to the worst-case
    enemy projection so the wave's depletion of source planets is priced
    correctly in the rollout.
    """
    __slots__ = ("shots", "tgt_id", "ship_cost_by_src", "total_ships",
                  "arrival_step", "roi", "extra_threats")

    def __init__(self, shots: tuple[intercept.Shot, ...], tgt_id: int,
                 arrival_step: int, roi: float,
                 extra_threats: tuple = ()):
        self.shots = shots
        self.tgt_id = tgt_id
        self.arrival_step = arrival_step
        self.roi = roi
        self.extra_threats = extra_threats
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


def plan_turn_v2_frozen(
    world: dict,
    deadline: float,
    max_picks: int = 15,
    horizon_steps: int = 200,
    enable_waves: bool = False,
) -> list[intercept.Shot]:
    """FROZEN snapshot of plan_turn after iteration 4 — pre-merge baseline.

    Kept for regression benchmarking (`tests/test_precision_v3_vs_v2_frozen.py`).
    The current `plan_turn` (defined below) IS the iteration-5 merge target.
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
        sg = fast_sim.plan_score(world, plan, horizon_steps=horizon_steps,
                                    extra_arrivals=enemy_greedy)
        sw = fast_sim.plan_score(world, plan, horizon_steps=horizon_steps,
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


# ---- v3 (merged): best-of-both-worlds, evolved across phases -------------

# Min-keep floor: each source must retain at least the configured fraction of
# its current garrison after picking a candidate. Tuned to balance defense
# (against the mid-game collapse seen vs main_v2) with offensive throughput.
# Empirical: 0.30 → too passive (0W-6L vs main); 0.10 → less passive; 0.0 →
# disabled (relies only on the reserve table).
MIN_KEEP_FRACTION = 0.0  # disabled by default; floor came from reserve table only


def _greedy_pick(world, candidates, reserve, robust_score, max_picks, deadline,
                 me, seeded_chosen=None, swap_passes: int = 1):
    """Greedy ROI selection with min-keep floor + optional swap-improvement.

    Returns (chosen_shots, chosen_cands, src_remaining, base_score).
    """
    src_remaining: dict[int, int] = {}
    src_keep_floor: dict[int, int] = {}
    for p in world["planets"]:
        if p.owner == me:
            r = reserve.get(p.id, 0)
            src_remaining[p.id] = p.ships - r
            # Floor is an ADDITIONAL safety margin on top of the reserve. The
            # reserve is already subtracted from src_remaining once — don't
            # apply it again here or we double-count. With MIN_KEEP_FRACTION=0
            # the floor is disabled, matching v2_frozen behaviour.
            src_keep_floor[p.id] = int(MIN_KEEP_FRACTION * p.ships)

    chosen_shots: list[intercept.Shot] = []
    chosen_cands: list = []
    chosen_targets: set[int] = set()

    # Pre-commit any seeded candidates (commitment continuation, Phase 3).
    if seeded_chosen:
        for cand in seeded_chosen:
            # Re-validate budget under current floor.
            ok = True
            for src, cost in cand.ship_cost_by_src.items():
                planet = world["planet_by_id"].get(src)
                if planet is None or planet.owner != me:
                    ok = False; break
                # After this commit, ensure source stays above floor.
                if src_remaining.get(src, 0) - cost < 0:
                    ok = False; break
                if planet.ships - cost < src_keep_floor.get(src, 0):
                    ok = False; break
            if not ok or cand.tgt_id in chosen_targets:
                continue
            chosen_shots.extend(cand.emit())
            chosen_cands.append(cand)
            chosen_targets.add(cand.tgt_id)
            for src, c in cand.ship_cost_by_src.items():
                src_remaining[src] -= c

    base = robust_score(chosen_shots)
    candidates_sorted = sorted(candidates, key=lambda c: c.roi, reverse=True)

    picked = len(chosen_cands)
    while picked < max_picks and time.perf_counter() < deadline:
        best_cand = None
        best_score = base
        for cand in candidates_sorted:
            if cand.tgt_id in chosen_targets:
                continue
            # Budget check + min-keep floor.
            ok = True
            for src, cost in cand.ship_cost_by_src.items():
                if src_remaining.get(src, 0) < cost:
                    ok = False; break
                planet = world["planet_by_id"].get(src)
                if planet is None:
                    ok = False; break
                already_spent = planet.ships - src_remaining[src]
                if planet.ships - already_spent - cost < src_keep_floor.get(src, 0):
                    ok = False; break
            if not ok:
                continue
            if time.perf_counter() >= deadline:
                break
            trial = chosen_shots + cand.emit()
            # Aggregate all chosen candidates' extra_threats + this candidate's.
            # Single-shot candidates have empty extra_threats; only waves carry
            # non-empty post-commitment threat lists (iter 6).
            agg_extras = tuple(t for c in chosen_cands for t in c.extra_threats) + cand.extra_threats
            score = robust_score(trial, extra_threats=agg_extras)
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

    # Swap-improvement pass: replace each chosen candidate with an alternative
    # if it improves robust_score. Matches v2_frozen's behaviour.
    for _ in range(swap_passes):
        if time.perf_counter() >= deadline:
            break
        improved = False
        for i in range(len(chosen_cands)):
            if time.perf_counter() >= deadline:
                break
            cur = chosen_cands[i]
            best_alt = None
            best_alt_score = base
            budget_without = dict(src_remaining)
            for src, c in cur.ship_cost_by_src.items():
                budget_without[src] = budget_without.get(src, 0) + c
            chosen_without = [c for j, c in enumerate(chosen_cands) if j != i]
            shots_without = [s for c in chosen_without for s in c.emit()]
            targets_without = {c.tgt_id for c in chosen_without}
            for alt in candidates_sorted:
                if alt is cur or alt.tgt_id in targets_without:
                    continue
                # Budget + min-keep floor check.
                ok = True
                for src, cost in alt.ship_cost_by_src.items():
                    if budget_without.get(src, 0) < cost:
                        ok = False; break
                    planet = world["planet_by_id"].get(src)
                    if planet is None:
                        ok = False; break
                    already = planet.ships - budget_without[src]
                    if planet.ships - already - cost < src_keep_floor.get(src, 0):
                        ok = False; break
                if not ok:
                    continue
                if time.perf_counter() >= deadline:
                    break
                trial = shots_without + alt.emit()
                # Same as greedy: aggregate extra_threats of the surviving picks + alt.
                agg_extras = tuple(t for c in chosen_without for t in c.extra_threats) + alt.extra_threats
                score = robust_score(trial, extra_threats=agg_extras)
                if score > best_alt_score:
                    best_alt_score = score
                    best_alt = alt
            if best_alt is not None:
                chosen_cands[i] = best_alt
                chosen_shots = [s for c in chosen_cands for s in c.emit()]
                chosen_targets = {c.tgt_id for c in chosen_cands}
                src_remaining = {p.id: p.ships - reserve.get(p.id, 0)
                                  for p in world["planets"] if p.owner == me}
                for c in chosen_cands:
                    for src, ships in c.ship_cost_by_src.items():
                        src_remaining[src] -= ships
                base = best_alt_score
                improved = True
        if not improved:
            break

    return chosen_shots, chosen_cands, src_remaining, base


def _world_after_wave(world: dict, wave) -> dict:
    """Build a shallow-cloned world dict with each wave source's `ships`
    debited by its contribution to the wave.

    Used by the iter-6 post-commitment projection: project the enemy's
    response to our DEPLETED state. Only `planets` + `planet_by_id` are
    re-built; other fields (omega, step, fleets, comets, …) are aliased,
    which is safe because the enemy projection only reads them.
    """
    new_planets = []
    new_by_id: dict[int, intercept.PlanetView] = {}
    for p in world["planets"]:
        delta = wave.shots and sum(
            s.ship_count for s in wave.shots if s.src_id == p.id
        )
        if delta:
            p_new = dataclasses.replace(p, ships=max(0, p.ships - delta))
        else:
            p_new = p
        new_planets.append(p_new)
        new_by_id[p_new.id] = p_new
    return {
        **world,
        "planets": new_planets,
        "planet_by_id": new_by_id,
    }


def plan_turn(
    world: dict,
    deadline: float,
    max_picks: int = 15,
    horizon_steps: int = 200,
) -> list[intercept.Shot]:
    """Iteration-5 merged planner: fast-sim + extended defense + min-keep floor.

    Phases applied:
      1. Fast forward-sim (already swapped in via scoring/planner imports).
      2. Defense horizon 60 ticks, in-flight enemy fleets folded in,
         k_shots_per_player=2, min-keep floor on each source.
      3. Commitment continuation (deferred to next phase).
      4. Wave bundling with post-commitment projection (deferred).
    """
    me = world["player"]
    obs_step = world["step"]
    by_id = world["planet_by_id"]
    end_step = obs_step + horizon_steps

    # 1. Project enemy actions (k=2 per player) under both hypotheses.
    enemy_greedy = enemy_model.project_enemy_actions_greedy(world, end_step=end_step)
    # Depth-2 enemy minimax: project two turns of worst-for-us enemy actions.
    # Catches cascading threats — same-aggression opponents (e.g. v2_frozen)
    # attack our depleted sources on turn t+1 then exploit further on t+2,
    # which the depth-1 projection couldn't see. Replaces the prior depth-1
    # `project_enemy_actions_worst_for_us` for ALL candidate scoring.
    enemy_worst = enemy_model.project_two_turns(world, end_step=end_step)

    # 2. Defense reserves: 30-tick horizon (matches v2_frozen baseline; longer
    # horizons made us too passive vs same-aggression opponents). In-flight
    # enemy fleets NOT included in the reserve — relying on production
    # growth + counter-attack as in v2.
    reserve = scoring.defense_reserve_table(world, enemy_worst, horizon=30,
                                             include_in_flight=False)

    # 3. Build candidate pool (single shots + strike-window only for now;
    #    waves return in Phase 4 with the fixed gate).
    candidates: list[_Candidate] = []
    best_single_roi_per_target: dict[int, float] = {}

    def reserve_fn(p):
        return reserve.get(p.id, 0)

    menu_deadline = time.perf_counter() + 0.25 * max(deadline - time.perf_counter(), 0)
    menu = intercept.build_shot_menu(world, defense_reserve_fn=reserve_fn,
                                       deadline=menu_deadline)
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

    # 3-bis. Strike-window candidates (iter 4 mechanic, kept).
    my_planets_owned = [p for p in world["planets"] if p.owner == me]
    sw_seen: set[tuple[int, int]] = set()
    sw_cache = intercept.SweepCache(world["omega"], obs_step)
    for arr in list(enemy_greedy) + list(enemy_worst):
        if (arr.planet_id, arr.step) in sw_seen:
            continue
        sw_seen.add((arr.planet_id, arr.step))
        target = by_id.get(arr.planet_id)
        if target is None or target.owner == me:
            continue
        for delta in STRIKE_WINDOW_DELTAS:
            arrival_step = arr.step + delta
            if arrival_step <= obs_step or arrival_step > obs_step + STRIKE_WINDOW_MAX_HORIZON:
                continue
            for src in my_planets_owned:
                if src.id == target.id:
                    continue
                avail = src.ships - reserve.get(src.id, 0)
                if avail < 1:
                    continue
                src_view = dataclasses.replace(src, ships=avail)
                shot = intercept.find_shot_for_arrival(
                    src_view, target, arrival_step, world, cache=sw_cache,
                )
                if shot is None:
                    continue
                roi_g = scoring.shot_roi(shot, target, world, end_step=end_step, enemy_arrivals=enemy_greedy)
                roi_w = scoring.shot_roi(shot, target, world, end_step=end_step, enemy_arrivals=enemy_worst)
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

    # 3b. Wave bundling — gated strictly: only emit a wave when NO single-source
    # shot for that target has positive ROI. Single shots dominate when they
    # suffice; bundling is reserved for targets beyond reach of any one source.
    #
    # Iter-6: each emitted wave carries `extra_threats` = the enemy's worst
    # response projected from the POST-WAVE world. This is folded into the
    # rollout's worst-case projection when the greedy scores this candidate,
    # so the wave pays for the depletion it creates at its sources.
    waves_deadline = time.perf_counter() + 0.15 * max(deadline - time.perf_counter(), 0)
    waves = bundling.candidate_waves(
        world, defense_reserve=reserve, extra_arrivals=enemy_worst,
        end_step=end_step, max_sources=2, deadline=waves_deadline,
    )
    # Cache the post-wave projection by (src_id, post_wave_ships) signature.
    # Many waves share source sets; this keeps the extra cost bounded.
    post_resp_cache: dict[tuple, tuple] = {}

    def _post_wave_threats(wv) -> tuple:
        # Cache key: (src_id, ships_after_wave) per participating source.
        # Different waves depleting the same sources to the same garrison level
        # produce the same enemy response (which only depends on our resources).
        per_src_cost: dict[int, int] = {}
        for s in wv.shots:
            per_src_cost[s.src_id] = per_src_cost.get(s.src_id, 0) + s.ship_count
        sig = tuple(sorted(
            (src, max(0, by_id[src].ships - cost))
            for src, cost in per_src_cost.items()
        ))
        if sig in post_resp_cache:
            return post_resp_cache[sig]
        # Skip the extra projection if we're tight on time; fall back to ().
        if time.perf_counter() >= deadline - 0.20:
            post_resp_cache[sig] = ()
            return ()
        post_world = _world_after_wave(world, wv)
        # Depth-2: project two turns of enemy response from the post-wave world.
        # Wave commits ships from two sources; same-aggression opponents pick on
        # the weaker of the two on t+1, then sweep further on t+2.
        resp = enemy_model.project_two_turns(post_world, end_step=end_step)
        resp_tup = tuple(resp)
        post_resp_cache[sig] = resp_tup
        return resp_tup

    for wv in waves:
        tgt = by_id.get(wv.target_id)
        if tgt is None:
            continue
        # Strict gate: skip the wave if any single-shot already captures it.
        if best_single_roi_per_target.get(wv.target_id, 0.0) > 0:
            continue
        # Robustness gate: each source must retain at least 3× its wave
        # contribution post-firing — i.e. spending ≤ 25% of garrison per
        # source per wave. Empirically tuned to close the v2_frozen
        # regression: the 50%-spend threshold was too generous and left
        # sources captureable by same-aggression opponents.
        per_src_cost: dict[int, int] = {}
        for s in wv.shots:
            per_src_cost[s.src_id] = per_src_cost.get(s.src_id, 0) + s.ship_count
        too_thin = False
        for src_id, cost in per_src_cost.items():
            src = by_id.get(src_id)
            if src is None or src.ships - cost < 3 * cost:
                too_thin = True
                break
        if too_thin:
            continue
        # Post-commitment projection: enemy's counter-strikes after our wave.
        post_threats = _post_wave_threats(wv)
        enriched_worst = list(enemy_worst) + list(post_threats)
        roi_g = scoring.wave_roi(list(wv.shots), tgt, world, end_step=end_step, enemy_arrivals=enemy_greedy)
        roi_w = scoring.wave_roi(list(wv.shots), tgt, world, end_step=end_step, enemy_arrivals=enriched_worst)
        roi = min(roi_g, roi_w)
        if roi <= 0:
            continue
        candidates.append(_Candidate(
            shots=wv.shots, tgt_id=wv.target_id,
            arrival_step=wv.arrival_step, roi=roi,
            extra_threats=post_threats,
        ))

    # 3c. Dedup by sorted shot signature.
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

    def robust_score(plan: list[intercept.Shot],
                     extra_threats: tuple = ()) -> float:
        """Score a plan under both enemy hypotheses.

        `extra_threats` is an optional list of additional projected enemy
        Arrivals (typically wave-specific post-commitment counter-strikes).
        Folded into the worst-case projection only, since they represent
        adversarial response to OUR commitments.
        """
        enriched_worst = enemy_worst + list(extra_threats)
        sg = fast_sim.plan_score(world, plan, horizon_steps=horizon_steps,
                                  extra_arrivals=enemy_greedy)
        sw = fast_sim.plan_score(world, plan, horizon_steps=horizon_steps,
                                  extra_arrivals=enriched_worst)
        return 0.7 * sg + 0.3 * sw

    chosen_shots, chosen_cands, src_remaining, base = _greedy_pick(
        world, candidates, reserve, robust_score, max_picks, deadline, me,
    )

    return chosen_shots


def emit_actions(plan: list[intercept.Shot]) -> list[list]:
    return [[shot.src_id, shot.angle, shot.ship_count] for shot in plan]
