"""iter — fast-iteration fork of v7_pv (= v7_0_drop_one + PV_GAMMA=0.99).

Day-zero behaviour: functionally equivalent to v7_pv (ladder mu=1064.4).
Edit the knobs below to A/B variants; add code under the PRE_FILTER /
POST_PROCESS hooks to fix specific bugs observed in live replays.

Eval cycle:
    python fast.py eval iter --vs-panel default --max-seeds 32       # 2P
    python -m scripts.ffa_panel --focals agents/iter/main.py --seeds 32   # 4P

Bundle + parity:
    python scripts/bundle_agent.py agents/iter
    pytest tests/test_iter_agent.py tests/test_bundle.py -q

See agents/iter/README.md for the four patch surfaces and submission gate.
"""

from __future__ import annotations

# ============================================================================
# ITER KNOBS — edit these for a knob sweep, one line per variant.
# ============================================================================
K = 10                          # lookahead horizon — matches iter_v1's shipped config
WALLCLOCK_MS = 700.0            # per-turn budget (ms); matches iter_v1
ENUMERATOR_MODE = "drop_one"    # iter_v1 default
OPP_TIERS = (1,)                # iter_v1 default
PV_GAMMA = 0.99                 # iter_v1 / v7_pv equivalent
VALUE_FN = "composite"          # iter_v1 VALIDATED head (composite_capture_value)
DEFENSIBILITY_ALPHA = 0.2       # inert (VALUE_FN=composite)
TERRITORY_WEIGHT = 0.01         # inert (VALUE_FN=composite)
K_4P = 8                        # 4P-branch lookahead — IS USED by 4P dispatch (the new piece)

# --- Adaptive K — DISABLED (2026-05-15 strip back) --------------------------
# K_CAP=K means K_eff = K = 10 always; relevance-filter is inert. Set to
# match iter_v1 behaviour exactly for the 2P branch. Adaptive K had neutral
# panel signal but adds variance; stripping for the conservative submit.
K_CAP = 10                      # = K → K_eff is always K
K_BUFFER = 0
RELEVANCE_PROD_FRACTION = 0.5   # inert when K_CAP = K

# --- Comet anti-panic (Bug 2 POST-PROCESS) ----------------------------------
COMET_MAX_LAUNCHES_PER_TURN = 999  # cap effectively disabled; ablation showed it costs 8-9 pp
                                    # vs v7_0/v4_planner (chooser's multi-source choice was usually right)

# --- Comet evacuation — DISABLED (2026-05-15 strip back) --------------------
# THRESHOLD=0 means evac never fires. Was neutral in panel; removing for the
# conservative submit to match iter_v1 behaviour exactly.
COMET_EVAC_THRESHOLD = 0
COMET_EVAC_RESERVE = 1

# --- Two-phase scoring (2026-05-15) -----------------------------------------
# Phase 1: cheap analytical leaf evaluation on ALL drop-one candidates via
# WorldModel timeline propagation (no rollout, no opp model). Phase 2: deep
# K_DEEP rollout via lib.v7_search.score_candidate on the top-PHASE2_TOP_K
# Phase-1 candidates only. Unlocks K_DEEP=18-22 within the 700 ms budget
# because we don't burn rollout cost on obviously-weak candidates.
TWO_PHASE = False               # default OFF — eval baseline (sizing fixes only) first
PHASE1_HORIZON = 50             # horizon for Phase 1 analytical evaluation
PHASE2_TOP_K = 3                # # of top Phase-1 finalists to deep-score (always includes incumbent)
K_DEEP = 18                     # deep K for Phase 2 (overrides K when TWO_PHASE=True; K_CAP still bounds)

# --- Latest-launch heuristic (2026-05-15) -----------------------------------
# POST-PROCESS shrink: for each launch in the chooser's action, binary-search
# the smallest ship count whose ETA still beats threat_eta - LATEST_LAUNCH_BUFFER_TURNS
# at the target. Conserves ships without losing the engagement. Default OFF.
LATEST_LAUNCH_ENABLED = False
LATEST_LAUNCH_BUFFER_TURNS = 1  # safety margin: arrive at threat_eta - this
LATEST_LAUNCH_MIN_FLEET = 2     # never shrink below this (avoid degenerate 1-ship near-zero-speed fleets)

# --- Mission persistence + multi-source coordination (2026-05-15) -----------
# Plan-style commitment across turns. At turn 0 (or whenever the plan
# invalidates) we pick a best opening target and allocate ships from multiple
# of our planets to combine against that target. Each source's launch is
# scheduled at its earliest feasible turn (now if affordable, growth-delayed
# otherwise). The plan persists across turns until the target is captured
# (by us or enemy) — fixes the "hesitate past OPENING_WINDOW" pattern in
# 61% of iter_v1's ladder losses + matches top players' multi-source
# whittling pattern.
MISSION_PERSISTENCE_ENABLED = False  # disabled 2026-05-15: panel -42 pp vs v7_0 in 32-seed eval
MP_OPENING_WINDOW = 12          # plan-builder only fires at step <= this (vs lib's OPENING_WINDOW=5)
MP_SHIPS_SAFETY = 2             # add this to base capture cost (= t.ships+1+safety) for buffer
MP_SOURCE_RESERVE = 0           # leave at least this many ships behind on each source. 0 in opening
                                # matches top players emptying homes; sweep candidate (0/2/5).
MP_DELAY_PENALTY_PER_TURN = 0.05  # plan-builder ROI penalises late launches
MP_MAX_SOURCES_PER_TARGET = 3   # cap how many of our planets contribute to one capture
# ============================================================================

# Dev-mode: override lib.scoring.PV_GAMMA BEFORE v7_search imports propagate
# the `from lib.scoring import PV_GAMMA` bindings into snipe/reinforce.
# Bundled form: lib.scoring is not a separate module (concatenated above),
# so this import raises ImportError and we rely on the module-scope
# PV_GAMMA rebind above — every callsite looks PV_GAMMA up by name at
# call time (verified across snipe.py + reinforce.py).
try:
    import lib.scoring as _scoring
    _scoring.PV_GAMMA = PV_GAMMA
except ImportError:
    pass

from lib.v7_search import choose, choose_4p
# Private helpers imported at module scope so the bundler preserves indent
# semantics inside `_choose_two_phase`.
from lib.v7_search import _build_incumbent_intents as _v7_build_incumbent
from lib.v7_search import _action_from_intents as _v7_action_from_intents
from lib.v7_search import _enumerate_drop_one as _v7_enumerate_drop_one
from lib.v7_search import score_candidate as _v7_score_candidate
from lib.fast_sim import from_obs as _v7_fs_from_obs
from lib.intent import World
from lib.world_model import fleet_target_planet, comet_remaining_lifetime, fleet_speed, WorldModel, simulate_planet_timeline
import math
import time


# ============================================================================
# Mission persistence — module-level plan state (carries across turns within
# one game; reset at turn 0). Architecturally distinct from per-turn missions.
# ============================================================================
_PLAN_STATE: dict = {
    "epoch": -1,              # turn when plan was built
    "target_id": None,         # planet id we committed to capture
    "src_ids": [],             # planet ids contributing ships to the capture
    "scheduled_launches": [],  # list of (fire_turn, src_id, angle, ships)
}


def _has_inflight_us_toward(world, my_id: int, target_id: int) -> bool:
    """True if any of our in-flight fleets ray-casts to `target_id`.

    Used by `_is_plan_invalid` to keep the plan alive while our fleets
    are still in transit toward the committed target (avoids re-planning
    mid-execution and double-allocating ships).
    """
    from kaggle_environments.envs.orbit_wars.orbit_wars import Fleet
    fleets_raw = _fleets_raw_from_world(world)
    if not fleets_raw:
        return False
    planets_list = list(world.planets_by_id.values())
    if not planets_list:
        return False
    for f_raw in fleets_raw:
        try:
            f = Fleet(*f_raw)
        except TypeError:
            continue
        if int(f.owner) != my_id:
            continue
        tgt, _eta = fleet_target_planet(f, planets_list)
        if tgt is not None and int(tgt.id) == int(target_id):
            return True
    return False


def _is_plan_invalid(world, my_id: int) -> bool:
    """Plan invalidates when: target is captured by anyone (us or enemy),
    target planet vanished, all our committed sources are lost, OR all
    scheduled launches have fired AND no inflight friendly fleet is still
    heading to the target (so the plan has fully resolved).
    """
    target_id = _PLAN_STATE.get("target_id")
    if target_id is None:
        return True
    target = world.planets_by_id.get(target_id)
    if target is None:
        return True
    if target.owner == my_id:
        # We captured it — plan succeeded; rebuild for next target.
        return True
    if target.owner != -1 and target.owner != my_id:
        # An enemy took it — plan failed; rebuild.
        return True
    src_ids = _PLAN_STATE.get("src_ids", [])
    if not src_ids:
        return True
    sources_alive = False
    for sid in src_ids:
        s = world.planets_by_id.get(int(sid))
        if s is not None and s.owner == my_id:
            sources_alive = True
            break
    if not sources_alive and not _has_inflight_us_toward(world, my_id, target_id):
        return True
    # Resolution check: all launches fired AND no friendly fleet still
    # heading to the target → time to plan the next move.
    if (not _PLAN_STATE.get("scheduled_launches")
            and not _has_inflight_us_toward(world, my_id, target_id)):
        return True
    return False


def _build_opening_plan(world, my_id: int) -> dict:
    """Multi-source coordinated opening plan.

    For each candidate target, greedily allocate ships from the closest
    `MP_MAX_SOURCES_PER_TARGET` sources until the sum exceeds the target's
    garrison + safety buffer. Each source's launch is scheduled at the
    earliest turn it can afford its allocation (`src.ships >= ships`),
    growth-delayed if needed. The plan persists across turns; each launch
    fires when its scheduled turn arrives.

    Multi-source coordination means: even when no single planet alone has
    enough ships, multiple planets combine their fleets toward one target.
    Fleets that bounce off the target's garrison still WHITTLE it, so a
    subsequent wave captures cheaply. (We don't model the bouncing
    explicitly here — we just sum the ship contributions; the mission
    framework's combat resolution handles the rest.)

    Returns a plan dict with `target_id`, `src_ids`, `scheduled_launches`.
    Returns an empty plan if no target is reachable with our combined
    ship budget within the opening window.
    """
    plan = {"epoch": int(world.step), "target_id": None, "src_ids": [],
            "scheduled_launches": []}
    if int(world.step) > MP_OPENING_WINDOW:
        return plan
    my_planets = [p for p in world.planets_by_id.values() if p.owner == my_id]
    targets = [
        p for p in world.planets_by_id.values()
        if p.owner == -1 and p.id not in world.comet_ids
    ]
    if not my_planets or not targets:
        return plan

    step_now = int(world.step)
    best_plan = None
    best_score = float("-inf")

    for t in targets:
        base_capture = int(t.ships) + 1 + MP_SHIPS_SAFETY
        sources_by_dist = sorted(
            my_planets,
            key=lambda s: math.hypot(t.x - s.x, t.y - s.y),
        )[:MP_MAX_SOURCES_PER_TARGET]
        # Find the earliest step at which our combined available ships
        # from these sources reach `base_capture`. All launches fire at
        # that single feasible turn (simpler than per-source timing;
        # whittle-and-finish is a future iteration).
        feasible_turn = None
        for step_t in range(step_now, MP_OPENING_WINDOW + 1):
            total_avail = 0
            for src in sources_by_dist:
                ships_at_t = int(src.ships) + (step_t - step_now) * max(1, int(src.production))
                total_avail += max(0, ships_at_t - MP_SOURCE_RESERVE)
                if total_avail >= base_capture:
                    break
            if total_avail >= base_capture:
                feasible_turn = step_t
                break
        if feasible_turn is None:
            continue   # can't cover this target within opening window
        # Allocate ships from sources, all firing at feasible_turn.
        allocations = []
        total_attack = 0
        for src in sources_by_dist:
            if total_attack >= base_capture:
                break
            ships_at_t = int(src.ships) + (feasible_turn - step_now) * max(1, int(src.production))
            avail = max(0, ships_at_t - MP_SOURCE_RESERVE)
            if avail <= 0:
                continue
            ships_send = min(avail, base_capture - total_attack)
            if ships_send <= 0:
                continue
            allocations.append((src, ships_send, feasible_turn))
            total_attack += ships_send
        if total_attack < base_capture:
            continue
        # ROI: production-of-target divided by closest source distance,
        # penalised by feasible launch turn (later is worse).
        closest_d = math.hypot(t.x - sources_by_dist[0].x, t.y - sources_by_dist[0].y)
        score = (float(t.production) / (closest_d + 1.0)
                 - feasible_turn * MP_DELAY_PENALTY_PER_TURN)
        if score > best_score:
            best_score = score
            best_plan = (t, allocations)

    if best_plan is None:
        return plan
    target, allocations = best_plan
    plan["target_id"] = int(target.id)
    plan["src_ids"] = sorted({int(s.id) for (s, _n, _l) in allocations})
    plan["scheduled_launches"] = [
        (int(launch_turn), int(s.id),
         float(math.atan2(target.y - s.y, target.x - s.x)),
         int(ships_send))
        for (s, ships_send, launch_turn) in allocations
    ]
    return plan


def _execute_planned_launches(world, current_turn: int, my_id: int) -> list:
    """Return launches in _PLAN_STATE scheduled for `current_turn` that pass
    a final-validity gate (source still ours with enough ships). Mutates
    `_PLAN_STATE["scheduled_launches"]` to drop fired/invalid entries."""
    scheduled = _PLAN_STATE.get("scheduled_launches", [])
    if not scheduled:
        return []
    fired: list = []
    remaining: list = []
    for entry in scheduled:
        try:
            fire_turn, src_id, angle, ships = entry
        except (ValueError, TypeError):
            continue
        if int(fire_turn) > current_turn:
            remaining.append(entry)
            continue
        if int(fire_turn) < current_turn:
            # Expired — skip (might have been blocked by an earlier turn's check).
            continue
        src = world.planets_by_id.get(int(src_id))
        if src is None or src.owner != my_id:
            continue
        if int(src.ships) < int(ships):
            continue
        fired.append([int(src_id), float(angle), int(ships)])
    _PLAN_STATE["scheduled_launches"] = remaining
    return fired


def _merge_planned_with_action(action: list, planned: list) -> list:
    """Merge planned launches with the chooser's action. Each source can only
    launch once per turn — planned launches take precedence and replace any
    chooser launches from the same source."""
    if not planned:
        return action or []
    plan_srcs = {int(p[0]) for p in planned}
    base = [a for a in (action or []) if a and int(a[0]) not in plan_srcs]
    return base + planned


def _resolve_value_fn(name):
    if name == "default":
        return None  # lib.v7_search.score_candidate defaults to delta_us_minus_them
    if name == "composite":
        from lib.value_heads import composite_capture_value
        return composite_capture_value
    if name == "defensibility":
        from lib.value_heads import defensibility_value
        return lambda obs, mid: defensibility_value(obs, mid, weight=DEFENSIBILITY_ALPHA)
    if name == "composite_plus_defensibility":
        from lib.value_heads import composite_plus_defensibility
        return lambda obs, mid: composite_plus_defensibility(
            obs, mid, defensibility_weight=DEFENSIBILITY_ALPHA
        )
    if name == "territory":
        from lib.value_heads import territory_value
        return lambda obs, mid: territory_value(obs, mid, weight=TERRITORY_WEIGHT)
    if name == "composite_plus_territory":
        from lib.value_heads import composite_plus_territory
        return lambda obs, mid: composite_plus_territory(
            obs, mid, territory_weight=TERRITORY_WEIGHT
        )
    raise ValueError(f"unknown VALUE_FN: {name!r}")


def _detect_num_seats(world) -> int:
    """Infer seat count from owner IDs on planets + in-flight fleets.

    Inline reimplementation of `lib.v7_search._infer_num_seats` so we don't
    reach into a private helper. 2P if max non-neutral owner ID is ≤ 1;
    4P if max is ≥ 2.
    """
    max_id = -1
    for p in world.planets_by_id.values():
        if p.owner >= 0 and p.owner > max_id:
            max_id = p.owner
    raw = world.obs_raw
    fleets = raw.get("fleets", []) if isinstance(raw, dict) else getattr(raw, "fleets", [])
    for f in fleets:
        owner = int(f[1])
        if owner >= 0 and owner > max_id:
            max_id = owner
    return 4 if max_id >= 2 else 2


def _fleets_raw_from_world(world):
    raw = world.obs_raw
    return raw.get("fleets", []) if isinstance(raw, dict) else getattr(raw, "fleets", [])


def _relevant_target_ids(world, prod_fraction: float) -> set:
    """Planet ids whose state changes matter to our short-term decisions.

    Includes (a) every planet we own (defensibility), (b) every planet with
    production >= prod_fraction × max_production (high-value to capture or
    deny). Excludes obscure low-prod neutrals far from action.
    """
    my_id = world.my_id
    planets = list(world.planets_by_id.values())
    if not planets:
        return set()
    max_prod = max(p.production for p in planets) or 1
    threshold = float(max_prod) * float(prod_fraction)
    out = set()
    for p in planets:
        if p.owner == my_id or float(p.production) >= threshold:
            out.add(int(p.id))
    return out


def _max_inflight_eta(world) -> int:
    """Max ETA of in-flight fleets whose RAY-CAST target is "relevant".

    Relevance filter: we extend K_eff only when there's a fleet inbound to
    one of OUR planets (defensibility) or to a high-production planet
    (capture/denial). Long fleets headed for obscure corners don't inflate
    K — so we can afford a higher K_CAP without blowing wallclock.

    Cost: O(N_fleets * N_planets) for the ray-cast loop + a single relevance
    set construction. Both cheap.
    """
    from kaggle_environments.envs.orbit_wars.orbit_wars import Fleet
    fleets_raw = _fleets_raw_from_world(world)
    if not fleets_raw:
        return 0
    planets_list = list(world.planets_by_id.values())
    if not planets_list:
        return 0
    relevant = _relevant_target_ids(world, RELEVANCE_PROD_FRACTION)
    max_eta = 0
    for f_raw in fleets_raw:
        try:
            f = Fleet(*f_raw)
        except TypeError:
            continue
        tgt, eta = fleet_target_planet(f, planets_list)
        if tgt is None or eta is None:
            continue
        if int(tgt.id) not in relevant:
            continue
        if int(eta) > max_eta:
            max_eta = int(eta)
    return max_eta


def _launch_eta(src_planet, angle_rad: float, ships: int, planets_list) -> tuple:
    """Predicted (target_planet, eta) for a NEW launch we're about to make.

    The env ray-casts every launch along its angle; we mirror that to map
    a chosen launch back to a target. Builds a synthetic Fleet at the
    source planet's centre + chosen angle + ship count.
    """
    from kaggle_environments.envs.orbit_wars.orbit_wars import Fleet
    spd = fleet_speed(ships)
    if spd <= 0:
        return None, None
    # Fleet schema (per env): id, owner, x, y, angle, speed, ships, ...
    # We only need fields ray-cast reads: x, y, angle, ships.
    # Build a partial Fleet with the minimum fields the helper consumes.
    f = Fleet(-1, int(src_planet.owner), float(src_planet.x), float(src_planet.y),
              float(angle_rad), float(spd), int(ships))
    return fleet_target_planet(f, planets_list)


def _cap_comet_launches(action, world, cap: int):
    """Strip excess launches targeting the SAME comet (Bug 2 POST-PROCESS).

    Action entries are `[src_id, angle_rad, ships]`. For each launch, ray-
    cast to find its predicted target via `_launch_eta`. Group by target
    planet id; for targets in `world.comet_ids` keep at most `cap` entries
    (prefer shortest ETA = first to arrive). Non-comet targets pass through.

    Hard fallback: if `cap >= len(action)` or no comets present, no-op.
    """
    if not action or cap <= 0:
        return action
    comet_ids = world.comet_ids
    if not comet_ids:
        return action
    planets_list = list(world.planets_by_id.values())
    by_comet: dict = {}      # target_id -> list of (eta, idx)
    other_idx = []
    for i, entry in enumerate(action):
        try:
            src_id, angle, ships = int(entry[0]), float(entry[1]), int(entry[2])
        except (ValueError, TypeError, IndexError):
            other_idx.append(i)
            continue
        src = world.planets_by_id.get(src_id)
        if src is None:
            other_idx.append(i)
            continue
        tgt, eta = _launch_eta(src, angle, ships, planets_list)
        if tgt is None or tgt.id not in comet_ids:
            other_idx.append(i)
            continue
        by_comet.setdefault(tgt.id, []).append((int(eta), i))
    if not by_comet:
        return action
    keep_idx = set(other_idx)
    for tid, entries in by_comet.items():
        entries.sort(key=lambda x: x[0])  # shortest ETA first
        keep_idx.update(idx for _, idx in entries[:cap])
    return [a for i, a in enumerate(action) if i in keep_idx]


def _comet_evacuation_launches(world, my_id: int):
    """Emit launches FROM our short-lifetime comets to a safe destination.

    For each planet we own that is also a comet with
    `remaining_lifetime <= COMET_EVAC_THRESHOLD`: emit a launch carrying
    `ships - COMET_EVAC_RESERVE` ships at the angle pointing to the
    nearest non-comet planet. Skip if no non-comet target exists or if
    the comet has too few ships.
    """
    launches = []
    sources = [
        p for p in world.planets_by_id.values()
        if p.owner == my_id and p.id in world.comet_ids
    ]
    if not sources:
        return launches
    non_comet_targets = [
        p for p in world.planets_by_id.values()
        if p.id not in world.comet_ids
    ]
    if not non_comet_targets:
        return launches
    for c in sources:
        rem = comet_remaining_lifetime(c.id, world)
        if rem is None or rem > COMET_EVAC_THRESHOLD:
            continue
        if int(c.ships) <= COMET_EVAC_RESERVE:
            continue
        # Nearest non-comet planet (any owner — destination is "off the comet").
        best = None
        best_d2 = None
        for t in non_comet_targets:
            dx = t.x - c.x
            dy = t.y - c.y
            d2 = dx * dx + dy * dy
            if best_d2 is None or d2 < best_d2:
                best_d2 = d2
                best = t
        if best is None:
            continue
        angle = math.atan2(best.y - c.y, best.x - c.x)
        ships_out = int(c.ships) - COMET_EVAC_RESERVE
        launches.append([int(c.id), float(angle), int(ships_out)])
    return launches


def _score_phase1_analytical(world, action, my_id: int, horizon: int) -> float:
    """Cheap analytical leaf score for one candidate action.

    Builds a SYNTHETIC arrival ledger by adding the candidate's launches
    (ray-cast to find their targets + ETAs) to the real in-flight fleets'
    ledger, then re-simulates per-planet timelines out to `horizon`. Score
    = production-weighted territorial differential summed across horizon
    turns: +production per turn our planets are predicted to be ours,
    -production per turn enemy planets stay enemy.

    No step-by-step rollout, no opp policy — pure deterministic propagation
    of CURRENT in-flight fleets + the candidate's hypothetical launches.
    """
    from kaggle_environments.envs.orbit_wars.orbit_wars import Fleet
    planets_list = list(world.planets_by_id.values())
    if not planets_list:
        return 0.0

    # Base ledger from real in-flight fleets.
    base_model = WorldModel.from_world(world, horizon=horizon)
    # Shallow-copy the ledger so we can extend per-target arrival lists.
    synthetic_ledger = {pid: list(arr) for pid, arr in base_model.ledger.items()}

    # Add the candidate's launches to the ledger.
    for entry in action or []:
        try:
            src_id = int(entry[0])
            angle = float(entry[1])
            ships = int(entry[2])
        except (ValueError, TypeError, IndexError):
            continue
        if ships <= 0:
            continue
        src = world.planets_by_id.get(src_id)
        if src is None:
            continue
        spd = fleet_speed(ships)
        if spd <= 0:
            continue
        f = Fleet(-1, my_id, float(src.x), float(src.y), float(angle), src_id, ships)
        tgt, eta = fleet_target_planet(f, planets_list)
        if tgt is None or eta is None:
            continue
        synthetic_ledger.setdefault(int(tgt.id), []).append((int(eta), my_id, ships))

    # Re-simulate each planet's timeline with the augmented ledger.
    # Cost: O(N_planets * horizon). Cheap (~few ms for N≈30, H=50).
    score = 0.0
    for p in planets_list:
        arrivals = synthetic_ledger.get(int(p.id), [])
        tl = simulate_planet_timeline(p, arrivals, horizon=horizon)
        owner_at = tl["owner_at"]
        for t in range(1, horizon + 1):
            owner = owner_at.get(t, p.owner)
            if owner == my_id:
                score += float(p.production)
            elif owner is not None and owner >= 0 and owner != my_id:
                score -= float(p.production)
    return score


def _choose_two_phase(obs, configuration, *, K_deep: int, wallclock_ms: float,
                       phase1_horizon: int, phase2_top_k: int,
                       opp_tier: int, value_fn, world):
    """Phase 1 analytical triage + Phase 2 deep rollout on top-K survivors.

    Always preserves the incumbent (candidate 0) as a parity floor — it
    is scored in Phase 2 first so the watchdog can't drop it. If Phase 1
    can't rank candidates, falls back to the incumbent.
    """
    t_start = time.perf_counter()
    model = WorldModel.from_world(world)
    incumbent_intents = _v7_build_incumbent(world, model, include_recapture=True)
    incumbent_action = _v7_action_from_intents(incumbent_intents, obs, model)
    candidates = _v7_enumerate_drop_one(incumbent_action)
    if len(candidates) <= 1:
        return incumbent_action

    my_id = world.my_id

    # Phase 1: analytical scores on every candidate.
    p1_scores = [
        _score_phase1_analytical(world, c, my_id, phase1_horizon)
        for c in candidates
    ]
    # Rank by Phase-1 score, descending. Survivors = top-K UNION incumbent (idx 0).
    ranked = sorted(range(len(candidates)), key=lambda i: p1_scores[i], reverse=True)
    survivors_idx = list(dict.fromkeys([0] + ranked[: max(1, phase2_top_k)]))

    # Phase 2: deep rollout on survivors.
    snap = _v7_fs_from_obs(obs, configuration, episode_seed=0, num_seats=2)
    best_action = incumbent_action
    best_score = float("-inf")
    incumbent_scored = False
    for i in survivors_idx:
        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        if elapsed_ms > wallclock_ms:
            break
        try:
            s = _v7_score_candidate(
                snap, candidates[i], my_id=my_id, K=K_deep,
                opp_tier=opp_tier, value_fn=value_fn,
            )
        except Exception:
            continue
        if not incumbent_scored:
            incumbent_scored = True
            best_score = s
            best_action = list(candidates[i])
            continue
        if s > best_score:
            best_score = s
            best_action = list(candidates[i])
    return best_action


def _shrink_to_min_viable(action, world, model):
    """Latest-launch heuristic: for each enemy-targeted launch, shrink the
    fleet to the smallest ship count that still arrives BEFORE the target's
    earliest enemy-threat (or before our predicted-flip-to-us if attacking).

    For each launch entry [src_id, angle, ships]:
    - Ray-cast (via _launch_eta) to find the target and our predicted ETA.
    - Look up the target's threat_eta via WorldModel.time_to_enemy_threat.
      Where the threat is OUR own threat (i.e., for an enemy-owned target
      that WE threaten), we instead use the planet's predicted-flip-to-us
      step.
    - Binary-search S in [LATEST_LAUNCH_MIN_FLEET, current_ships] for the
      smallest S where eta_at_S < threat_eta - LATEST_LAUNCH_BUFFER_TURNS
      AND S still covers the predicted defenders at eta_at_S (via
      model.ships_at).

    Skip launches that touch unknown / safe planets (no threat_eta).
    """
    if not action:
        return action
    planets_list = list(world.planets_by_id.values())
    if not planets_list:
        return action

    def _sufficient(planet_id, eta, S):
        pred = model.ships_at(planet_id, eta)
        if pred is None:
            return True   # no info ⇒ assume safe
        return S >= int(math.ceil(float(pred))) + 1

    out = []
    for entry in action:
        try:
            src_id = int(entry[0])
            angle = float(entry[1])
            ships = int(entry[2])
        except (ValueError, TypeError, IndexError):
            out.append(entry)
            continue
        src = world.planets_by_id.get(src_id)
        if src is None or ships <= LATEST_LAUNCH_MIN_FLEET:
            out.append(entry)
            continue
        tgt, eta = _launch_eta(src, angle, ships, planets_list)
        if tgt is None or eta is None or tgt.id == src_id:
            out.append(entry)
            continue
        # Threat ETA: for non-our targets, query enemy threat (other than us).
        # For our targets (reinforce), we don't shrink — would risk losing
        # the defense.
        if tgt.owner == world.my_id:
            out.append(entry)
            continue
        threat_eta = model.time_to_enemy_threat(int(tgt.id), int(tgt.owner), world)
        if threat_eta is None:
            out.append(entry)
            continue
        deadline = int(threat_eta) - int(LATEST_LAUNCH_BUFFER_TURNS)
        if deadline <= 0:
            out.append(entry)
            continue

        # Binary search smallest viable S.
        d = math.hypot(tgt.x - src.x, tgt.y - src.y)
        lo, hi = LATEST_LAUNCH_MIN_FLEET, ships
        best_S = ships  # default: keep original size
        while lo <= hi:
            mid = (lo + hi) // 2
            v_mid = fleet_speed(mid)
            eta_mid = int(math.ceil(d / max(v_mid, 1e-6))) if v_mid > 0 else 999
            if eta_mid <= deadline and _sufficient(int(tgt.id), eta_mid, mid):
                best_S = mid
                hi = mid - 1
            else:
                lo = mid + 1
        if best_S < ships:
            out.append([src_id, angle, best_S])
        else:
            out.append(entry)
    return out


def agent(obs, configuration=None):
    global _PLAN_STATE
    world = World.from_obs(obs)
    current_turn = int(world.step)

    # ------------------------------------------------------------------
    # Mission persistence — opening commitment across turns.
    # Reset at turn 0 (new game), rebuild on invalidation. Scheduled
    # launches for THIS turn are extracted and merged with the chooser
    # action below (taking precedence per source).
    # ------------------------------------------------------------------
    planned_launches: list = []
    if MISSION_PERSISTENCE_ENABLED:
        if current_turn == 0 or _is_plan_invalid(world, world.my_id):
            _PLAN_STATE = _build_opening_plan(world, world.my_id)
        planned_launches = _execute_planned_launches(world, current_turn, world.my_id)

    # ------------------------------------------------------------------
    # PRE-FILTER HOOK — Bug 3: comet evacuation
    # Compute evacuation launches from owned comets about to leave the
    # board. Merged into the action AFTER the chooser runs, but only for
    # source planets the chooser didn't already use (avoid double-spend).
    # ------------------------------------------------------------------
    evac_launches = _comet_evacuation_launches(world, world.my_id)

    # ------------------------------------------------------------------
    # Adaptive K (Option A): scale lookahead to cover the longest in-flight
    # fleet. The wallclock watchdog inside choose() caps total cost.
    # ------------------------------------------------------------------
    max_eta = _max_inflight_eta(world)
    K_eff = min(K_CAP, max(K, max_eta + K_BUFFER))
    K_eff_4p = min(K_CAP, max(K_4P, max_eta + K_BUFFER))

    # Dispatch: 2P uses iter_v1's validated choose(enumerator_mode="drop_one",
    # opp_tiers=[1]) path. 4P uses choose_4p() instead of falling back to the
    # v3.5.1 incumbent (the pre-fix behaviour). choose_maximin is NOT used in
    # 2P because v7.1+ maximin variants regressed historically; we preserve
    # iter_v1's 2P behaviour exactly while adding 4P competence.
    n_seats = _detect_num_seats(world)
    value_fn = _resolve_value_fn(VALUE_FN)
    if n_seats == 4:
        action = choose_4p(
            obs, configuration,
            K=K_eff_4p,
            wallclock_ms=WALLCLOCK_MS,
            include_recapture=True,
            value_fn=value_fn,
        )
    elif TWO_PHASE:
        # Two-phase: cheap analytical Phase 1 ranks ALL candidates,
        # deep K_DEEP rollout on the top survivors. Bounded by K_CAP
        # for wallclock safety.
        K_deep_eff = min(K_CAP, max(K_DEEP, max_eta + K_BUFFER))
        action = _choose_two_phase(
            obs, configuration,
            K_deep=K_deep_eff,
            wallclock_ms=WALLCLOCK_MS,
            phase1_horizon=PHASE1_HORIZON,
            phase2_top_k=PHASE2_TOP_K,
            opp_tier=OPP_TIERS[0],
            value_fn=value_fn,
            world=world,
        )
    else:
        action = choose(
            obs, configuration,
            enumerator_mode=ENUMERATOR_MODE,
            K=K_eff,
            wallclock_ms=WALLCLOCK_MS,
            opp_tiers=list(OPP_TIERS),
            value_fn=value_fn,
        )

    # ------------------------------------------------------------------
    # POST-PROCESS HOOK — Bug 2: cap launches per comet target
    # Multi-source pile-on at a single comet is the panic; cap at
    # COMET_MAX_LAUNCHES_PER_TURN (shortest ETA wins the slots).
    # ------------------------------------------------------------------
    action = _cap_comet_launches(action, world, COMET_MAX_LAUNCHES_PER_TURN)

    # ------------------------------------------------------------------
    # Merge planned launches (mission persistence) — take precedence per
    # source over the chooser's launches.
    # ------------------------------------------------------------------
    action = _merge_planned_with_action(action, planned_launches)

    # ------------------------------------------------------------------
    # Merge evacuation launches (Bug 3) — only for sources the chooser
    # didn't already launch from. Each `entry[0]` is the src planet id.
    # ------------------------------------------------------------------
    if evac_launches:
        chosen_sources = {int(entry[0]) for entry in action if entry}
        for ev in evac_launches:
            if int(ev[0]) not in chosen_sources:
                action.append(ev)

    # ------------------------------------------------------------------
    # POST-PROCESS — latest-launch shrink (default OFF).
    # For each enemy-targeted launch, shrink to the smallest fleet that
    # still arrives in time. Conserves ships. Reuses the same WorldModel
    # we built earlier for adaptive K relevance filtering.
    # ------------------------------------------------------------------
    if LATEST_LAUNCH_ENABLED:
        model = WorldModel.from_world(world)
        action = _shrink_to_min_viable(action, world, model)

    return action
