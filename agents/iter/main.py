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
CLUSTER_WEIGHT = 1.0            # cluster_value weight (inert unless VALUE_FN uses cluster)
CLUSTER_FRONTIER_DISCOUNT = 0.5 # frontier planets contribute at this fraction in cluster head
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

# --- Multi-step plan ROI scorer (2026-05-15) --------------------------------
# Score MULTI-TURN action sequences ("first conquer A from X, then B from Y in
# 4 turns once Y has grown, then T1→T3 once captured") as a whole, not per-
# turn atomic missions. The plan candidate is emitted as ONE entry in the
# chooser's drop-one candidate list; the K=10 rollout scores it like any
# other candidate. No cross-turn state.
MULTI_STEP_PLAN_ENABLED = False         # default OFF; flip to True for A/B
MSP_TEMPLATES = ("saturation_strike", "near_chain", "high_prod_chain", "cluster_complete")
MSP_PLAN_LENGTH = 3                     # max missions per plan (saturation_strike uses 1)
MSP_HORIZON = 50                        # match PHASE1_HORIZON
MSP_DELAY_BUDGET = 12                   # max future-turn delay we'll schedule into
MSP_TOP_K_TARGETS = 4                   # per-template target shortlist size
MSP_MAX_SOURCES_PER_TARGET = 3          # saturation_strike: planets contributing per target
MSP_SHIPS_SAFETY = 2                    # add this to base capture cost for buffer
# ============================================================================

# --- Geo allocator candidate (2026-05-15) -----------------------------------
# Generates ONE additional candidate via lib.geo.allocator.allocate_greedy_multi
# under posture-aware reserves. Joint multi-launch generation that snipe's
# settle_plan often misses (snipe is one launch per source per turn). Default
# OFF; requires TWO_PHASE=True (only _choose_two_phase exposes the candidate
# list). Live-game evidence: iter already empties opening garrisons (1.7
# ships at home); geo's value here is multi-source coordination, NOT lower
# garrisons.
GEO_ALLOCATOR_CANDIDATE_ENABLED = False   # default OFF; flip for A/B
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
    if name == "cluster":
        from lib.value_heads import cluster_value
        return lambda obs, mid: cluster_value(
            obs, mid, weight=CLUSTER_WEIGHT,
            frontier_discount=CLUSTER_FRONTIER_DISCOUNT
        )
    if name == "composite_plus_cluster":
        from lib.value_heads import composite_plus_cluster
        return lambda obs, mid: composite_plus_cluster(
            obs, mid, cluster_weight=CLUSTER_WEIGHT,
            cluster_frontier_discount=CLUSTER_FRONTIER_DISCOUNT
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


# ============================================================================
# Multi-step plan ROI scorer (2026-05-15)
# ============================================================================
# Score MULTI-TURN action sequences as a whole, not per-turn atomic missions.
# A "plan" is a list of (fire_turn, src_id, angle, ships) tuples representing
# scheduled launches across the next MSP_DELAY_BUDGET turns. The plan scorer
# extends _score_phase1_analytical to support FUTURE-turn launches.
# The orchestrator emits ONE candidate (the first-turn launches of the best
# plan) into the chooser's candidate list; the K=10 rollout argmax-selects.
# No cross-turn state. No post-hoc merging. Strictly additive.


def _plan_first_turn_action(plan_launches, world):
    """Extract launches that fire THIS turn as a [src_id, angle, ships] list.

    Plan format: list of (fire_turn_absolute, src_id, angle, ships).
    Anything with fire_turn == world.step is emitted as a chooser candidate.
    """
    now = int(world.step)
    out = []
    for fire_turn, src_id, angle, ships in plan_launches:
        if int(fire_turn) != now:
            continue
        out.append([int(src_id), float(angle), int(ships)])
    return out


def _score_plan_analytical(world, plan_launches, my_id: int, horizon: int) -> float:
    """Analytical leaf score for a multi-turn plan.

    Extends _score_phase1_analytical to support future-turn launches.
    Each plan launch (fire_turn, src_id, angle, ships) contributes one
    synthetic arrival at (delay + ray_cast_eta) where delay = fire_turn -
    world.step. simulate_planet_timeline correctly buckets arrivals at any
    future step <= horizon.

    Returns production-weighted us-minus-them differential to `horizon`.
    """
    from kaggle_environments.envs.orbit_wars.orbit_wars import Fleet
    planets_list = list(world.planets_by_id.values())
    if not planets_list:
        return 0.0

    base_model = WorldModel.from_world(world, horizon=horizon)
    synthetic_ledger = {pid: list(arr) for pid, arr in base_model.ledger.items()}

    now = int(world.step)
    for fire_turn, src_id, angle, ships in plan_launches:
        delay = int(fire_turn) - now
        if delay < 0 or ships <= 0:
            continue
        src = world.planets_by_id.get(int(src_id))
        if src is None:
            continue
        spd = fleet_speed(int(ships))
        if spd <= 0:
            continue
        f = Fleet(-1, my_id, float(src.x), float(src.y), float(angle), int(src_id), int(ships))
        tgt, eta = fleet_target_planet(f, planets_list)
        if tgt is None or eta is None:
            continue
        arrival_turn = delay + int(eta)
        if arrival_turn > horizon:
            continue
        synthetic_ledger.setdefault(int(tgt.id), []).append(
            (arrival_turn, my_id, int(ships)))

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


def _eligible_neutral_targets(world):
    """Non-comet neutral planets, sorted by (distance-to-nearest-ours / production)."""
    my_id = world.my_id
    my_planets = [p for p in world.planets_by_id.values() if p.owner == my_id]
    if not my_planets:
        return []
    targets = []
    for t in world.planets_by_id.values():
        if t.owner != -1:
            continue
        if t.id in world.comet_ids:
            continue
        min_d = min(math.hypot(t.x - s.x, t.y - s.y) for s in my_planets)
        roi = float(t.production) / (min_d + 1.0)
        targets.append((roi, min_d, t))
    targets.sort(key=lambda x: (-x[0], x[1]))
    return [t for _r, _d, t in targets]


def _build_saturation_strike_plan(world, my_id: int) -> list:
    """Single-target multi-source saturation. Addresses opening fleet-size gap.

    Picks the highest-ROI neutral target reachable RIGHT NOW (turn=world.step)
    with combined ships from up to MSP_MAX_SOURCES_PER_TARGET nearest sources.
    All launches fire at world.step. Returns [] if no target is fundable now.
    """
    targets = _eligible_neutral_targets(world)
    if not targets:
        return []
    my_planets = [p for p in world.planets_by_id.values() if p.owner == my_id]
    if not my_planets:
        return []
    now = int(world.step)
    for t in targets[:MSP_TOP_K_TARGETS]:
        base_capture = int(t.ships) + 1 + MSP_SHIPS_SAFETY
        sources = sorted(my_planets,
                         key=lambda s: math.hypot(t.x - s.x, t.y - s.y)
                         )[:MSP_MAX_SOURCES_PER_TARGET]
        total_avail = sum(int(s.ships) for s in sources)
        if total_avail < base_capture:
            continue
        plan_launches = []
        remaining = base_capture
        for s in sources:
            if remaining <= 0:
                break
            send = min(int(s.ships), remaining)
            if send <= 0:
                continue
            angle = math.atan2(t.y - s.y, t.x - s.x)
            plan_launches.append((now, int(s.id), float(angle), int(send)))
            remaining -= send
        if remaining > 0:
            continue
        return plan_launches
    return []


def _build_chain_plan(world, my_id: int, target_order, length: int, delay_budget: int):
    """Generic chain builder used by near_chain and high_prod_chain.

    `target_order` is a pre-sorted list of target planets. For each in order,
    allocate the closest owned source not yet locked, schedule fire_turn at
    earliest turn where source has ships >= capture cost (or fail if outside
    delay_budget). Stop after `length` missions.
    """
    if not target_order:
        return []
    my_planets = [p for p in world.planets_by_id.values() if p.owner == my_id]
    if not my_planets:
        return []
    now = int(world.step)
    plan_launches: list = []
    used_src_turns: set = set()   # (src_id, fire_turn) — one launch per slot

    for t in target_order[:length]:
        base_capture = int(t.ships) + 1 + MSP_SHIPS_SAFETY
        # Try sources nearest to t.
        candidates = sorted(my_planets,
                            key=lambda s: math.hypot(t.x - s.x, t.y - s.y))
        scheduled = False
        for src in candidates:
            # Earliest turn this source has enough ships (production-grown).
            prod = max(1, int(src.production))
            for delay in range(delay_budget + 1):
                fire_turn = now + delay
                if (int(src.id), fire_turn) in used_src_turns:
                    continue
                ships_at_t = int(src.ships) + delay * prod
                if ships_at_t >= base_capture:
                    angle = math.atan2(t.y - src.y, t.x - src.x)
                    plan_launches.append((fire_turn, int(src.id), float(angle), base_capture))
                    used_src_turns.add((int(src.id), fire_turn))
                    scheduled = True
                    break
            if scheduled:
                break
        # If no source can fund this target within delay_budget, skip it.
    return plan_launches


def _build_near_chain_plan(world, my_id: int, length: int, delay_budget: int):
    """Capture `length` closest non-comet neutrals in distance order."""
    my_planets = [p for p in world.planets_by_id.values() if p.owner == my_id]
    if not my_planets:
        return []
    neutrals = [p for p in world.planets_by_id.values()
                if p.owner == -1 and p.id not in world.comet_ids]
    if not neutrals:
        return []
    def _min_d(t):
        return min(math.hypot(t.x - s.x, t.y - s.y) for s in my_planets)
    neutrals.sort(key=_min_d)
    return _build_chain_plan(world, my_id, neutrals, length, delay_budget)


def _build_high_prod_chain_plan(world, my_id: int, length: int, delay_budget: int):
    """Capture `length` highest-production non-comet neutrals (production desc,
    distance tiebreak)."""
    my_planets = [p for p in world.planets_by_id.values() if p.owner == my_id]
    if not my_planets:
        return []
    neutrals = [p for p in world.planets_by_id.values()
                if p.owner == -1 and p.id not in world.comet_ids]
    if not neutrals:
        return []
    def _key(t):
        min_d = min(math.hypot(t.x - s.x, t.y - s.y) for s in my_planets)
        return (-float(t.production), min_d)
    neutrals.sort(key=_key)
    return _build_chain_plan(world, my_id, neutrals, length, delay_budget)


def _build_cluster_complete_plan(world, model, my_id: int, length: int, delay_budget: int):
    """Find an our-cluster, capture adjacent unclaimed neutrals to complete it.

    Uses lib.geo.sense.sense_state for cluster geometry. Picks the cluster
    with the highest production-weighted neighbour shortlist; captures up to
    `length` of those neutrals in distance order. Returns [] if sense_state
    finds no qualifying cluster.
    """
    try:
        from lib.geo.sense import sense_state
    except ImportError:
        return []
    try:
        sense = sense_state(world, model)
    except Exception:
        return []
    our_clusters = getattr(sense, "our_clusters", None) or getattr(sense, "my_clusters", None)
    if not our_clusters:
        return []
    # For each cluster, find adjacent unclaimed neutrals (within some radius
    # of the cluster's centroid). Sort by total production captured.
    best_cluster_targets = []
    best_score = 0.0
    for cluster in our_clusters:
        member_ids = getattr(cluster, "planet_ids", None) or getattr(cluster, "members", None)
        if not member_ids:
            continue
        members = [world.planets_by_id.get(int(pid)) for pid in member_ids]
        members = [m for m in members if m is not None]
        if not members:
            continue
        cx = sum(m.x for m in members) / len(members)
        cy = sum(m.y for m in members) / len(members)
        # Adjacent neutrals = closest non-comet neutrals to cluster centroid.
        neutrals = [p for p in world.planets_by_id.values()
                    if p.owner == -1 and p.id not in world.comet_ids]
        if not neutrals:
            continue
        neutrals.sort(key=lambda t: math.hypot(t.x - cx, t.y - cy))
        candidates = neutrals[:length]
        score = sum(float(t.production) for t in candidates)
        if score > best_score:
            best_score = score
            best_cluster_targets = candidates
    if not best_cluster_targets:
        return []
    return _build_chain_plan(world, my_id, best_cluster_targets, length, delay_budget)


def multi_step_plan_candidate(world, model, my_id: int, incumbent_action):
    """Build one plan per enabled template; score each analytically; return
    the FIRST-TURN action of the highest-scoring plan as ONE chooser candidate.

    Returns None if no template yields a plan that scores higher than the
    incumbent's analytical Phase-1 score.
    """
    incumbent_score = _score_phase1_analytical(world, incumbent_action, my_id, MSP_HORIZON)
    builders = {
        "saturation_strike": lambda: _build_saturation_strike_plan(world, my_id),
        "near_chain": lambda: _build_near_chain_plan(
            world, my_id, MSP_PLAN_LENGTH, MSP_DELAY_BUDGET),
        "high_prod_chain": lambda: _build_high_prod_chain_plan(
            world, my_id, MSP_PLAN_LENGTH, MSP_DELAY_BUDGET),
        "cluster_complete": lambda: _build_cluster_complete_plan(
            world, model, my_id, MSP_PLAN_LENGTH, MSP_DELAY_BUDGET),
    }
    best_first_action = None
    best_score = incumbent_score
    for template_name in MSP_TEMPLATES:
        builder = builders.get(template_name)
        if builder is None:
            continue
        try:
            plan_launches = builder()
        except Exception:
            continue
        if not plan_launches:
            continue
        first_action = _plan_first_turn_action(plan_launches, world)
        if not first_action:
            continue
        try:
            plan_score = _score_plan_analytical(world, plan_launches, my_id, MSP_HORIZON)
        except Exception:
            continue
        if plan_score > best_score:
            best_score = plan_score
            best_first_action = first_action
    return best_first_action


def geo_allocator_candidate(world, model, my_id, obs):
    """Build one joint-multi-launch candidate via lib.geo.allocator.

    Uses posture-aware reserves: OPENING/EXPAND/BREAK leave reserve=0;
    DEFEND uses per-planet threat_budget. Builds a Mission pool from the
    same proposers iter's incumbent uses (snipe aggressive + reinforce +
    opening), then asks `allocate_greedy_multi` to assign sources greedily
    under the posture's budget. Returns the launch list or None if the
    allocator yields nothing.
    """
    try:
        from lib.geo.sense import sense_state
        from lib.geo.posture import decide_posture
        from lib.geo.allocator import allocate_greedy_multi
        from lib.missions.snipe import propose_snipe_missions
        from lib.missions.reinforce import propose_reinforce_missions
        from lib.missions.opening import propose_opening_missions
    except ImportError:
        return None
    try:
        sense = sense_state(world, model)
        posture = decide_posture(world, sense, model)
    except Exception:
        return None
    try:
        missions = (
            propose_opening_missions(world, model)
            + propose_snipe_missions(world, model, aggressive=True)
            + propose_reinforce_missions(world, model)
        )
        # Strip comet targets — geo's _drop_comet_missions does this; we
        # mirror it to avoid the chooser proposing pile-on at comets.
        missions = [m for m in missions if m.target_id not in world.comet_ids]
    except Exception:
        return None
    if not missions:
        return None
    try:
        new_intents = allocate_greedy_multi(missions, world, sense, posture, model)
    except Exception:
        return None
    if not new_intents:
        return None
    return _v7_action_from_intents(new_intents, obs, model)


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
    my_id = world.my_id

    # Multi-step plan candidate: emits one extra candidate from the best-
    # scoring multi-turn plan. The chooser still gates via K=10 rollout.
    if MULTI_STEP_PLAN_ENABLED:
        try:
            plan_action = multi_step_plan_candidate(world, model, my_id, incumbent_action)
        except Exception:
            plan_action = None
        if plan_action is not None:
            candidates.append(plan_action)

    # Geo allocator candidate: joint multi-launch under posture-aware reserves
    # (OPENING/EXPAND/BREAK = 0; DEFEND = per-planet threat budget).
    if GEO_ALLOCATOR_CANDIDATE_ENABLED:
        try:
            geo_action = geo_allocator_candidate(world, model, my_id, obs)
        except Exception:
            geo_action = None
        if geo_action is not None:
            candidates.append(geo_action)

    if len(candidates) <= 1:
        return incumbent_action

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
    world = World.from_obs(obs)

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
        # Use score_candidate_4p's BUILT-IN 4P-aware leaf scorer
        # ("our ships - max(other seat ships)") instead of the 2P-tuned
        # composite_capture_value. composite's base (delta_us_minus_them)
        # is "us - sum(3 opps)" in 4P — biases toward defensive play
        # because we look outnumbered 3:1. The 4P default is structurally
        # aligned with first-place: beat the SINGLE strongest opponent.
        action = choose_4p(
            obs, configuration,
            K=K_eff_4p,
            wallclock_ms=WALLCLOCK_MS,
            include_recapture=True,
            value_fn=None,
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
