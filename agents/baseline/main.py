"""baseline — clean modular re-implementation of v15 (live champion μ=1115.5).

Pipeline (per turn):
  1. proposer.propose       enumerate fire-now + multi-wait grid, cheap-rank,
                            dedup by (src, tgt, wait_band).
  2. chooser.build_idle_baseline   precompute favor under (me-idle, opp-reactive).
  3. chooser.choose         validate top candidates with fast_sim K-step rollout,
                            emit greedy non-dogpile moves.

Knobs (env var overrides, all optional):
  BASELINE_GAMMA              PV-discount γ for favor() and cheap-rank.   default 0.99
  BASELINE_WALLCLOCK_MS       per-turn validate budget (env actTimeout=1000).
                                                                          default 600
  ORBIT_WARS_PARITY_WALLCLOCK_MS    bundle-parity override (very large
                                    value disables mid-loop deadline bail
                                    so the agent is a pure function of obs).
"""

from __future__ import annotations

import math
import os

# Production default: hybrid value head (composite in 2P, A2-favor in 4P).
# `setdefault` lets local A/B drivers (fast.py) override via env var without
# patching source, while submission-bundle / Kaggle-runner sees hybrid out
# of the box. See agents/baseline/value.select_favor_fn for the dispatch.
os.environ.setdefault("BASELINE_VALUE_HEAD", "hybrid")

# Production default: trajectory chooser. v4 with wait_N>0 + wallclock
# budgeting hits 42/64 = 65.6pct Wlo=0.534 vs v15 (n=64), point-estimate
# +3pp over composite_a2's 40/64 = 62.5pct in the same A/B, with better
# max-turn-ms (1077 vs 1292). The trajectory path is deterministic on
# sun/oob/expired-comet failure modes (predict_fleet_fate filter) and
# was the architectural reframe completed in this session. Local A/B
# drivers can force the composite path by setting BASELINE_CHOOSER to
# any value other than "trajectory" (e.g. "composite").
os.environ.setdefault("BASELINE_CHOOSER", "trajectory")

# Direction B v3 (2026-05-18 PM): joint candidate enumeration enabled
# by default. 2P A/B: joint vs hybrid = 38/64 = 59.4pct, Wlo=0.471,
# Whi=0.705 (INCONCLUSIVE-but-positive). 2P-only gate in chooser
# (num_seats <= 2 check) preserves 4P behaviour (4P regressed without
# gate at 12.5pct first-place). Wallclock OK: bench max=891ms,
# p95=703ms, zero >1000ms. Set BASELINE_JOINT=0 to disable.
os.environ.setdefault("BASELINE_JOINT", "1")

# H1 — post-chooser idle drain (2026-05-18) — DISABLED BY DEFAULT.
# Audit `audit/replays/idle-trajectory-2026-05-17.md` measured 43.8pct
# isolated ship-turns in trajectory champion (mu=1271.8). H1 attempted
# to drain rear sources via post-chooser reinforce launches. A/B vs
# hybrid reference at n=32: **11/32 = 34.4pct, Wlo=0.204, max-ms=1528
# — FAIL**. The chooser's decision to leave rear planets idle is
# CORRECTLY calibrated reserve-holding; H1's forced emissions weaken
# defense without compensating capture-EV. Spatial-leaf head (commit
# b5f5296) failed for the same root cause. The 43.8 pct isolated is
# not a leak — it's correctly-held reserve. See audit/2026-05-18-
# spatial-leaf-negative-result.md and audit/2026-05-18-h1-idle-drain-
# negative-result.md. Default OFF; opt-in via BASELINE_IDLE_DRAIN=1.
IDLE_DRAIN_THRESHOLD = int(os.environ.get("BASELINE_IDLE_DRAIN_THRESHOLD", "30"))
IDLE_REAR_THRESHOLD = float(os.environ.get("BASELINE_IDLE_REAR_THRESHOLD", "35.0"))
IDLE_DRAIN_RESERVE = int(os.environ.get("BASELINE_IDLE_DRAIN_RESERVE", "5"))
IDLE_DRAIN_ENABLED = os.environ.get("BASELINE_IDLE_DRAIN", "0") == "1"

# Reinforce-emit post-pass (2026-05-21). Wires `propose_reinforce_missions`
# (lib/missions/reinforce.py) into the chooser's emit path. Distinct from
# `drain_idle_rear` (which the 2026-05-18 audit falsified as weakening
# defense): this only fires for OUR planets predicted to flip to enemy
# within model.horizon. Triggered by PI live-game observation (4P seed
# 914393430): a +5 prod planet fell while rear sources held reserves.
# Default OFF; opt-in via BASELINE_REINFORCE_EMIT=1.
REINFORCE_EMIT_ENABLED = os.environ.get("BASELINE_REINFORCE_EMIT", "0") == "1"
REINFORCE_MIN_PROD = int(os.environ.get("BASELINE_REINFORCE_MIN_PROD", "2"))
REINFORCE_MAX_LAUNCHES = int(os.environ.get("BASELINE_REINFORCE_MAX", "3"))

# Anticipated-threat (preemptive) reinforce — direction (b) from
# PI 2026-05-21 directive "mobilize idle planets toward planets that
# need them." Fires for friendly destinations with inbound enemy fleets
# that thin defenders below safety margin, even if T_loss isn't predicted
# yet. Distinct from strict propose_reinforce_missions (T_loss < horizon
# only) and from drain_idle_rear (blanket "rear -> closer friend").
ANTICIPATE_ENABLED = os.environ.get("BASELINE_REINFORCE_ANTICIPATE", "0") == "1"
ANTICIPATE_MIN_PROD = int(os.environ.get("BASELINE_REINFORCE_ANTICIPATE_MIN_PROD", "3"))
ANTICIPATE_MARGIN = float(os.environ.get("BASELINE_REINFORCE_ANTICIPATE_MARGIN", "1.3"))

# Stateful commit ledger (2026-05-20). When `BASELINE_LEDGER=on`, the
# chooser's wait_N>0 winners are remembered across turns instead of
# being silently dropped. Each entry ticks down each turn; when
# wait_remaining hits 0 the agent emits the launch (re-aimed against
# current src/tgt geometry). See plan
# /root/.claude/plans/so-now-research-and-zany-widget.md and audit
# audit/2026-05-20-filter-rejection-trace.md.
#
# Module-level state keyed by `obs.player` so independent seats in the
# same process (eg local A/B harnesses spinning up both seats) don't
# share commitments. Cleared on `obs.step == 0` (new-match detection).
LEDGER_ENABLED = os.environ.get("BASELINE_LEDGER", "off").strip().lower() == "on"
# Mode for the ledger: "hard" (default) reserves the src across the
# wait, blocking chooser emits from it. "soft" leaves the src free
# (chooser can fire fire-now from it) and only requires enough ships
# at emit time. Set via env var BASELINE_LEDGER_MODE.
LEDGER_MODE = os.environ.get("BASELINE_LEDGER_MODE", "hard").strip().lower()
_PENDING_LAUNCHES: dict[int, list[dict]] = {}

# Opening override (2026-05-21). Cherry-picked from analytical track
# (origin/claude/strategy-axis-decision-3437). For step < OPENING_HORIZON
# (=30), run the one-shot multi-turn MILP `opening_plan` and emit
# fire_step==step_now entries from its schedule. Same three-case dispatch
# as `lib/pipeline/opening.opening_default`: (a) emit schedule entries
# fired now, (b) empty fire-now list, (c) empty schedule → fall through
# to standard chooser. Default OFF; opt-in via BASELINE_OPENING_MILP=1.
OPENING_MILP_ENABLED = os.environ.get("BASELINE_OPENING_MILP", "0") == "1"

from kaggle_environments.envs.orbit_wars.orbit_wars import Planet, Fleet

from lib.fast_sim import from_obs as fs_from_obs
from lib.fleet import speed as fleet_speed
from lib.intent import World
from lib.joint_solver.opening_planner import OPENING_HORIZON, opening_plan
from lib.missions.reinforce import propose_reinforce_missions
from lib.orbit import predict_relative
from lib.world_model import WorldModel

# Import by explicit names so the bundler's per-line import-stripping regex
# can handle them. Single-line form is mandatory — the regex matches one
# line at a time, so multi-line parenthesised imports would leak their
# continuation lines as indented orphans. Friction tag
# `bundler-modular-agent-namespace-access-breaks-bundle` (2026-05-17).
from agents.baseline.chooser import build_idle_baseline, choose, WALLCLOCK_BUDGET_MS
from agents.baseline.proposer import propose, MAX_HORIZON, MIN_HORIZON


_PARITY_ENV_VAR = "ORBIT_WARS_PARITY_WALLCLOCK_MS"


def _as_dict(obs) -> dict:
    if isinstance(obs, dict):
        return obs
    return {
        "player": getattr(obs, "player", 0),
        "step": getattr(obs, "step", 0),
        "planets": list(getattr(obs, "planets", []) or []),
        "fleets": list(getattr(obs, "fleets", []) or []),
        "comets": list(getattr(obs, "comets", []) or []),
        "comet_planet_ids": list(getattr(obs, "comet_planet_ids", []) or []),
        "angular_velocity": float(getattr(obs, "angular_velocity", 0.0)),
    }


def _num_seats(planets, fleets) -> int:
    max_owner = -1
    for p in planets:
        if int(p.owner) > max_owner:
            max_owner = int(p.owner)
    for f in fleets:
        if int(f.owner) > max_owner:
            max_owner = int(f.owner)
    return 4 if max_owner >= 2 else 2


def _wallclock_ms() -> float:
    override = os.environ.get(_PARITY_ENV_VAR)
    if override:
        try:
            return float(override)
        except ValueError:
            pass
    try:
        return float(os.environ.get("BASELINE_WALLCLOCK_MS", WALLCLOCK_BUDGET_MS))
    except ValueError:
        return WALLCLOCK_BUDGET_MS


def _gamma() -> float:
    try:
        return float(os.environ.get("BASELINE_GAMMA", 0.99))
    except ValueError:
        return 0.99


def _tick_ledger(me: int, world, model, omega: float) -> tuple[list[list], list[dict]]:
    """Tick pending wait commitments for `me`.

    Returns `(due_moves, surviving_pending)`:
      `due_moves`           — actions to emit this turn (one per due commit
                              that validated successfully). Re-aimed against
                              current src/tgt geometry.
      `surviving_pending`   — entries still in flight (wait_remaining > 0
                              after the decrement) plus entries whose
                              wait_remaining hit 0 but failed validation
                              (NOT included — silently dropped).

    Tick semantics:
      - Decrement every entry's `wait_remaining` by 1.
      - If `wait_remaining` reaches 0 (or already <= 0):
          * Drop if src no longer ours.
          * Drop if tgt now ours (capture goal moot — chooser may have
            redirected or another src took it).
          * Drop if src has 0 ships (nothing to send).
          * Otherwise re-aim using `proposer.aim_and_eta` and emit
            `min(ships_planned, src.ships)` toward tgt.
      - Else: keep entry alive (decrement only).

    Re-aim is essential because planets orbit between commit time and
    emit time. The proposer's original `angle_original` was correct for
    geometry at commit time; firing at the same angle N turns later
    would miss.
    """
    pending = _PENDING_LAUNCHES.get(int(me), [])
    if not pending:
        return [], []

    from agents.baseline.proposer import aim_and_eta as _aim_and_eta

    due_moves: list[list] = []
    survivors: list[dict] = []
    for entry in pending:
        entry["wait_remaining"] = int(entry["wait_remaining"]) - 1
        if entry["wait_remaining"] > 0:
            survivors.append(entry)
            continue

        # Time to fire — validate. Record drop reason on the entry for
        # downstream telemetry (the entry is otherwise discarded after
        # this loop).
        sid = int(entry["src_id"])
        tid = int(entry["tgt_id"])
        src = world.planets_by_id.get(sid)
        tgt = world.planets_by_id.get(tid)
        if src is None or tgt is None:
            entry["drop_reason"] = "planet_missing"
            continue
        if int(src.owner) != int(me):
            entry["drop_reason"] = "src_lost"
            continue
        if int(tgt.owner) == int(me):
            entry["drop_reason"] = "tgt_now_ours"
            continue
        available = int(src.ships)
        if available <= 0:
            entry["drop_reason"] = "src_empty"
            continue
        ships = min(int(entry["ships_planned"]), available)
        if ships <= 0:
            entry["drop_reason"] = "size_zero"
            continue
        # Re-aim against the geometry that holds RIGHT NOW (planets
        # have orbited during the wait). wait_N=0 because we're firing
        # this turn.
        try:
            angle, _eta = _aim_and_eta(src, tgt, ships, omega, wait_N=0,
                                       world=world)
        except Exception:
            entry["drop_reason"] = "aim_failed"
            continue
        entry["fired_at_step"] = int(world.step)
        entry["fired_ships"] = int(ships)
        due_moves.append([sid, float(angle), int(ships)])

    return due_moves, survivors


def emit_threat_reinforcements(
    moves, planets, my_id: int, world, model, omega: float,
) -> list:
    """Append reinforce launches for OUR planets predicted to fall.

    Defense-directed: uses `propose_reinforce_missions` which scans the
    WorldModel timeline for the first `T_loss` per friendly planet, then
    proposes (src, defended) candidates feasible to arrive before the
    flip. Skips sources already in `moves` so the chooser's offensive
    plan isn't disrupted. Caps total reinforce launches at
    REINFORCE_MAX_LAUNCHES per turn.
    """
    if not REINFORCE_EMIT_ENABLED:
        return moves
    candidates = propose_reinforce_missions(world, model)
    if not candidates:
        return moves
    used_srcs: set[int] = set()
    for m in moves:
        try:
            used_srcs.add(int(m[0]))
        except (TypeError, IndexError):
            pass
    planet_by_id = {int(p.id): p for p in planets}

    def tgt_prod(M):
        p = planet_by_id.get(int(M.target_id))
        return float(p.production) if p is not None else 0.0

    candidates.sort(key=lambda M: (-tgt_prod(M), -float(M.score)))
    extras = []
    fired = 0
    for mission in candidates:
        if fired >= REINFORCE_MAX_LAUNCHES:
            break
        if mission.mission_class != "reinforce":
            continue
        sid = int(mission.src_id)
        if sid in used_srcs:
            continue
        src = planet_by_id.get(sid)
        tgt = planet_by_id.get(int(mission.target_id))
        if src is None or tgt is None:
            continue
        if int(tgt.production) < REINFORCE_MIN_PROD:
            continue
        ships = int(mission.ships)
        if int(src.ships) < ships:
            continue
        try:
            tx, ty = predict_relative(tgt, int(mission.eta), omega)
        except Exception:
            tx, ty = float(tgt.x), float(tgt.y)
        angle = math.atan2(float(ty) - float(src.y), float(tx) - float(src.x))
        extras.append([sid, float(angle), int(ships)])
        used_srcs.add(sid)
        fired += 1

    if ANTICIPATE_ENABLED and fired < REINFORCE_MAX_LAUNCHES:
        extras2 = _propose_anticipated_reinforces(
            planets, used_srcs, my_id, world, model, omega,
            slots_left=REINFORCE_MAX_LAUNCHES - fired,
        )
        extras.extend(extras2)
    return list(moves) + extras


def _propose_anticipated_reinforces(
    planets, used_srcs: set[int], my_id: int, world, model, omega: float,
    slots_left: int,
) -> list:
    """Preemptive reinforce: defenders thinned by inbound enemy fleets.

    For each friendly D with prod >= ANTICIPATE_MIN_PROD and at least one
    inbound enemy fleet within model.horizon, check whether projected
    defenders cover the inbound threat by ANTICIPATE_MARGIN. If not,
    propose a launch from the nearest viable friendly source whose
    arrival ETA precedes the earliest enemy ETA.
    """
    if slots_left <= 0:
        return []
    my_planets = [p for p in planets if int(p.owner) == my_id]
    if len(my_planets) < 2:
        return []
    horizon = int(getattr(model, "horizon", 40))
    out = []
    fired = 0
    # Pre-compute friendly index for source iteration.
    friendly_by_id = {int(p.id): p for p in my_planets}
    # Score destinations by (production desc, thinness ratio asc — most-thin first).
    destinations: list[tuple] = []
    for d in my_planets:
        if int(d.production) < ANTICIPATE_MIN_PROD:
            continue
        arrivals = (model.ledger.get(int(d.id)) or []) if hasattr(model, "ledger") else []
        if not arrivals:
            continue
        enemy_inbound = 0
        friendly_inbound = 0
        earliest_enemy_eta: int | None = None
        for (eta_arr, owner_arr, ships_arr) in arrivals:
            if int(ships_arr) <= 0:
                continue
            if int(eta_arr) > horizon:
                continue
            if int(owner_arr) == my_id:
                friendly_inbound += int(ships_arr)
            else:
                enemy_inbound += int(ships_arr)
                if earliest_enemy_eta is None or int(eta_arr) < earliest_enemy_eta:
                    earliest_enemy_eta = int(eta_arr)
        if enemy_inbound <= 0 or earliest_enemy_eta is None:
            continue
        # Projected defenders at earliest enemy arrival (ignoring
        # accruing production from the enemy's perspective; production
        # accrues for us between now and arrival).
        proj_defenders = (
            int(d.ships) + int(d.production) * int(earliest_enemy_eta)
            + friendly_inbound
        )
        # Already comfortable margin → skip.
        if proj_defenders >= enemy_inbound * ANTICIPATE_MARGIN:
            continue
        deficit = int(enemy_inbound * ANTICIPATE_MARGIN) - proj_defenders + 1
        if deficit <= 0:
            continue
        destinations.append((deficit, d, earliest_enemy_eta))
    # Highest deficit first.
    destinations.sort(key=lambda x: -x[0])
    for deficit, d, earliest_enemy_eta in destinations:
        if fired >= slots_left:
            break
        # Find nearest friendly source not already used, with enough
        # ships AND able to arrive before earliest_enemy_eta.
        best_src = None
        best_eta = None
        for s in my_planets:
            if int(s.id) == int(d.id):
                continue
            if int(s.id) in used_srcs:
                continue
            if int(s.ships) < deficit:
                continue
            dist = math.hypot(float(d.x) - float(s.x), float(d.y) - float(s.y))
            v = fleet_speed(deficit)
            if v <= 0:
                continue
            eta = int(math.ceil(dist / v))
            if eta >= int(earliest_enemy_eta):
                continue
            if best_src is None or eta < best_eta:
                best_src = s
                best_eta = eta
        if best_src is None:
            continue
        try:
            tx, ty = predict_relative(d, int(best_eta), omega)
        except Exception:
            tx, ty = float(d.x), float(d.y)
        angle = math.atan2(
            float(ty) - float(best_src.y), float(tx) - float(best_src.x),
        )
        out.append([int(best_src.id), float(angle), int(deficit)])
        used_srcs.add(int(best_src.id))
        fired += 1
    return out


def drain_idle_rear(moves, planets, my_id: int, world, model) -> list:
    """H1: append reinforce launches for rear sources the chooser didn't use.

    Idempotent post-chooser pass. Fires only when ALL of:
      - source is one of MY planets AND not in `moves`
      - source.ships > IDLE_DRAIN_THRESHOLD
      - source's min-distance to any non-our planet > IDLE_REAR_THRESHOLD
      - source has no enemy threat (model.time_to_enemy_threat is None)
      - there is an own planet strictly closer to the action than source
    Emits one launch toward that closer own planet, ships = source.ships
    minus IDLE_DRAIN_RESERVE. Each `move` is `[src_id, angle, ships]`.
    """
    if not IDLE_DRAIN_ENABLED:
        return moves
    used_srcs = set()
    for m in moves:
        try:
            used_srcs.add(int(m[0]))
        except (TypeError, IndexError):
            pass
    non_our_xy = [(float(p.x), float(p.y)) for p in planets
                  if int(p.owner) != my_id]
    if not non_our_xy:
        return moves
    my_planets = [p for p in planets if int(p.owner) == my_id]
    if len(my_planets) < 2:
        return moves  # no closer own target available

    def d_action(p):
        return min(math.hypot(float(p.x) - tx, float(p.y) - ty)
                   for tx, ty in non_our_xy)

    extras = []
    for src in my_planets:
        if int(src.id) in used_srcs:
            continue
        if int(src.ships) <= IDLE_DRAIN_THRESHOLD:
            continue
        src_d = d_action(src)
        if src_d <= IDLE_REAR_THRESHOLD:
            continue
        if model.time_to_enemy_threat(int(src.id), my_id, world) is not None:
            continue
        best_target = None
        best_d = src_d  # strict-less-than → require improvement
        for q in my_planets:
            if int(q.id) == int(src.id):
                continue
            qd = d_action(q)
            if qd >= best_d:
                continue
            best_d = qd
            best_target = q
        if best_target is None:
            continue
        ships = int(src.ships) - IDLE_DRAIN_RESERVE
        if ships < 1:
            continue
        angle = math.atan2(float(best_target.y) - float(src.y),
                           float(best_target.x) - float(src.x))
        extras.append([int(src.id), float(angle), int(ships)])
    return list(moves) + extras


def agent(obs, configuration=None):
    obs_d = _as_dict(obs)
    me = int(obs_d.get("player", 0))
    step = int(obs_d.get("step", 0))

    # New-match detection — clear this seat's commit ledger on step 0.
    # Both `LEDGER_ENABLED` and `BASELINE_LEDGER=on` are checked at call
    # time so harnesses can flip the env var mid-process without
    # restarting the agent module.
    ledger_on = (
        LEDGER_ENABLED
        or os.environ.get("BASELINE_LEDGER", "off").strip().lower() == "on"
    )
    if ledger_on and step == 0:
        _PENDING_LAUNCHES.pop(me, None)

    raw_planets = obs_d.get("planets", []) or []
    raw_fleets = obs_d.get("fleets", []) or []
    if not raw_planets:
        return []

    planets = [Planet(*p) for p in raw_planets]
    fleets = [Fleet(*f) for f in raw_fleets]
    my_planets = [p for p in planets if int(p.owner) == me]
    other_planets = [p for p in planets if int(p.owner) != me]
    if not my_planets or not other_planets:
        return []

    world = World.from_obs(obs_d)
    model = WorldModel.from_world(world)
    omega = float(obs_d.get("angular_velocity", 0.0))
    num_seats = _num_seats(planets, fleets)
    gamma = _gamma()
    wallclock_ms = _wallclock_ms()

    threatened_mine = [
        p for p in my_planets
        if model.time_to_enemy_threat(int(p.id), me, world) is not None
    ]
    target_pool = other_planets + threatened_mine

    # Opening override (2026-05-21, hybrid). For step < OPENING_HORIZON
    # AND when opening_plan produced fire_step==step_now entries, emit
    # those. Cases (b) "MILP wants to wait" and (c) "empty schedule"
    # both fall through to AGGR's standard chooser — AGGR's aggressive
    # opening attacks outperform the MILP's "wait" recommendations in
    # empirical 4P testing (variant_open n=16 5/16 vs pre-patch 6/16
    # when intentional-waits were honoured).
    if OPENING_MILP_ENABLED and int(step) < OPENING_HORIZON:
        try:
            op = opening_plan(world, model, me, num_seats)
        except Exception:
            op = None
        if op is not None and op.schedule:
            opening_moves = [
                [int(e.src_id), float(e.angle), int(e.ships)]
                for e in op.schedule if int(e.fire_step) == int(step)
            ]
            if opening_moves:
                # Case (a): MILP has fire-now entries — emit and return.
                return opening_moves
            # Cases (b) and (c) fall through.

    snap_base = fs_from_obs(obs, num_seats=num_seats)

    # Trajectory-first chooser opt-in (2026-05-17). Deterministic
    # admissibility + single-tick combat prediction; no K-step rollout,
    # no leaf-value approximation. See knowledge-base/concepts/
    # trajectory-first-architecture.md. Default chooser remains the
    # K-step rollout for backward compat with the v15-line A/B baseline.
    if os.environ.get("BASELINE_CHOOSER", "").strip().lower() == "trajectory":
        # Trajectory chooser doesn't need baseline_favors (no idle baseline);
        # propose still wants a baseline_len for shape but value doesn't
        # affect the trajectory chooser's scoring.
        prerank = propose(
            my_planets, target_pool, world, model, me, omega,
            baseline_len=MAX_HORIZON + 1,
        )
        from agents.baseline.chooser_trajectory import choose_trajectory

        # 1. Tick + emit the ledger's due commitments (if any). Build
        #    the reserved-srcs set so the chooser doesn't double-commit
        #    on srcs we've already scheduled.
        #
        # Mode "hard" (default): reserve src for the whole wait window —
        # chooser cannot emit anything from that src until the commit
        # fires.
        # Mode "soft": only reserve sources whose commit is FIRING this
        # turn (so the chooser can't fire-now on top of the commit's
        # emit). Sources with surviving (in-flight) entries are NOT
        # reserved, leaving them free to opportunistically fire-now via
        # the chooser. The pending commit just needs `ships_planned`
        # ships still available when wait_remaining hits 0; if not
        # enough remain, the commit drops at emit time.
        due_moves: list[list] = []
        surviving_pending: list[dict] = []
        reserved_srcs: set[int] = set()
        reserved_for_new_commits: set[int] = set()
        if ledger_on:
            due_moves, surviving_pending = _tick_ledger(
                me, world, model, omega,
            )
            mode = os.environ.get("BASELINE_LEDGER_MODE",
                                  LEDGER_MODE).strip().lower()
            # Sources firing via the ledger this turn — chooser must not
            # fire-now on top of those (duplicate-src emit).
            firing_srcs = {int(m[0]) for m in due_moves}
            pending_srcs = {int(e["src_id"]) for e in surviving_pending}
            # Always block stacking a second wait-commit on a src that
            # already has a surviving commit (regardless of mode).
            reserved_for_new_commits = firing_srcs | pending_srcs
            # Hard mode: also block fire-now from pending srcs (preserve
            # the ship reserve for the future commit). Soft mode: leave
            # pending srcs free to fire-now (commit drops at emit time
            # if not enough ships remain).
            reserved_srcs = firing_srcs if mode == "soft" \
                else firing_srcs | pending_srcs

        moves, new_commits = choose_trajectory(
            snap_base, prerank, None,
            me, num_seats, wallclock_ms,
            MIN_HORIZON, MAX_HORIZON, gamma,
            world, model,
            reserved_srcs=reserved_srcs,
            reserved_for_new_commits=reserved_for_new_commits,
        )

        # 2. Persist updated ledger (surviving + new commits) when on.
        if ledger_on:
            _PENDING_LAUNCHES[me] = surviving_pending + new_commits

        moves = due_moves + moves
        moves = emit_threat_reinforcements(moves, planets, me, world, model, omega)
        return drain_idle_rear(moves, planets, me, world, model)

    # ROI chooser opt-in (2026-05-19). Closed-form ROI prior + N-way
    # coalition + opp-modifier posterior; no fast_sim rollout. See
    # agents/baseline/chooser_roi.py and the plan at
    # /root/.claude/plans/okay-we-can-do-elegant-lampson.md.
    if os.environ.get("BASELINE_CHOOSER", "").strip().lower() == "roi":
        prerank = propose(
            my_planets, target_pool, world, model, me, omega,
            baseline_len=MAX_HORIZON + 1,
        )
        from agents.baseline.chooser_roi import choose_roi
        step = int(obs_d.get("step", 0))
        moves = choose_roi(
            snap_base, prerank,
            me, num_seats, wallclock_ms,
            MIN_HORIZON, MAX_HORIZON, gamma,
            world, model, step,
        )
        moves = emit_threat_reinforcements(moves, planets, me, world, model, omega)
        return drain_idle_rear(moves, planets, me, world, model)

    baseline_favors = build_idle_baseline(
        snap_base, me, num_seats, MAX_HORIZON, gamma,
    )

    prerank = propose(
        my_planets, target_pool, world, model, me, omega,
        baseline_len=len(baseline_favors),
    )

    # Ledger lifecycle for the composite chooser path (parallel to the
    # trajectory branch above). Tick first; pass reservation sets;
    # merge with chooser output.
    composite_due: list[list] = []
    composite_surviving: list[dict] = []
    composite_reserved: set[int] = set()
    composite_reserved_new: set[int] = set()
    if ledger_on:
        composite_due, composite_surviving = _tick_ledger(
            me, world, model, omega,
        )
        mode = os.environ.get("BASELINE_LEDGER_MODE",
                              LEDGER_MODE).strip().lower()
        firing_srcs = {int(m[0]) for m in composite_due}
        pending_srcs = {int(e["src_id"]) for e in composite_surviving}
        composite_reserved_new = firing_srcs | pending_srcs
        composite_reserved = firing_srcs if mode == "soft" \
            else firing_srcs | pending_srcs

    moves, new_commits = choose(
        snap_base, prerank, baseline_favors,
        me, num_seats, wallclock_ms,
        MIN_HORIZON, MAX_HORIZON, gamma,
        world=world,
        reserved_srcs=composite_reserved,
        reserved_for_new_commits=composite_reserved_new,
    )

    if ledger_on:
        _PENDING_LAUNCHES[me] = composite_surviving + new_commits

    moves = composite_due + moves
    moves = emit_threat_reinforcements(moves, planets, me, world, model, omega)
    return drain_idle_rear(moves, planets, me, world, model)
