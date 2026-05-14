"""v7 minimax — richer-candidate enumeration + forward-sim scoring.

What this delivers: a 1-ply "minimax" chooser that evaluates a richer
set of candidate this-turn action bundles via `lib/fast_sim`'s K-step
rollout, with the opponent modelled by `lib/opp_model`. Picks
`argmax` of `delta_us_minus_them` at the rollout's terminal state.

Why now: the v3_lookahead MVP (audit/2026-05-11-v3-lookahead-mvp-
parity.md) plateaued at 50/50 vs v2 because drop-one is a strict
subset of the incumbent. With `fast_sim` at 0.12 ms/step (183× the
previous budget) we can afford 8–12 candidates × K=10 rollout
(~960 ms) per turn — enough headroom for additive enumeration.

The module is shared by all v7 variants. Each variant supplies only
its `enumerator_mode`; the scorer and watchdog are the same.

Enumerator modes (one variant per mode):

- `"drop_one"`     — incumbent + (incumbent minus each launch).
                     Re-runs the MVP under the new scorer.
- `"target_swap"`  — per-source runner-up snipe target.
- `"ship_sweep"`   — per-source same-target at {min, half, 0.95×}.
- `"archetype"`    — four siblings under preset archetypes
                     (baseline / concentrated / saturation / defensive).
- `"hungarian"`    — one global bipartite assignment alternative.
- `"combined"`     — union of every mode that passed its gate (the
                     final-form variant assembled by run_v7_ablation).

The scorer signature (`score_candidate`) is identical across modes —
the variation lives in `enumerate_candidates`.
"""

from __future__ import annotations

import math
import os
import time
from contextlib import contextmanager
from typing import Any, Callable, Iterable

# Env-var override for the wallclock budget used by every `choose_*` entry
# point. **Only consulted at the top of each chooser**, never inside the
# search loop, so the production path's `time.perf_counter()` watchdog
# is unchanged. Set by `scripts/bundle_agent.py::_parity_gate` to make
# source-vs-bundle parity tests deterministic: with the default 700 ms
# budget, a chooser may bail mid-candidate-list on system jitter, leaving
# argmax to pick over a different subset of candidates each run. Setting
# the budget effectively unbounded lets every candidate be scored, so
# the agent becomes a pure function of its inputs.
_WALLCLOCK_ENV_VAR = "ORBIT_WARS_PARITY_WALLCLOCK_MS"


def _effective_wallclock_ms(wallclock_ms: float) -> float:
    """Return `wallclock_ms` unless the parity-test env var is set, in
    which case use the env-var value. Invalid values fall back to the
    caller's number rather than crashing the agent."""
    override = os.environ.get(_WALLCLOCK_ENV_VAR)
    if not override:
        return wallclock_ms
    try:
        return float(override)
    except ValueError:
        return wallclock_ms

from lib.fast_sim import Snapshot, delta_us_minus_them
from lib.fast_sim import clone as fs_clone
from lib.fast_sim import from_obs as fs_from_obs
from lib.fast_sim import step as fs_step
from lib.fleet import speed as fleet_speed
from lib.intent import Intent, World, realize
from lib.mechanism import DEFAULT_MECHANISMS
from lib.mission import Mission
from lib.missions.opening import propose_opening_missions
from lib.missions.recapture import propose_recapture_missions
from lib.missions.reinforce import propose_reinforce_missions
from lib.missions.snipe import propose_snipe_missions
from lib.opp_model import make_opp_policy, top_tier_mirror_policy
from lib.planner import settle_plan
from lib.world_model import WorldModel


# ---------------------------------------------------------------------------
# Archetype presets — frozen constants from the top-10 fingerprint
# (knowledge-base/concepts/top-performer-strategies.md).
# ---------------------------------------------------------------------------

# Each preset is (aggressive_fraction, max_targets_per_source, reinforce_priority_boost).
# Reading: concentrated artillery (Isaiah / bowwowforeach) empties the
# source onto one big target; saturation skirmisher (flg / Ebi) spreads
# medium fleets over multiple targets; defensive boosts reinforce
# missions over snipe.
ARCHETYPE_PRESETS = {
    "baseline":     (0.7,  1, 1.0),   # v3.5.1 default
    "concentrated": (0.95, 1, 1.0),   # Isaiah-style: full source onto top-1
    "saturation":   (0.5,  3, 1.0),   # flg-style: medium fleets, 3 targets
    "defensive":    (0.7,  1, 3.0),   # reinforce priority × 3
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ranked_snipe_missions_by_source(
    world: World, model: WorldModel,
) -> dict[int, list[Mission]]:
    """{src_id: [Mission descending by score]} for every owned source.

    Uses the existing `propose_snipe_missions(aggressive=True)` — same
    set v3.5.1 considers. We re-rank per source so each source has its
    own top-K ordering."""
    missions = propose_snipe_missions(world, model, aggressive=True)
    by_src: dict[int, list[Mission]] = {}
    for m in missions:
        by_src.setdefault(m.src_id, []).append(m)
    for src_id in by_src:
        by_src[src_id].sort(key=lambda m: -m.score)
    return by_src


def _action_from_intents(
    intents: list[Intent], obs: Any, model: WorldModel | None = None,
) -> list[list]:
    """Run the realize pipeline to convert Intents into env-format actions."""
    return realize(intents, obs, mechanisms=DEFAULT_MECHANISMS, model=model)


def _override_one_source(
    incumbent_intents: list[Intent], override: Intent,
) -> list[Intent]:
    """Replace incumbent's intent from `override.src_id` with `override`.
    If the incumbent had no launch from that source, append."""
    out = [i for i in incumbent_intents if i.src_id != override.src_id]
    out.append(override)
    return out


def _aggressive_size(
    src_ships: int, target_min: int,
    *, fraction: float, reserve: int = 5,
) -> int:
    """v3.5.1's aggressive sizing formula, parameterized by fraction.

    Mirrors `lib/missions/snipe.py::AGGRESSIVE_FRACTION` semantics:
    base = min(src.ships * fraction, src.ships - reserve), clamped
    above target_min. With fraction=0.7 / reserve=5 this matches
    v3.5.1 exactly. Other fractions produce the concentrated /
    saturation archetypes."""
    if src_ships <= 12:
        return target_min
    fraction_size = max(1, int(src_ships * fraction))
    cap = max(1, int(src_ships) - reserve)
    return max(target_min, min(fraction_size, cap))


def _build_incumbent_intents(
    world: World, model: WorldModel, *, include_recapture: bool = False,
) -> list[Intent]:
    """v3.5.1's mission set: aggressive snipe + reinforce, run through
    settle_plan. Used as the parity-floor candidate.

    `include_recapture=True` (v7.2+) also adds recapture missions.
    The recapture proposer's score is calibrated to snipe scale and
    top-K capped (see lib/missions/recapture.py); without those fixes
    recapture dominates settle_plan and regresses
    (audit/2026-05-12-recapture-wireup-ab.md).
    """
    missions = (
        propose_opening_missions(world, model)
        + propose_snipe_missions(world, model, aggressive=True)
        + propose_reinforce_missions(world, model)
    )
    if include_recapture:
        missions = missions + propose_recapture_missions(world, model)
    chosen = settle_plan(missions, world, model)
    return chosen


# ---------------------------------------------------------------------------
# Enumerators (one per mode)
# ---------------------------------------------------------------------------


@contextmanager
def _bind_shared_world_model(obs_list, model):
    """Temporarily attach ``model`` to each observation in ``obs_list`` as
    the ``_shared_world_model`` attribute so mirror-style policies can
    skip the expensive ``WorldModel.from_world`` rebuild
    (`lib/opp_model.py:76, 112`). Exception-safe — the attribute is
    always removed on exit, even if a policy raises.

    Why this matters: the previous bare ``del obs._shared_world_model``
    cleanup ran only on the happy path. A raise inside the followup
    policy left the attribute on the cloned observation Struct, which
    is then garbage-collected — but in the parity-gate setting where
    the source and bundle agents are called back-to-back in the same
    process on the same input ``obs``, any leaked side-channel state on
    a Struct that participates in both calls is a parity risk. Encoding
    the lifetime as a context manager makes the invariant impossible
    to violate accidentally.
    """
    if model is None or not obs_list:
        yield
        return
    for obs in obs_list:
        obs._shared_world_model = model
    try:
        yield
    finally:
        for obs in obs_list:
            # Use try/except (rather than __dict__.pop) to mirror the
            # same access path that __setattr__ took; Struct's attribute
            # storage isn't guaranteed to be __dict__.
            try:
                del obs._shared_world_model
            except AttributeError:
                pass


def _enumerate_drop_one(incumbent_action: list[list]) -> list[list[list]]:
    """Incumbent + each drop-one variant. Floor for v7 framework lift."""
    if not incumbent_action:
        return [[]]
    cands: list[list[list]] = [list(incumbent_action)]
    for i in range(len(incumbent_action)):
        cands.append([m for j, m in enumerate(incumbent_action) if j != i])
    return cands


def _enumerate_target_swap(
    world: World, model: WorldModel,
    incumbent_intents: list[Intent], incumbent_action: list[list],
    obs: Any,
) -> list[list[list]]:
    """For each owned source in the incumbent, swap to its runner-up
    snipe target. Generates at most N additional candidates."""
    cands: list[list[list]] = [list(incumbent_action)]
    by_src = _ranked_snipe_missions_by_source(world, model)
    incumbent_by_src = {i.src_id: i for i in incumbent_intents}
    for src_id, ranked in by_src.items():
        if len(ranked) < 2:
            continue
        # Find the top mission that's NOT the incumbent's target choice.
        cur_target = (
            incumbent_by_src[src_id].target_id
            if src_id in incumbent_by_src else None
        )
        alt = next((m for m in ranked if m.target_id != cur_target), None)
        if alt is None:
            continue
        new_intents = _override_one_source(incumbent_intents, alt.to_intent())
        cands.append(_action_from_intents(new_intents, obs, model))
    return cands


def _enumerate_ship_sweep(
    world: World, model: WorldModel,
    incumbent_intents: list[Intent], incumbent_action: list[list],
    obs: Any,
) -> list[list[list]]:
    """For each owned source's top-target mission, sweep ships in
    {min-viable, half, 0.95×}. At most 3N additional candidates."""
    cands: list[list[list]] = [list(incumbent_action)]
    by_src = _ranked_snipe_missions_by_source(world, model)
    incumbent_by_src = {i.src_id: i for i in incumbent_intents}
    for src_id, ranked in by_src.items():
        if not ranked:
            continue
        src = world.planets_by_id.get(src_id)
        if src is None or src.ships <= 1:
            continue
        # Use the incumbent's chosen target if it exists for this source,
        # otherwise the top-ranked target from the missions.
        chosen_target_id = (
            incumbent_by_src[src_id].target_id
            if src_id in incumbent_by_src
            else ranked[0].target_id
        )
        target = world.planets_by_id.get(chosen_target_id)
        if target is None:
            continue
        target_min = max(1, int(target.ships) + 1)
        for fraction in (0.5, 0.95):
            ships = _aggressive_size(
                int(src.ships), target_min, fraction=fraction,
            )
            if ships <= 0 or ships > src.ships:
                continue
            # Build the swap intent at this fraction.
            new_intent = Intent(
                src_id=src_id, target_id=chosen_target_id, ships=ships,
            )
            new_intents = _override_one_source(incumbent_intents, new_intent)
            cand = _action_from_intents(new_intents, obs, model)
            if cand and cand != incumbent_action:
                cands.append(cand)
    return cands


def _enumerate_archetype(
    world: World, model: WorldModel, obs: Any,
) -> list[list[list]]:
    """Generate four full-action bundles under preset archetypes.

    Each preset re-derives the snipe+reinforce mission set and runs it
    through settle_plan. The realize pipeline is the same; only the
    per-mission scoring weights and ship-sizing change.
    """
    cands: list[list[list]] = []
    # The "baseline" preset == v3.5.1 incumbent, which the caller
    # always includes first via enumerate_candidates.
    for name, (fraction, max_per_src, reinforce_boost) in ARCHETYPE_PRESETS.items():
        if name == "baseline":
            continue
        # Re-derive missions with the preset's aggressive_fraction.
        snipe = _snipe_missions_with_fraction(
            world, model, fraction=fraction, max_targets_per_source=max_per_src,
        )
        reinforce = propose_reinforce_missions(world, model)
        if reinforce_boost != 1.0:
            for r in reinforce:
                r.score = r.score * reinforce_boost
        chosen = settle_plan(snipe + reinforce, world, model)
        cand = _action_from_intents(chosen, obs, model)
        cands.append(cand)
    return cands


def _snipe_missions_with_fraction(
    world: World, model: WorldModel,
    *, fraction: float, max_targets_per_source: int,
) -> list[Mission]:
    """Reuse propose_snipe_missions(aggressive=True) for ship sizing,
    then post-filter each source to only its top-N missions to enforce
    the archetype's `max_targets_per_source` policy.

    A full re-implementation would pass `fraction` into the proposer,
    but the proposer's `AGGRESSIVE_FRACTION` is a module constant. We
    re-size each emitted Mission's `ships` here to match the preset's
    fraction — same effect, no patching of the lib module.
    """
    base = propose_snipe_missions(world, model, aggressive=True)
    # Re-size to the preset's fraction.
    resized: list[Mission] = []
    for m in base:
        src = world.planets_by_id.get(m.src_id)
        if src is None:
            continue
        target = world.planets_by_id.get(m.target_id)
        if target is None:
            continue
        target_min = max(1, int(target.ships) + 1)
        ships = _aggressive_size(int(src.ships), target_min, fraction=fraction)
        m.ships = ships
        resized.append(m)
    # Per-source top-N filter.
    by_src: dict[int, list[Mission]] = {}
    for m in resized:
        by_src.setdefault(m.src_id, []).append(m)
    out: list[Mission] = []
    for src_id, ranked in by_src.items():
        ranked.sort(key=lambda m: -m.score)
        out.extend(ranked[:max_targets_per_source])
    return out


def _enumerate_hungarian(
    world: World, model: WorldModel,
    incumbent_action: list[list], obs: Any,
) -> list[list[list]]:
    """Globally-coordinated (source × target) assignment as one
    additional candidate. Uses `scipy.optimize.linear_sum_assignment`.

    The score matrix is `propose_snipe_missions` scores filtered to
    our owned sources × non-owned targets. The assignment forces each
    source to ONE target (no double-commit), unlike settle_plan's
    per-source greedy with a same-turn ledger.

    Returns `[incumbent, hungarian_alternative]` — caller appends.
    """
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError:
        # Bundle environment doesn't ship scipy — fall back to incumbent
        # only. This is an acceptable degradation.
        return [list(incumbent_action)]

    cands: list[list[list]] = [list(incumbent_action)]

    missions = propose_snipe_missions(world, model, aggressive=True)
    if not missions:
        return cands
    sources = sorted({m.src_id for m in missions})
    targets = sorted({m.target_id for m in missions})
    if not sources or not targets:
        return cands

    src_to_row = {s: i for i, s in enumerate(sources)}
    tgt_to_col = {t: j for j, t in enumerate(targets)}
    # Initialise with a very negative score (linear_sum_assignment minimises
    # cost; we negate scores so high score = low cost).
    NEG = 1e6
    cost = [[NEG for _ in targets] for _ in sources]
    by_pair: dict[tuple[int, int], Mission] = {}
    for m in missions:
        i = src_to_row[m.src_id]
        j = tgt_to_col[m.target_id]
        if cost[i][j] > -m.score:
            cost[i][j] = -m.score
            by_pair[(m.src_id, m.target_id)] = m

    # Pad to square matrix if rectangular — linear_sum_assignment supports
    # rectangular but a small problem is fine.
    row_ind, col_ind = linear_sum_assignment(cost)
    chosen: list[Mission] = []
    for i, j in zip(row_ind, col_ind):
        if cost[i][j] >= NEG / 2:
            continue
        src_id = sources[i]
        tgt_id = targets[j]
        m = by_pair.get((src_id, tgt_id))
        if m is not None:
            chosen.append(m)
    if not chosen:
        return cands
    # settle_plan still applies per-source uniqueness + same-turn ledger;
    # run it on `chosen` so the bundle uses the same gating discipline.
    intents = settle_plan(chosen, world, model)
    cand = _action_from_intents(intents, obs, model)
    if cand:
        cands.append(cand)
    return cands


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def enumerate_candidates(
    world: World,
    model: WorldModel,
    *,
    enumerator_mode: str,
    incumbent_intents: list[Intent],
    incumbent_action: list[list],
    obs: Any,
) -> list[list[list]]:
    """Generate candidate action bundles. Incumbent is ALWAYS index 0
    so the watchdog fallback never regresses below v3.5.1."""
    if enumerator_mode == "drop_one":
        return _enumerate_drop_one(incumbent_action)
    if enumerator_mode == "target_swap":
        return _enumerate_target_swap(
            world, model, incumbent_intents, incumbent_action, obs,
        )
    if enumerator_mode == "ship_sweep":
        return _enumerate_ship_sweep(
            world, model, incumbent_intents, incumbent_action, obs,
        )
    if enumerator_mode == "archetype":
        archetypes = _enumerate_archetype(world, model, obs)
        # baseline first, then the other three.
        return [list(incumbent_action)] + archetypes
    if enumerator_mode == "hungarian":
        return _enumerate_hungarian(world, model, incumbent_action, obs)
    if enumerator_mode == "combined":
        # Union of every mode's candidates, with the incumbent only once.
        seen: list[list[list]] = [list(incumbent_action)]
        seen_keys = {_action_key(incumbent_action)}
        for mode in ("drop_one", "target_swap", "ship_sweep",
                     "archetype", "hungarian"):
            for cand in enumerate_candidates(
                world, model, enumerator_mode=mode,
                incumbent_intents=incumbent_intents,
                incumbent_action=incumbent_action,
                obs=obs,
            ):
                k = _action_key(cand)
                if k not in seen_keys:
                    seen.append(cand)
                    seen_keys.add(k)
        return seen
    raise ValueError(f"unknown enumerator_mode: {enumerator_mode}")


def _action_key(action: list[list]) -> tuple:
    """Hashable key for deduplicating action bundles. Coarse rounding on
    the angle so jittered duplicates aren't double-counted."""
    return tuple(
        (int(m[0]), round(float(m[1]), 5), int(m[2])) for m in action
    )


def _infer_num_seats(world: World) -> int:
    """Best-effort player-count inference from the obs.

    The kaggle_environments obs doesn't directly expose `num_agents` —
    only `player` (our seat). We infer from the highest owner ID seen
    across planets + fleets. 2P games yield owners in {0, 1} ∪ {-1};
    4P yields {0, 1, 2, 3} ∪ {-1}.
    """
    max_owner = world.my_id
    for p in world.planets_by_id.values():
        if p.owner > max_owner:
            max_owner = p.owner
    raw = world.obs_raw
    fleets = (
        raw.get("fleets", [])
        if isinstance(raw, dict)
        else getattr(raw, "fleets", [])
    )
    for f in fleets or []:
        owner = int(f[1])
        if owner > max_owner:
            max_owner = owner
    return max_owner + 1 if max_owner >= 0 else 1


def score_candidate(
    snap: Snapshot,
    action: list[list],
    *,
    my_id: int = 0,
    K: int = 10,
    opp_tier: int = 1,
    value_fn: Callable | None = None,
    followup_policy: Callable | None = None,
) -> float:
    """Rollout score for `action` under our seat.

    The opponent plays the requested tier policy throughout the
    rollout. Our seat plays `action` on the first tick, then the
    top-tier mirror policy thereafter.

    `value_fn(observation, my_id) -> float` is the leaf-state scoring
    head. Defaults to `delta_us_minus_them` (our minus their total
    ships) — the Phase-2-validated scalar. v7.3+ passes
    `lib.lookahead_planner.evaluate_value` for production-share +
    denial + survivor bonus.

    `followup_policy(observation) -> list` is the policy applied to
    BOTH seats for the K-1 follow-up steps after our forced action.
    Defaults to `top_tier_mirror_policy` (v3.5.1 pipeline, ~10 ms /
    call). Pass `lite_greedy_policy` (~1 ms / call) when the rollout
    only needs to estimate trajectory direction and bit-fidelity to
    v3.5.1 isn't required — typically the case for wider/deeper
    multi-candidate search.
    """
    if snap.num_seats != 2:
        raise ValueError(f"v7 score_candidate is 2P only (got {snap.num_seats})")
    clone = fs_clone(snap)
    opp_id = 1 - my_id

    opp_policy = make_opp_policy(opp_tier)
    if followup_policy is None:
        followup_policy = top_tier_mirror_policy

    # First step: forced action for us; opp plays its policy.
    a_opp = opp_policy(clone.state[opp_id].observation)
    actions = [None, None]
    actions[my_id] = action
    actions[opp_id] = a_opp
    if not clone.done:
        clone = fs_step(clone, actions, in_place=True)

    # Remaining K-1 steps: both seats play follow-up policy.
    # OPTIMIZATION (Phase 3c): both seats see the same planets/fleets/
    # comets/angular_velocity, so `WorldModel.from_world` produces the
    # SAME object regardless of which seat's obs we built it from. We
    # build it once per step and stash it on both seats' observations
    # via the `_shared_world_model` attribute; mirror-style policies
    # (lib.opp_model.top_tier_mirror_policy / mirror_self_policy) check
    # for this attribute and skip the expensive rebuild (~3.8 ms each).
    # Net savings: ~3.8 ms per rollout step. Bit-exact parity preserved
    # because the same model object is used either way.
    for _ in range(max(0, K - 1)):
        if clone.done:
            break
        obs0 = clone.state[0].observation
        obs1 = clone.state[1].observation
        # Build shared World/Model once. Cheap to construct World per
        # seat (it's a tiny dataclass); the expensive part is WorldModel.
        shared_world = World.from_obs(obs0)
        shared_model = (
            WorldModel.from_world(shared_world)
            if shared_world.planets_by_id else None
        )
        with _bind_shared_world_model((obs0, obs1), shared_model):
            a0 = followup_policy(obs0)
            a1 = followup_policy(obs1)
        clone = fs_step(clone, [a0, a1], in_place=True)

    if value_fn is None:
        return delta_us_minus_them(clone, my_id)
    return value_fn(clone.state[my_id].observation, my_id)


def score_candidate_symmetric(
    snap: Snapshot,
    action: list[list],
    *,
    K: int = 10,
    opp_tier: int = 1,
) -> float:
    """Seat-symmetric variant of `score_candidate`.

    Runs the rollout twice — once with us at seat 0 (opp at seat 1)
    and once with us at seat 1 (opp at seat 0) — and averages the
    `delta_us_minus_them` results from our POV in each. Cancels the
    env's documented P1-favoring tie-break bias that otherwise leaks
    into the maximin payoff matrix.

    Ported from `score_joint_action_symmetric` in
    `origin/claude/game-theory-strategy-analysis-0oH4N` but adapted
    to operate on Snapshots (so we keep fast_sim's 183× speedup).

    Cost: 2× score_candidate.
    """
    a = score_candidate(snap, action, my_id=0, K=K, opp_tier=opp_tier)
    b = score_candidate(snap, action, my_id=1, K=K, opp_tier=opp_tier)
    return (a + b) / 2.0


def score_joint(
    snap: Snapshot,
    our_action: list[list],
    opp_action: list[list],
    *,
    my_id: int = 0,
    K: int = 10,
    value_fn: Callable | None = None,
) -> float:
    """Snapshot variant of `lib/lookahead.score_joint_action`.

    Both first-turn actions are forced; turns 2..K both seats play
    top_tier_mirror. Returns the leaf-state value via `value_fn`
    (default: `delta_us_minus_them`).
    """
    if snap.num_seats != 2:
        raise ValueError(f"score_joint is 2P only (got {snap.num_seats})")
    clone = fs_clone(snap)
    opp_id = 1 - my_id
    actions = [None, None]
    actions[my_id] = our_action
    actions[opp_id] = opp_action
    if not clone.done:
        clone = fs_step(clone, actions, in_place=True)
    for _ in range(max(0, K - 1)):
        if clone.done:
            break
        a0 = top_tier_mirror_policy(clone.state[0].observation)
        a1 = top_tier_mirror_policy(clone.state[1].observation)
        clone = fs_step(clone, [a0, a1], in_place=True)
    if value_fn is None:
        return delta_us_minus_them(clone, my_id)
    return value_fn(clone.state[my_id].observation, my_id)


def score_joint_symmetric(
    snap: Snapshot,
    our_action: list[list],
    opp_action: list[list],
    *,
    K: int = 10,
    value_fn: Callable | None = None,
) -> float:
    """Seat-symmetric joint scorer. Used by the maximin overlay."""
    a = score_joint(snap, our_action, opp_action, my_id=0, K=K, value_fn=value_fn)
    b = score_joint(snap, our_action, opp_action, my_id=1, K=K, value_fn=value_fn)
    return (a + b) / 2.0


def _drop_smallest(action: list[list]) -> list[list]:
    """Return `action` with its smallest-ship launch removed.

    Mirrors the drop_smallest function in v7_minimax (ported from
    `origin/claude/game-theory-strategy-analysis-0oH4N`'s
    agents/v7_minimax/main.py:98-117). Ties broken by removing the
    EARLIEST launch among smallest, which is σ-deterministic given
    upstream ordering.
    """
    if not action:
        return []
    if len(action) == 1:
        return []
    min_idx = 0
    min_ships = int(action[0][2])
    for i, la in enumerate(action[1:], start=1):
        if int(la[2]) < min_ships:
            min_ships = int(la[2])
            min_idx = i
    return [la for i, la in enumerate(action) if i != min_idx]


def _opp_incumbent_action(world: World, obs: Any, opp_id: int) -> list[list]:
    """Compute the opponent's incumbent action via v3.5.1 pipeline
    from the opp's POV.

    We don't have a clean way to swap `world.my_id` (it's frozen at
    construction), so we rebuild World from a copy of obs with
    `player=opp_id`. This is the same technique v7_minimax uses
    (`_swap_obs_player` in their main.py).
    """
    if isinstance(obs, dict):
        obs2 = dict(obs)
        obs2["player"] = opp_id
    else:
        keys = (
            "player", "planets", "fleets", "angular_velocity",
            "initial_planets", "comet_planet_ids", "comets",
            "step", "next_fleet_id",
        )
        obs2 = {}
        for k in keys:
            v = getattr(obs, k, None)
            if v is not None:
                obs2[k] = v
        obs2["player"] = opp_id
    opp_world = World.from_obs(obs2)
    if not opp_world.planets_by_id:
        return []
    opp_model = WorldModel.from_world(opp_world)
    missions = (
        propose_snipe_missions(opp_world, opp_model, aggressive=True)
        + propose_reinforce_missions(opp_world, opp_model)
    )
    intents = settle_plan(missions, opp_world, opp_model)
    return realize(intents, obs2, mechanisms=DEFAULT_MECHANISMS, model=opp_model)


def choose_maximin(
    obs: Any,
    configuration: Any = None,
    *,
    K: int = 10,
    wallclock_ms: float = 700.0,
    my_id: int | None = None,
    use_symmetric: bool = True,
    include_recapture: bool = False,
    value_fn: Callable | None = None,
) -> list[list]:
    """v7.1 maximin overlay.

    Per turn:
      1. Build N=N+1 our candidates via `_enumerate_drop_one(incumbent)`
         (incumbent + drop-each-launch).
      2. Build M=2 opp candidates: opp's v3.5.1 incumbent + drop-smallest.
      3. Score every (our_i, opp_j) cell via `score_joint_symmetric`
         (Snapshot, K-step rollout, symmetric average).
      4. Pick i* = argmax_i (min_j P[i,j]). Tie-break: prefer row 0
         (= our incumbent) — σ-equivariant fallback.

    Wallclock watchdog (`wallclock_ms`, default 700) bails the inner
    loop if budget exhausted. Row 0 (incumbent) is ALWAYS evaluated
    in full first so its worst-case is honest. 4P games fall back to
    the incumbent (no maximin guarantee at n>2).
    """
    wallclock_ms = _effective_wallclock_ms(wallclock_ms)
    t_start = time.perf_counter()

    world = World.from_obs(obs)
    if not world.planets_by_id:
        return []
    if my_id is None:
        my_id = world.my_id
    model = WorldModel.from_world(world)

    incumbent_intents = _build_incumbent_intents(
        world, model, include_recapture=include_recapture,
    )
    incumbent_action = _action_from_intents(incumbent_intents, obs, model)

    # 4P fallback — maximin is 2P-only.
    if _infer_num_seats(world) != 2:
        return incumbent_action

    opp_id = 1 - my_id
    # Our candidate class: incumbent + each drop-one variant.
    C = _enumerate_drop_one(incumbent_action)
    if len(C) <= 1:
        return incumbent_action
    # Opp candidate class M=2.
    O_inc = _opp_incumbent_action(world, obs, opp_id)
    O_drop = _drop_smallest(O_inc)
    O = [O_inc] if not O_drop or O_drop == O_inc else [O_inc, O_drop]

    snap = fs_from_obs(obs, configuration, episode_seed=0, num_seats=2)
    if use_symmetric:
        def score_fn(s, ours, opps, *, K=K):
            return score_joint_symmetric(s, ours, opps, K=K, value_fn=value_fn)
    else:
        def score_fn(s, ours, opps, *, K=K):
            return score_joint(s, ours, opps, my_id=my_id, K=K, value_fn=value_fn)

    N = len(C)
    M = len(O)
    P: list[list[float]] = [[float("-inf")] * M for _ in range(N)]
    unfilled: list[list[bool]] = [[True] * M for _ in range(N)]

    # Row 0 (incumbent) first, full row. Then i>=1 row-by-row with bail.
    for i in range(N):
        for j in range(M):
            elapsed_ms = (time.perf_counter() - t_start) * 1000.0
            if i > 0 and elapsed_ms > wallclock_ms:
                break
            try:
                P[i][j] = score_fn(snap, C[i], O[j], K=K)
                unfilled[i][j] = False
            except Exception:
                P[i][j] = float("-inf")
                unfilled[i][j] = False
        else:
            continue
        break  # exited inner via budget bail

    # Maximin: argmax_i (min_j P[i,j]) over evaluated cells, tie → row 0.
    best_i = 0
    best_worst = float("-inf")
    for i in range(N):
        evaluated = [P[i][j] for j in range(M) if not unfilled[i][j]]
        if not evaluated:
            worst = float("-inf")
        else:
            worst = min(evaluated)
        if worst > best_worst:
            best_worst = worst
            best_i = i
    return C[best_i]


def score_candidate_4p(
    snap: Snapshot,
    action: list[list],
    *,
    my_id: int,
    K: int = 8,
    value_fn: Callable | None = None,
) -> float:
    """Rollout score for a 4P candidate action.

    All 3 non-pov seats play `top_tier_mirror_policy`. Our seat plays
    `action` on tick 0, then `top_tier_mirror_policy` for the rest.
    Scoring head: `value_fn(state[my_id].observation, my_id)` at
    terminal — defaults to "our ships − max(other seat ships)" which
    rewards keeping the lead vs the best-remaining-opponent (better
    proxy for 4P first-place than total-sum-of-them).
    """
    if snap.num_seats != 4:
        raise ValueError(f"score_candidate_4p needs num_seats=4 (got {snap.num_seats})")
    clone = fs_clone(snap)

    # First step: forced action for us; all 3 opps play top_tier_mirror.
    actions: list[list[list]] = [[] for _ in range(4)]
    for seat in range(4):
        if seat == my_id:
            actions[seat] = action
        else:
            actions[seat] = top_tier_mirror_policy(clone.state[seat].observation)
    if not clone.done:
        clone = fs_step(clone, actions, in_place=True)

    # Remaining K-1 steps: all 4 seats play top_tier_mirror.
    for _ in range(max(0, K - 1)):
        if clone.done:
            break
        acts = [top_tier_mirror_policy(clone.state[seat].observation) for seat in range(4)]
        clone = fs_step(clone, acts, in_place=True)

    if value_fn is not None:
        return value_fn(clone.state[my_id].observation, my_id)

    # Default 4P scoring: our ships − max(other seat ships).
    # Better proxy for "did we keep the lead vs the best-remaining-
    # opponent" than (our − sum_others), which is dominated by total
    # ship counts.
    from collections import defaultdict
    totals: dict[int, float] = defaultdict(float)
    obs0 = clone.state[my_id].observation
    for p in obs0.planets:
        if int(p[1]) >= 0:
            totals[int(p[1])] += float(p[5])
    for f in obs0.fleets:
        if int(f[1]) >= 0:
            totals[int(f[1])] += float(f[6])
    ours = totals.get(my_id, 0.0)
    others = [v for k, v in totals.items() if k != my_id and k >= 0]
    best_opp = max(others) if others else 0.0
    return ours - best_opp


def choose_4p(
    obs: Any,
    configuration: Any = None,
    *,
    K: int = 8,
    wallclock_ms: float = 700.0,
    my_id: int | None = None,
    include_recapture: bool = True,
    value_fn: Callable | None = None,
) -> list[list]:
    """v7.4 — 4P drop-one chooser.

    No maximin (no Nash guarantee at n>2). All 3 opps modeled as
    top_tier_mirror; we score drop-one candidates and pick argmax.
    Falls back to incumbent if the watchdog trips or no candidate
    strictly beats it.
    """
    wallclock_ms = _effective_wallclock_ms(wallclock_ms)
    t_start = time.perf_counter()

    world = World.from_obs(obs)
    if not world.planets_by_id:
        return []
    if my_id is None:
        my_id = world.my_id
    model = WorldModel.from_world(world)

    incumbent_intents = _build_incumbent_intents(
        world, model, include_recapture=include_recapture,
    )
    incumbent_action = _action_from_intents(incumbent_intents, obs, model)

    # Build candidate set: incumbent + drop-each-launch.
    candidates = _enumerate_drop_one(incumbent_action)
    if len(candidates) <= 1:
        return incumbent_action

    snap = fs_from_obs(obs, configuration, episode_seed=0, num_seats=4)

    best_action = incumbent_action
    best_score = float("-inf")
    incumbent_scored = False
    for cand in candidates:
        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        if elapsed_ms > wallclock_ms:
            break
        try:
            score = score_candidate_4p(
                snap, cand, my_id=my_id, K=K, value_fn=value_fn,
            )
        except Exception:
            continue
        if not incumbent_scored:
            incumbent_scored = True
            best_score = score
            best_action = list(cand)
            continue
        if score > best_score:
            best_score = score
            best_action = list(cand)
    return best_action


def choose_simple_2p(
    obs: Any,
    configuration: Any = None,
    *,
    K: int = 10,
    wallclock_ms: float = 700.0,
    my_id: int | None = None,
    include_recapture: bool = True,
    value_fn: Callable | None = None,
) -> list[list]:
    """2P drop-one chooser WITHOUT maximin overlay.

    This is what v7.1 maximin should have been but wasn't: pure
    argmax over drop-one candidates with σ-equiv-enabled incumbent.
    The maximin variant (`choose_maximin`) lost the A/B because
    its 2×N × symmetric-scoring budget blew the wallclock; the
    simple variant has the same per-candidate cost as v7_0 (proven
    fast enough at 746-816 ms p95) while still getting σ-equiv,
    recapture, and value_fn for free.
    """
    wallclock_ms = _effective_wallclock_ms(wallclock_ms)
    t_start = time.perf_counter()

    world = World.from_obs(obs)
    if not world.planets_by_id:
        return []
    if my_id is None:
        my_id = world.my_id
    model = WorldModel.from_world(world)

    incumbent_intents = _build_incumbent_intents(
        world, model, include_recapture=include_recapture,
    )
    incumbent_action = _action_from_intents(incumbent_intents, obs, model)

    candidates = _enumerate_drop_one(incumbent_action)
    if len(candidates) <= 1:
        return incumbent_action

    snap = fs_from_obs(obs, configuration, episode_seed=0, num_seats=2)
    best_action = incumbent_action
    best_score = float("-inf")
    incumbent_scored = False
    for cand in candidates:
        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        if elapsed_ms > wallclock_ms:
            break
        try:
            score = score_candidate(
                snap, cand, my_id=my_id, K=K, opp_tier=1, value_fn=value_fn,
            )
        except Exception:
            continue
        if not incumbent_scored:
            incumbent_scored = True
            best_score = score
            best_action = list(cand)
            continue
        if score > best_score:
            best_score = score
            best_action = list(cand)
    return best_action


def _score_after_opp_response(
    snap_i: Snapshot,
    opp_act: list[list],
    *,
    my_id: int,
    opp_id: int,
    K_tail: int,
    value_fn: Callable | None = None,
) -> float:
    """Score after a forced opp response on turn 2.

    From `snap_i` (a snapshot that has already been advanced one turn by
    our forced action paired with the opp's incumbent), force the opp's
    response action `opp_act` on this turn (we pass — we've committed),
    then run `K_tail` mirror-mirror follow-up steps. Score from `my_id`'s
    POV via `value_fn` (default `delta_us_minus_them`).

    Used by `choose_depth2` to fill the maximin payoff matrix.
    """
    clone = fs_clone(snap_i)
    if not clone.done:
        actions: list[Any] = [None, None]
        actions[my_id] = []  # we pass
        actions[opp_id] = opp_act
        clone = fs_step(clone, actions, in_place=True)

    for _ in range(max(0, K_tail)):
        if clone.done:
            break
        a0 = top_tier_mirror_policy(clone.state[0].observation)
        a1 = top_tier_mirror_policy(clone.state[1].observation)
        clone = fs_step(clone, [a0, a1], in_place=True)

    if value_fn is None:
        return delta_us_minus_them(clone, my_id)
    return value_fn(clone.state[my_id].observation, my_id)


def choose_depth2(
    obs: Any,
    configuration: Any = None,
    *,
    K: int = 6,
    wallclock_ms: float = 700.0,
    my_id: int | None = None,
    include_recapture: bool = True,
    value_fn: Callable | None = None,
    max_our_candidates: int = 8,
    max_opp_candidates: int = 4,
) -> list[list]:
    """v7 depth-2 maximin (action-SEQUENCE depth-2, not joint-1-ply).

    Algorithm:
    1. Enumerate our drop-one candidate set (≤ `max_our_candidates`).
    2. For each our candidate i:
       a. Step the snapshot one turn with [our_i, opp_initial_incumbent].
       b. From the post-step state, recompute the opp's incumbent and
          enumerate the opp's drop-one set (≤ `max_opp_candidates`).
       c. For each opp candidate j, force it on turn 2 (we pass), then
          rollout `K-2` mirror-mirror steps. Record payoff[i][j].
    3. Maximin: argmax_i min_j payoff[i][j]. Tie → row 0 (incumbent).

    Budget shape (defaults): 8 × 4 × ~15 ms = ~500 ms wall; under 700 ms
    actTimeout. Watchdog: bail outer rows past 0.5 × wallclock_ms (row 0
    always evaluated in full first), bail inner cells past wallclock_ms.

    4P fallback: return the incumbent (depth-2 minimax is 2P-only —
    Nash maximin doesn't generalise cleanly to n > 2).
    """
    wallclock_ms = _effective_wallclock_ms(wallclock_ms)
    t_start = time.perf_counter()

    world = World.from_obs(obs)
    if not world.planets_by_id:
        return []
    if my_id is None:
        my_id = world.my_id
    model = WorldModel.from_world(world)

    incumbent_intents = _build_incumbent_intents(
        world, model, include_recapture=include_recapture,
    )
    incumbent_action = _action_from_intents(incumbent_intents, obs, model)

    # 4P fallback — depth-2 maximin is 2P only.
    if _infer_num_seats(world) != 2:
        return incumbent_action

    our_C = _enumerate_drop_one(incumbent_action)
    if max_our_candidates and len(our_C) > max_our_candidates:
        our_C = our_C[:max_our_candidates]
    if len(our_C) <= 1:
        return incumbent_action

    snap = fs_from_obs(obs, configuration, episode_seed=0, num_seats=2)
    opp_id = 1 - my_id

    # Opp plays its v3.5.1 incumbent on turn 1 against every one of our
    # candidates (same opp action across all rows — keeps the matrix
    # comparable to v7_0_drop_one's evaluation).
    opp_initial_action = _opp_incumbent_action(world, obs, opp_id)

    K_tail = max(0, K - 2)
    N = len(our_C)
    P: list[list[float]] = [[] for _ in range(N)]

    for i in range(N):
        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        if i > 0 and elapsed_ms > 0.5 * wallclock_ms:
            # Row 0 (incumbent) is always evaluated in full first; later
            # rows bail if half the budget is gone.
            P[i] = []
            continue

        try:
            snap_i = fs_clone(snap)
            if not snap_i.done:
                actions: list[Any] = [None, None]
                actions[my_id] = our_C[i]
                actions[opp_id] = opp_initial_action
                snap_i = fs_step(snap_i, actions, in_place=True)
        except Exception:
            P[i] = []
            continue

        if snap_i.done:
            # Game ended in turn 1 — score the leaf directly. Same
            # payoff for any opp_C[j] since there's no turn 2.
            try:
                terminal = (
                    delta_us_minus_them(snap_i, my_id)
                    if value_fn is None
                    else value_fn(snap_i.state[my_id].observation, my_id)
                )
            except Exception:
                terminal = float("-inf")
            P[i] = [terminal]
            continue

        # Recompute opp's incumbent from the post-turn-1 state.
        opp_obs_after = snap_i.state[opp_id].observation
        try:
            opp_world = World.from_obs(opp_obs_after)
            opp_model = WorldModel.from_world(opp_world)
            opp_inc_intents = _build_incumbent_intents(
                opp_world, opp_model, include_recapture=include_recapture,
            )
            opp_inc_action = _action_from_intents(
                opp_inc_intents, opp_obs_after, opp_model,
            )
            opp_C = _enumerate_drop_one(opp_inc_action)
            if max_opp_candidates and len(opp_C) > max_opp_candidates:
                opp_C = opp_C[:max_opp_candidates]
        except Exception:
            opp_C = [[]]
        if not opp_C:
            opp_C = [[]]

        row_scores: list[float] = []
        for j, opp_act in enumerate(opp_C):
            elapsed_ms = (time.perf_counter() - t_start) * 1000.0
            if elapsed_ms > wallclock_ms:
                break
            try:
                payoff = _score_after_opp_response(
                    snap_i, opp_act,
                    my_id=my_id, opp_id=opp_id, K_tail=K_tail,
                    value_fn=value_fn,
                )
            except Exception:
                payoff = float("-inf")
            row_scores.append(payoff)

        P[i] = row_scores

    # Maximin over the evaluated rows.
    NEG_INF = float("-inf")
    best_i = 0
    best_worst = NEG_INF
    for i in range(N):
        if not P[i]:
            continue
        worst = min(P[i])
        if worst > best_worst:
            best_worst = worst
            best_i = i

    return our_C[best_i] if best_worst > NEG_INF else incumbent_action


def choose_archetype_minregret(
    obs: Any,
    configuration: Any = None,
    *,
    K: int = 6,
    wallclock_ms: float = 700.0,
    my_id: int | None = None,
    use_min_regret: bool = True,
    include_recapture: bool = True,
    value_fn: Callable | None = None,
    max_our_candidates: int = 8,
) -> list[list]:
    """Depth-2 chooser using hand-crafted opp archetypes (not v3.5.1
    drop-ones) as the opp candidate set, with either min-regret or
    maximin row aggregation.

    Why this exists: the prior `choose_depth2` derives `opp_C` from
    v3.5.1's incumbent via drop-one. Both v7_1 and v7_2 failed in
    scalar A/B against v7_0_drop_one — strong evidence the v3.5.1
    opp assumption is biased (the live ladder is heterogeneous). The
    archetype set (`lib.missions.opp_archetypes`) gives 5 distinct
    opp threat patterns: no-launch / v3.5.1 / counter-reinforce /
    counter-snipe / cross-attack. Min-regret aggregation picks our
    action with the smallest worst-case gap from its best response
    over any of those archetypes — robust under opp uncertainty.

    `use_min_regret=True` (default) uses min-regret aggregation:
      regret[i] = max_j (max_k P[k][j] - P[i][j])
      return argmin_i regret[i]
    `use_min_regret=False` falls back to maximin:
      return argmax_i min_j P[i][j]

    4P fallback: depth-2 game-theory is 2P only; 4P returns incumbent.
    """
    wallclock_ms = _effective_wallclock_ms(wallclock_ms)
    t_start = time.perf_counter()

    world = World.from_obs(obs)
    if not world.planets_by_id:
        return []
    if my_id is None:
        my_id = world.my_id
    model = WorldModel.from_world(world)

    incumbent_intents = _build_incumbent_intents(
        world, model, include_recapture=include_recapture,
    )
    incumbent_action = _action_from_intents(incumbent_intents, obs, model)

    if _infer_num_seats(world) != 2:
        return incumbent_action

    our_C = _enumerate_drop_one(incumbent_action)
    if max_our_candidates and len(our_C) > max_our_candidates:
        our_C = our_C[:max_our_candidates]
    if len(our_C) <= 1:
        return incumbent_action

    snap = fs_from_obs(obs, configuration, episode_seed=0, num_seats=2)
    opp_id = 1 - my_id

    # Opp plays their natural incumbent on turn 1. Same as choose_depth2.
    opp_initial_action = _opp_incumbent_action(world, obs, opp_id)

    K_tail = max(0, K - 2)
    N = len(our_C)
    P: list[list[float]] = [[] for _ in range(N)]

    # Lazy import — keeps the existing lib graph clean.
    from lib.missions.opp_archetypes import build_opp_archetypes, opp_pov_obs

    for i in range(N):
        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        if i > 0 and elapsed_ms > 0.5 * wallclock_ms:
            P[i] = []
            continue

        # Forced turn 1.
        try:
            snap_i = fs_clone(snap)
            if not snap_i.done:
                actions: list[Any] = [None, None]
                actions[my_id] = our_C[i]
                actions[opp_id] = opp_initial_action
                snap_i = fs_step(snap_i, actions, in_place=True)
        except Exception:
            P[i] = []
            continue

        if snap_i.done:
            try:
                terminal = (
                    delta_us_minus_them(snap_i, my_id)
                    if value_fn is None
                    else value_fn(snap_i.state[my_id].observation, my_id)
                )
            except Exception:
                terminal = float("-inf")
            P[i] = [terminal]
            continue

        # Build opp archetypes from the POST-turn-1 state. Counter-
        # reinforce uses the OPP's intents that would best counter our
        # turn-1 launches, so we pass the incumbent intents (we already
        # committed to a subset of these on this candidate row).
        opp_obs_after = opp_pov_obs(snap_i.state[opp_id].observation, opp_id)
        try:
            opp_archetypes = build_opp_archetypes(
                opp_obs_after, our_intents=incumbent_intents,
            )
        except Exception:
            opp_archetypes = [[]]
        if not opp_archetypes:
            opp_archetypes = [[]]

        row_scores: list[float] = []
        for j, opp_act in enumerate(opp_archetypes):
            elapsed_ms = (time.perf_counter() - t_start) * 1000.0
            if elapsed_ms > wallclock_ms:
                break
            try:
                payoff = _score_after_opp_response(
                    snap_i, opp_act,
                    my_id=my_id, opp_id=opp_id, K_tail=K_tail,
                    value_fn=value_fn,
                )
            except Exception:
                payoff = float("-inf")
            row_scores.append(payoff)

        P[i] = row_scores

    # Aggregate the payoff matrix.
    NEG_INF = float("-inf")

    if not use_min_regret:
        # Maximin: argmax_i min_j P[i][j].
        best_i = 0
        best_worst = NEG_INF
        for i in range(N):
            if not P[i]:
                continue
            worst = min(P[i])
            if worst > best_worst:
                best_worst = worst
                best_i = i
        return our_C[best_i] if best_worst > NEG_INF else incumbent_action

    # Min-regret: column-wise best response (only over fully-scored rows
    # so a budget-bailed row doesn't poison the column best), then pick
    # the row whose worst regret is smallest.
    M = max((len(row) for row in P), default=0)
    if M == 0:
        return incumbent_action
    col_best: list[float] = []
    for j in range(M):
        vals = [P[i][j] for i in range(N) if j < len(P[i])]
        col_best.append(max(vals) if vals else NEG_INF)

    best_i = 0
    best_regret = float("inf")
    for i in range(N):
        if not P[i] or len(P[i]) < M:
            # Skip rows that didn't complete every column — would give
            # pessimistic regret. The incumbent (i=0) is fully evaluated
            # by the budget-first guarantee above, so the safe default
            # of `best_i = 0` survives.
            continue
        regret_i = max(col_best[j] - P[i][j] for j in range(M))
        # Tie-break: lower index wins (=row 0 incumbent).
        if regret_i < best_regret:
            best_regret = regret_i
            best_i = i

    return our_C[best_i] if best_regret < float("inf") else incumbent_action


def choose_archetype_minregret_with_4p(
    obs: Any,
    configuration: Any = None,
    *,
    K: int = 6,
    K_4p: int = 8,
    wallclock_ms: float = 700.0,
    use_min_regret: bool = True,
    include_recapture: bool = True,
    value_fn: Callable | None = None,
) -> list[list]:
    """v7.3 entry that auto-routes 2P → `choose_archetype_minregret`,
    4P → `choose_4p` (no maximin / regret in 4P)."""
    world = World.from_obs(obs)
    if not world.planets_by_id:
        return []
    n_seats = _infer_num_seats(world)
    if n_seats == 2:
        return choose_archetype_minregret(
            obs, configuration,
            K=K, wallclock_ms=wallclock_ms,
            use_min_regret=use_min_regret,
            include_recapture=include_recapture,
            value_fn=value_fn,
        )
    if n_seats == 4:
        return choose_4p(
            obs, configuration,
            K=K_4p, wallclock_ms=wallclock_ms,
            include_recapture=include_recapture,
            value_fn=value_fn,
        )
    model = WorldModel.from_world(world)
    intents = _build_incumbent_intents(
        world, model, include_recapture=include_recapture,
    )
    return _action_from_intents(intents, obs, model)


def choose_depth2_with_4p(
    obs: Any,
    configuration: Any = None,
    *,
    K_2p: int = 6,
    K_4p: int = 8,
    wallclock_ms: float = 700.0,
    include_recapture: bool = True,
    value_fn: Callable | None = None,
) -> list[list]:
    """v7 depth-2 entry that auto-routes 2P → choose_depth2, 4P → choose_4p.

    Use this as the agent's `agent(obs, configuration)` entry point when
    bundling a depth-2 variant. 4P games fall through to the drop-one
    chooser (no maximin guarantee at n > 2).
    """
    world = World.from_obs(obs)
    if not world.planets_by_id:
        return []
    n_seats = _infer_num_seats(world)
    if n_seats == 2:
        return choose_depth2(
            obs, configuration,
            K=K_2p, wallclock_ms=wallclock_ms,
            include_recapture=include_recapture,
            value_fn=value_fn,
        )
    if n_seats == 4:
        return choose_4p(
            obs, configuration,
            K=K_4p, wallclock_ms=wallclock_ms,
            include_recapture=include_recapture,
            value_fn=value_fn,
        )
    # 3P or 1P: rare; fall back to incumbent.
    model = WorldModel.from_world(world)
    intents = _build_incumbent_intents(
        world, model, include_recapture=include_recapture,
    )
    return _action_from_intents(intents, obs, model)


def choose_simple_with_4p(
    obs: Any,
    configuration: Any = None,
    *,
    K_2p: int = 10,
    K_4p: int = 8,
    wallclock_ms: float = 700.0,
    include_recapture: bool = True,
    value_fn: Callable | None = None,
) -> list[list]:
    """v7.5 entry — auto-routes 2P→choose_simple_2p, 4P→choose_4p.

    No maximin overlay (which regressed at v7.1 A/B). σ-equiv layer
    is library-level (lib/planner + lib/geometry + lib/missions/snipe)
    so it's automatically present. Recapture + value_fn pluggable.
    """
    world = World.from_obs(obs)
    if not world.planets_by_id:
        return []
    n_seats = _infer_num_seats(world)
    if n_seats == 2:
        return choose_simple_2p(
            obs, configuration,
            K=K_2p, wallclock_ms=wallclock_ms,
            include_recapture=include_recapture,
            value_fn=value_fn,
        )
    if n_seats == 4:
        return choose_4p(
            obs, configuration,
            K=K_4p, wallclock_ms=wallclock_ms,
            include_recapture=include_recapture,
            value_fn=value_fn,
        )
    # 3P or 1P: rare; fall back to incumbent.
    model = WorldModel.from_world(world)
    intents = _build_incumbent_intents(world, model, include_recapture=include_recapture)
    return _action_from_intents(intents, obs, model)


def choose_with_4p(
    obs: Any,
    configuration: Any = None,
    *,
    K_2p: int = 10,
    K_4p: int = 8,
    wallclock_ms: float = 700.0,
    use_symmetric: bool = True,
    include_recapture: bool = True,
    value_fn: Callable | None = None,
) -> list[list]:
    """Auto-routes 2P → choose_maximin, 4P → choose_4p.

    Combines the v7.1 maximin overlay (with σ-equiv + symmetric
    scoring) for 2P and the v7.4 4-seat drop-one rollout for 4P. v7.5
    final agent uses this entry point.
    """
    world = World.from_obs(obs)
    if not world.planets_by_id:
        return []
    n_seats = _infer_num_seats(world)
    if n_seats == 2:
        return choose_maximin(
            obs, configuration,
            K=K_2p, wallclock_ms=wallclock_ms,
            use_symmetric=use_symmetric,
            include_recapture=include_recapture,
            value_fn=value_fn,
        )
    if n_seats == 4:
        return choose_4p(
            obs, configuration,
            K=K_4p, wallclock_ms=wallclock_ms,
            include_recapture=include_recapture,
            value_fn=value_fn,
        )
    # 3P or 1P: rare; fall back to incumbent (safest).
    model = WorldModel.from_world(world)
    intents = _build_incumbent_intents(world, model, include_recapture=include_recapture)
    return _action_from_intents(intents, obs, model)


def choose(
    obs: Any,
    configuration: Any = None,
    *,
    enumerator_mode: str,
    K: int = 10,
    wallclock_ms: float = 700.0,
    my_id: int | None = None,
    opp_tiers: list[int] | None = None,
    value_fn: Callable | None = None,
    followup_policy: Callable | None = None,
) -> list[list]:
    """End-to-end: build incumbent, enumerate, score with watchdog,
    return argmax. Always returns the incumbent if no candidate
    scores strictly higher (parity floor).

    `my_id` defaults to `obs.player`. `configuration` is forwarded to
    `fs_from_obs`; if `None`, defaults are used (live ladder
    scrubs the episode seed anyway).

    `opp_tiers` is the opponent-policy pool used to score each
    candidate. Defaults to `[1]` (Tier-1 v3.5.1 mirror, v7_0 default).
    With multiple tiers, the chooser uses MAXIMIN — pick the candidate
    whose MIN-over-tiers score is highest. Robust to opp-policy
    uncertainty across the live ladder.

    `value_fn(observation, my_id) -> float` is the leaf-state scoring
    head. Defaults to `delta_us_minus_them` (Phase-2-validated baseline).
    Phase 3c uses a composite (ship_delta + denial + survivor) blend.
    """
    wallclock_ms = _effective_wallclock_ms(wallclock_ms)
    t_start = time.perf_counter()
    if opp_tiers is None:
        opp_tiers = [1]

    world = World.from_obs(obs)
    if not world.planets_by_id:
        return []
    if my_id is None:
        my_id = world.my_id
    model = WorldModel.from_world(world)

    incumbent_intents = _build_incumbent_intents(world, model)
    incumbent_action = _action_from_intents(incumbent_intents, obs, model)

    # 2P-only guard: the rollout's opp_model assumes a single opponent
    # seat. In 4P games we'd need to model 3 opponents simultaneously,
    # and the 2P Snapshot of a 4P obs systematically prefers "do
    # nothing" (the other 2 opponents are invisible to the simulator).
    # Fall back to the v3.5.1 incumbent — parity floor preserved.
    if _infer_num_seats(world) != 2:
        return incumbent_action

    # Snapshot for rollout. Episode seed unknown on the live ladder; this
    # is the same caveat documented in lib/lookahead.py and lib/fast_sim.
    snap = fs_from_obs(obs, configuration, episode_seed=0, num_seats=2)

    candidates = enumerate_candidates(
        world, model,
        enumerator_mode=enumerator_mode,
        incumbent_intents=incumbent_intents,
        incumbent_action=incumbent_action,
        obs=obs,
    )
    if len(candidates) <= 1:
        return incumbent_action

    best_action = incumbent_action  # incumbent is always candidates[0]
    best_score = float("-inf")
    incumbent_scored = False
    for cand in candidates:
        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        if elapsed_ms > wallclock_ms:
            break
        # Maximin: score this candidate against every opp tier in the
        # pool, take the WORST (min) score. Picking the candidate that
        # maximises this worst-case is the maximin / game-theoretic
        # robust choice against opp-policy uncertainty. For a single
        # tier (the v7_0 default) min-of-one is just the score itself.
        per_tier = []
        for tier in opp_tiers:
            s = score_candidate(
                snap, cand, my_id=my_id, K=K,
                opp_tier=tier, value_fn=value_fn,
                followup_policy=followup_policy,
            )
            per_tier.append(s)
        score = min(per_tier)
        if not incumbent_scored:
            # The first candidate is the incumbent — pin its score so
            # ties prefer the parity floor.
            incumbent_scored = True
            best_score = score
            best_action = list(cand)
            continue
        if score > best_score:
            best_score = score
            best_action = list(cand)
    return best_action
