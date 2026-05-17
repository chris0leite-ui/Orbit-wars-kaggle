"""Fast_sim-based scoring + greedy non-dogpile selection.

Replaces v8_analytic's JAX K-rollout value head (`analytic_score.py`'s
`score_candidates_vmap_value_prod`). Root cause diagnosed via
`/tmp/micro_trace.py` (2026-05-17): the JAX leaf evaluator was blind
to captures with ETA > K — 38 of 40 atoms scored exactly equal to
no-op because in-flight fleets count as `my_ships`, so launches with
ETA > K=8 produced bit-identical leaf states. Empty action set won
argmax by default; agent launched on 19% of turns vs nearest's 49%.

This module replaces the leaf with a Python `fast_sim` K-step rollout
that:
1. Applies my candidate action at turn 0 (paired with opp's lite_greedy
   reactive action — opp recomputes its policy against the EVOLVING
   state each step, not once).
2. Continues for K-1 turns with me idle and opp continuing to react.
3. Evaluates leaf via `_favor` (F1 ship delta + F2 PV-discounted prod
   delta) — matches v8_scavenge's value head exactly.

Greedy non-dogpile merge (v8_scavenge `agents/v8_scavenge/main.py:723-
739` pattern) picks one launch per source, one per target, sorted by
score descending. Drops beam search entirely — beam over fast_sim
busts the 1000 ms CPU budget at any reasonable width × depth.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

from lib.fast_sim import (
    Snapshot, clone as fs_clone, from_obs as fs_from_obs, step as fs_step,
)
from lib.foundation.actions import ActionSpec
from lib.game.jax.jax_types import MAX_LAUNCH_PER_AGENT
from lib.opp_model import lite_greedy_policy
from lib.scoring import pv_horizon

# v8_scavenge defaults — kept aligned so the leaf evaluator behaves
# identically when our enumerator hands the same candidate to the same
# leaf as v8_scavenge would.
_EPISODE_STEPS = 500
_FAVOR_GAMMA = 0.99


def _favor(snap: Snapshot, my_id: int, num_seats: int = 2) -> float:
    """F1 + F2 favor — direct port of v8_scavenge `_favor`
    (/tmp/v8_scavenge.py:447-503).

    F1 = my_ships_total − max(opp_ships_totals)  (2P; sum in 4P)
    F2 = (my_prod − opp_prod) × pv_horizon(step, 0, γ=0.99)

    Reads from `snap.state[0].observation` — the kaggle obs dict that
    fast_sim already maintains per step.
    """
    obs = snap.state[0].observation
    planets = obs.planets if hasattr(obs, "planets") else obs.get("planets", [])
    fleets = obs.fleets if hasattr(obs, "fleets") else obs.get("fleets", [])
    step = obs.step if hasattr(obs, "step") else obs.get("step", 0)

    ships_by_owner: dict[int, float] = {}
    prod_by_owner: dict[int, float] = {}
    for p in planets:
        owner = int(p[1])
        if owner < 0:
            continue
        ships_by_owner[owner] = ships_by_owner.get(owner, 0.0) + float(p[5])
        prod_by_owner[owner] = prod_by_owner.get(owner, 0.0) + float(p[6])
    for f in fleets:
        owner = int(f[1])
        if owner < 0:
            continue
        ships_by_owner[owner] = ships_by_owner.get(owner, 0.0) + float(f[6])

    my_ships = ships_by_owner.get(my_id, 0.0)
    my_prod = prod_by_owner.get(my_id, 0.0)
    if num_seats <= 2:
        opp_ships = max(
            (v for k, v in ships_by_owner.items() if k != my_id),
            default=0.0,
        )
        opp_prod = max(
            (v for k, v in prod_by_owner.items() if k != my_id),
            default=0.0,
        )
    else:
        opp_ships = sum(v for k, v in ships_by_owner.items() if k != my_id)
        opp_prod = sum(v for k, v in prod_by_owner.items() if k != my_id)

    pv = pv_horizon(int(step), 0, gamma=_FAVOR_GAMMA, t_total=_EPISODE_STEPS)
    return (my_ships - opp_ships) + (my_prod - opp_prod) * pv


def _action_for_seat(launches: Sequence[Sequence[float]]) -> list:
    """Convert iterable of (src_id, angle, ships) into the env's action
    format `[[src_id, angle, ships], ...]`."""
    return [[int(s), float(a), int(n)] for (s, a, n) in launches]


def _empty_actions(num_seats: int) -> list:
    return [[] for _ in range(num_seats)]


def _rollout_with_opp_idle_me(
    snap_base: Snapshot, my_id: int, opp_id: int, K: int,
) -> float:
    """K-step rollout with me idle, opp playing reactive lite_greedy.
    Returns leaf favor. Used as baseline so candidate scores are
    DELTA-favor relative to "do nothing while opp keeps acting".
    """
    snap = fs_clone(snap_base)
    for _ in range(K):
        opp_action = lite_greedy_policy(snap.state[opp_id].observation)
        actions = _empty_actions(2)
        actions[opp_id] = opp_action
        fs_step(snap, actions, in_place=True)
    return _favor(snap, my_id)


def score_candidates_fastsim(
    snap_base: Snapshot,
    atoms_with_targets: Sequence[tuple[ActionSpec, int]],
    my_id: int,
    K: int,
) -> tuple[list[float], float]:
    """Score each candidate atom as a singleton action via fast_sim.

    Returns (`scores`, `baseline_favor`). Scores are delta-favor:
    `score[i] = _favor(after_K_steps_with_atom_i_at_turn_0) − baseline`.
    Positive = launch is better than doing nothing; <=0 = no signal or
    actively bad.

    `baseline_favor` is the K-step rollout with me idle (opp keeps
    playing lite_greedy reactively at each step). Computed once.
    """
    opp_id = 1 - my_id
    baseline_favor = _rollout_with_opp_idle_me(snap_base, my_id, opp_id, K)

    scores: list[float] = []
    for atom, _tgt_id in atoms_with_targets:
        snap = fs_clone(snap_base)
        # Turn 0: my candidate launch + opp's reactive lite_greedy.
        opp_action = lite_greedy_policy(snap.state[opp_id].observation)
        actions = _empty_actions(2)
        actions[my_id] = [[int(atom.from_planet_id),
                           float(atom.dir_angle), int(atom.ships)]]
        actions[opp_id] = opp_action
        fs_step(snap, actions, in_place=True)
        # Turns 1..K-1: me idle, opp continues reacting.
        for _ in range(K - 1):
            opp_action = lite_greedy_policy(snap.state[opp_id].observation)
            actions = _empty_actions(2)
            actions[opp_id] = opp_action
            fs_step(snap, actions, in_place=True)
        scores.append(_favor(snap, my_id) - baseline_favor)
    return scores, baseline_favor


def _greedy_select_non_dogpile(
    atoms_with_targets: Sequence[tuple[ActionSpec, int]],
    scores: Sequence[float],
    pre_committed: Sequence[ActionSpec],
    max_launches: int,
) -> list[ActionSpec]:
    """Mirror v8_scavenge `/tmp/v8_scavenge.py:723-739` greedy non-
    dogpile emit: max 1 launch per source, max 1 per target, sorted by
    score descending, only positive-delta candidates included.

    `pre_committed` launches are output first; their sources are
    auto-reserved so subsequent atoms can't dogpile the same source.
    """
    used_sources: set[int] = {a.from_planet_id for a in pre_committed}
    # pre_committed targets aren't tracked here — we don't know their
    # target_id (ActionSpec doesn't carry it). Greedy may dogpile a
    # pre_committed target; in practice pre_committed waves come from
    # mission re-aim and re-aim correctly so a second launch is
    # usually beneficial, not harmful. Acceptable.
    used_targets: set[int] = set()

    chosen: list[ActionSpec] = list(pre_committed)

    # Filter to positive-delta candidates and sort by score desc.
    ranked = sorted(
        ((a, t, s) for (a, t), s in zip(atoms_with_targets, scores) if s > 0),
        key=lambda triple: -triple[2],
    )
    for atom, target_id, _score in ranked:
        if len(chosen) >= max_launches:
            break
        if atom.from_planet_id in used_sources:
            continue
        if target_id in used_targets:
            continue
        chosen.append(atom)
        used_sources.add(atom.from_planet_id)
        used_targets.add(target_id)
    return chosen


def score_and_select_via_fastsim(
    raw_obs,
    atoms_with_targets: Sequence[tuple[ActionSpec, int]],
    my_id: int,
    *,
    pre_committed: Sequence[ActionSpec] = (),
    K: int = 8,
    max_n: int = 40,
    max_launches: int = MAX_LAUNCH_PER_AGENT,
) -> list[ActionSpec]:
    """Top-level entry: enumerate-capped atoms in, action set out.

    Pipeline:
    1. Trim atoms to top `max_n` (atoms come pre-ranked by cheap-rank
       from `enumerate_capped`; just slice).
    2. Build a single base snapshot from `raw_obs` (~5 ms).
    3. Score each remaining atom as a singleton via fast_sim K-step
       rollout with reactive lite_greedy opp.
    4. Greedy non-dogpile merge keeping max 1 launch per source / per
       target, only positive-delta candidates.
    5. Prepend `pre_committed` waves (chainer-supplied; their sources
       are auto-excluded from greedy).
    """
    if not atoms_with_targets and not pre_committed:
        return list(pre_committed)

    # Step 1: cap.
    if max_n is not None and 0 < max_n < len(atoms_with_targets):
        atoms_with_targets = atoms_with_targets[:max_n]

    # Step 2: snap once.
    snap_base = fs_from_obs(raw_obs, num_seats=2)

    # Step 3: score.
    scores, _baseline = score_candidates_fastsim(
        snap_base, atoms_with_targets, my_id, K,
    )

    # Step 4/5: greedy merge + pre_committed prefix.
    return _greedy_select_non_dogpile(
        atoms_with_targets, scores, pre_committed, max_launches,
    )
