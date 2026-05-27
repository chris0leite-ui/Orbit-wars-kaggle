"""Idle baseline + per-candidate Δ rollout, greedy emit.

Pipeline:
  baseline[h] = favor over HORIZON steps with (me=idle, opp=reactive)
  for each candidate:
    Δ = favor(me-fires-at-step-0, opp=reactive, HORIZON steps) - baseline[H]
  emit Δ>0 candidates sorted desc, 1 launch per source per turn.

The reactive opp model is lib.opp_model.lite_greedy_policy — a cheap
production/distance ROI scorer that approximates what spam-launchers
like `nearest` and `roi` actually do. Penalising captures that
predictably trigger counter-attacks keeps the chooser honest about
which moves actually hold.
"""

from __future__ import annotations

from lib.fast_sim import clone as fs_clone
from lib.fast_sim import step as fs_step
from lib.opp_model import lite_greedy_policy as opp_policy

from agents.minimal.value import favor

HORIZON = 40
N_VALIDATE = 48


def _opp_actions(snap, me: int, num_seats: int) -> list[list]:
    out: list[list] = [[] for _ in range(num_seats)]
    for i in range(num_seats):
        if i == me:
            continue
        try:
            out[i] = opp_policy(snap.state[i].observation) or []
        except Exception:
            out[i] = []
    return out


def _idle_baseline(snap_base, me: int, num_seats: int) -> list[float]:
    snap = fs_clone(snap_base)
    favors = [favor(snap.state[me].observation, me, num_seats)]
    for _ in range(HORIZON):
        if snap.fake_env.done:
            favors.append(favors[-1])
            continue
        snap = fs_step(snap, _opp_actions(snap, me, num_seats), in_place=True)
        favors.append(favor(snap.state[me].observation, me, num_seats))
    return favors


def _score_action(snap_base, me: int, num_seats: int,
                  src_id: int, angle: float, ships: int,
                  baseline: list[float]) -> float:
    snap = fs_clone(snap_base)
    for h in range(HORIZON):
        if snap.fake_env.done:
            break
        actions = _opp_actions(snap, me, num_seats)
        if h == 0:
            actions[me] = [[int(src_id), float(angle), int(ships)]]
        snap = fs_step(snap, actions, in_place=True)
    return favor(snap.state[me].observation, me, num_seats) - baseline[HORIZON]


def choose(snap_base, cands, me: int, num_seats: int) -> list[list]:
    if not cands:
        return []
    baseline = _idle_baseline(snap_base, me, num_seats)
    scored: list[tuple] = []
    for src, tgt, ships, angle in cands[:N_VALIDATE]:
        delta = _score_action(snap_base, me, num_seats,
                              int(src.id), angle, ships, baseline)
        if delta > 0:
            scored.append((delta, src, tgt, ships, angle))
    if not scored:
        return []
    scored.sort(key=lambda c: -c[0])
    used_s: set[int] = set()
    moves: list[list] = []
    for _d, src, _tgt, ships, angle in scored:
        sid = int(src.id)
        if sid in used_s:
            continue
        used_s.add(sid)
        moves.append([sid, float(angle), int(ships)])
    return moves
