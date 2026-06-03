"""Forward-rollout chooser — pick the turn whose SIMULATED end-state wins.

The closed-form chooser (`chooser.select`) proposes a value-ordered set of
launches, but it can't tell a capture that STICKS from one that gets retaken in
five turns (churn), nor see that spreading thin loses the end-state — those are
multi-turn, emergent. So we SIMULATE.

Candidate turns = PREFIXES of the committed launch set (idle .. full spread).
For each, we apply it on turn 0 and roll the game forward K ticks with every
seat playing a cheap competent policy (`lite_greedy_policy`, ROI-greedy — close
to the roi opponent we're losing to, so we optimise against the right thing),
then read the estimated turn-500 differential. The prefix that simulates best is
emitted. Concentration + anti-churn EMERGE: a thin-spread plan simulates to a
worse end-state than a short concentrated prefix, so we don't pick it.

This differs from the saturated champion's rollout in two ways that matter: it
scores whole TURNS (not individual launches in isolation), and both seats play
competently (no passive-self baseline that undervalues territory).
"""

from __future__ import annotations

import os
import time

from lib.fast_sim import clone, delta_us_minus_them, from_obs, rollout as fs_rollout, step
from lib.intent import realize
from lib.mechanism import DEFAULT_MECHANISMS
from lib.opp_model import lite_greedy_policy

from agents.holdgrab.chooser import select


def _budget_ms(cfg) -> float:
    override = os.environ.get("ORBIT_WARS_PARITY_WALLCLOCK_MS")
    if override:
        try:
            return float(override)
        except ValueError:
            pass
    return cfg.rollout_budget_ms


def _leaf_score(snap, my_id: int, game_horizon: int) -> float:
    """Estimated turn-500 differential at the rollout horizon: current ship lead
    + production lead x turns remaining (production we control keeps making ships
    for the rest of the game). No passive baseline."""
    ships = float(delta_us_minus_them(snap, my_id))
    obs0 = snap.state[0].observation
    step_now = int(getattr(obs0, "step", 0)) if not isinstance(obs0, dict) else int(obs0.get("step", 0))
    planets = getattr(obs0, "planets", None) if not isinstance(obs0, dict) else obs0.get("planets")
    my_prod = 0.0
    opp_prod = 0.0
    for p in (planets or []):
        owner = int(p[1])
        if owner == my_id:
            my_prod += float(p[6])
        elif owner >= 0:
            opp_prod += float(p[6])
    remaining = max(0, game_horizon - step_now)
    return ships + (my_prod - opp_prod) * float(remaining)


def _evaluate(base, my_actions, my_id: int, num_seats: int, K: int, game_horizon: int) -> float:
    """Apply `my_actions` on my seat at turn 0 (opponents play lite_greedy), then
    roll K-1 ticks with every seat playing lite_greedy; return the leaf score."""
    snap = clone(base)
    acts = []
    for s in range(num_seats):
        if s == my_id:
            acts.append(my_actions)
        else:
            acts.append(lite_greedy_policy(snap.state[s].observation))
    snap = step(snap, acts, in_place=True)
    if K > 1 and not snap.done:
        snap = fs_rollout(snap, K - 1, [lite_greedy_policy] * num_seats, in_place=True)
    return _leaf_score(snap, my_id, game_horizon)


def choose(view, cfg, obs, configuration):
    """Return env actions for the best-simulating prefix of the committed set."""
    intents = select(view, cfg)
    model = view.model
    full_actions = realize(intents, obs, mechanisms=DEFAULT_MECHANISMS, model=model)
    if not intents:
        return full_actions

    my_id = int(view.me)
    num_seats = int(view.num_seats)
    base = from_obs(obs, configuration, episode_seed=0, num_seats=num_seats)

    n = len(intents)
    ks = sorted({k for k in cfg.rollout_prefix_ks if 0 <= k <= n} | {n})
    deadline = time.perf_counter() + _budget_ms(cfg) / 1000.0

    best_actions = full_actions
    best_score = None
    for k in ks:
        if best_score is not None and time.perf_counter() > deadline:
            break
        if k == n:
            actions = full_actions
        elif k == 0:
            actions = []
        else:
            actions = realize(intents[:k], obs, mechanisms=DEFAULT_MECHANISMS, model=model)
        score = _evaluate(base, actions, my_id, num_seats, cfg.rollout_K, cfg.game_horizon)
        if best_score is None or score > best_score:   # strict: ties -> smaller k (concentration)
            best_score = score
            best_actions = actions

    return best_actions
