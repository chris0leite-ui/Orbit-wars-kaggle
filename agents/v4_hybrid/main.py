"""v4_hybrid — practical synthesis: v3_snipe base + mirror defensive overlay.

The Tier 0-2 mirror iterations falsified the strict cannot-lose floor
empirically in 2P vs v3_snipe (0/4 W/L). Root causes:

  • 1-turn lag + tight fleet sizing → mirror always fails to capture
    σ(target) by ~1 production unit (Tier 1 fixed this).
  • CASCADE: once we lose any planet P, all future mirrors of opp
    launches from σ(P) silently drop because we no longer own σ(P).
    Tier 2's sun/oob veto doesn't address cascade.
  • Combat rule 1 sums same-owner arrivals, so opp can't self-kamikaze
    — Tier 3 from the plan doesn't apply in this game.

Pivot: use v3_snipe (current main agent) as the intelligent baseline,
and add mirror counter-launches as an additive overlay. The result has
v3's competence as the floor + mirror's symmetric counter-attacks when
opp commits, IF the garrison allows both.

Algorithm:
  1. Compute mirror actions via the Tier 2 logic (bijection + lag bump +
     sun-veto). These are added to the action list.
  2. Subtract mirror's garrison usage from a snapshot of obs.planets.
  3. Call v3_snipe with the modified obs so v3 plans against the
     remaining garrison.
  4. Concatenate (mirror_actions + v3_actions). Env validates per launch.

This is NOT cannot-lose in any strict sense, but it inherits v3's empirical
strength while extending it with structural reciprocity. The expected
win rate is bounded below by v3-vs-v3 ≈ 50% draws and above by v3's
empirical edge.

Plan reference: /root/.claude/plans/you-are-a-top-parallel-swan.md
"""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

from kaggle_environments.envs.orbit_wars.orbit_wars import Planet

from lib.intent import World
from lib.mirror import (
    build_bijection,
    detect_num_players,
    diagonal_opponent,
    rotate_angle,
)
from lib.trajectory import predict_fleet_fate


_STATE: dict[int, dict] = {}
_V3_AGENT = None
MAX_PRODUCTION = 5


def _obs_get(obs, key, default=None):
    if isinstance(obs, dict):
        return obs.get(key, default)
    return getattr(obs, key, default)


def _obs_set(obs, key, value):
    if isinstance(obs, dict):
        obs[key] = value
    else:
        setattr(obs, key, value)


def _load_v3():
    global _V3_AGENT
    if _V3_AGENT is not None:
        return _V3_AGENT
    path = Path(__file__).resolve().parents[2] / "agents" / "v3_snipe" / "main.py"
    spec = importlib.util.spec_from_file_location("_hybrid_v3", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _V3_AGENT = mod.agent
    return _V3_AGENT


def _reset_state(my_id: int, obs) -> dict:
    initial = _obs_get(obs, "initial_planets", []) or []
    planets = _obs_get(obs, "planets", []) or []
    n = detect_num_players(planets)
    st = {
        "num_players": n,
        "bijection": build_bijection(initial) if n == 2 else {},
        "opp_id": diagonal_opponent(my_id, n) if n in (2, 4) else None,
        "prev_fleet_ids": set(),
    }
    _STATE[my_id] = st
    return st


def _predict_opp_fate(fleet_raw, world: World):
    fx, fy = float(fleet_raw[2]), float(fleet_raw[3])
    angle = float(fleet_raw[4])
    ships = int(fleet_raw[6])
    synth_src = Planet(-1, -1, fx, fy, 0.0, ships, 0)
    any_planet = next(iter(world.planets_by_id.values()), None)
    if any_planet is None:
        return ("oob", None)
    fate = predict_fleet_fate(synth_src, any_planet, angle, ships, world)
    hit = world.planets_by_id.get(fate.hit_planet_id) if fate.hit_planet_id else None
    return (fate.outcome, hit)


def _compute_mirror_actions(obs, st):
    """Tier 2 mirror logic (lag-comp + sun-veto). Returns
    (actions, garrison_used_by_src_id)."""
    raw_planets = _obs_get(obs, "planets", []) or []
    raw_fleets = _obs_get(obs, "fleets", []) or []
    my_id = int(_obs_get(obs, "player", 0))

    new_opp = [
        f for f in raw_fleets
        if f[0] not in st["prev_fleet_ids"] and f[1] == st["opp_id"]
    ]
    st["prev_fleet_ids"] = {f[0] for f in raw_fleets}

    if not new_opp:
        return [], {}

    world = World.from_obs(obs)
    garrison: dict[int, int] = {p[0]: int(p[5]) for p in raw_planets if p[1] == my_id}
    used: dict[int, int] = {}

    actions: list[list] = []
    for f_raw in new_opp:
        opp_from = int(f_raw[5])
        our_src = st["bijection"].get(opp_from)
        if our_src is None:
            continue
        avail = garrison.get(our_src, 0)
        if avail <= 0:
            continue

        outcome, hit_planet = _predict_opp_fate(f_raw, world)
        if outcome in ("sun", "oob"):
            continue

        if hit_planet is not None:
            sigma_target_id = st["bijection"].get(hit_planet.id)
            sigma_target = world.planets_by_id.get(sigma_target_id) if sigma_target_id else None
            bump = int(sigma_target.production) if sigma_target is not None else MAX_PRODUCTION
        else:
            bump = MAX_PRODUCTION

        desired = int(f_raw[6]) + bump
        emit = min(desired, avail)
        if emit <= 0:
            continue
        actions.append([our_src, rotate_angle(float(f_raw[4])), emit])
        garrison[our_src] = avail - emit
        used[our_src] = used.get(our_src, 0) + emit

    return actions, used


def _obs_with_deducted_garrison(obs, used: dict[int, int]):
    """Shallow-clone obs with planets' ships reduced by `used[src_id]`,
    so v3_snipe plans against the remaining garrison.
    """
    if not used:
        return obs

    raw_planets = _obs_get(obs, "planets", []) or []
    new_planets = []
    for p in raw_planets:
        delta = used.get(p[0], 0)
        if delta > 0:
            # planet tuple is mutable list per env [id, owner, x, y, r, ships, prod]
            np_ = list(p)
            np_[5] = max(0, int(p[5]) - delta)
            new_planets.append(np_)
        else:
            new_planets.append(p)

    # Lightweight wrapper that proxies all other attributes.
    if isinstance(obs, dict):
        new_obs = dict(obs)
        new_obs["planets"] = new_planets
        return new_obs

    class _ObsWrap:
        pass

    wrap = _ObsWrap()
    for k in dir(obs):
        if k.startswith("_"):
            continue
        try:
            v = getattr(obs, k)
        except Exception:
            continue
        if callable(v):
            continue
        setattr(wrap, k, v)
    wrap.planets = new_planets
    return wrap


def agent(obs):
    my_id = int(_obs_get(obs, "player", 0))
    step = int(_obs_get(obs, "step", 0))
    st = _STATE.get(my_id)
    if step == 0 or st is None:
        st = _reset_state(my_id, obs)

    if st["num_players"] != 2:
        # 4P → just defer to v3_snipe; mirror has no guarantee.
        return _load_v3()(obs)

    # v3 first — it's our intelligent baseline (draws against v3 in
    # self-play). Mirror layer sees v3's garrison usage as already
    # committed and only launches from planets with residual capacity.
    v3_actions = _load_v3()(obs)

    v3_used: dict[int, int] = {}
    for a in v3_actions:
        if not a:
            continue
        src = int(a[0]); ships = int(a[2])
        v3_used[src] = v3_used.get(src, 0) + ships

    obs_after_v3 = _obs_with_deducted_garrison(obs, v3_used)
    mirror_actions, _ = _compute_mirror_actions(obs_after_v3, st)
    return list(v3_actions) + list(mirror_actions)
