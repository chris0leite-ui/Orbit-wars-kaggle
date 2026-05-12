"""v4_mirror (Tier 0 of the strategy ladder) — pure 180° mirror in 2P.

Strict cannot-lose construction: at turn t we replay each new opponent
launch from turn t-1 through the 180°-rotation bijection σ. By the
symmetric-game value theorem, this guarantees expected payoff ≥ 0 in
2P zero-sum games (modulo the engine's documented P0/P1 tie-break leak
and 1-turn lag artifact).

In 4P FFA the theorem doesn't apply — fall back to v3_snipe (current
main agent on origin/main).

State (bijection, prev fleet ids, opponent id, num players) is kept in
a per-player module dict so two MirrorBots can self-play inside one
process without cross-contamination. Reset on `obs.step == 0`.

See `/root/.claude/plans/you-are-a-top-parallel-swan.md` for the
tier ladder this is iteration 0 of (Tier 1-3 floor-preserving, Tier 4+
floor-trading).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

from lib.mirror import (
    build_bijection,
    detect_num_players,
    diagonal_opponent,
    rotate_angle,
)


# Per-player state. Keyed by obs.player so self-play doesn't cross-contaminate.
_STATE: dict[int, dict] = {}
_V3_AGENT = None  # lazy-loaded


def _obs_get(obs, key, default=None):
    if isinstance(obs, dict):
        return obs.get(key, default)
    return getattr(obs, key, default)


def _load_v3_fallback():
    """Lazy-load agents/v3_snipe/main.py without requiring it to be a package."""
    global _V3_AGENT
    if _V3_AGENT is not None:
        return _V3_AGENT
    repo = Path(__file__).resolve().parents[2]
    path = repo / "agents" / "v3_snipe" / "main.py"
    spec = importlib.util.spec_from_file_location("_mirror_v3_fallback", path)
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


def agent(obs):
    my_id = int(_obs_get(obs, "player", 0))
    step = int(_obs_get(obs, "step", 0))
    st = _STATE.get(my_id)
    if step == 0 or st is None:
        st = _reset_state(my_id, obs)

    # 4P → fall back to v3_snipe; the cannot-lose theorem is 2P-only.
    if st["num_players"] != 2:
        return _load_v3_fallback()(obs)

    fleets = _obs_get(obs, "fleets", []) or []
    planets = _obs_get(obs, "planets", []) or []
    bij = st["bijection"]
    opp_id = st["opp_id"]
    prev_ids = st["prev_fleet_ids"]

    # New opponent fleets that appeared since last call.
    new_opp = [
        f for f in fleets
        if f[0] not in prev_ids and f[1] == opp_id
    ]
    st["prev_fleet_ids"] = {f[0] for f in fleets}

    if not new_opp:
        return []

    # Current garrison budget per owned planet — depleted as we emit.
    garrison: dict[int, int] = {p[0]: int(p[5]) for p in planets if p[1] == my_id}

    actions: list[list] = []
    for f in new_opp:
        # Fleet layout: [id, owner, x, y, angle, from_planet_id, ships]
        opp_from = int(f[5])
        our_src = bij.get(opp_from)
        if our_src is None:
            continue
        avail = garrison.get(our_src, 0)
        if avail <= 0:
            continue
        emit_ships = min(int(f[6]), avail)
        if emit_ships <= 0:
            continue
        actions.append([our_src, rotate_angle(float(f[4])), emit_ships])
        garrison[our_src] = avail - emit_ships

    return actions
