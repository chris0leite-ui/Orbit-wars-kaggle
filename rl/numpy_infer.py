"""Pure-numpy eval-time inference: obs dict -> launch actions.

Standalone by design: only numpy + math, no lib/ or jax imports, so the
submission bundler can concatenate this file directly. Mirrors
rl/features.py + rl/net.py + rl/aim.py exactly (parity-tested by
tests/test_rl_numpy_parity.py).
"""
from __future__ import annotations

import math

import numpy as np

BOARD_SIZE = 100.0
CENTER = 50.0
SUN_RADIUS = 10.0
ROTATION_RADIUS_LIMIT = 50.0
COMET_SPAWN_STEPS = (50, 150, 250, 350, 450)

MAX_PLANETS = 80
MAX_AGENTS = 4
MAX_LAUNCH = 20
AIM_ITERS = 4
ETA_BUCKETS = np.array([3.0, 7.0, 12.0, 18.0, 26.0, 48.0])
N_BUCKETS = 6
T_HORIZON = 48
N_FRACS = 4
FRACS = np.array([0.25, 0.5, 0.75, 1.0])

D_MODEL = 64
N_LAYERS = 3
N_HEADS = 4
D_HEAD = D_MODEL // N_HEADS
N_TOKENS = MAX_PLANETS + 1


# ---------------------------------------------------------------- obs
def obs_to_arrays(obs) -> dict:
    """Pack a kaggle obs (dict or Struct) into padded numpy arrays
    mirroring the JAX GameState fields used by features."""
    def get(o, k, default=None):
        try:
            v = o[k]
        except (KeyError, TypeError, IndexError):
            v = getattr(o, k, default)
        return v if v is not None else default

    planets = [list(p) for p in get(obs, "planets", [])]
    fleets = [list(f) for f in get(obs, "fleets", [])]
    initial = [list(p) for p in get(obs, "initial_planets", [])]
    comets = get(obs, "comets", []) or []
    comet_ids = set(int(c) for c in (get(obs, "comet_planet_ids", []) or []))
    step = int(get(obs, "step", 0) or 0)
    omega = float(get(obs, "angular_velocity", 0.0) or 0.0)

    P = MAX_PLANETS
    a = {
        "x": np.zeros(P), "y": np.zeros(P),
        "pid": -np.ones(P, np.int64), "owner": -np.ones(P, np.int64),
        "ships": np.zeros(P), "prod": np.zeros(P),
        "radius": np.zeros(P), "alive": np.zeros(P, bool),
        "ix": np.zeros(P), "iy": np.zeros(P),
        "is_comet": np.zeros(P, bool),
        "comet_remain": np.full(P, 1e6),
        # per-planet future path for comets: (P, T_HORIZON+61, 2)
        "step": step, "omega": omega,
    }
    pid_to_idx = {}
    for i, p in enumerate(planets[:P]):
        pid = int(p[0])
        pid_to_idx[pid] = i
        a["pid"][i] = pid
        a["owner"][i] = int(p[1])
        a["x"][i] = float(p[2]); a["y"][i] = float(p[3])
        a["radius"][i] = float(p[4])
        a["ships"][i] = float(p[5]); a["prod"][i] = float(p[6])
        a["alive"][i] = True
        a["is_comet"][i] = pid in comet_ids
        a["ix"][i] = float(p[2]); a["iy"][i] = float(p[3])
    for ip in initial:
        idx = pid_to_idx.get(int(ip[0]))
        if idx is not None:
            a["ix"][idx] = float(ip[2]); a["iy"][idx] = float(ip[3])

    # Comet future positions: dense per-comet lookup table, relative
    # offsets 0..L. comet_pos[i, t] = position of comet i at t ticks
    # ahead (clamped to path end); remain[i] = steps until expiry.
    TT = T_HORIZON + 61
    comet_pos = np.zeros((P, TT, 2))
    for g in comets:
        gids = list(g["planet_ids"]) if hasattr(g, "keys") else list(g.planet_ids)
        gpaths = g["paths"] if hasattr(g, "keys") else g.paths
        gidx = int(g["path_index"]) if hasattr(g, "keys") else int(g.path_index)
        for j, pid in enumerate(gids):
            slot = pid_to_idx.get(int(pid))
            if slot is None:
                continue
            path = gpaths[j]
            L = len(path)
            a["comet_remain"][slot] = max(L - 1 - gidx, 0)
            for t in range(TT):
                k = min(gidx + t, L - 1)
                k = max(k, 0)
                comet_pos[slot, t, 0] = path[k][0]
                comet_pos[slot, t, 1] = path[k][1]
    a["comet_pos"] = comet_pos

    F = 256
    a["f_x"] = np.zeros(F); a["f_y"] = np.zeros(F)
    a["f_angle"] = np.zeros(F); a["f_owner"] = -np.ones(F, np.int64)
    a["f_ships"] = np.zeros(F); a["f_alive"] = np.zeros(F, bool)
    for i, f in enumerate(fleets[:F]):
        a["f_owner"][i] = int(f[1])
        a["f_x"][i] = float(f[2]); a["f_y"][i] = float(f[3])
        a["f_angle"][i] = float(f[4]); a["f_ships"][i] = float(f[6])
        a["f_alive"][i] = True
    return a


# ------------------------------------------------------------- physics
def fleet_speed(ships, max_speed=6.0):
    s = np.maximum(np.asarray(ships, dtype=np.float64), 1.0)
    v = 1.0 + (max_speed - 1.0) * (np.log(s) / math.log(1000.0)) ** 1.5
    return np.minimum(v, max_speed)


def planet_pos_at(a, t_rel):
    """t_rel: scalar or (P,) or (S,P) — future positions (..., P, 2)."""
    t = np.asarray(t_rel, dtype=np.float64)
    dx = a["ix"] - CENTER
    dy = a["iy"] - CENTER
    r = np.sqrt(dx * dx + dy * dy)
    theta0 = np.arctan2(dy, dx)
    abs_step = a["step"] + t - 1.0
    theta = theta0 + a["omega"] * abs_step
    is_rot = (r + a["radius"] < ROTATION_RADIUS_LIMIT) & ~a["is_comet"]
    rx = CENTER + r * np.cos(theta)
    ry = CENTER + r * np.sin(theta)
    ti = np.clip(t.astype(np.int64), 0, a["comet_pos"].shape[1] - 1)
    ti_b = np.broadcast_to(ti, theta.shape).astype(np.int64)
    pidx = np.broadcast_to(np.arange(MAX_PLANETS), theta.shape)
    cx = a["comet_pos"][pidx, ti_b, 0]
    cy = a["comet_pos"][pidx, ti_b, 1]
    px = np.where(a["is_comet"], cx, np.where(is_rot, rx, a["x"]))
    py = np.where(a["is_comet"], cy, np.where(is_rot, ry, a["y"]))
    return np.stack([px, py], axis=-1)


def solve_intercept(a, ships_grid):
    """(P,P) lead-aim. Returns dict angle/eta/sun_hit/valid."""
    src = np.stack([a["x"], a["y"]], axis=-1)
    speed = fleet_speed(ships_grid)
    tgt_now = planet_pos_at(a, 0.0)
    eta = np.linalg.norm(tgt_now[None, :, :] - src[:, None, :], axis=-1) / speed
    for _ in range(AIM_ITERS):
        tgt_fut = planet_pos_at(a, eta)  # (P,P,2) — eta (P,P) broadcasts
        eta = np.linalg.norm(tgt_fut - src[:, None, :], axis=-1) / speed
    tgt_fut = planet_pos_at(a, eta)
    delta = tgt_fut - src[:, None, :]
    angle = np.arctan2(delta[..., 1], delta[..., 0])

    sx = src[:, None, 0]; sy = src[:, None, 1]
    tx = tgt_fut[..., 0]; ty = tgt_fut[..., 1]
    vx = tx - sx; vy = ty - sy
    l2 = vx * vx + vy * vy
    tpar = np.clip(((CENTER - sx) * vx + (CENTER - sy) * vy)
                   / np.maximum(l2, 1e-9), 0.0, 1.0)
    cx = sx + tpar * vx - CENTER
    cy = sy + tpar * vy - CENTER
    sun_hit = np.sqrt(cx * cx + cy * cy) < (SUN_RADIUS + 0.5)

    on_board = ((tx >= 1.0) & (tx <= BOARD_SIZE - 1.0)
                & (ty >= 1.0) & (ty <= BOARD_SIZE - 1.0))
    comet_ok = eta <= a["comet_remain"][None, :] - 1.0
    valid = on_board & comet_ok & a["alive"][None, :] & a["alive"][:, None]
    return {"angle": angle, "eta": eta, "sun_hit": sun_hit, "valid": valid}


def solve_intercept_rows(a, tgt_idx, ships):
    src = np.stack([a["x"], a["y"]], axis=-1)
    speed = fleet_speed(ships)
    safe_t = np.clip(tgt_idx, 0, MAX_PLANETS - 1)

    def pos_at(idx, t):
        full = planet_pos_at(a, _scatter_t(idx, t))
        return full[np.arange(len(idx)), idx]

    tgt_now = planet_pos_at(a, 0.0)[safe_t]
    eta = np.linalg.norm(tgt_now - src, axis=-1) / speed
    for _ in range(AIM_ITERS):
        tgt_fut = pos_at(safe_t, eta)
        eta = np.linalg.norm(tgt_fut - src, axis=-1) / speed
    tgt_fut = pos_at(safe_t, eta)
    delta = tgt_fut - src
    return np.arctan2(delta[:, 1], delta[:, 0]), eta


def _scatter_t(idx, t):
    out = np.zeros(MAX_PLANETS)
    out[idx] = t
    return out


def fleet_arrivals(a):
    f_pos = np.stack([a["f_x"], a["f_y"]], axis=-1)
    speed = fleet_speed(a["f_ships"])
    vel = speed[:, None] * np.stack(
        [np.cos(a["f_angle"]), np.sin(a["f_angle"])], axis=-1)
    F = f_pos.shape[0]
    hit_t = -np.ones(F, np.int64)
    hit_p = np.zeros(F, np.int64)
    for t in range(1, T_HORIZON + 1):
        fp = f_pos + vel * t
        pp = planet_pos_at(a, float(t))
        d = np.linalg.norm(fp[:, None, :] - pp[None, :, :], axis=-1)
        thresh = a["radius"][None, :] + 0.5 * speed[:, None]
        hits = (d < thresh) & a["alive"][None, :]
        any_hit = hits.any(axis=1)
        first_p = np.argmax(hits, axis=1)
        new = any_hit & (hit_t < 0)
        hit_t[new] = t
        hit_p[new] = first_p[new]

    live = a["f_alive"] & (hit_t > 0) & (a["f_owner"] >= 0)
    bucket = np.clip(np.searchsorted(ETA_BUCKETS, hit_t.astype(np.float64)),
                     0, N_BUCKETS - 1)
    arrive = np.zeros((MAX_PLANETS, MAX_AGENTS, N_BUCKETS))
    for i in np.where(live)[0]:
        arrive[hit_p[i], a["f_owner"][i], bucket[i]] += a["f_ships"][i]
    cum = np.cumsum(arrive, axis=-1)
    return arrive, cum


# ------------------------------------------------------------ features
def seat_features(a, seat, num_agents):
    P = MAX_PLANETS
    alive = a["alive"]
    owner = a["owner"]
    mine = (owner == seat) & alive
    neutral = (owner == -1) & alive
    enemy = (~mine) & (~neutral) & (owner >= 0) & alive

    arrive, cum = fleet_arrivals(a)
    half = np.maximum(a["ships"] // 2, 1)
    ships_grid = np.broadcast_to(half[:, None], (P, P))
    aim = solve_intercept(a, ships_grid)

    ships = a["ships"]
    log_ships = np.log1p(np.maximum(ships, 0.0)) / 5.0
    prod = a["prod"] / 5.0
    radius = a["radius"] / 3.0
    dx = a["x"] - CENTER
    dy = a["y"] - CENTER
    r_orb = np.sqrt(dx * dx + dy * dy)
    theta = np.arctan2(dy, dx)
    idx0 = np.sqrt((a["ix"] - CENTER) ** 2 + (a["iy"] - CENTER) ** 2)
    is_rot = (idx0 + a["radius"] < ROTATION_RADIUS_LIMIT) & ~a["is_comet"] & alive
    remain = np.minimum(a["comet_remain"], 999.0) / 40.0
    remain = np.where(a["is_comet"], remain, 0.0)

    seat_ids = np.arange(MAX_AGENTS)
    enemy_seats = (seat_ids != seat) & (seat_ids < num_agents)
    inc_mine = arrive[:, seat, :]
    inc_enemy = (arrive * enemy_seats[None, :, None]).sum(axis=1)

    nodes = np.concatenate([
        mine[:, None], enemy[:, None], neutral[:, None],
        log_ships[:, None], prod[:, None], radius[:, None],
        (dx / 50.0)[:, None], (dy / 50.0)[:, None], (r_orb / 50.0)[:, None],
        np.cos(theta)[:, None], np.sin(theta)[:, None],
        is_rot[:, None], a["is_comet"][:, None], remain[:, None],
        np.log1p(inc_mine) / 5.0, np.log1p(inc_enemy) / 5.0,
    ], axis=-1).astype(np.float64)
    nodes = np.where(alive[:, None], nodes, 0.0)

    eta = aim["eta"]
    eta_n = np.clip(eta, 0.0, 60.0) / 40.0
    sun = aim["sun_hit"].astype(np.float64)
    valid = aim["valid"].astype(np.float64)
    bucket_e = np.clip(np.searchsorted(ETA_BUCKETS, eta), 0, N_BUCKETS - 1)
    cum_mine_t = cum[:, seat, :]
    cum_enemy_t = (cum * enemy_seats[None, :, None]).sum(axis=1)
    tgt_idx = np.broadcast_to(np.arange(P)[None, :], (P, P))
    my_before = cum_mine_t[tgt_idx, bucket_e]
    en_before = cum_enemy_t[tgt_idx, bucket_e]
    owned_t = (owner >= 0) & alive
    garr_proj = ships[None, :] + np.where(owned_t, a["prod"], 0)[None, :] * eta
    net_def = garr_proj + np.where(mine[None, :], my_before, en_before) \
        - np.where(mine[None, :], en_before, my_before)
    net_def_n = np.sign(net_def) * np.log1p(np.abs(net_def)) / 5.0
    dist = eta * fleet_speed(np.maximum(a["ships"] // 2, 1))[:, None]
    edges = np.stack([
        eta_n, sun, valid, net_def_n,
        np.clip(dist, 0.0, 150.0) / 100.0,
        np.eye(P),
    ], axis=-1)

    my_ships_tot = ships[mine].sum()
    en_ships_tot = ships[enemy].sum()
    my_prod_tot = a["prod"][mine].sum()
    en_prod_tot = a["prod"][enemy].sum()
    fl_mine = a["f_ships"][(a["f_owner"] == seat) & a["f_alive"]].sum()
    fl_enemy = a["f_ships"][(a["f_owner"] != seat) & (a["f_owner"] >= 0)
                            & a["f_alive"]].sum()
    my_mat = my_ships_tot + fl_mine
    en_mat = en_ships_tot + fl_enemy
    tot = my_mat + en_mat + 1e-6
    step_f = float(a["step"])
    future = [s for s in COMET_SPAWN_STEPS if s > step_f]
    next_spawn = (future[0] - step_f) if future else 999.0
    globals_ = np.array([
        step_f / 500.0,
        1.0 if num_agents == 2 else 0.0,
        1.0 if num_agents == 4 else 0.0,
        my_mat / tot, en_mat / tot,
        (my_prod_tot - en_prod_tot) / (my_prod_tot + en_prod_tot + 1e-6),
        mine.sum() / 10.0, enemy.sum() / 10.0, neutral.sum() / 10.0,
        min(next_spawn, 200.0) / 100.0,
        a["omega"] * 20.0,
        math.log1p(my_mat) / 8.0,
    ])

    src_mask = mine & (ships >= 1)
    pair_ok = (aim["valid"] & ~aim["sun_hit"] & alive[None, :]
               & ~np.eye(P, dtype=bool)) & src_mask[:, None]
    tgt_mask = np.concatenate([pair_ok, np.ones((P, 1), bool)], axis=1)
    return nodes, edges, globals_, src_mask, tgt_mask


# ----------------------------------------------------------------- net
def _ln(x, g, b):
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + 1e-5) * g + b


def _gelu(x):
    return 0.5 * x * (1.0 + np.tanh(
        math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))


def _softmax(x, axis=-1):
    m = x.max(axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis=axis, keepdims=True)


def forward(params, nodes, edges, globals_, alive_mask, tgt_mask):
    p = params
    x_p = nodes @ p["embed_w"] + p["embed_b"]
    x_g = (globals_ @ p["gembed_w"] + p["gembed_b"])[None, :]
    x = np.concatenate([x_p, x_g], axis=0)
    tok_mask = np.concatenate([alive_mask, np.ones(1, bool)])
    for i in range(N_LAYERS):
        h = _ln(x, p[f"l{i}_ln1_g"], p[f"l{i}_ln1_b"])
        qkv = h @ p[f"l{i}_qkv_w"] + p[f"l{i}_qkv_b"]
        q, k, v = np.split(qkv, 3, axis=-1)
        q = q.reshape(N_TOKENS, N_HEADS, D_HEAD).transpose(1, 0, 2)
        k = k.reshape(N_TOKENS, N_HEADS, D_HEAD).transpose(1, 0, 2)
        v = v.reshape(N_TOKENS, N_HEADS, D_HEAD).transpose(1, 0, 2)
        att = q @ k.transpose(0, 2, 1) / math.sqrt(D_HEAD)
        att = np.where(tok_mask[None, None, :], att, -1e9)
        att = _softmax(att, axis=-1)
        out = (att @ v).transpose(1, 0, 2).reshape(N_TOKENS, D_MODEL)
        x = x + out @ p[f"l{i}_proj_w"] + p[f"l{i}_proj_b"]
        h = _ln(x, p[f"l{i}_ln2_g"], p[f"l{i}_ln2_b"])
        h = _gelu(h @ p[f"l{i}_ff1_w"] + p[f"l{i}_ff1_b"])
        x = x + h @ p[f"l{i}_ff2_w"] + p[f"l{i}_ff2_b"]
    x = _ln(x, p["lnf_g"], p["lnf_b"])
    x = np.where(tok_mask[:, None], x, 0.0)

    pe = x[:MAX_PLANETS]
    q = pe @ p["tq_w"] + p["tq_b"]
    k = pe @ p["tk_w"] + p["tk_b"]
    ptr = q @ k.T / math.sqrt(D_MODEL)
    e = _gelu(edges @ p["edge1_w"] + p["edge1_b"])
    ebias = (e @ p["edge2_w"] + p["edge2_b"])[..., 0]
    hold = pe @ p["hold_w"] + p["hold_b"]
    logits = np.concatenate([ptr + ebias, hold], axis=1)
    logits = np.where(tgt_mask, logits, -1e9)
    g = x[-1]
    v = _gelu(g @ p["v1_w"] + p["v1_b"])
    value = float((v @ p["v2_w"] + p["v2_b"])[0])
    return value, logits, x


def frac_logits_for(params, emb, edges, tgt_choice):
    p = params
    pe = emb[:MAX_PLANETS]
    safe_t = np.clip(tgt_choice, 0, MAX_PLANETS - 1)
    tgt_emb = pe[safe_t]
    e = edges[np.arange(MAX_PLANETS), safe_t]
    h = np.concatenate([pe, tgt_emb, e], axis=-1)
    h = _gelu(h @ p["f1_w"] + p["f1_b"])
    return h @ p["f2_w"] + p["f2_b"]


# --------------------------------------------------------------- agent
class RLAgent:
    """Stateful per-episode agent. Call .act(obs) -> action list."""

    def __init__(self, params):
        self.params = {k: np.asarray(v, dtype=np.float64)
                       for k, v in params.items()}
        self.num_agents = None
        self.last_step = None

    def act(self, obs):
        a = obs_to_arrays(obs)
        try:
            me = int(obs["player"])
        except (KeyError, TypeError, IndexError):
            me = int(getattr(obs, "player", 0))
        # New-episode detection for in-process multi-game harnesses
        # (step counter resets); Kaggle eval is fresh-process anyway.
        if self.last_step is not None and a["step"] <= self.last_step:
            self.num_agents = None
        self.last_step = a["step"]
        if self.num_agents is None:
            owners = set(int(o) for o in a["owner"][a["alive"]] if o >= 0)
            self.num_agents = 4 if len(owners) > 2 or me > 1 else 2

        nodes, edges, globals_, src_mask, tgt_mask = seat_features(
            a, me, self.num_agents)
        _, logits, emb = forward(self.params, nodes, edges, globals_,
                                 a["alive"], tgt_mask)
        tgt_choice = np.argmax(logits, axis=-1)
        fl = frac_logits_for(self.params, emb, edges, tgt_choice)
        frac_choice = np.argmax(fl, axis=-1)

        is_launch = (tgt_choice < MAX_PLANETS) & src_mask
        garrison = a["ships"]
        frac_val = FRACS[frac_choice]
        ships = np.floor(frac_val * garrison)
        ships = np.clip(ships, 1.0, np.maximum(garrison, 1.0))
        ships = np.where(is_launch & (garrison >= 1.0), ships, 0.0)
        is_launch = is_launch & (ships > 0)
        if not is_launch.any():
            return []

        angle, _ = solve_intercept_rows(
            a, tgt_choice.astype(np.int64), ships.astype(np.int64))
        order = np.argsort(-np.where(is_launch, ships, -1))
        actions = []
        for s in order[:MAX_LAUNCH]:
            if not is_launch[s]:
                break
            actions.append([int(a["pid"][s]), float(angle[s]), int(ships[s])])
        return actions
