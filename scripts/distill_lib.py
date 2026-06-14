"""Distilled fast rollout policy: shared featurization + the policy class.

The teacher (top_tier_mirror) launches sparsely/selectively (~0.44/turn). We
learn a per-(source, target) launch score so the student reproduces WHEN and
WHERE to launch (the main edge over lite_greedy's spray). Inference is a
standardized linear/logit dot-product -> fast.

Candidate = (owned source planet, non-mine target among its nearest-K).
Label (training) = 1 if the teacher launched from src toward that target.
"""
from __future__ import annotations
import math
import numpy as np

K_TARGETS = 10
SPEED_EST = 3.5
FEAT_NAMES = [
    "src_ships", "src_prod", "tgt_ships", "tgt_prod",
    "is_enemy", "is_neutral", "dist", "eta", "roi",
    "ship_margin", "capturable", "step_frac", "is_4p",
    "my_share", "enemy_near",
]
NF = len(FEAT_NAMES)


def _angle(sx, sy, tx, ty):
    return math.atan2(ty - sy, tx - sx)


def featurize(obs, num_seats: int):
    """Return (cand_meta, X) for one observation.
    cand_meta: list of (src_id, tgt_id, angle, src_ships, tgt_ships).
    X: (n_cand, NF) float32 raw features.
    """
    me = int(obs["player"]) if "player" in obs else 0
    planets = obs["planets"]
    step = int(obs.get("step", 0))
    is_4p = 1.0 if num_seats >= 3 else 0.0

    mine, others = [], []
    my_ships = 0.0
    tot_ships = 0.0
    for p in planets:
        owner = int(p[1])
        sh = float(p[5])
        tot_ships += sh if owner >= 0 else 0.0
        if owner == me:
            mine.append(p); my_ships += sh
        else:
            others.append(p)
    my_share = my_ships / tot_ships if tot_ships > 0 else 0.0

    cand_meta = []
    rows = []
    for s in mine:
        s_sh, s_pr, sx, sy = float(s[5]), float(s[6]), float(s[2]), float(s[3])
        if s_sh < 2:
            continue
        # nearest-K non-mine targets
        ds = sorted(others, key=lambda t: (sx - t[2])**2 + (sy - t[3])**2)[:K_TARGETS]
        enemy_near = sum(1.0 for t in ds if int(t[1]) >= 0)
        for t in ds:
            t_owner = int(t[1]); t_sh, t_pr = float(t[5]), float(t[6])
            tx, ty = float(t[2]), float(t[3])
            dist = math.hypot(tx - sx, ty - sy)
            eta = dist / SPEED_EST
            roi = t_pr / max(eta, 1.0)
            rows.append([
                s_sh, s_pr, t_sh, t_pr,
                1.0 if t_owner >= 0 else 0.0,
                1.0 if t_owner < 0 else 0.0,
                dist, eta, roi,
                s_sh - t_sh, 1.0 if s_sh > t_sh else 0.0,
                step / 500.0, is_4p, my_share, enemy_near,
            ])
            cand_meta.append((int(s[0]), int(t[0]), _angle(sx, sy, tx, ty), s_sh, t_sh))
    X = np.asarray(rows, dtype=np.float32) if rows else np.zeros((0, NF), np.float32)
    return cand_meta, X


def infer_target_id(src, angle, planets, me):
    """Which non-mine planet did a teacher launch (src, angle) aim at?
    Nearest angular match among non-mine planets."""
    sx, sy = float(src[2]), float(src[3])
    best, bestd = None, 1e9
    for p in planets:
        if int(p[1]) == me:
            continue
        a = math.atan2(float(p[3]) - sy, float(p[2]) - sx)
        d = abs((a - angle + math.pi) % (2 * math.pi) - math.pi)
        if d < bestd:
            bestd, best = d, int(p[0])
    return best if bestd < 0.20 else None  # ~11deg tolerance


class DistillPolicy:
    """Fast linear launch policy. weights: dict(w, b, mean, std, threshold)."""
    def __init__(self, weights: dict, num_seats: int = 2):
        self.w = np.asarray(weights["w"], np.float32)
        self.b = float(weights["b"])
        self.mean = np.asarray(weights["mean"], np.float32)
        self.std = np.asarray(weights["std"], np.float32)
        self.threshold = float(weights.get("threshold", 0.5))
        self.num_seats = num_seats

    def score(self, X):
        if X.shape[0] == 0:
            return np.zeros((0,), np.float32)
        Z = (X - self.mean) / self.std
        return 1.0 / (1.0 + np.exp(-(Z @ self.w + self.b)))

    def __call__(self, obs):
        cand_meta, X = featurize(obs, self.num_seats)
        if X.shape[0] == 0:
            return []
        p = self.score(X)
        action, used_src = [], set()
        # one launch per source per turn (the strongest candidate above thresh)
        order = np.argsort(-p)
        for i in order:
            if p[i] < self.threshold:
                break
            src_id, tgt_id, angle, s_sh, t_sh = cand_meta[i]
            if src_id in used_src:
                continue
            ships = int(min(s_sh, max(t_sh + 2.0, 0.7 * s_sh)))
            if ships < 1:
                continue
            action.append([src_id, float(angle), ships])
            used_src.add(src_id)
        return action
