"""Oracle agent — policy net inference (numpy).

Maps per-candidate policy features to (P(fire now), size fraction of the
source garrison), behavior-cloned from 1500+ rated ladder replays
(scripts/oracle_policy_train.py -> policy_weights.py).

Untrained fallback: a transparent prior so the agent stays playable —
fires affordable captures by a simple price rank, never transfers.
"""

import numpy as np

from .policy_features import POLICY_FEATURES, N_GLOBALS

_I = {n: k for k, n in enumerate(POLICY_FEATURES)}


class PolicyNet:
    def __init__(self):
        self.loaded = False
        PW = None
        try:
            from . import policy_weights as PW
        except Exception:
            # flattened bundle: the builder injects _POLICY_WEIGHTS
            PW = globals().get("_POLICY_WEIGHTS")
        try:
            self.mu = PW.MU
            self.sigma = PW.SIGMA
            self.layers = PW.LAYERS
            self.head_fire = PW.HEAD_FIRE
            self.head_frac = PW.HEAD_FRAC
            self.loaded = True
        except Exception:
            pass
        # state-level initiation head (trained on the global feature slice)
        self.state_loaded = False
        try:
            self.s_mu = PW.S_MU
            self.s_sigma = PW.S_SIGMA
            self.s_layers = PW.S_LAYERS
            self.s_head = PW.HEAD_STATE
            self.state_loaded = True
        except Exception:
            pass

    def state_fire_p(self, any_pair_feats):
        """P(an expert launches anything this turn) from one pair's
        state-global slice (identical across the state's pairs)."""
        if not self.state_loaded:
            return 1.0          # without the head, never block on state
        g = np.asarray(any_pair_feats[-N_GLOBALS:], dtype=np.float32)
        z = (g - self.s_mu) / self.s_sigma
        for W, b in self.s_layers:
            z = np.maximum(z @ W + b, 0.0)
        logit = float(z @ self.s_head[0] + self.s_head[1])
        return 1.0 / (1.0 + np.exp(-logit))

    def batch(self, feats_list):
        """-> (p_fire array, size_frac array)."""
        if len(feats_list) == 0:
            z = np.zeros(0, dtype=np.float32)
            return z, z
        X = np.asarray(feats_list, dtype=np.float32)
        if not self.loaded:
            # fallback prior: capture when affordable and close
            req = X[:, _I["tgt_required"]]
            spare = np.maximum(X[:, _I["src_spare"]], 1.0)
            eta = np.maximum(X[:, _I["eta_full"]], 1.0)
            afford = (req > 0) & (req <= spare)
            tgt_own = X[:, _I["tgt_owner_me"]] > 0.5
            prio = (X[:, _I["tgt_prod"]] + 0.3) / (req + 2.0 * eta + 1.0)
            p = np.where(afford & ~tgt_own,
                         np.clip(0.4 + 3.0 * prio, 0.0, 0.95), 0.05)
            frac = np.clip((req + 3.0) / spare, 0.2, 1.0)
            return p.astype(np.float32), frac.astype(np.float32)
        Z = (X - self.mu) / self.sigma
        for W, b in self.layers:
            Z = Z @ W + b
            np.maximum(Z, 0.0, out=Z)
        fire_logit = Z @ self.head_fire[0] + self.head_fire[1]
        frac = Z @ self.head_frac[0] + self.head_frac[1]
        p = 1.0 / (1.0 + np.exp(-fire_logit))
        return p, np.clip(frac, 0.02, 1.0)
