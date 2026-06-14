"""Train the distilled launch scorer from teacher self-play data."""
import sys, pickle, json, time
import numpy as np
sys.path.insert(0, "scripts")
from distill_lib import featurize, infer_target_id, FEAT_NAMES, NF

def build_xy(samples, max_turns=24000):
    Xs, ys = [], []
    n_pos = n_turn = 0
    for (ns, obs, action) in samples[:max_turns]:
        cand_meta, X = featurize(obs, ns)
        if X.shape[0] == 0:
            continue
        n_turn += 1
        planets = obs["planets"]
        pos = set()
        for launch in action:
            src_id = int(launch[0]); angle = float(launch[1])
            src = next((p for p in planets if int(p[0]) == src_id), None)
            if src is None:
                continue
            tid = infer_target_id(src, angle, planets, me=0)
            if tid is not None:
                pos.add((src_id, tid))
        y = np.array([1.0 if (m[0], m[1]) in pos else 0.0 for m in cand_meta], np.float32)
        n_pos += int(y.sum())
        Xs.append(X); ys.append(y)
    X = np.concatenate(Xs); y = np.concatenate(ys)
    print(f"turns={n_turn} candidates={len(y)} positives={int(y.sum())} "
          f"({100*y.mean():.2f}%)  matched_launches={n_pos}")
    return X, y

def train_logreg(X, y, iters=300, lr=0.5, l2=1e-4):
    mean = X.mean(0); std = X.std(0) + 1e-6
    Z = (X - mean) / std
    n, d = Z.shape
    w = np.zeros(d, np.float64); b = 0.0
    pos_w = (len(y) - y.sum()) / max(y.sum(), 1.0)  # balance classes
    sw = np.where(y > 0, pos_w, 1.0)
    for it in range(iters):
        p = 1.0 / (1.0 + np.exp(-(Z @ w + b)))
        g = (p - y) * sw
        gw = Z.T @ g / n + l2 * w
        gb = g.mean()
        w -= lr * gw; b -= lr * gb
    return w, b, mean, std

def auc(y, p):
    pos = p[y > 0]; neg = p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    # rank-based AUC, subsample neg for speed
    if len(neg) > 200000:
        neg = np.random.choice(neg, 200000, replace=False)
    return float((pos[:, None] > neg[None, :]).mean())

if __name__ == "__main__":
    data = sys.argv[1] if len(sys.argv) > 1 else "/tmp/distill_data.pkl"
    out = sys.argv[2] if len(sys.argv) > 2 else "/tmp/distill_weights.json"
    samples = pickle.load(open(data, "rb"))
    np.random.seed(0)
    idx = np.random.permutation(len(samples))
    tr = [samples[i] for i in idx[: int(0.85 * len(idx))]]
    va = [samples[i] for i in idx[int(0.85 * len(idx)):]]
    t0 = time.perf_counter()
    Xtr, ytr = build_xy(tr)
    w, b, mean, std = train_logreg(Xtr, ytr)
    ptr = 1.0 / (1.0 + np.exp(-(((Xtr - mean) / std) @ w + b)))
    print(f"train AUC={auc(ytr, ptr):.3f}  ({time.perf_counter()-t0:.0f}s)")
    Xva, yva = build_xy(va, max_turns=6000)
    pva = 1.0 / (1.0 + np.exp(-(((Xva - mean) / std) @ w + b)))
    print(f"val   AUC={auc(yva, pva):.3f}")
    # pick threshold: match teacher ~0.44 launches/turn is implicit; aim for
    # precision/recall balance -> threshold where val launch rate ~ teacher.
    for thr in [0.3, 0.4, 0.5, 0.6, 0.7]:
        sel = pva > thr
        prec = yva[sel].mean() if sel.sum() else 0.0
        rec = (yva[sel].sum() / max(yva.sum(), 1))
        print(f"  thr={thr}: precision={prec:.3f} recall={rec:.3f} "
              f"sel/turn~{sel.sum()/6000:.2f}")
    weights = {"w": w.tolist(), "b": float(b), "mean": mean.tolist(),
               "std": std.tolist(), "threshold": 0.5, "feat": FEAT_NAMES}
    json.dump(weights, open(out, "w"))
    print(f"saved -> {out}")
