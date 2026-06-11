"""Train the oracle value model on the ladder-replay dataset.

Input:  data/external/oracle_ds.npz  (from scripts/oracle_dataset.py)
Output: agents/oracle/value_weights.py  (numpy weights, base64-embedded)

Split is by EPISODE (no rows from one game cross folds). Reports val AUC /
logloss vs two baselines: (a) always-0.5, (b) single-feature logistic on the
current score share — the net must clearly beat (b) to be worth shipping.

Usage:
  python scripts/oracle_train.py [--ds path] [--epochs 40] [--hidden 256]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def auc(y, p):
    order = np.argsort(p)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(p))
    pos = y > 0.5
    n_pos = pos.sum()
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return (ranks[pos].sum() - n_pos * (n_pos - 1) / 2) / (n_pos * n_neg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", default=str(REPO / "data" / "external" / "oracle_ds.npz"))
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--out", default=str(REPO / "agents" / "oracle" / "value_weights.py"))
    args = ap.parse_args()

    import torch
    import torch.nn as nn
    torch.manual_seed(0)
    torch.set_num_threads(max(1, (torch.get_num_threads() or 4)))

    d = np.load(args.ds, allow_pickle=True)
    X, yw, ys, meta = d["X"], d["y_win"], d["y_share"], d["meta"]
    # drop exact ties/draws from the win head? keep, label 0.5 is fine for BCE
    ep = meta[:, 0]
    fold = (ep % 10)
    tr = fold <= 7
    va = fold == 8
    te = fold == 9
    print(f"rows: train {tr.sum()}, val {va.sum()}, test {te.sum()}, "
          f"features {X.shape[1]}")

    mu = X[tr].mean(0)
    sigma = X[tr].std(0)
    sigma[sigma < 1e-3] = 1e-3
    Z = (X - mu) / sigma

    # baseline (b): logistic on current score share
    from agents.oracle.features import FEATURE_NAMES
    i_share = FEATURE_NAMES.index("share_score_t0")
    s = X[:, i_share]
    # closed-form-ish 1D logistic via grid on scale
    best_b = None
    for k in (2, 4, 6, 8, 12, 20):
        p = 1 / (1 + np.exp(-k * (s - 0.5)))
        ll = -np.mean(yw[va] * np.log(p[va] + 1e-9)
                      + (1 - yw[va]) * np.log(1 - p[va] + 1e-9))
        if best_b is None or ll < best_b[0]:
            best_b = (ll, k, auc(yw[va], p[va]))
    print(f"baseline share-only: val logloss {best_b[0]:.4f}, "
          f"AUC {best_b[2]:.4f} (scale {best_b[1]})")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    layers = []
    d_in = X.shape[1]
    for _ in range(args.layers):
        layers += [nn.Linear(d_in, args.hidden), nn.ReLU()]
        d_in = args.hidden

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.body = nn.Sequential(*layers)
            self.win = nn.Linear(d_in, 1)
            self.share = nn.Linear(d_in, 1)

        def forward(self, z):
            h = self.body(z)
            return self.win(h).squeeze(-1), self.share(h).squeeze(-1)

    net = Net().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-5)
    bce = nn.BCEWithLogitsLoss()
    mse = nn.MSELoss()

    Zt = torch.tensor(Z[tr], dtype=torch.float32)
    yw_t = torch.tensor(yw[tr], dtype=torch.float32)
    ys_t = torch.tensor(ys[tr], dtype=torch.float32)
    Zv = torch.tensor(Z[va], dtype=torch.float32, device=dev)
    n_tr = Zt.shape[0]

    best = (1e9, None)
    patience, bad = 6, 0
    for epoch in range(args.epochs):
        net.train()
        perm = torch.randperm(n_tr)
        tot = 0.0
        for k in range(0, n_tr, args.batch):
            idx = perm[k:k + args.batch]
            z = Zt[idx].to(dev)
            lw, lsh = net(z)
            loss = bce(lw, yw_t[idx].to(dev)) \
                + mse(lsh, ys_t[idx].to(dev))
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss) * len(idx)
        net.eval()
        with torch.no_grad():
            lw, lsh = net(Zv)
            pv = torch.sigmoid(lw).cpu().numpy()
            ll = -np.mean(yw[va] * np.log(pv + 1e-9)
                          + (1 - yw[va]) * np.log(1 - pv + 1e-9))
            a = auc(yw[va], pv)
        print(f"epoch {epoch}: train {tot/n_tr:.4f}, val logloss {ll:.4f}, "
              f"val AUC {a:.4f}")
        if ll < best[0] - 1e-4:
            best = (ll, {k: v.detach().cpu().clone()
                         for k, v in net.state_dict().items()})
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(best[1])
    net.eval()

    # test fold report
    Zte = torch.tensor(Z[te], dtype=torch.float32, device=dev)
    with torch.no_grad():
        lw, lsh = net(Zte)
        pt = torch.sigmoid(lw).cpu().numpy()
    ll_te = -np.mean(yw[te] * np.log(pt + 1e-9)
                     + (1 - yw[te]) * np.log(1 - pt + 1e-9))
    print(f"TEST: logloss {ll_te:.4f}, AUC {auc(yw[te], pt):.4f} "
          f"(n={te.sum()})")
    # per-phase AUC
    step = meta[:, 2]
    for lo, hi in ((0, 50), (50, 150), (150, 300), (300, 500)):
        m = te & (step >= lo) & (step < hi)
        if m.sum() > 100:
            with torch.no_grad():
                lws, _ = net(torch.tensor(Z[m], dtype=torch.float32,
                                          device=dev))
                pm = torch.sigmoid(lws).cpu().numpy()
            print(f"  steps {lo}-{hi}: AUC {auc(yw[m], pm):.4f} (n={m.sum()})")

    sd = net.state_dict()
    hidden = []
    for k in range(args.layers):
        W = sd[f"body.{2*k}.weight"].numpy().T   # torch is (out,in) -> (in,out)
        b = sd[f"body.{2*k}.bias"].numpy()
        hidden.append((W, b))
    arrays = {
        "mu": mu.astype(np.float32),
        "sigma": sigma.astype(np.float32),
        "layers": hidden,
        "head_win": (sd["win.weight"].numpy().reshape(-1),
                     sd["win.bias"].numpy()),
        "head_share": (sd["share.weight"].numpy().reshape(-1),
                       sd["share.bias"].numpy()),
    }
    from agents.oracle.value import encode_weights_py
    src = encode_weights_py(arrays)
    with open(args.out, "w") as f:
        f.write(src)
    print(f"wrote {args.out} ({len(src)//1024} KB)")


if __name__ == "__main__":
    main()
