"""Train the oracle policy net (fire + size heads) on the replay dataset.

Negatives were subsampled at 30% during dataset build; training weights
restore the true prior so P(fire) stays calibrated. Reports the metrics the
planner cares about:
  - per-state top-k hit rate (states with >=1 expert launch: is one of the
    expert's pairs ranked in our top-1/top-3?)
  - calibration: expected vs actual fires at several thresholds, with a
    recommended FIRE_THETA that reproduces the expert per-state fire rate.

Usage:
  python scripts/oracle_policy_train.py [--ds path] [--epochs 30]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

NEG_KEEP = 0.3   # must match the builder's subsampling rate


def pr_auc(y, p):
    order = np.argsort(-p)
    y_s = y[order]
    tp = np.cumsum(y_s)
    fp = np.cumsum(1 - y_s)
    prec = tp / np.maximum(tp + fp, 1)
    rec = tp / max(y_s.sum(), 1)
    return float(np.trapezoid(prec, rec))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", default=str(
        REPO / "data" / "external" / "oracle_policy_ds.npz"))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--out", default=str(
        REPO / "agents" / "oracle" / "policy_weights.py"))
    args = ap.parse_args()

    import torch
    import torch.nn as nn
    torch.manual_seed(0)

    d = np.load(args.ds)
    X, y, frac, meta = d["X"], d["y"], d["frac"], d["meta"]
    ep = meta[:, 0]
    state = meta[:, 0] * 100000 + meta[:, 3]      # unique state key
    fold = ep % 10
    tr, va, te = fold <= 7, fold == 8, fold == 9
    print(f"rows {X.shape}, positives {100*y.mean():.2f}%, "
          f"train/val/test {tr.sum()}/{va.sum()}/{te.sum()}")

    mu = X[tr].mean(0)
    sigma = X[tr].std(0)
    sigma[sigma < 1e-3] = 1e-3
    Z = ((X - mu) / sigma).astype(np.float32)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    blocks = []
    d_in = X.shape[1]
    for _ in range(args.layers):
        blocks += [nn.Linear(d_in, args.hidden), nn.ReLU()]
        d_in = args.hidden

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.body = nn.Sequential(*blocks)
            self.fire = nn.Linear(d_in, 1)
            self.frac = nn.Linear(d_in, 1)

        def forward(self, z):
            h = self.body(z)
            return self.fire(h).squeeze(-1), self.frac(h).squeeze(-1)

    net = Net().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-5)
    bce = nn.BCEWithLogitsLoss(reduction="none")

    Zt = torch.tensor(Z[tr])
    yt = torch.tensor(y[tr])
    ft = torch.tensor(frac[tr])
    wt = torch.where(yt > 0.5, torch.tensor(1.0),
                     torch.tensor(1.0 / NEG_KEEP))
    Zv = torch.tensor(Z[va], device=dev)
    n = Zt.shape[0]

    best = (1e9, None)
    bad = 0
    for epoch in range(args.epochs):
        net.train()
        perm = torch.randperm(n)
        tot = 0.0
        for k in range(0, n, args.batch):
            idx = perm[k:k + args.batch]
            z = Zt[idx].to(dev)
            yy = yt[idx].to(dev)
            ww = wt[idx].to(dev)
            ff = ft[idx].to(dev)
            lf, lr_ = net(z)
            loss = (bce(lf, yy) * ww).mean()
            mask = ff >= 0
            if mask.any():
                loss = loss + 2.0 * ((lr_[mask] - ff[mask]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * len(idx)
        net.eval()
        with torch.no_grad():
            lf, _ = net(Zv)
            pv = torch.sigmoid(lf).cpu().numpy()
        wv = np.where(y[va] > 0.5, 1.0, 1.0 / NEG_KEEP)
        ll = -np.average(y[va] * np.log(pv + 1e-9)
                         + (1 - y[va]) * np.log(1 - pv + 1e-9), weights=wv)
        print(f"epoch {epoch}: train {tot/n:.4f}, val wlogloss {ll:.4f}, "
              f"val PR-AUC {pr_auc(y[va], pv):.4f}")
        if ll < best[0] - 1e-4:
            best = (ll, {k_: v.detach().cpu().clone()
                         for k_, v in net.state_dict().items()})
            bad = 0
        else:
            bad += 1
            if bad >= 5:
                break
    net.load_state_dict(best[1])
    net.eval()

    # ---- planner-relevant metrics on the test fold -------------------
    with torch.no_grad():
        lf, lr_ = net(torch.tensor(Z[te], device=dev))
        pt = torch.sigmoid(lf).cpu().numpy()
        fr_pred = lr_.cpu().numpy()
    yt_ = y[te]
    st_ = state[te]
    print(f"TEST PR-AUC {pr_auc(yt_, pt):.4f} "
          f"(base rate {yt_.mean():.4f})")
    order = np.argsort(st_, kind="stable")
    st_o, y_o, p_o = st_[order], yt_[order], pt[order]
    bounds = np.flatnonzero(np.diff(st_o)) + 1
    groups = np.split(np.arange(len(st_o)), bounds)
    hit1 = hit3 = n_pos_states = 0
    for g in groups:
        if y_o[g].sum() < 1:
            continue
        n_pos_states += 1
        rank = np.argsort(-p_o[g])
        hit1 += int(y_o[g][rank[0]] > 0.5)
        hit3 += int(y_o[g][rank[:3]].max() > 0.5)
    print(f"states with expert launch: {n_pos_states}; "
          f"expert pair in our top-1: {hit1/n_pos_states:.3f}, "
          f"top-3: {hit3/n_pos_states:.3f}")
    # threshold calibration vs true fire rate (subsampling-corrected)
    w_te = np.where(yt_ > 0.5, 1.0, 1.0 / NEG_KEEP)
    true_rate = np.average(yt_, weights=w_te)
    print(f"true fire rate {true_rate:.4f}; fires predicted at thresholds:")
    for th in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6):
        pred_rate = np.average(pt > th, weights=w_te)
        print(f"  theta {th:.1f}: {pred_rate:.4f}")
    # size head quality on positives
    mpos = yt_ > 0.5
    if mpos.any():
        err = np.abs(np.clip(fr_pred[mpos], 0, 1) - frac[te][mpos])
        print(f"size-frac MAE on positives: {err.mean():.3f} "
              f"(label mean {frac[te][mpos].mean():.3f})")

    sd = net.state_dict()
    hidden = []
    for k in range(args.layers):
        hidden.append((sd[f"body.{2*k}.weight"].numpy().T,
                       sd[f"body.{2*k}.bias"].numpy()))
    arrays = {
        "mu": mu.astype(np.float32),
        "sigma": sigma.astype(np.float32),
        "layers": hidden,
        "head_fire": (sd["fire.weight"].numpy().reshape(-1),
                      sd["fire.bias"].numpy()),
        "head_frac": (sd["frac.weight"].numpy().reshape(-1),
                      sd["frac.bias"].numpy()),
    }
    from agents.oracle.value import encode_weights_py
    src = encode_weights_py(arrays, head_names=("FIRE", "FRAC"))
    with open(args.out, "w") as f:
        f.write(src)
    print(f"wrote {args.out} ({len(src)//1024} KB)")


if __name__ == "__main__":
    main()
