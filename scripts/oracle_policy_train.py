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
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

NEG_KEEP = 0.3   # must match the builder's subsampling rate


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
    ap.add_argument("--extra-ds", default="",
                    help="second dataset (self-play/live lessons) mixed in")
    ap.add_argument("--extra-weight", type=float, default=4.6,
                    help="per-row weight multiplier for the extra dataset")
    args = ap.parse_args()

    import torch
    import torch.nn as nn
    torch.manual_seed(0)

    d = np.load(args.ds)
    X, y, frac, meta = d["X"], d["y"], d["frac"], d["meta"]
    extra_mask = np.zeros(len(X), dtype=bool)
    row_w = np.ones(len(X), dtype=np.float32)
    if args.extra_ds:
        d2 = np.load(args.extra_ds)
        m2 = d2["meta"]
        if m2.shape[1] < meta.shape[1]:        # pad missing flag columns
            pad = np.zeros((len(m2), meta.shape[1] - m2.shape[1]), np.int64)
            m2 = np.concatenate([m2, pad], axis=1)
        n2 = len(d2["X"])
        X = np.concatenate([X, d2["X"]])
        y = np.concatenate([y, d2["y"]])
        frac = np.concatenate([frac, d2["frac"]])
        meta = np.concatenate([meta, m2])
        # per-row AWR advantage weights if present, else flat extra-weight
        rw2 = d2["row_w"] if "row_w" in d2.files else np.ones(n2, np.float32)
        row_w = np.concatenate([row_w, rw2.astype(np.float32)])
        extra_mask = np.zeros(len(X), dtype=bool)
        extra_mask[-n2:] = True
        kind = "AWR advantage-weighted" if "row_w" in d2.files else "flat"
        print(f"mixed in {n2} extra rows (x{args.extra_weight}, {kind})")
    if os.environ.get("ORACLE_ZERO_COMMIT"):
        # no-chain variant: train with the same-turn commitment features
        # neutralized (runtime must then pass committed=None and skip
        # re-scoring — ORACLE_NO_CHAIN=1 in the planner)
        from agents.oracle.policy_features import POLICY_FEATURES
        idx = [POLICY_FEATURES.index(n) for n in
               ("committed_from_src", "committed_to_tgt",
                "committed_total", "committed_n")]
        X = X.copy()
        X[:, idx] = 0.0
        print("commitment features ZEROED (no-chain variant)")
    max_rows = int(os.environ.get("ORACLE_TRAIN_MAX_ROWS", "0"))
    if max_rows and len(X) > max_rows:
        # deterministic thinning that keeps whole states together
        state_key = (meta[:, 0] * 100000 + meta[:, 3]).astype(np.uint64)
        mixed = (state_key * np.uint64(2654435761)) % np.uint64(1 << 32)
        keep = (mixed % np.uint64(1000)) < np.uint64(
            int(1000 * max_rows / len(X)))
        X, y, frac, meta, extra_mask, row_w = (
            X[keep], y[keep], frac[keep], meta[keep],
            extra_mask[keep], row_w[keep])
        print(f"thinned to {len(X)} rows (ORACLE_TRAIN_MAX_ROWS={max_rows})")
    ep = meta[:, 0]
    state = meta[:, 0] * 100000 + meta[:, 3]      # unique state key
    fold = ep % 10
    tr, va, te = fold <= 7, fold == 8, fold == 9
    if va.sum() == 0 or te.sum() == 0:
        print("WARNING: empty val/test fold (tiny dataset) — "
              "reporting on train rows")
        va = tr.copy()
        te = tr.copy()
    print(f"rows {X.shape}, positives {100*y.mean():.2f}%, "
          f"train/val/test {tr.sum()}/{va.sum()}/{te.sum()}")

    mu = X[tr].mean(0)
    sigma = X[tr].std(0)
    sigma[sigma < 1e-3] = 1e-3
    Z = ((X - mu) / sigma).astype(np.float32)
    n_features = X.shape[1]
    from agents.oracle.policy_features import N_GLOBALS
    X_globals = X[:, -N_GLOBALS:].copy()   # state head trains on raw globals
    del X, d                      # 23M-row datasets: free aggressively

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    blocks = []
    d_in = n_features
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

    Zt = torch.from_numpy(Z[tr])
    yt = torch.tensor(y[tr])
    ft = torch.tensor(frac[tr])
    wt = torch.where(yt > 0.5, torch.tensor(1.0),
                     torch.tensor(1.0 / NEG_KEEP))
    if os.environ.get("ORACLE_REWEIGHT_THREAT") and meta.shape[1] > 4:
        # the defense-dense builder samples threat states at stride 1 vs
        # calm at stride 3 (2 in the opening); reweight rows back to the
        # natural state distribution so cadence calibration survives the
        # oversampling (the unweighted version collapsed calm judgment)
        t_col = torch.tensor(meta[tr, 2])
        thr_col = torch.tensor(meta[tr, 4] > 0)
        w_nat = torch.where(
            thr_col, torch.tensor(1.0),
            torch.where(t_col < 60, torch.tensor(2.0), torch.tensor(3.0)))
        wt = wt * w_nat
        print("threat-state importance reweighting ON")
    if args.extra_ds:
        ew = np.where(extra_mask[tr],
                      args.extra_weight * row_w[tr], 1.0).astype(np.float32)
        wt = wt * torch.from_numpy(ew)
    Zv = torch.from_numpy(Z[va]).to(dev)
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
        if os.environ.get("ORACLE_REWEIGHT_THREAT") and meta.shape[1] > 4:
            wv = wv * np.where(meta[va, 4] > 0, 1.0,
                               np.where(meta[va, 2] < 60, 2.0, 3.0))
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
    # defense-state split (meta col 4 = under booked attack)
    if meta.shape[1] > 4:
        thr_te = meta[te, 4] > 0
        for label, mrows in (("THREAT", thr_te), ("calm", ~thr_te)):
            st_t, y_t, p_t = st_[mrows], yt_[mrows], pt[mrows]
            o2 = np.argsort(st_t, kind="stable")
            st_o2, y_o2, p_o2 = st_t[o2], y_t[o2], p_t[o2]
            b2 = np.flatnonzero(np.diff(st_o2)) + 1
            h1 = npos = 0
            for g in np.split(np.arange(len(st_o2)), b2):
                if y_o2[g].sum() < 1:
                    continue
                npos += 1
                h1 += int(y_o2[g][np.argsort(-p_o2[g])[0]] > 0.5)
            if npos:
                print(f"  {label} states: top-1 {h1/npos:.3f} (n={npos})")
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

    # ---- state-level initiation head --------------------------------
    # One row per state: the global slice (identical across a state's
    # pairs, so negative subsampling cannot skew it), label = expert
    # launched anything this turn. This is what lets the agent INITIATE
    # from a cold board — the per-pair head is dominated by follow-up
    # conditioning (fleets already in flight).
    print("training state-initiation head...")
    skey = state
    order_s = np.argsort(skey, kind="stable")
    sk_o = skey[order_s]
    first = np.r_[True, np.diff(sk_o) != 0]
    idx_first = order_s[first]                     # one row index per state
    G = X_globals[idx_first]
    # label: any positive among the state's rows
    import collections
    pos_states = set(skey[y > 0.5].tolist())
    ys_state = np.array([1.0 if k in pos_states else 0.0
                         for k in skey[idx_first]], dtype=np.float32)
    folds = (meta[idx_first, 0] % 10)
    s_tr, s_va = folds <= 7, folds == 8
    print(f"  states {len(G)}, fire rate {ys_state.mean():.3f}")
    s_mu = G[s_tr].mean(0)
    s_sigma = G[s_tr].std(0)
    s_sigma[s_sigma < 1e-3] = 1e-3
    Gz = ((G - s_mu) / s_sigma).astype(np.float32)

    class SNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.body = nn.Sequential(nn.Linear(N_GLOBALS, 32), nn.ReLU())
            self.out = nn.Linear(32, 1)

        def forward(self, z):
            return self.out(self.body(z)).squeeze(-1)

    snet = SNet().to(dev)
    sopt = torch.optim.AdamW(snet.parameters(), lr=1e-3)
    Gt = torch.tensor(Gz[s_tr], device=dev)
    yt_s = torch.tensor(ys_state[s_tr], device=dev)
    ws_state = torch.ones_like(yt_s)
    if os.environ.get("ORACLE_REWEIGHT_THREAT") and meta.shape[1] > 4:
        # the state head's fire-rate calibration is exactly what the
        # unweighted defense-dense data broke — reweight to natural
        thr_s = torch.tensor(meta[idx_first][s_tr, 4] > 0)
        t_s = torch.tensor(meta[idx_first][s_tr, 2])
        ws_state = torch.where(
            thr_s, torch.tensor(1.0),
            torch.where(t_s < 60, torch.tensor(2.0), torch.tensor(3.0)))
    Gv = torch.tensor(Gz[s_va], device=dev)
    bce_m = nn.BCEWithLogitsLoss(reduction="none")
    for ep_ in range(60):
        snet.train()
        perm = torch.randperm(len(Gt))
        for k in range(0, len(Gt), 16384):
            idx = perm[k:k + 16384]
            loss = (bce_m(snet(Gt[idx]), yt_s[idx])
                    * ws_state[idx]).mean()
            sopt.zero_grad()
            loss.backward()
            sopt.step()
    snet.eval()
    with torch.no_grad():
        ps = torch.sigmoid(snet(Gv)).cpu().numpy()
    print(f"  state head val AUC {auc(ys_state[s_va], ps):.4f}, "
          f"logloss {-np.mean(ys_state[s_va]*np.log(ps+1e-9)+(1-ys_state[s_va])*np.log(1-ps+1e-9)):.4f}")
    for th in (0.3, 0.4, 0.5, 0.6):
        print(f"    theta {th}: predicted fire-state rate "
              f"{(ps > th).mean():.3f} (true {ys_state[s_va].mean():.3f})")

    sd = net.state_dict()
    hidden = []
    for k in range(args.layers):
        hidden.append((sd[f"body.{2*k}.weight"].numpy().T,
                       sd[f"body.{2*k}.bias"].numpy()))
    ssd = snet.state_dict()
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
    # append the state head (own normalization + stack)
    s_arrays = {
        "mu": s_mu.astype(np.float32),
        "sigma": s_sigma.astype(np.float32),
        "layers": [(ssd["body.0.weight"].numpy().T,
                    ssd["body.0.bias"].numpy())],
        "head_state": (ssd["out.weight"].numpy().reshape(-1),
                       ssd["out.bias"].numpy()),
    }
    s_src = encode_weights_py(s_arrays, head_names=("STATE",))
    s_src = (s_src
             .replace("MU =", "S_MU =").replace("SIGMA =", "S_SIGMA =")
             .replace("LAYERS =", "S_LAYERS =")
             .replace("_W0", "_SW0").replace("_b0", "_Sb0"))
    s_src = "\n".join(ln for ln in s_src.splitlines()
                      if not ln.startswith(("# Generated", "import ")))
    with open(args.out, "w") as f:
        f.write(src + "\n# ---- state-initiation head ----\n" + s_src)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
