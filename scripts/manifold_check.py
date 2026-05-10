"""Manifold diagnostic for the strategy zoo.

See plan: /root/.claude/plans/read-the-handover-next-imperative-whisper.md
Phase 1 §gate. Companion: audit/2026-05-10-meta-strategy-prior-art.md.

Question this script answers:
    Given a corpus of (replay, opponent_strategy_label) pairs, can we
    classify the opponent's strategy from a K-turn behavioural fingerprint?

If random-forest CV-by-seed accuracy >= 90% at K <= 100, the manifold
hypothesis (small-dim, classifiable strategy space) is supported and we
proceed to Phase 2 (zoo expansion). If it fails, we either expand the
fingerprint feature set, or move to a learned embedding (Grover et al.,
ICML 2018).

CV-by-seed (NOT random row CV) is load-bearing: a per-game fingerprint
shares a seed across both players, so random row-splits leak info; we
split by SEED so the held-out fold has unseen game seeds entirely.

Usage:
    python -m scripts.manifold_check --replay-dir audit/replays/<utc>
    python -m scripts.manifold_check --replay-dir <path> --prefixes 50 100 200

Outputs:
    audit/manifold/<utc>/report.md
    audit/manifold/<utc>/accuracy_vs_K.png
    audit/manifold/<utc>/pca_K{25,50,100,200}.png
    audit/manifold/<utc>/umap_K100.png   (skipped if umap-learn missing)
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from lib.fingerprint import FEATURE_NAMES, FEATURE_VERSION, batch_fingerprints  # noqa: E402


def load_replays(replay_dir: Path) -> list[dict]:
    """Load every `*.json.gz` under `replay_dir` as a parsed dict."""
    replays: list[dict] = []
    for path in sorted(replay_dir.glob("*.json.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            replays.append(json.load(fh))
    return replays


def cv_accuracy(
    X: np.ndarray, y: np.ndarray, seeds: np.ndarray, n_folds: int, classifier: str
) -> tuple[float, np.ndarray, list[str]]:
    """Group-K-fold cross-validation grouped by seed.

    Returns (mean_accuracy, confusion_matrix, class_labels).
    Confusion matrix is summed over folds so the row-fractions are stable.
    """
    unique_seeds = np.unique(seeds)
    n_folds = min(n_folds, len(unique_seeds))
    if n_folds < 2:
        # Fallback: single train-test split with stratification by seed.
        # Pathological dataset; warn upstream.
        return float("nan"), np.zeros((1, 1)), []

    gkf = GroupKFold(n_splits=n_folds)
    classes = sorted(set(y.tolist()))
    cm_total = np.zeros((len(classes), len(classes)), dtype=int)
    accs: list[float] = []
    for train_idx, test_idx in gkf.split(X, y, groups=seeds):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_te, y_te = X[test_idx], y[test_idx]
        if classifier == "rf":
            clf = make_pipeline(
                StandardScaler(),
                RandomForestClassifier(
                    n_estimators=200, random_state=0, n_jobs=-1
                ),
            )
        elif classifier == "lr":
            clf = make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=2000),
            )
        else:
            raise ValueError(classifier)
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_te)
        accs.append(accuracy_score(y_te, y_pred))
        cm_total += confusion_matrix(y_te, y_pred, labels=classes)
    return float(np.mean(accs)), cm_total, classes


def plot_accuracy_vs_k(
    prefixes: Sequence[int],
    rf_accs: Sequence[float],
    lr_accs: Sequence[float],
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(prefixes, rf_accs, "o-", label="Random Forest")
    ax.plot(prefixes, lr_accs, "s-", label="Logistic Regression")
    ax.axhline(0.9, color="gray", linestyle="--", alpha=0.5, label="Phase-1 gate (90%)")
    ax.set_xlabel("Prefix turns K")
    ax.set_ylabel("CV-by-seed accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("Strategy classification vs prefix length")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_pca(X: np.ndarray, y: np.ndarray, K: int, out_path: Path) -> None:
    classes = sorted(set(y.tolist()))
    Xs = StandardScaler().fit_transform(X)
    Z = PCA(n_components=2).fit_transform(Xs)
    fig, ax = plt.subplots(figsize=(7, 6))
    cmap = plt.get_cmap("tab10")
    for i, cls in enumerate(classes):
        mask = y == cls
        ax.scatter(Z[mask, 0], Z[mask, 1], label=cls, alpha=0.7, s=22, color=cmap(i % 10))
    ax.set_title(f"PCA (2D) of fingerprints, K={K}")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_umap(X: np.ndarray, y: np.ndarray, K: int, out_path: Path) -> None:
    """UMAP scatter; skipped silently if umap-learn isn't installed."""
    try:
        import umap  # type: ignore
    except ImportError:
        return
    classes = sorted(set(y.tolist()))
    Xs = StandardScaler().fit_transform(X)
    Z = umap.UMAP(n_components=2, random_state=0, n_jobs=1).fit_transform(Xs)
    fig, ax = plt.subplots(figsize=(7, 6))
    cmap = plt.get_cmap("tab10")
    for i, cls in enumerate(classes):
        mask = y == cls
        ax.scatter(Z[mask, 0], Z[mask, 1], label=cls, alpha=0.7, s=22, color=cmap(i % 10))
    ax.set_title(f"UMAP (2D) of fingerprints, K={K}")
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def confusion_to_markdown(cm: np.ndarray, classes: list[str]) -> str:
    """Render a confusion matrix as a markdown table."""
    if cm.size == 0:
        return "(empty)"
    header = "| true \\ pred | " + " | ".join(classes) + " |"
    sep = "|" + "|".join(["---"] * (len(classes) + 1)) + "|"
    rows = [header, sep]
    for i, cls in enumerate(classes):
        cells = [str(int(cm[i, j])) for j in range(len(classes))]
        rows.append(f"| **{cls}** | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def render_report(
    *,
    out_dir: Path,
    prefixes: list[int],
    rf_accs: list[float],
    lr_accs: list[float],
    n_replays: int,
    n_classes: int,
    classes: list[str],
    n_features: int,
    feature_version: int,
    cm_at_best_k: tuple[int, np.ndarray, list[str]] | None,
    classifier_for_cm: str,
    gate_target: float = 0.9,
    gate_max_k: int = 100,
) -> Path:
    """Write report.md summarising the diagnostic. Returns the path."""
    lines: list[str] = []
    lines.append(f"# Manifold check — {out_dir.name}")
    lines.append("")
    lines.append(
        f"- replays: {n_replays}  |  classes: {n_classes}  |  "
        f"features: {n_features} (version {feature_version})"
    )
    lines.append(f"- class labels: {', '.join(classes)}")
    lines.append("")
    lines.append("## Accuracy vs K (CV-by-seed)")
    lines.append("")
    lines.append("| K (prefix turns) | Random Forest | Logistic Regression |")
    lines.append("|---|---|---|")
    for k, rf, lr in zip(prefixes, rf_accs, lr_accs):
        lines.append(f"| {k} | {rf:.1%} | {lr:.1%} |")
    lines.append("")
    lines.append(
        f"![accuracy vs K](accuracy_vs_K.png)"
    )
    lines.append("")
    # Gate decision.
    rf_passing = [
        (k, rf) for k, rf in zip(prefixes, rf_accs)
        if rf >= gate_target and k <= gate_max_k
    ]
    if rf_passing:
        k_pass, rf_pass = rf_passing[0]
        gate_line = (
            f"**Phase 1 gate: ✅ CLEARED.** Random-forest CV-by-seed accuracy "
            f"{rf_pass:.1%} at K={k_pass} (≥ {gate_target:.0%} target, K ≤ {gate_max_k})."
        )
    else:
        gate_line = (
            f"**Phase 1 gate: ❌ NOT CLEARED.** No K ≤ {gate_max_k} hits "
            f"{gate_target:.0%} on random forest. "
            "Either extend `lib/fingerprint.FEATURE_NAMES` (and bump "
            "FEATURE_VERSION) or move to a learned embedding "
            "(Grover et al. ICML 2018 protocol)."
        )
    lines.append("## Gate")
    lines.append("")
    lines.append(gate_line)
    lines.append("")
    if cm_at_best_k is not None:
        k_cm, cm, cm_classes = cm_at_best_k
        lines.append(f"## Confusion matrix ({classifier_for_cm}, K={k_cm})")
        lines.append("")
        lines.append("Rows = true label, columns = predicted, cells summed across folds.")
        lines.append("")
        lines.append(confusion_to_markdown(cm, cm_classes))
        lines.append("")
        # Confusable-pairs callout: any off-diagonal cell >= 25% of its row.
        confusable: list[tuple[str, str, float]] = []
        for i, cls_i in enumerate(cm_classes):
            row_total = int(cm[i].sum())
            if row_total == 0:
                continue
            for j, cls_j in enumerate(cm_classes):
                if i == j:
                    continue
                frac = cm[i, j] / row_total
                if frac >= 0.25:
                    confusable.append((cls_i, cls_j, frac))
        if confusable:
            lines.append("### Confusable pairs (≥25% of true class predicted as other)")
            lines.append("")
            for true_cls, pred_cls, frac in confusable:
                lines.append(f"- `{true_cls}` → `{pred_cls}`: {frac:.1%}")
            lines.append("")
    lines.append("## Visualisations")
    lines.append("")
    for k in prefixes:
        lines.append(f"- PCA at K={k}: ![pca K={k}](pca_K{k}.png)")
    umap_png = out_dir / f"umap_K{prefixes[len(prefixes) // 2]}.png"
    if umap_png.exists():
        lines.append(f"- UMAP at K={prefixes[len(prefixes) // 2]}: ![umap](umap_K{prefixes[len(prefixes) // 2]}.png)")
    lines.append("")
    lines.append(
        "## Method"
    )
    lines.append("")
    lines.append(
        "- Fingerprints from `lib/fingerprint.py` (15 hand-designed features)."
    )
    lines.append(
        "- Each replay contributes 2 rows (one per seat); label = the "
        "agent name as recorded in the replay header (`agent_p0` / `agent_p1`)."
    )
    lines.append(
        "- CV is `GroupKFold` on **seed** (not random row split) so the "
        "held-out fold has fully unseen game seeds. Same-seed rows for "
        "two seats stay together in the same fold."
    )
    lines.append(
        "- Both classifiers run inside a `StandardScaler` pipeline so "
        "feature-scale differences do not dominate."
    )
    report_path = out_dir / "report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--replay-dir", type=Path, required=True,
        help="Directory of `*.json.gz` replays (output of `--capture-replays`).",
    )
    parser.add_argument(
        "--prefixes", type=int, nargs="+", default=[25, 50, 100, 200],
        help="Prefix lengths in turns (default: 25 50 100 200).",
    )
    parser.add_argument(
        "--n-folds", type=int, default=5,
        help="Group-K-fold count (default: 5).",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Output directory (default: audit/manifold/<utc>/).",
    )
    parser.add_argument(
        "--strategies", nargs="*", default=None,
        help="Filter to these strategy labels (default: keep all).",
    )
    args = parser.parse_args(argv)

    if not args.replay_dir.exists():
        raise SystemExit(f"replay-dir not found: {args.replay_dir}")

    print(f"--- loading replays from {args.replay_dir}")
    replays = load_replays(args.replay_dir)
    if not replays:
        raise SystemExit(f"no `*.json.gz` replays under {args.replay_dir}")

    if args.strategies is not None:
        keep = set(args.strategies)
        before = len(replays)
        replays = [
            r for r in replays
            if r.get("agent_p0") in keep and r.get("agent_p1") in keep
        ]
        print(f"--- filtered {before} -> {len(replays)} replays via --strategies")

    # Set up output dir.
    out_dir = args.output_dir
    if out_dir is None:
        utc = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_dir = REPO / "audit" / "manifold" / utc
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"--- output: {out_dir}")

    # Per-prefix accuracy + viz.
    rf_accs: list[float] = []
    lr_accs: list[float] = []
    cm_at_best: tuple[int, np.ndarray, list[str]] | None = None
    classes_global: list[str] = []
    n_classes_global = 0
    for K in args.prefixes:
        X, labels, seeds, _players = batch_fingerprints(replays, prefix_turns=K)
        y = np.array(labels)
        seeds_arr = np.array(seeds)
        rf_acc, cm, classes = cv_accuracy(X, y, seeds_arr, args.n_folds, "rf")
        lr_acc, _, _ = cv_accuracy(X, y, seeds_arr, args.n_folds, "lr")
        rf_accs.append(rf_acc)
        lr_accs.append(lr_acc)
        classes_global = classes
        n_classes_global = len(classes)
        print(f"    K={K:>3}  RF={rf_acc:.1%}  LR={lr_acc:.1%}  "
              f"(n_rows={X.shape[0]}, n_classes={len(classes)})")
        # Save PCA for every K.
        plot_pca(X, y, K, out_dir / f"pca_K{K}.png")
        # CM only at the best (smallest) K that hits the gate; or at the largest K.
        if cm_at_best is None and rf_acc >= 0.9 and K <= 100:
            cm_at_best = (K, cm, classes)
    if cm_at_best is None and rf_accs:
        # Fall back: store CM at the largest K we ran.
        K_last = args.prefixes[-1]
        X, labels, seeds, _ = batch_fingerprints(replays, prefix_turns=K_last)
        y = np.array(labels)
        seeds_arr = np.array(seeds)
        _, cm, classes = cv_accuracy(X, y, seeds_arr, args.n_folds, "rf")
        cm_at_best = (K_last, cm, classes)

    # Accuracy vs K.
    plot_accuracy_vs_k(args.prefixes, rf_accs, lr_accs, out_dir / "accuracy_vs_K.png")

    # Optional UMAP at the median prefix.
    K_mid = args.prefixes[len(args.prefixes) // 2]
    X_mid, labels_mid, _, _ = batch_fingerprints(replays, prefix_turns=K_mid)
    plot_umap(X_mid, np.array(labels_mid), K_mid, out_dir / f"umap_K{K_mid}.png")

    report_path = render_report(
        out_dir=out_dir,
        prefixes=list(args.prefixes),
        rf_accs=rf_accs,
        lr_accs=lr_accs,
        n_replays=len(replays),
        n_classes=n_classes_global,
        classes=classes_global,
        n_features=len(FEATURE_NAMES),
        feature_version=FEATURE_VERSION,
        cm_at_best_k=cm_at_best,
        classifier_for_cm="random forest",
    )
    print(f"--- report: {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
