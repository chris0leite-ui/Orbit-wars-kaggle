"""Reframe B.1 — pv_eta leaf-residual diagnostic probe.

For each accepted candidate in pv_eta self-play, compare the chooser's
predicted Δ (raw pv_eta-discounted leaf − idle baseline) to the focal
seat's ACTUAL ship-delta over the next K turns. Decision gate:

  - σ(residual) / σ(actual) > 0.5 AND any stratification ANOVA F > 4
    → Reframe B.2 (per-target value head) is GREENLIT.
  - Otherwise → pv_eta's chooser is already value-optimal at its
    accepted set; pivot to Reframe C (opponent-emit predictor).

Stratifications: ship-count quintile, eta bucket [0]/[1-3]/[4-8]/[9+],
target ownership at launch (me/enemy/neutral), top-5 target_id.

Inputs:
  <probe_data_dir>/
    game-<seed>/
      accepted.jsonl  — chooser's trace_accepted records
      replay.jsonl    — per-tick {step, focal0_ships_total,
                                  focal1_ships_total, planets, fleets}

Output: markdown report at --out (default stdout only).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

SEAT0 = 0
DEFAULT_KS = (5, 10, 20)


def _spearman(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 3:
        return 0.0
    rx = np.argsort(np.argsort(np.asarray(xs)))
    ry = np.argsort(np.argsort(np.asarray(ys)))
    if rx.std() == 0 or ry.std() == 0:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def _r2(pred: np.ndarray, actual: np.ndarray) -> float:
    """OLS R² of `actual = a + b·pred`. Returns 0.0 on degenerate input."""
    if len(pred) < 3:
        return 0.0
    if pred.std() == 0 or actual.std() == 0:
        return 0.0
    b, a = np.polyfit(pred, actual, 1)
    yhat = a + b * pred
    ss_res = float(np.sum((actual - yhat) ** 2))
    ss_tot = float(np.sum((actual - actual.mean()) ** 2))
    if ss_tot <= 0:
        return 0.0
    return 1.0 - ss_res / ss_tot


def _anova_f(values: np.ndarray, groups: list) -> tuple[float, int]:
    """One-way ANOVA F-statistic across `groups` (list of bucket labels
    per row). Returns (F, n_groups). Manual implementation: F =
    (between-var/df_between) / (within-var/df_within). Returns 0.0 if
    fewer than 2 non-empty groups."""
    if len(values) < 4:
        return 0.0, 0
    by: dict = defaultdict(list)
    for v, g in zip(values, groups):
        by[g].append(float(v))
    by = {k: v for k, v in by.items() if len(v) >= 2}
    if len(by) < 2:
        return 0.0, len(by)
    grand_mean = float(values.mean())
    n_total = len(values)
    df_b = len(by) - 1
    df_w = n_total - len(by)
    if df_w <= 0:
        return 0.0, len(by)
    ss_b = sum(len(v) * (np.mean(v) - grand_mean) ** 2 for v in by.values())
    ss_w = sum(float(np.sum((np.asarray(v) - np.mean(v)) ** 2))
               for v in by.values())
    if ss_w <= 0:
        return float("inf"), len(by)
    f = (ss_b / df_b) / (ss_w / df_w)
    return float(f), len(by)


def load_game_dir(game_dir: Path) -> tuple[list[dict], list[dict]]:
    accepted_path = game_dir / "accepted.jsonl"
    replay_path = game_dir / "replay.jsonl"
    if not accepted_path.exists() or not replay_path.exists():
        return [], []
    accepted: list[dict] = []
    with accepted_path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                accepted.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    replay: list[dict] = []
    with replay_path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                replay.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return accepted, replay


def pair_records(accepted: list[dict], replay: list[dict],
                 ks: tuple[int, ...]) -> list[dict]:
    """For each seat-0 accepted record at turn T, emit one paired row
    per K with `actual_delta_K` from replay[T+K]−replay[T]. Skips
    records where T+K is past game end."""
    if not replay:
        return []
    # Index replay by step value (robust to potential missing ticks).
    replay_by_step: dict[int, dict] = {int(r["step"]): r for r in replay}
    steps_sorted = sorted(replay_by_step.keys())
    last_step = steps_sorted[-1] if steps_sorted else -1

    rows: list[dict] = []
    for rec in accepted:
        if int(rec.get("me", -1)) != SEAT0:
            continue
        t = int(rec["step"])
        if t not in replay_by_step:
            continue
        ships_t = int(replay_by_step[t]["focal0_ships_total"])
        # Resolve target ownership at launch from replay[T].planets.
        planets_t = replay_by_step[t].get("planets") or []
        tgt_id = int(rec["tgt_id"])
        owner_at_launch = -2  # sentinel: not found
        for p in planets_t:
            if int(p[0]) == tgt_id:
                owner_at_launch = int(p[1])
                break
        for K in ks:
            t_future = t + K
            if t_future > last_step or t_future not in replay_by_step:
                continue
            ships_future = int(replay_by_step[t_future]["focal0_ships_total"])
            actual_delta = ships_future - ships_t
            rows.append({
                "K": K,
                "step": t,
                "src_id": int(rec["src_id"]),
                "tgt_id": tgt_id,
                "ships": int(rec["ships"]),
                "eta": int(rec.get("eta", 0)),
                "kind": str(rec.get("kind", "solo")),
                "delta_pred": float(rec["delta_pred"]),
                "actual_delta": float(actual_delta),
                "owner_at_launch": owner_at_launch,
            })
    return rows


def _eta_bucket(eta: int) -> str:
    if eta <= 0:
        return "[0]"
    if eta <= 3:
        return "[1-3]"
    if eta <= 8:
        return "[4-8]"
    return "[9+]"


def _ownership_label(owner: int) -> str:
    if owner == 0:
        return "me"
    if owner == -1 or owner is None:
        return "neutral"
    return "enemy"


def per_k_stats(rows: list[dict], K: int) -> dict:
    sub = [r for r in rows if r["K"] == K]
    if len(sub) < 3:
        return {"K": K, "n": len(sub), "degenerate": True}
    pred = np.asarray([r["delta_pred"] for r in sub], dtype=np.float64)
    actual = np.asarray([r["actual_delta"] for r in sub], dtype=np.float64)
    residual = actual - pred
    sigma_actual = float(actual.std())
    sigma_pred = float(pred.std())
    sigma_residual = float(residual.std())
    ratio = sigma_residual / sigma_actual if sigma_actual > 0 else float("nan")
    rho = _spearman(pred.tolist(), actual.tolist())
    r2 = _r2(pred, actual)
    # Stratifications.
    if len(sub) >= 10:
        ship_quintiles = np.quantile(
            np.asarray([r["ships"] for r in sub]),
            [0.2, 0.4, 0.6, 0.8],
        ).tolist()

        def ship_bucket(s: int) -> str:
            for i, q in enumerate(ship_quintiles):
                if s < q:
                    return f"Q{i+1}"
            return f"Q{len(ship_quintiles)+1}"
    else:
        ship_bucket = lambda s: "all"  # noqa: E731

    ship_groups = [ship_bucket(r["ships"]) for r in sub]
    eta_groups = [_eta_bucket(r["eta"]) for r in sub]
    own_groups = [_ownership_label(r["owner_at_launch"]) for r in sub]
    # Top-5 target_id by frequency.
    tgt_counts: dict = defaultdict(int)
    for r in sub:
        tgt_counts[r["tgt_id"]] += 1
    top5 = {t for t, _ in sorted(tgt_counts.items(), key=lambda x: -x[1])[:5]}
    tgt_groups = [
        f"tgt_{r['tgt_id']}" if r["tgt_id"] in top5 else "other"
        for r in sub
    ]

    f_ship, n_ship = _anova_f(residual, ship_groups)
    f_eta, n_eta = _anova_f(residual, eta_groups)
    f_own, n_own = _anova_f(residual, own_groups)
    f_tgt, n_tgt = _anova_f(residual, tgt_groups)

    # Per-bucket residual mean/std for the report tables.
    def bucket_table(groups: list) -> list[dict]:
        by: dict = defaultdict(list)
        for v, g in zip(residual, groups):
            by[g].append(float(v))
        out = []
        for g, vs in sorted(by.items(), key=lambda kv: (-len(kv[1]), kv[0])):
            arr = np.asarray(vs)
            out.append({
                "bucket": g,
                "n": int(len(vs)),
                "mean": float(arr.mean()),
                "std": float(arr.std()),
            })
        return out

    return {
        "K": K,
        "n": len(sub),
        "degenerate": False,
        "sigma_actual": sigma_actual,
        "sigma_pred": sigma_pred,
        "sigma_residual": sigma_residual,
        "residual_ratio": ratio,
        "spearman_rho": rho,
        "r2": r2,
        "anova": {
            "ship_quintile": {"F": f_ship, "n_groups": n_ship},
            "eta_bucket": {"F": f_eta, "n_groups": n_eta},
            "owner_at_launch": {"F": f_own, "n_groups": n_own},
            "target_id_top5": {"F": f_tgt, "n_groups": n_tgt},
        },
        "tables": {
            "ship_quintile": bucket_table(ship_groups),
            "eta_bucket": bucket_table(eta_groups),
            "owner_at_launch": bucket_table(own_groups),
            "target_id_top5": bucket_table(tgt_groups),
        },
    }


def evaluate_gate(per_k: list[dict]) -> dict:
    """B.2 GREENLIT iff for at least one K: residual_ratio > 0.5 AND any
    stratification F > 4."""
    triggers: list[dict] = []
    for s in per_k:
        if s.get("degenerate"):
            continue
        ratio_ok = s["residual_ratio"] > 0.5
        f_axes = s["anova"]
        f_ok = any(v["F"] > 4.0 for v in f_axes.values())
        if ratio_ok and f_ok:
            triggers.append({
                "K": s["K"],
                "residual_ratio": s["residual_ratio"],
                "max_f": max(v["F"] for v in f_axes.values()),
            })
    greenlit = bool(triggers)
    return {
        "greenlit_b2": greenlit,
        "triggers": triggers,
        "verdict": (
            "GREENLIT Reframe B.2 (per-target value head)"
            if greenlit else "PIVOT to Reframe C (opponent-emit predictor)"
        ),
    }


def render_report(data_dir: Path, per_k: list[dict], gate: dict,
                  n_games: int, total_accepted_seat0: int) -> str:
    L = []
    L.append("# Reframe B.1 — pv_eta leaf-residual diagnostic probe")
    L.append("")
    L.append(f"Data dir: `{data_dir}`")
    L.append(f"Games analysed: **{n_games}**  "
             f"Seat-0 accepted candidates: **{total_accepted_seat0}**")
    L.append("")
    L.append(f"## Verdict: **{gate['verdict']}**")
    L.append("")
    if gate["triggers"]:
        L.append("Triggers (K → residual_ratio, max F):")
        for t in gate["triggers"]:
            L.append(f"- K={t['K']}: ratio={t['residual_ratio']:.3f}, "
                     f"max F={t['max_f']:.2f}")
        L.append("")
    L.append("Gate rule: any K with σ(residual)/σ(actual) > 0.5 AND "
             "any stratification ANOVA F > 4.")
    L.append("")
    L.append("## Per-K stats")
    L.append("")
    L.append("| K | n | σ(actual) | σ(pred) | σ(residual) | ratio | R² | ρ |")
    L.append("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for s in per_k:
        if s.get("degenerate"):
            L.append(f"| {s['K']} | {s['n']} | — | — | — | — | — | — |")
            continue
        L.append(
            f"| {s['K']} | {s['n']} | "
            f"{s['sigma_actual']:.2f} | {s['sigma_pred']:.2f} | "
            f"{s['sigma_residual']:.2f} | {s['residual_ratio']:.3f} | "
            f"{s['r2']:.3f} | {s['spearman_rho']:.3f} |"
        )
    L.append("")
    L.append("## ANOVA F-stats by stratification axis")
    L.append("")
    L.append("| K | ship_quintile | eta_bucket | owner_at_launch | top-5 tgt |")
    L.append("|---:|---:|---:|---:|---:|")
    for s in per_k:
        if s.get("degenerate"):
            continue
        a = s["anova"]
        L.append(
            f"| {s['K']} | "
            f"{a['ship_quintile']['F']:.2f} (g={a['ship_quintile']['n_groups']}) | "
            f"{a['eta_bucket']['F']:.2f} (g={a['eta_bucket']['n_groups']}) | "
            f"{a['owner_at_launch']['F']:.2f} (g={a['owner_at_launch']['n_groups']}) | "
            f"{a['target_id_top5']['F']:.2f} (g={a['target_id_top5']['n_groups']}) |"
        )
    L.append("")
    # Bucket tables — show the largest-K stratification breakdown so
    # the report stays compact.
    last = next((s for s in per_k if not s.get("degenerate")), None)
    if last is not None:
        for axis_name in ("ship_quintile", "eta_bucket",
                          "owner_at_launch", "target_id_top5"):
            tbl = last["tables"][axis_name]
            L.append(f"## Residual by {axis_name} "
                     f"(K={last['K']})")
            L.append("")
            L.append("| bucket | n | mean residual | std |")
            L.append("|---|---:|---:|---:|")
            for row in tbl:
                L.append(
                    f"| {row['bucket']} | {row['n']} | "
                    f"{row['mean']:+.2f} | {row['std']:.2f} |"
                )
            L.append("")
    L.append("## Interpretation")
    L.append("")
    L.append("σ(residual)/σ(actual) measures the fraction of future "
             "ship-delta variance that the chooser's leaf-Δ does NOT "
             "explain. Ratio → 0 means the leaf already predicts the "
             "outcome perfectly; ratio → 1 means the leaf is noise "
             "against the ground truth.")
    L.append("")
    L.append("ANOVA F > 4 on an axis means residuals systematically "
             "differ across that axis's buckets. A non-flat residual "
             "structure is the headroom a per-target value head can "
             "exploit; if all F-stats are small, the leaf's errors "
             "are unstructured noise that no per-target head can fix.")
    return "\n".join(L)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("data_dir", type=Path,
                   help="Probe data dir (contains game-<seed>/ subdirs)")
    p.add_argument("--out", type=Path, default=None,
                   help="Optional markdown report destination")
    p.add_argument("--K", default="5,10,20",
                   help="Comma-separated K-horizons (default 5,10,20)")
    args = p.parse_args()

    ks = tuple(int(x) for x in args.K.split(",") if x.strip())
    if not args.data_dir.is_dir():
        print(f"ERROR: data dir not found: {args.data_dir}", file=sys.stderr)
        return 2

    game_dirs = sorted(d for d in args.data_dir.iterdir()
                       if d.is_dir() and d.name.startswith("game-"))
    if not game_dirs:
        print(f"ERROR: no game-* subdirs in {args.data_dir}", file=sys.stderr)
        return 2

    all_rows: list[dict] = []
    total_accepted_seat0 = 0
    for g in game_dirs:
        accepted, replay = load_game_dir(g)
        total_accepted_seat0 += sum(
            1 for r in accepted if int(r.get("me", -1)) == SEAT0
        )
        all_rows.extend(pair_records(accepted, replay, ks))

    if not all_rows:
        print("ERROR: no paired rows produced — check accepted.jsonl "
              "and replay.jsonl contents.", file=sys.stderr)
        return 3

    per_k = [per_k_stats(all_rows, K) for K in ks]
    gate = evaluate_gate(per_k)
    report = render_report(args.data_dir, per_k, gate,
                           n_games=len(game_dirs),
                           total_accepted_seat0=total_accepted_seat0)

    print(report)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report + "\n")
        print(f"\nReport written to {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
