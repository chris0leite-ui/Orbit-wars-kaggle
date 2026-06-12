"""One-command verdict for the shot-filter live probe (sub 53595717).

Pulls the probe's live episodes, scores every attack launch we made with
the trained model, and compares against the no-filter baseline sub
(53577315, identical config minus the filter). Self-match episodes are
excluded on BOTH sides (episodes.csv carries no seat info, so in
multi-our-sub games the probe's seat is ambiguous).

What to expect if the filter works as designed:
  - low-P (<0.15) attack share collapses toward ~0 (those waves are now
    vetoed before launch, so they never appear in the action stream)
  - overall attack success rate rises from the ~0.42 baseline
  - ships spent on failed attacks per episode drops

Usage:
    python -m scripts.shot_probe_verdict [--probe 53595717]
                                         [--baseline 53577315]
                                         [--no-pull] [--threshold 0.15]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
LIVE_DIR = REPO / "audit" / "live-episodes"
OUR_TEAM = "ChrisLeiteScha"

sys.path.insert(0, str(REPO / "agents" / "producer_plus"))
import shot_mlp  # noqa: E402

from scripts.label_shot_outcomes import _label_seat  # noqa: E402

MIN_EPISODES = 20


def pull(sub_id: str) -> None:
    subprocess.run(
        [sys.executable, "-m", "scripts.live_episode_summary", sub_id, "--pull"],
        cwd=REPO, capture_output=True, text=True, check=False,
    )


def collect(sub_id: str) -> dict:
    """Per-sub stats over non-self-match episodes."""
    sub_dir = LIVE_DIR / sub_id
    feats, labels = [], []
    n_eps = n_self = 0
    for f in sorted(sub_dir.glob("episode-*-replay.json")):
        try:
            replay = json.loads(f.read_text())
        except Exception:
            continue
        teams = replay.get("info", {}).get("TeamNames", [])
        ours = [i for i, t in enumerate(teams) if t == OUR_TEAM]
        if len(ours) != 1:
            n_self += 1
            continue
        n_eps += 1
        for ex in _label_seat(f, replay.get("steps", []), ours[0], OUR_TEAM):
            feats.append(ex["features"])
            labels.append(ex["label"])
    if not feats:
        return {"n_episodes": n_eps, "n_self_excluded": n_self, "n_launches": 0}
    X = np.asarray(feats, dtype=np.float32)
    y = np.asarray(labels, dtype=np.float32)
    p = shot_mlp.predict_success(X)
    attack = X[:, 6] < 0.5          # owner_mine feature == 0
    ships = X[:, 9] * shot_mlp.NORM["max_ships"]
    failed_attack_ships = float(ships[attack & (y < 0.5)].sum())
    return {
        "n_episodes": n_eps,
        "n_self_excluded": n_self,
        "n_launches": int(len(y)),
        "n_attacks": int(attack.sum()),
        "attacks_per_ep": attack.sum() / max(1, n_eps),
        "attack_success": float(y[attack].mean()) if attack.any() else float("nan"),
        "lowp_attack_share": float((p[attack] < ARGS.threshold).mean()) if attack.any() else float("nan"),
        "lowp_attack_success": float(y[attack & (p < ARGS.threshold)].mean())
                               if (attack & (p < ARGS.threshold)).any() else float("nan"),
        "failed_attack_ships_per_ep": failed_attack_ships / max(1, n_eps),
    }


def main() -> int:
    global ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", default="53595717")
    ap.add_argument("--baseline", default="53577315")
    ap.add_argument("--threshold", type=float, default=0.15,
                    help="the threshold baked into the probe bundle")
    ap.add_argument("--no-pull", action="store_true")
    ARGS = ap.parse_args()

    if not ARGS.no_pull:
        for sid in (ARGS.probe, ARGS.baseline):
            print(f"pulling episodes for {sid} ...", file=sys.stderr)
            pull(sid)

    rows = {name: collect(sid) for name, sid in
            (("PROBE " + ARGS.probe, ARGS.probe),
             ("BASE  " + ARGS.baseline, ARGS.baseline))}

    print(f"\n=== shot-filter probe verdict (threshold {ARGS.threshold}) ===")
    keys = [
        ("episodes (vs field, self-matches excluded)", "n_episodes", "{:d}"),
        ("launches", "n_launches", "{:d}"),
        ("attack launches / episode", "attacks_per_ep", "{:.1f}"),
        (f"attack share with model P<{ARGS.threshold}", "lowp_attack_share", "{:.1%}"),
        ("  ...success of that share", "lowp_attack_success", "{:.1%}"),
        ("attack success rate (overall)", "attack_success", "{:.1%}"),
        ("ships lost to failed attacks / episode", "failed_attack_ships_per_ep", "{:.0f}"),
    ]
    name_w = max(len(n) for n in rows)
    print(f"{'metric':<44} | " + " | ".join(f"{n:>{name_w}}" for n in rows))
    for label, key, fmt in keys:
        cells = []
        for r in rows.values():
            v = r.get(key)
            cells.append(fmt.format(v) if v is not None and v == v else "—")
        print(f"{label:<44} | " + " | ".join(f"{c:>{name_w}}" for c in cells))

    probe = list(rows.values())[0]
    if probe["n_episodes"] < MIN_EPISODES:
        print(f"\n⚠ probe has {probe['n_episodes']} usable episodes "
              f"(< {MIN_EPISODES}) — directional only, re-run later.")
    else:
        print("\nRead: low-P share ↓ toward 0 + attack success ↑ = filter "
              "working as designed; settled μ (24h) is the outcome metric.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
