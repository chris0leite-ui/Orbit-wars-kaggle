"""validate_favor.py — sample mid-game states, check that favor predicts the winner.

Pipeline
--------
1. Run N games per matchup, parallel. Matchups span the skill range
   (random / nearest / v7_0 / v4_planner).
2. Snapshot the observation at turns {50, 150, 250, 350, 450} per game,
   plus the eventual winner. Save to `audit/favor-states.jsonl`.
3. Score each state with `favor(obs, 0) − favor(obs, 1)`.
4. Report:
     - overall accuracy: % of states where sign(favor_diff) matches winner
     - overall AUC: rank-correlation of favor_diff with outcome
     - per-turn-bucket breakdown
     - per-matchup breakdown

Re-score mode
-------------
`python validate_favor.py --replay` re-reads the saved states and
recomputes AUC with the current favor() — no game simulation. This is
the inner loop for tuning features: change `favor.py`, re-run replay,
read AUC, decide.

Gate
----
Overall AUC ≥ 0.75 → favor is informative enough to drive agent moves.
At turn ≥ 250 we want ≥ 0.85 — past mid-game the right answer is
visible from the state, and a weaker AUC there means the function is
missing the decisive feature.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from favor import favor, FavorConfig

STATES_PATH = REPO / "audit" / "favor-states.jsonl"
SNAPSHOTS_AT = [50, 150, 250, 350, 450]

BASELINES = {
    "random":     "random",
    "nearest":    str(REPO / "baselines" / "nearest.py"),
    "v7_0":       str(REPO / "baselines" / "v7_0.py"),
    "v4_planner": str(REPO / "baselines" / "v4_planner.py"),
}

# Matchups span skill regimes. Tuned for ~10 min wallclock at 4 workers.
DEFAULT_MATCHUPS = [
    ("random", "nearest"),
    ("nearest", "nearest"),
    ("nearest", "v7_0"),
    ("v7_0",    "v4_planner"),
]


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------


def _to_jsonable(o):
    """Convert kaggle_environments Struct / nested obs → json-safe dict."""
    if o is None or isinstance(o, (bool, int, float, str)):
        return o
    if isinstance(o, dict):
        return {k: _to_jsonable(v) for k, v in o.items()}
    if hasattr(o, "items"):  # Struct-style
        return {k: _to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_to_jsonable(x) for x in o]
    return str(o)


# ---------------------------------------------------------------------------
# Game generation (worker)
# ---------------------------------------------------------------------------


def _run_game_with_snapshots(args):
    a_name, b_name, a_spec, b_spec, seed = args
    try:
        from kaggle_environments import make
        env = make("orbit_wars", configuration={"seed": seed}, debug=False)
        env.run([a_spec, b_spec])
    except Exception as e:  # noqa: BLE001 — a bundle could break at runtime
        return {"matchup": f"{a_name}_vs_{b_name}", "seed": seed, "error": str(e), "snaps": []}

    final = env.steps[-1]
    rewards = [s["reward"] if s.get("reward") is not None else -1 for s in final]
    if rewards[0] == rewards[1]:  # tie — drop from validation set
        return {"matchup": f"{a_name}_vs_{b_name}", "seed": seed, "error": "tie", "snaps": []}
    winner = 0 if rewards[0] > rewards[1] else 1

    snaps = []
    for t in SNAPSHOTS_AT:
        if t >= len(env.steps):
            break
        try:
            obs = env.steps[t][0]["observation"]
        except (KeyError, IndexError, TypeError):
            continue
        snaps.append(
            {
                "matchup": f"{a_name}_vs_{b_name}",
                "seed": seed,
                "turn": t,
                "winner": winner,
                "obs": _to_jsonable(obs),
            }
        )
    return {"matchup": f"{a_name}_vs_{b_name}", "seed": seed, "error": None, "snaps": snaps}


def generate_states(n_per_matchup: int, workers: int, seed0: int = 5000) -> int:
    """Run games, write snapshots to STATES_PATH. Returns total snapshots saved."""
    STATES_PATH.parent.mkdir(parents=True, exist_ok=True)
    tasks = []
    for mi, (a, b) in enumerate(DEFAULT_MATCHUPS):
        for i in range(n_per_matchup):
            tasks.append((a, b, BASELINES[a], BASELINES[b], seed0 + mi * 1000 + i))

    written = 0
    skipped = 0
    t0 = time.perf_counter()
    with STATES_PATH.open("w") as out:
        if workers <= 1:
            results = (_run_game_with_snapshots(t) for t in tasks)
        else:
            ex = ProcessPoolExecutor(max_workers=workers)
            futs = [ex.submit(_run_game_with_snapshots, t) for t in tasks]
            results = (f.result() for f in as_completed(futs))

        done = 0
        for res in results:
            done += 1
            if res["error"]:
                skipped += 1
            for snap in res["snaps"]:
                out.write(json.dumps(snap) + "\n")
                written += 1
            if done % 10 == 0:
                el = time.perf_counter() - t0
                print(
                    f"  [{done}/{len(tasks)}] games done; "
                    f"{written} snaps saved; "
                    f"{skipped} skipped (tie/error); "
                    f"{el:.1f} s elapsed",
                    flush=True,
                )

    print(
        f"\ngenerated {written} state snapshots from {len(tasks)} games "
        f"({skipped} skipped) in {time.perf_counter() - t0:.1f} s"
    )
    return written


# ---------------------------------------------------------------------------
# Scoring + AUC
# ---------------------------------------------------------------------------


def auc_score(scores: list[float], labels: list[int]) -> float:
    """AUC via Mann-Whitney U with average-rank tie handling. labels ∈ {0,1}."""
    n = len(scores)
    n_pos = sum(labels)
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    pairs = sorted(zip(scores, labels), key=lambda x: x[0])
    rank_sum_pos = 0.0
    i = 0
    rank = 1
    while i < n:
        j = i
        while j < n and pairs[j][0] == pairs[i][0]:
            j += 1
        avg_rank = (rank + rank + (j - i) - 1) / 2.0
        for k in range(i, j):
            if pairs[k][1] == 1:
                rank_sum_pos += avg_rank
        rank += j - i
        i = j
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def score_snapshots(config: FavorConfig | None = None) -> list[dict]:
    """Read STATES_PATH, compute favor_diff per snapshot."""
    if not STATES_PATH.exists():
        raise FileNotFoundError(
            f"{STATES_PATH} not found. Run without --replay first to generate."
        )
    scored = []
    with STATES_PATH.open() as f:
        for line in f:
            snap = json.loads(line)
            obs = snap["obs"]
            f0 = favor(obs, 0, config)
            f1 = favor(obs, 1, config)
            scored.append(
                {
                    "matchup": snap["matchup"],
                    "turn": snap["turn"],
                    "winner": snap["winner"],
                    "favor_diff": f0 - f1,
                    "favor_0": f0,
                    "favor_1": f1,
                }
            )
    return scored


def report(scored: list[dict]) -> dict:
    """Print accuracy + AUC overall and per (matchup, turn). Return dict."""
    if not scored:
        print("no snapshots scored — nothing to report.")
        return {}

    def summarise(rows):
        if not rows:
            return {"n": 0}
        labels = [1 if r["winner"] == 0 else 0 for r in rows]
        scores = [r["favor_diff"] for r in rows]
        acc = sum(
            1 for r in rows if (r["favor_diff"] > 0) == (r["winner"] == 0)
        ) / len(rows)
        auc = auc_score(scores, labels)
        return {"n": len(rows), "acc": acc, "auc": auc}

    overall = summarise(scored)
    by_turn = {
        t: summarise([r for r in scored if r["turn"] == t]) for t in SNAPSHOTS_AT
    }
    matchups = sorted({r["matchup"] for r in scored})
    by_matchup = {m: summarise([r for r in scored if r["matchup"] == m]) for m in matchups}

    print("\n=== FAVOR VALIDATION ===")
    print(
        f"OVERALL: n={overall['n']:>4}  "
        f"acc={overall['acc']:.3f}  AUC={overall['auc']:.3f}"
    )

    print("\n--- by turn ---")
    print(f"  {'turn':>5}  {'n':>5}  {'acc':>6}  {'AUC':>6}")
    for t in SNAPSHOTS_AT:
        r = by_turn[t]
        if r["n"]:
            print(f"  {t:>5}  {r['n']:>5}  {r['acc']:.3f}  {r['auc']:.3f}")
        else:
            print(f"  {t:>5}  {0:>5}  {'-':>6}  {'-':>6}")

    print("\n--- by matchup ---")
    print(f"  {'matchup':<30}  {'n':>5}  {'acc':>6}  {'AUC':>6}")
    for m in matchups:
        r = by_matchup[m]
        print(f"  {m:<30}  {r['n']:>5}  {r['acc']:.3f}  {r['auc']:.3f}")

    # Gate verdict.
    print("\n--- gate ---")
    auc = overall["auc"]
    late_auc = by_turn[250].get("auc", float("nan"))
    if auc >= 0.85:
        print(f"GREEN. Overall AUC={auc:.3f} ≥ 0.85; wire into agent.")
    elif auc >= 0.75:
        print(f"YELLOW. Overall AUC={auc:.3f} ≥ 0.75; wire it, but expect tuning.")
    elif auc >= 0.65:
        print(f"ORANGE. Overall AUC={auc:.3f} < 0.75; add F3 defensibility, re-validate.")
    else:
        print(f"RED. Overall AUC={auc:.3f} < 0.65; framework may be wrong, escalate.")
    if not (late_auc != late_auc):  # not NaN
        print(f"  late-game (turn 250) AUC={late_auc:.3f} (target ≥ 0.85)")

    return {"overall": overall, "by_turn": by_turn, "by_matchup": by_matchup}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=25, help="games per matchup (default 25 → 100 total)")
    ap.add_argument("-w", "--workers", type=int, default=4)
    ap.add_argument(
        "--replay",
        action="store_true",
        help="skip game generation; re-score the saved states only",
    )
    args = ap.parse_args(argv)

    if not args.replay:
        print(
            f"--- generating states: {len(DEFAULT_MATCHUPS)} matchups × "
            f"{args.n} games = {len(DEFAULT_MATCHUPS) * args.n} games ---"
        )
        for a, b in DEFAULT_MATCHUPS:
            print(f"    {a} vs {b}")
        n = generate_states(args.n, args.workers)
        if n == 0:
            print("ERROR: no states generated.")
            return 1

    scored = score_snapshots()
    report(scored)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
