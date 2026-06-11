"""scripts/knob_tune.py — joint knob tuning on the paired-margin objective.

The live stack's constants were each set by a single triage and never tuned
jointly (reactive-floor weight 0.5 / lag 2, enemy overkill, veto margin,
horizon). This runs a small cross-entropy search: sample knob vectors from
a Gaussian, build a bundle per candidate (bundler --set), score it with the
fast 2P margin harness against the namespaced live-stack referee, refit the
Gaussian to the elites, repeat. Seeds rotate per generation so the tune
can't overfit one map set.

Objective per candidate: mean of (paired share lead @120) over seeds, plus
0.25 * (win rate - 0.5). Margins carry the information (margin_ab docs);
the win-rate term breaks ties toward conversion.

Run overnight, ONE instance, nothing else running (measurement discipline):
    python scripts/knob_tune.py --generations 5 --pop 6 --seeds-per-eval 3 \
        --referee submissions/_ns_veto_rf.py
Results: audit/tune/knob_tune_<utc>.jsonl (one line per eval, best-so-far
summary at the end).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import random
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# knob -> (env var, low, high, is_int)
KNOBS = {
    "rf_weight": ("PRODUCER_PLUS_REACTIVE_FLOOR", 0.2, 1.2, False),
    "rf_lag": ("PRODUCER_PLUS_REACTIVE_FLOOR_LAG", 0.5, 4.0, False),
    "overkill_enemy": ("PRODUCER_PLUS_OVERKILL_FACTOR_ENEMY", 1.0, 4.0, False),
    "veto_margin": ("PRODUCER_PLUS_RESPONSE_VETO_MARGIN", 0.0, 6.0, False),
    "horizon_2p": ("PRODUCER_PLUS_HORIZON_2P", 14, 24, True),
}

# The stack being tuned = the live 2P stack (veto + reactive floor).
BASE_VARIANT = "veto_rf"

_PAIRED_RE = re.compile(r"lead@120: .*paired-mean=\s*([+-][\d.]+)%")
_WINS_RE = re.compile(r"focal_wins=(\d+)/(\d+)")


def build_bundle(env_sets: dict[str, str], out: Path) -> None:
    cmd = [sys.executable, str(REPO / "scripts" / "bundle_producer_plus.py"),
           "--variant", BASE_VARIANT, "--out", str(out)]
    for k, v in env_sets.items():
        cmd += ["--set", f"{k}={v}"]
    subprocess.run(cmd, check=True, capture_output=True, text=True)


def evaluate(bundle: Path, referee: Path, seed0: int, n_seeds: int,
             max_steps: int, workers: int) -> dict:
    cmd = [sys.executable, str(REPO / "scripts" / "margin_ab.py"),
           str(bundle), str(referee),
           "--seeds", str(n_seeds), "--seed-start", str(seed0),
           "--workers", str(workers), "--max-steps", str(max_steps)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
    out = r.stdout
    m_pair = _PAIRED_RE.search(out)
    m_wins = _WINS_RE.search(out)
    if not (m_pair and m_wins):
        return {"objective": -1e9, "error": out[-300:] + (r.stderr or "")[-200:]}
    paired120 = float(m_pair.group(1)) / 100.0
    w, n = int(m_wins.group(1)), int(m_wins.group(2))
    winrate = w / n if n else 0.0
    return {"objective": paired120 + 0.25 * (winrate - 0.5),
            "paired120": paired120, "wins": f"{w}/{n}"}


def _git_push_log(log_path: Path) -> None:
    """Commit + push the tune log so progress survives container restarts.

    Best-effort: any git failure (lock contention with the main session,
    transient network) is swallowed — the next eval retries.
    """
    try:
        subprocess.run(["git", "add", str(log_path)], cwd=REPO,
                       capture_output=True, timeout=60)
        subprocess.run(
            ["git", "-c", "user.email=noreply@anthropic.com",
             "-c", "user.name=Claude", "commit", "-q", "-m",
             "audit: knob-tune progress (auto)"],
            cwd=REPO, capture_output=True, timeout=60,
        )
        r = subprocess.run(
            ["git", "push", "-q", "origin", "HEAD"],
            cwd=REPO, capture_output=True, timeout=120,
        )
        if r.returncode != 0:
            subprocess.run(["git", "pull", "--rebase", "-q", "origin",
                            "claude/awesome-clarke-ixy57v"],
                           cwd=REPO, capture_output=True, timeout=120)
            subprocess.run(["git", "push", "-q", "origin", "HEAD"],
                           cwd=REPO, capture_output=True, timeout=120)
    except Exception:
        pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--referee", default="submissions/_ns_veto_rf.py")
    ap.add_argument("--generations", type=int, default=5)
    ap.add_argument("--pop", type=int, default=6)
    ap.add_argument("--elite", type=int, default=2)
    ap.add_argument("--seeds-per-eval", type=int, default=3)
    ap.add_argument("--max-steps", type=int, default=150)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--rng", type=int, default=0)
    ap.add_argument("--resume", default=None,
                    help="existing jsonl: completed (gen,cand) evals are "
                    "replayed from the log instead of re-run; the rng stream "
                    "stays aligned because candidates are resampled in the "
                    "same order either way")
    args = ap.parse_args()

    rng = random.Random(args.rng)
    referee = REPO / args.referee
    log_dir = REPO / "audit" / "tune"
    log_dir.mkdir(parents=True, exist_ok=True)
    done: dict[tuple[int, int], dict] = {}
    if args.resume:
        log_path = Path(args.resume)
        if log_path.exists():
            for line in log_path.open():
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "objective" in rec and rec["objective"] > -1e8:
                    done[(rec["gen"], rec["cand"])] = rec
        print(f"resuming: {len(done)} completed evals replayed from {log_path}",
              flush=True)
    else:
        stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        log_path = log_dir / f"knob_tune_{stamp}.jsonl"
    log = log_path.open("a")
    tmp = REPO / "submissions" / "_tune_candidate.py"

    names = list(KNOBS)
    # Init distribution centred on the shipped values.
    shipped = {"rf_weight": 0.5, "rf_lag": 2.0, "overkill_enemy": 1.3,
               "veto_margin": 1.5, "horizon_2p": 18}
    mu = {k: float(shipped[k]) for k in names}
    sigma = {k: (KNOBS[k][2] - KNOBS[k][1]) / 4.0 for k in names}

    best = {"objective": -1e9, "knobs": dict(mu)}
    seed0 = 0
    for gen in range(args.generations):
        scored = []
        for i in range(args.pop):
            knobs = {}
            for k in names:
                env, lo, hi, is_int = KNOBS[k]
                v = min(hi, max(lo, rng.gauss(mu[k], sigma[k])))
                knobs[k] = int(round(v)) if is_int else round(v, 3)
            if (gen, i) in done:
                rec = done[(gen, i)]
                knobs = rec["knobs"]
                res = {"objective": rec["objective"],
                       "wins": rec.get("wins", "")}
                print(f"gen {gen} cand {i}: obj={res['objective']:+.3f} "
                      f"{res.get('wins', '')} knobs={knobs} (replayed)",
                      flush=True)
                scored.append((res["objective"], knobs))
                if res["objective"] > best["objective"]:
                    best = {"objective": res["objective"], "knobs": knobs}
                continue
            env_sets = {KNOBS[k][0]: str(knobs[k]) for k in names}
            build_bundle(env_sets, tmp)
            res = evaluate(tmp, referee, seed0, args.seeds_per_eval,
                           args.max_steps, args.workers)
            rec = {"gen": gen, "cand": i, "seed0": seed0, "knobs": knobs, **res}
            log.write(json.dumps(rec) + "\n"); log.flush()
            _git_push_log(log_path)
            print(f"gen {gen} cand {i}: obj={res['objective']:+.3f} "
                  f"{res.get('wins', '')} knobs={knobs}", flush=True)
            scored.append((res["objective"], knobs))
            if res["objective"] > best["objective"]:
                best = {"objective": res["objective"], "knobs": knobs}
        scored.sort(key=lambda t: t[0], reverse=True)
        elites = [k for _o, k in scored[: args.elite]]
        for k in names:
            vals = [e[k] for e in elites]
            mu[k] = sum(vals) / len(vals)
            var = sum((v - mu[k]) ** 2 for v in vals) / max(1, len(vals) - 1)
            sigma[k] = max(math.sqrt(var), sigma[k] * 0.5)   # floor: keep exploring
        seed0 += args.seeds_per_eval * 2          # rotate maps each generation
        print(f"== gen {gen} done. mu={mu} best={best}", flush=True)

    print(f"\nBEST: obj={best['objective']:+.3f} knobs={best['knobs']}")
    print("Confirm the winner with a full n>=32 A/B before any submission "
          "(Rule 45; the tune objective is a triage signal).")


if __name__ == "__main__":
    main()
