"""Continuous-score wide-map A/B for the smart-dropout line.

WHY THIS EXISTS
---------------
The DROPOUT_PLAN A/B table ranks variants by binary win/loss over 28 maps
(base 15/28, incentive 13/28, ...). Binary outcomes throw away most of the
signal: a map won 23786-to-456 and a map won 510-to-509 both count "1", so a
1-2 map swing between variants is indistinguishable from noise and needs huge n.

This harness scores each game by the CONTINUOUS ship-margin instead — the exact
quantity the engine argmaxes to decide the winner, normalised to [-1, 1]:

    margin = (focal_ships - best_rival_ships) / (focal_ships + best_rival_ships)

Its sign reproduces win/loss; its magnitude measures dominance. Crucially, every
variant is run on the SAME seed+seat as `base`, so we can PAIR by map and report
the mean per-map margin difference  Δ = margin_variant - margin_base  with the
map-to-map variance differenced out. That paired Δ has far lower variance than
the difference of two win-rate proportions, so a real effect shows up at n≈32
where the binary table only saw 13 vs 15.

METHODOLOGY (DROPOUT_PLAN "EVALUATION" section)
-----------------------------------------------
- ONE game per FRESH SUBPROCESS — producer_plus bundles set knobs via
  os.environ.setdefault, which leak across variants in one process. Each game is
  a clean `_continuous_game_worker.py` invocation with the variant's knobs set
  as real env vars.
- Outcome is MAP-determined and seat-invariant — seat is rotated ACROSS seeds
  (seat = seed_index % players), never within a seed. base and every variant
  share each seed+seat, giving a valid paired diff.
- Diverse wide maps, one game per seed, vs Producer V2 (the discriminating peer).

USAGE
-----
    python scripts/continuous_ab.py --run                 # run + report
    python scripts/continuous_ab.py --run --seeds 48      # n=48 maps
    python scripts/continuous_ab.py --report-only LOG     # re-analyse a JSONL
    python scripts/continuous_ab.py --run --variants base,incentive
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
WORKER = REPO / "scripts" / "_continuous_game_worker.py"
V2 = REPO / "audit" / "external" / "agents" / "slawekbiel_the-producer-v2" / "main.py"

# dropout_repl base (the DROPOUT_PLAN A/B baseline) — dropout REPLACING the
# opponent model: multi_size + reactive_floor + FFA(uniform) + dropout.
_DROP_BASE = {
    "PRODUCER_PLUS_MULTI_SIZE": "1",
    "PRODUCER_PLUS_REACTIVE_FLOOR": "0.5",
    "PRODUCER_PLUS_FFA_SCORE": "1",
    "PRODUCER_PLUS_FFA_WEIGHTS": "uniform",
    "PRODUCER_PLUS_DROPOUT": "1",
}


def _v(base, **extra):
    d = dict(base)
    d.update({k: str(v) for k, v in extra.items()})
    return d


# least_resistance (the LIVE champion). The refuted default-OFF levers are
# pinned OFF (matching scripts/verify_confirm.py) so OFF/ON isolates exactly the
# shipped take-and-hold pair. "off" = pre-take-and-hold shipped agent; "champion"
# = the live default (LR_HOLD_MARGIN=0.5, LR_DEFEND=1). Re-measuring the KNOWN
# +7/32 take-and-hold lift in MARGIN space is a calibration probe (Rule 45 exempt)
# AND maps the champion's remaining loss landscape vs V2 (the headroom).
_LR_OFF = {
    "LR_LEADER_RELATIVE_4P": "0", "LR_VALUE_COMMIT": "0", "LR_ANYTIME": "0",
    "LR_ENEMY_BOOST": "1.0", "LR_ROLLOUT_DEPTH": "0",
}


# A variant-set bundles a focal agent with its named knob variants and the
# pairing reference. Select with --set.
VARIANT_SETS = {
    "dropout": {
        "focal": REPO / "agents" / "producer_plus" / "main.py",
        "ref": "base",
        "variants": {
            "base":         _v(_DROP_BASE),
            # Dropout-NATIVE Phase A: same base candidate stack, but the mean-
            # field flip-hazard ownership value RANKS the candidates instead of
            # the bolt-on reflip. DROPOUT off (native replaces it). KILL-GATE:
            # must beat base on the paired continuous margin.
            "native":       _v({k: v for k, v in _DROP_BASE.items()
                                if k != "PRODUCER_PLUS_DROPOUT"},
                               PRODUCER_PLUS_NATIVE_HAZARD=1),
            "native_s8":    _v({k: v for k, v in _DROP_BASE.items()
                                if k != "PRODUCER_PLUS_DROPOUT"},
                               PRODUCER_PLUS_NATIVE_HAZARD=1,
                               PRODUCER_PLUS_NATIVE_STEEPNESS=8.0),
            # Self-consistency: concentrated adversary (opponent commits mass to
            # each candidate's single worst planet) -> threat is candidate-
            # dependent, so defending a weak spot can reorder the ranking.
            "native_sc":    _v({k: v for k, v in _DROP_BASE.items()
                                if k != "PRODUCER_PLUS_DROPOUT"},
                               PRODUCER_PLUS_NATIVE_HAZARD=1,
                               PRODUCER_PLUS_NATIVE_SELFCONSIST=1),
            # ship-margin λ (production credit) + steepness sweep.
            "native_t6":    _v({k: v for k, v in _DROP_BASE.items()
                                if k != "PRODUCER_PLUS_DROPOUT"},
                               PRODUCER_PLUS_NATIVE_HAZARD=1,
                               PRODUCER_PLUS_NATIVE_TERMINAL=6),
            "native_t24":   _v({k: v for k, v in _DROP_BASE.items()
                                if k != "PRODUCER_PLUS_DROPOUT"},
                               PRODUCER_PLUS_NATIVE_HAZARD=1,
                               PRODUCER_PLUS_NATIVE_TERMINAL=24),
            "native_st8":   _v({k: v for k, v in _DROP_BASE.items()
                                if k != "PRODUCER_PLUS_DROPOUT"},
                               PRODUCER_PLUS_NATIVE_HAZARD=1,
                               PRODUCER_PLUS_NATIVE_STEEPNESS=8),
            "native_st3":   _v({k: v for k, v in _DROP_BASE.items()
                                if k != "PRODUCER_PLUS_DROPOUT"},
                               PRODUCER_PLUS_NATIVE_HAZARD=1,
                               PRODUCER_PLUS_NATIVE_STEEPNESS=3),
            # force concentration: multiple coordinated waves per target, to crack
            # defended high-value (corner) neutrals a single source can't take.
            "native_fc":    _v({k: v for k, v in _DROP_BASE.items()
                                if k != "PRODUCER_PLUS_DROPOUT"},
                               PRODUCER_PLUS_NATIVE_HAZARD=1,
                               PRODUCER_PLUS_FORCE_CONCENTRATION=1),
            # anticipatory threat: grow enemy reservoir by opp production over the
            # horizon (alpha bracket), to hold the frontier vs the mid-game army.
            "native_grow":  _v({k: v for k, v in _DROP_BASE.items()
                                if k != "PRODUCER_PLUS_DROPOUT"},
                               PRODUCER_PLUS_NATIVE_HAZARD=1,
                               PRODUCER_PLUS_NATIVE_THREAT_GROWTH=0.5),
            "native_grow25": _v({k: v for k, v in _DROP_BASE.items()
                                 if k != "PRODUCER_PLUS_DROPOUT"},
                                PRODUCER_PLUS_NATIVE_HAZARD=1,
                                PRODUCER_PLUS_NATIVE_THREAT_GROWTH=0.25),
            "native_grow10": _v({k: v for k, v in _DROP_BASE.items()
                                 if k != "PRODUCER_PLUS_DROPOUT"},
                                PRODUCER_PLUS_NATIVE_HAZARD=1,
                                PRODUCER_PLUS_NATIVE_THREAT_GROWTH=1.0),
            # Bracket the hazard steepness to test whether the flip-hazard term
            # is load-bearing at all (s0.5 ~ flat hazard ~ pure ownership margin;
            # s20 ~ hard contest). If all three ~equal, the hazard is inert and
            # Phase A is really testing ownership-margin vs the tuned scorer.
            "native_s0.5":  _v({k: v for k, v in _DROP_BASE.items()
                                if k != "PRODUCER_PLUS_DROPOUT"},
                               PRODUCER_PLUS_NATIVE_HAZARD=1,
                               PRODUCER_PLUS_NATIVE_STEEPNESS=0.5),
            "native_s20":   _v({k: v for k, v in _DROP_BASE.items()
                                if k != "PRODUCER_PLUS_DROPOUT"},
                               PRODUCER_PLUS_NATIVE_HAZARD=1,
                               PRODUCER_PLUS_NATIVE_STEEPNESS=20.0),
            "more_sims4":   _v(_DROP_BASE, PRODUCER_PLUS_DROPOUT_SCENARIOS=4),
            "incentive":    _v(_DROP_BASE, PRODUCER_PLUS_DROPOUT_INCENTIVE=1),
            "winprob_g0.5": _v(_DROP_BASE, PRODUCER_PLUS_DROPOUT_WINPROB=0.5),
            "winprob_g1.0": _v(_DROP_BASE, PRODUCER_PLUS_DROPOUT_WINPROB=1.0),
            "deeper_h30":   _v(_DROP_BASE, PRODUCER_PLUS_HORIZON_2P=30),
        },
    },
    "champion": {
        "focal": REPO / "agents" / "least_resistance" / "main.py",
        "ref": "off",
        "variants": {
            "off":      _v(_LR_OFF, LR_HOLD_MARGIN="0.0", LR_DEFEND="0"),
            "champion": _v(_LR_OFF, LR_HOLD_MARGIN="0.5", LR_DEFEND="1"),
        },
    },
}


# --------------------------------------------------------------------------
# stats
# --------------------------------------------------------------------------

def wilson_ci(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = wins / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def mean_ci(xs: list[float], z: float = 1.96) -> tuple[float, float, float]:
    """Mean and normal-approx CI half-width for a sample. Returns (mean, lo, hi)."""
    n = len(xs)
    if n == 0:
        return (0.0, 0.0, 0.0)
    m = sum(xs) / n
    if n < 2:
        return (m, m, m)
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    se = math.sqrt(var / n)
    return (m, m - z * se, m + z * se)


def bootstrap_ci(xs: list[float], iters: int = 10000, seed: int = 12345
                 ) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean (paired Δ is just a 1-sample mean)."""
    n = len(xs)
    if n == 0:
        return (0.0, 0.0)
    rng = random.Random(seed)
    means = []
    for _ in range(iters):
        s = 0.0
        for _ in range(n):
            s += xs[rng.randrange(n)]
        means.append(s / n)
    means.sort()
    lo = means[int(0.025 * iters)]
    hi = means[int(0.975 * iters)]
    return (lo, hi)


def sign_test_p(wins: int, losses: int) -> float:
    """Two-sided exact sign test p-value (ties excluded)."""
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    # P(X<=k) + P(X>=n-k) under Binom(n, 0.5), two-sided.
    cum = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    p = 2 * cum
    return min(1.0, p)


# --------------------------------------------------------------------------
# run
# --------------------------------------------------------------------------

def _play(variant: str, knobs: dict, seed: int, seat: int, players: int,
          opps: list[str], focal: str) -> dict:
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    cmd = [
        sys.executable, str(WORKER),
        "--seed", str(seed), "--focal-seat", str(seat),
        "--players", str(players),
        "--focal", str(focal),
        "--opps", ",".join(opps),
        "--knobs", json.dumps(knobs),
    ]
    try:
        out = subprocess.run(cmd, env=env, capture_output=True, text=True,
                             timeout=600)
    except subprocess.TimeoutExpired:
        return {"variant": variant, "seed": seed, "seat": seat,
                "error": "timeout"}
    rec = None
    for line in out.stdout.splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
    if rec is None:
        return {"variant": variant, "seed": seed, "seat": seat,
                "error": "no-json", "stderr": out.stderr[-200:]}
    rec["variant"] = variant
    return rec


def load_done(log: Path) -> set[tuple[str, int]]:
    done = set()
    if log.is_file():
        for line in log.read_text().splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "variant" in r and "seed" in r and "error" not in r:
                done.add((r["variant"], int(r["seed"])))
    return done


def run(args) -> Path:
    seeds = list(range(args.seed_start, args.seed_start + args.seeds))
    players = args.players
    opps = [str(V2)]
    if players == 4:
        EXT = REPO / "audit" / "external" / "agents"
        opps = [str(V2),
                str(EXT / "romantamrazov_orbit-star-wars-lb-max-1224" / "main.py"),
                str(EXT / "konbu17_orbit-wars-rule-base-ml-shot-validator-hybrid" / "main.py")]

    vset = VARIANT_SETS[args.set]
    VARIANTS = vset["variants"]
    focal = str(args.focal) if args.focal else str(vset["focal"])
    variants = (args.variants.split(",") if args.variants
                else list(VARIANTS.keys()))
    for v in variants:
        if v not in VARIANTS:
            raise SystemExit("unknown variant %r in set %r; known: %s"
                             % (v, args.set, ", ".join(VARIANTS)))

    log = args.log or (REPO / "audit" / ("continuous-ab-%s.jsonl"
            % datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")))
    log = Path(log)
    log.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(log)

    # Build the task list (paired: same seed+seat across all variants).
    tasks = []
    for i, seed in enumerate(seeds):
        seat = i % players
        for v in variants:
            if (v, seed) in done:
                continue
            tasks.append((v, VARIANTS[v], seed, seat))

    total = len(tasks)
    header = ("# continuous-ab  %s  | %dP vs %s | seeds %d..%d (n=%d) | "
              "variants=%s | %d games to run (%d already done)"
              % (datetime.now(timezone.utc).isoformat(), players,
                 ",".join(Path(o).parent.name for o in opps),
                 seeds[0], seeds[-1], len(seeds), ",".join(variants),
                 total, len(done) and len(done)))
    print(header, flush=True)
    with open(log, "a") as f:
        f.write(header + "\n")

    t0 = time.perf_counter()
    completed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_play, v, knobs, seed, seat, players, opps, focal):
                (v, seed) for (v, knobs, seed, seat) in tasks}
        for fut in as_completed(futs):
            rec = fut.result()
            completed += 1
            with open(log, "a") as f:
                f.write(json.dumps(rec) + "\n")
            tag = ("ERR " + rec.get("error", "")) if "error" in rec else (
                "WIN " if rec.get("win") else "loss")
            el = time.perf_counter() - t0
            rate = completed / el if el > 0 else 0
            eta = (total - completed) / rate if rate > 0 else 0
            print("  [%3d/%3d] %-13s seed=%d seat=%d  %s  margin=%+.3f  "
                  "steps=%s max_ms=%s  (eta %.0fs)"
                  % (completed, total, rec["variant"], rec["seed"],
                     rec.get("seat", -1), tag, rec.get("margin", 0.0),
                     rec.get("steps", "?"), rec.get("max_ms", "?"), eta),
                  flush=True)
    print("# run done in %.0fs -> %s" % (time.perf_counter() - t0, log))
    return log


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------

def report(log: Path, ref: str | None = None, landscape: str | None = None) -> None:
    rows = []
    seen_order: list[str] = []
    for line in Path(log).read_text().splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "variant" in r and "seed" in r:
            rows.append(r)
            if r["variant"] not in seen_order:
                seen_order.append(r["variant"])

    # variant -> seed -> record (last wins on dup)
    by_var: dict[str, dict[int, dict]] = {}
    errors: list[dict] = []
    for r in rows:
        if "error" in r:
            errors.append(r)
            continue
        by_var.setdefault(r["variant"], {})[int(r["seed"])] = r

    order = [v for v in seen_order if v in by_var]
    # pairing reference: explicit --ref, else "base"/"off" if present, else first.
    if ref is None:
        ref = next((c for c in ("base", "off") if c in by_var), order[0] if order else None)
    if ref not in by_var:
        print("ref %r not in log — cannot pair." % ref)
    base = by_var.get(ref, {})
    print("(pairing reference = %r)" % ref)

    pl5 = next((g for g in by_var.values() if g), {})
    opp_hint = "Producer V2" if not pl5 or next(iter(pl5.values())).get(
        "players", 2) == 2 else "the panel"
    print("\n" + "=" * 92)
    print("CONTINUOUS-SCORE A/B  (margin = (focal_ships - rival_ships)/"
          "(focal_ships + rival_ships) in [-1,1]; sign = win)")
    print("vs %s" % opp_hint)
    print("=" * 92)
    print("%-14s %5s | %-15s | %-22s | %-26s"
          % ("variant", "n", "wins (Wilson lo-hi)", "mean margin [95% CI]",
             "paired vs base  Δmargin"))
    print("-" * 92)

    for v in order:
        games = by_var[v]
        seeds = sorted(games)
        margins = [games[s]["margin"] for s in seeds]
        wins = sum(1 for s in seeds if games[s]["win"])
        n = len(seeds)
        wlo, whi = wilson_ci(wins, n)
        mm, mlo, mhi = mean_ci(margins)

        paired = ""
        if v != ref and base:
            # Pair only seeds played at the SAME seat in both (guards against a
            # resumed log with a shifted seed window pairing seat-mismatched
            # games into an invalid Δ).
            common = [s for s in seeds if s in base
                      and games[s].get("seat") == base[s].get("seat")]
            deltas = [games[s]["margin"] - base[s]["margin"] for s in common]
            up = sum(1 for d in deltas if d > 1e-9)
            dn = sum(1 for d in deltas if d < -1e-9)
            dmean, dlo, dhi = mean_ci(deltas)
            blo, bhi = bootstrap_ci(deltas)
            p = sign_test_p(up, dn)
            sig = "  *" if (dlo > 0 or dhi < 0) else ""
            paired = ("%+.3f [%+.3f,%+.3f] n=%d up/dn=%d/%d p=%.2f%s"
                      % (dmean, blo, bhi, len(common), up, dn, p, sig))
        elif v == ref:
            paired = "(reference)"

        print("%-14s %5d | %3d/%-3d (%.2f-%.2f) | %+.3f [%+.3f,%+.3f]  | %s"
              % (v, n, wins, n, wlo, whi, mm, mlo, mhi, paired))

    if errors:
        print("\n%d errored games:" % len(errors))
        seen = set()
        for e in errors[:12]:
            key = (e["variant"], e.get("error", "")[:40])
            if key in seen:
                continue
            seen.add(key)
            print("   %-13s seed=%s  %s" % (e["variant"], e.get("seed"),
                                            e.get("error", "")[:80]))

    if landscape and landscape in by_var:
        games = by_var[landscape]
        items = sorted(games.items(), key=lambda kv: kv[1]["margin"])
        losses = [(s, r) for s, r in items if not r["win"]]
        close = [(s, r) for s, r in items if r["win"] and r["margin"] < 0.5]
        print("\nLOSS LANDSCAPE for %r (the headroom): %d losses, %d close wins "
              "(margin<0.5) of %d maps" % (landscape, len(losses), len(close),
                                           len(items)))
        print("  losing maps (margin asc) — these are where the work is:")
        for s, r in losses:
            print("    seed=%-6d margin=%+.3f  scores=%s steps=%d"
                  % (s, r["margin"], r.get("scores"), r.get("steps", 0)))
        if close:
            print("  close wins (could flip):")
            for s, r in close:
                print("    seed=%-6d margin=%+.3f  steps=%d"
                      % (s, r["margin"], r.get("steps", 0)))

    print("\nReading the paired column:")
    print("  Δmargin>0 = variant dominates base on the same maps; [lo,hi] is the")
    print("  bootstrap 95%% CI on the mean per-map margin shift; up/dn = maps the")
    print("  variant beat / lost to base on margin; '*' = CI excludes 0 (real).")
    print("  Win-rate Wilson CI is the OLD coarse signal, shown for continuity.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--report-only", default=None, metavar="JSONL")
    ap.add_argument("--set", default="dropout", choices=list(VARIANT_SETS),
                    help="variant set / focal agent (default: dropout)")
    ap.add_argument("--focal", default=None,
                    help="override the set's focal agent path")
    ap.add_argument("--ref", default=None,
                    help="pairing reference variant (default: set's ref)")
    ap.add_argument("--seeds", type=int, default=40, help="number of maps (n)")
    ap.add_argument("--seed-start", type=int, default=5000)
    ap.add_argument("--players", type=int, default=2, choices=(2, 4))
    ap.add_argument("--variants", default=None,
                    help="comma list (default: all in the set)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--log", default=None)
    ap.add_argument("--landscape", default=None,
                    help="print the per-map loss landscape for this variant "
                         "(the headroom view); e.g. --landscape champion")
    args = ap.parse_args()

    ref = args.ref or VARIANT_SETS[args.set].get("ref")
    if args.report_only:
        report(Path(args.report_only), ref=ref, landscape=args.landscape)
        return 0
    if args.run:
        log = run(args)
        report(log, ref=ref, landscape=args.landscape)
        return 0
    ap.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
