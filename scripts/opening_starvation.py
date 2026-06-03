"""scripts/opening_starvation.py — does the horizon-K filter starve our OPENING?

Hypothesis (PI, seed 722289020 replay): on SPARSE maps (planets ringed around the
perimeter, big central sun forcing long arc-paths) the nearest neutral from a
starting planet is 20-30+ turns away. The proposer hard-drops any launch with
arrival ETA > K (proposer.py:1162  `if eta > _k_tgt: continue`), there is NO
fallback, so when every reachable neutral sits beyond K the agent generates ZERO
candidates and sits IDLE while the opponent expands.

The shipped champion runs adaptive-K (K_OPEN=20 -> floor 10 by step 30). But the
adaptive-K audit itself notes the OPENING median neutral ETA is ~22 — i.e. K_OPEN=20
is BELOW the median, so even the fix under-reaches on the sparse tail.

This diagnostic measures, on REAL opening boards (steps 0..WINDOW), from the focal
seat:
  * usable sources (own, ships >= MIN_FLEET_SIZE)
  * global-min neutral ETA (the single easiest expansion move available)
  * reach@K = (source, neutral) pairs whose full-budget ETA <= K, for K in
    {10, 20, 30, 40}  (10=static floor, 20=adaptive K_OPEN, 30/40=headroom)
  * propose_count = candidates the LIVE agent actually gets post-K-filter
  * starved = usable_sources>0 AND propose_count==0  (forced idle)

The opening board is essentially a function of the MAP (seed), not the opponent
(no contact yet), so seed 722289020's opening starvation reproduces locally
regardless of the local opponent.

Usage:
  PYTHONPATH=. python scripts/opening_starvation.py <opp> <seeds csv> [window] [trace_seed]
  e.g. PYTHONPATH=. python scripts/opening_starvation.py submissions/baseline.py 722289020 30 722289020
"""
from __future__ import annotations

import math
import os
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# Live champion config (the "similar prior agent" that played 722289020): champion
# + adaptive horizon K. We measure starvation UNDER this real config.
for k, v in {
    "BASELINE_JOINT_AGGR": "1", "BASELINE_JOINT_TOP_K": "5",
    "BASELINE_JOINT_MAX_PAIRS": "60", "BASELINE_REINFORCE_EMIT": "1",
    "BASELINE_REINFORCE_ANTICIPATE": "1", "BASELINE_NEUTRAL_BONUS": "2.0",
    "BASELINE_NEUTRAL_EARLY_EXTRA": "1.5", "BASELINE_NEUTRAL_EARLY_HORIZON": "50",
    "BASELINE_ORBITAL_SAFETY": "1", "BASELINE_PV_ETA": "1",
    "BASELINE_LAUNCH_RULES": "1", "BASELINE_CAPTURE_HORIZON_K": "10",
    "BASELINE_VALUE_HEAD": "hybrid", "BASELINE_JOINT": "1",
    "BASELINE_CHOOSER": "trajectory", "BASELINE_ADAPTIVE_K": "1",
    "BASELINE_KINEMATIC_TABLE": "1",
}.items():
    os.environ.setdefault(k, v)

from kaggle_environments import make  # noqa: E402
from lib.intent import World  # noqa: E402
from lib.world_model import WorldModel  # noqa: E402
from agents.baseline.proposer import (  # noqa: E402
    MIN_FLEET_SIZE, NUM_TARGETS_PER_SOURCE, aim_and_eta, nearest_k, propose,
)
from agents.baseline.launch_rules import capture_horizon_k  # noqa: E402

try:
    from agents.baseline.main import KinematicTable  # noqa: E402
except Exception:  # pragma: no cover
    KinematicTable = None

FOCAL = str(REPO / "agents/baseline/main.py")
OPP = sys.argv[1] if len(sys.argv) > 1 else str(REPO / "submissions/baseline.py")


def _safe_int(s, default=0):
    try:
        return int(s)
    except (ValueError, TypeError):
        return default


SEEDS = [_safe_int(s) for s in (sys.argv[2].split(",")
         if len(sys.argv) > 2 and not sys.argv[2].startswith("-") else ["0"])]
WINDOW = _safe_int(sys.argv[3], 30) if len(sys.argv) > 3 else 30
TRACE_SEED = _safe_int(sys.argv[4], SEEDS[0] if SEEDS else 0) if len(sys.argv) > 4 \
    else (SEEDS[0] if SEEDS else 0)
# Optional cheap step-0 sparsity scan over a seed RANGE: --scan START COUNT
SCAN = None
if "--scan" in sys.argv:
    i = sys.argv.index("--scan")
    SCAN = (int(sys.argv[i + 1]), int(sys.argv[i + 2]))
K_GRID = (10, 20, 30, 40)


def _is_neutral(p, me):
    return int(p.owner) == -1


def reach_only(obs, me, step):
    """Cheap step-0/board reach metrics (no propose, no model) — map sparsity."""
    world = World.from_obs(obs)
    omega = float(obs.get("angular_velocity", 0.0))
    planets = list(world.planets_by_id.values())
    usable = [p for p in planets if int(p.owner) == me and int(p.ships) >= MIN_FLEET_SIZE]
    neutrals = [p for p in planets if _is_neutral(p, me)]
    reach = {k: 0 for k in K_GRID}
    min_eta = math.inf
    for src in usable:
        budget = int(src.ships)
        for tgt in nearest_k(neutrals, src, NUM_TARGETS_PER_SOURCE):
            if int(tgt.id) == int(src.id):
                continue
            _ang, eta = aim_and_eta(src, tgt, budget, omega, world=world)
            eta = int(eta)
            min_eta = min(min_eta, eta)
            for k in K_GRID:
                if eta <= k:
                    reach[k] += 1
    return {"usable": len(usable), "neutrals": len(neutrals), "reach": reach,
            "min_eta": min_eta if min_eta < math.inf else None,
            "k_live": capture_horizon_k(step)}


def board_metrics(obs, me, step):
    """Compute opening-starvation metrics for one board from the focal seat."""
    world = World.from_obs(obs)
    if KinematicTable is not None:
        world._kt = KinematicTable()
        world._kt.begin_turn(world)
    model = WorldModel.from_world(world)
    omega = float(obs.get("angular_velocity", 0.0))

    planets = list(world.planets_by_id.values())
    my_planets = [p for p in planets if int(p.owner) == me]
    usable = [p for p in my_planets if int(p.ships) >= MIN_FLEET_SIZE]
    neutrals = [p for p in planets if _is_neutral(p, me)]

    # reach@K: (source, neutral) pairs reachable within K at FULL-budget fleet
    # (largest fleet = fastest = shortest ETA = best-case reachability), mirroring
    # the proposer's nearest_k(target_pool, src, 8) enumeration but neutral-only.
    reach = {k: 0 for k in K_GRID}
    min_eta = math.inf
    etas = []  # per-source nearest-neutral eta
    for src in usable:
        budget = int(src.ships)
        best_src = math.inf
        for tgt in nearest_k(neutrals, src, NUM_TARGETS_PER_SOURCE):
            if int(tgt.id) == int(src.id):
                continue
            _ang, eta = aim_and_eta(src, tgt, budget, omega, world=world)
            eta = int(eta)
            best_src = min(best_src, eta)
            for k in K_GRID:
                if eta <= k:
                    reach[k] += 1
        if best_src < math.inf:
            etas.append(best_src)
            min_eta = min(min_eta, best_src)

    # What the LIVE agent actually gets to choose from (post-K-filter).
    other_planets = [p for p in planets if int(p.owner) != me]
    threatened_mine = [
        p for p in my_planets
        if model.time_to_enemy_threat(int(p.id), me, world) is not None
    ]
    target_pool = other_planets + threatened_mine
    try:
        prerank = propose(my_planets, target_pool, world, model, me, omega,
                          baseline_len=0)
        propose_count = len(prerank)
    except Exception:
        propose_count = -1  # error sentinel

    k_live = capture_horizon_k(step)
    starved = len(usable) > 0 and propose_count == 0
    return {
        "usable": len(usable), "neutrals": len(neutrals), "reach": reach,
        "min_eta": min_eta if min_eta < math.inf else None,
        "median_src_eta": (sorted(etas)[len(etas) // 2] if etas else None),
        "k_live": k_live, "propose_count": propose_count, "starved": starved,
    }


def run_seed(seed, focal_p0, window):
    p0, p1 = (FOCAL, OPP) if focal_p0 else (OPP, FOCAL)
    me = 0 if focal_p0 else 1
    env = make("orbit_wars", configuration={"seed": seed}, debug=False)
    env.run([p0, p1])
    rows = []
    for t, step in enumerate(env.steps):
        if t > window:
            break
        obs = step[me].get("observation")
        if not obs or not obs.get("planets"):
            continue
        action = step[me].get("action")
        n_launch = len(action) if isinstance(action, list) else 0
        try:
            m = board_metrics(obs, me, t)
        except Exception as e:  # noqa: BLE001
            m = {"error": str(e)}
        m["n_launch"] = n_launch
        rows.append((t, m))
    return env, rows


def scan_step0(start, count):
    """Cheap step-0 sparsity scan over a seed range (no game play, both seats)."""
    print(f"== STEP-0 sparsity scan: seeds {start}..{start+count-1} (both seats) ==")
    print("   nearest-neutral ETA per seat at the opening board; K_OPEN=20 line.\n")
    eta_hist = defaultdict(int)
    n = 0
    over20 = 0
    over10 = 0
    reach20_zero = 0
    for seed in range(start, start + count):
        env = make("orbit_wars", configuration={"seed": seed}, debug=False)
        env.reset(num_agents=2)
        obs0 = env.steps[0]
        for me in (0, 1):
            obs = obs0[me].get("observation")
            if not obs or not obs.get("planets"):
                continue
            m = reach_only(obs, me, 0)
            if m["usable"] == 0:
                continue
            n += 1
            me_eta = m["min_eta"]
            if me_eta is not None:
                eta_hist[min(50, (me_eta // 5) * 5)] += 1
                if me_eta > 20:
                    over20 += 1
                if me_eta > 10:
                    over10 += 1
            if m["reach"][20] == 0:
                reach20_zero += 1
    print(f"  scanned {n} (seed,seat) opening boards with >=1 usable source")
    print(f"  nearest neutral ETA > 10 (past static floor): {over10}/{n} = {over10/max(1,n):.0%}")
    print(f"  nearest neutral ETA > 20 (past adaptive K_OPEN): {over20}/{n} = {over20/max(1,n):.0%}")
    print(f"  ZERO neutral reachable within K_OPEN=20: {reach20_zero}/{n} = {reach20_zero/max(1,n):.0%}")
    print("\n  nearest-neutral ETA histogram (the easiest grab from any source):")
    for b in sorted(eta_hist):
        label = f"{b}-{b+4}" if b < 50 else "50+"
        bar = "#" * min(60, eta_hist[b])
        print(f"    ETA {label:>6}: {bar} ({eta_hist[b]})")


def main():
    if SCAN is not None:
        scan_step0(*SCAN)
        return

    print(f"== opening-starvation diag  vs {Path(OPP).name}  seeds={SEEDS}  "
          f"window=0-{WINDOW}  K_live=adaptive(20->10) ==")
    print(f"   reach@K = (source,neutral) pairs with full-budget ETA <= K  "
          f"(K grid {K_GRID})\n")

    # ---- per-step trace for the focal seed (the sparse repro) ----
    if TRACE_SEED is not None:
        print(f"--- TRACE seed {TRACE_SEED} focal@P0 (steps 0..{WINDOW}) ---")
        print("   LAUNCH = fleets the agent actually emitted; propose = candidates "
              "available; WAIT = had candidates, launched nothing (chooser decision)")
        print(f"  {'step':>4} {'K':>3} {'usable':>6} {'neut':>4} {'minETA':>6} "
              f"{'r@10':>4} {'r@20':>4} {'r@30':>4} {'propose':>7} {'LAUNCH':>6}  note")
        _, rows = run_seed(TRACE_SEED, True, WINDOW)
        n_wait = n_starved = n_launch_turns = 0
        for t, m in rows:
            if "error" in m:
                print(f"  {t:>4}  ERROR {m['error'][:50]}")
                continue
            r = m["reach"]
            me_eta = m["min_eta"] if m["min_eta"] is not None else -1
            nl = m.get("n_launch", 0)
            note = ""
            if m["starved"]:
                note = "STARVED(0 cand)"; n_starved += 1
            elif m["propose_count"] > 0 and nl == 0:
                note = "WAIT(had cand)"; n_wait += 1
            if nl > 0:
                n_launch_turns += 1
            print(f"  {t:>4} {m['k_live']:>3} {m['usable']:>6} {m['neutrals']:>4} "
                  f"{me_eta:>6} {r[10]:>4} {r[20]:>4} {r[30]:>4} "
                  f"{m['propose_count']:>7} {nl:>6}  {note}")
        tot = len(rows)
        print(f"\n  opening {tot} turns: LAUNCHED on {n_launch_turns}; "
              f"WAITED-with-candidates {n_wait}; STARVED-zero-candidates {n_starved}")
        print("  -> STARVED>0 means horizon-K filter; WAIT>>STARVED means the lever "
              "is the chooser/value function, not the horizon.")
        print()

    # ---- aggregate over the panel (both seats) ----
    agg = defaultdict(int)
    n_boards = 0
    reach_sum = {k: 0 for k in K_GRID}
    eta_hist = defaultdict(int)
    starved_at_k = {k: 0 for k in K_GRID}  # would-be-starved if K were fixed at k
    for seed in SEEDS:
        for fp0 in (True, False):
            _, rows = run_seed(seed, fp0, WINDOW)
            for _t, m in rows:
                if "error" in m or m["usable"] == 0:
                    continue
                n_boards += 1
                if m["starved"]:
                    agg["starved_live"] += 1
                for k in K_GRID:
                    reach_sum[k] += m["reach"][k]
                    if m["reach"][k] == 0:
                        starved_at_k[k] += 1
                me_eta = m["min_eta"]
                if me_eta is not None:
                    bucket = min(50, (me_eta // 5) * 5)
                    eta_hist[bucket] += 1

    print(f"=== AGGREGATE opening boards (usable-source boards only): {n_boards} ===")
    if n_boards:
        print(f"  starved-LIVE (adaptive K, propose gave 0 candidates): "
              f"{agg['starved_live']}/{n_boards} = {agg['starved_live']/n_boards:.0%}")
        print("\n  mean reach@K (expansion launches admitted per board):")
        for k in K_GRID:
            print(f"    K={k:>2}: {reach_sum[k]/n_boards:5.2f} pairs   "
                  f"|  boards with ZERO neutral reachable @K: "
                  f"{starved_at_k[k]}/{n_boards} = {starved_at_k[k]/n_boards:.0%}")
        print("\n  global-min neutral ETA histogram (the easiest available grab):")
        for b in sorted(eta_hist):
            label = f"{b}-{b+4}" if b < 50 else "50+"
            print(f"    ETA {label:>6}: {'#'*eta_hist[b]} ({eta_hist[b]})")
    print("\nDECISION: high starved-LIVE% with reach@30/40 >> reach@20 means the "
          "adaptive K_OPEN=20 under-reaches the opening expansion map (Rule-40 fix: "
          "make the opening horizon predictability-/contest-bounded, not a fixed clock).")


if __name__ == "__main__":
    main()
