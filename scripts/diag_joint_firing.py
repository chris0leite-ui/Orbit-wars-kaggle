"""Phase E Phase 1 follow-up: count how many turns REAL distinct-source
joints actually fire in a full bundle-vs-baseline game.

Instruments BundleEvaluator._detect_joint_captures (scorer-side detector)
and BundleSearch._enumerate_joint_seeds (search-side seeder), counts:
  - turns where seed-enum produced >=1 joint seed
  - turns where the detector found >=1 distinct-source joint
  - turns where the chosen bundle contained a joint that was bonus-eligible

Decision context:
  Post-fix Phase 1 A/B was NULL (identical to cands=5-only baseline)
  vs v7_0 and vs baseline. Either:
  (a) Real joints rarely arise -> Phase 1 is structurally a no-op
  (b) Real joints arise but joint_bonus isn't enough to flip the
      chooser away from solos -> tune

Usage:
    BUNDLE_JOINT_BONUS=0.5 BUNDLE_JOINT_SEEDS=10 \\
        python scripts/diag_joint_firing.py [--seed 42] [--opponent baseline]
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from kaggle_environments import make  # noqa: E402
from lib.trajectory_layer import (  # noqa: E402
    BundleEvaluator, BundleSearch,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--opponent", type=str, default="baseline",
                    choices=["baseline", "v7_0"])
    args = ap.parse_args()

    opp_path = {
        "baseline": REPO / "submissions" / "baseline.py",
        "v7_0":     REPO / "submissions" / "v7_0_drop_one.py",
    }[args.opponent]

    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        m = importlib.util.module_from_spec(spec)
        sys.modules[name] = m
        spec.loader.exec_module(m)
        return m.agent

    # Load bundle as a module so we can monkeypatch its evaluator/search
    # classes (the bundled file has its own `BundleEvaluator` / `BundleSearch`
    # — patching the local lib copy won't reach them).
    bundle_path = REPO / "submissions" / "bundle.py"
    bundle_spec = importlib.util.spec_from_file_location("_bundle", bundle_path)
    bundle_mod = importlib.util.module_from_spec(bundle_spec)
    sys.modules["_bundle"] = bundle_mod
    bundle_spec.loader.exec_module(bundle_mod)
    bundle = bundle_mod.agent

    opp = load("_opp", opp_path)

    # Instrument the bundled file's classes.
    Eval = bundle_mod.BundleEvaluator
    Search = bundle_mod.BundleSearch
    orig_detect = Eval._detect_joint_captures
    orig_enum = Search._enumerate_joint_seeds
    orig_search = Search.search

    stats = {
        "detect_calls": 0,
        "detect_hits": 0,
        "detect_hit_sizes": [],
        "enum_calls": 0,
        "enum_total_seeds": 0,
        "enum_calls_with_seeds": 0,
        "chosen_is_joint_turns": 0,
        "chosen_bundle_size_at_join_turns": [],
        "by_turn": [],  # (turn, detected_count, seeded_count, chosen_is_joint)
    }
    current_turn = [-1]
    per_turn = {"detect_count": 0, "seed_count": 0, "chosen_is_joint": False}

    def logged_detect(self, world, bundle_obj, my_id):
        stats["detect_calls"] += 1
        result = orig_detect(self, world, bundle_obj, my_id)
        if result:
            stats["detect_hits"] += 1
            stats["detect_hit_sizes"].append(len(result))
            per_turn["detect_count"] += 1
        return result

    def logged_enum(self, world, my_id, *, max_seeds=10):
        stats["enum_calls"] += 1
        result = orig_enum(self, world, my_id, max_seeds=max_seeds)
        n = len(result)
        stats["enum_total_seeds"] += n
        if n:
            stats["enum_calls_with_seeds"] += 1
            per_turn["seed_count"] += n
        return result

    def logged_search(self, world, *, my_id=None, seed_bundle=None,
                      opp_overlays=None, deadline=None):
        chosen = orig_search(
            self, world,
            my_id=my_id, seed_bundle=seed_bundle,
            opp_overlays=opp_overlays, deadline=deadline,
        )
        # Inspect the chosen bundle for a real distinct-source joint.
        # `my_id` defaults to world.my_id inside search; reproduce that.
        actual_my_id = my_id if my_id is not None else world.my_id
        if not chosen.is_empty:
            # Mirror the only-our-search behavior: skip opp-search calls
            # (in mirror mode, search() is called per-opp from inside
            # _build_searches; those should NOT count toward "our" chosen).
            if actual_my_id == world.my_id:
                joints_in_chosen = orig_detect(
                    self.evaluator, world, chosen, actual_my_id,
                )
                if joints_in_chosen:
                    per_turn["chosen_is_joint"] = True
                    stats["chosen_bundle_size_at_join_turns"].append(
                        len(chosen.launches)
                    )
        return chosen

    Eval._detect_joint_captures = logged_detect
    Search._enumerate_joint_seeds = logged_enum
    Search.search = logged_search

    def wrap_agent(obs, configuration=None):
        t = int(obs.get("step", -1) if isinstance(obs, dict)
                else getattr(obs, "step", -1))
        # New turn — flush prior stats
        if t != current_turn[0]:
            if current_turn[0] >= 0:
                stats["by_turn"].append((current_turn[0],
                                         per_turn["detect_count"],
                                         per_turn["seed_count"],
                                         per_turn["chosen_is_joint"]))
                if per_turn["chosen_is_joint"]:
                    stats["chosen_is_joint_turns"] += 1
            current_turn[0] = t
            per_turn["detect_count"] = 0
            per_turn["seed_count"] = 0
            per_turn["chosen_is_joint"] = False
        return bundle(obs, configuration)

    print(f"Running bundle vs {args.opponent} (seed={args.seed}) with "
          f"joint instrumentation...")
    env = make("orbit_wars", configuration={"seed": args.seed}, debug=False)
    t0 = time.time()
    env.run([wrap_agent, opp])
    elapsed = time.time() - t0
    # Flush final turn
    stats["by_turn"].append((current_turn[0],
                             per_turn["detect_count"],
                             per_turn["seed_count"],
                             per_turn["chosen_is_joint"]))
    if per_turn["chosen_is_joint"]:
        stats["chosen_is_joint_turns"] += 1

    final = env.steps[-1]
    n_turns = len(env.steps)
    bundle_reward = final[0].reward

    print(f"\nGame finished: bundle reward={bundle_reward} "
          f"in {n_turns} turns ({elapsed:.0f}s elapsed)")

    # Aggregate
    turns_with_seed = sum(1 for _, _, s, _ in stats["by_turn"] if s > 0)
    turns_with_detect = sum(1 for _, d, _, _ in stats["by_turn"] if d > 0)
    turns_with_chosen_joint = stats["chosen_is_joint_turns"]
    # Conditional rate: of the turns where the detector found something,
    # how often did the chooser actually PICK a joint?
    if turns_with_detect:
        pick_rate = turns_with_chosen_joint / turns_with_detect * 100
    else:
        pick_rate = 0.0
    sizes = Counter(stats["detect_hit_sizes"])

    print(f"\n=== Joint firing summary ({n_turns} game turns) ===")
    print(f"  Turns where _enumerate_joint_seeds produced >=1 seed:")
    print(f"    {turns_with_seed} / {n_turns} = {turns_with_seed/n_turns*100:.1f}%")
    print(f"  Turns where _detect_joint_captures found >=1 joint:")
    print(f"    {turns_with_detect} / {n_turns} = {turns_with_detect/n_turns*100:.1f}%")
    print(f"  Turns where CHOSEN bundle contained a joint:")
    print(f"    {turns_with_chosen_joint} / {n_turns} = "
          f"{turns_with_chosen_joint/n_turns*100:.1f}%")
    print(f"  Pick rate (chosen-is-joint / detect-fires):")
    print(f"    {turns_with_chosen_joint} / {turns_with_detect} = {pick_rate:.1f}%")
    print(f"  Total seeds generated across all enum calls: {stats['enum_total_seeds']}")
    print(f"  Total detector calls: {stats['detect_calls']}")
    print(f"  Detector hits (>=1 joint found): {stats['detect_hits']}")
    print(f"  Hit-size distribution: {dict(sizes)}")
    print(f"  Chosen-bundle sizes when chosen-is-joint: "
          f"{Counter(stats['chosen_bundle_size_at_join_turns'])}")

    # Show first 25 turns with activity (mark which were chosen-is-joint)
    active = [(t, d, s, c) for t, d, s, c in stats["by_turn"]
              if d > 0 or s > 0 or c]
    print(f"\n=== Per-turn activity (first 25 active turns of {len(active)} total) ===")
    print(f"  {'turn':>4}  {'detect':>6}  {'seed':>5}  {'chose':>5}")
    for t, d, s, c in active[:25]:
        print(f"  {t:4d}  {d:6d}  {s:5d}  {'YES' if c else '   ':>5}")


if __name__ == "__main__":
    main()
