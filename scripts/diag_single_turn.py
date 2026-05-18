"""Single-turn diagnostic: at a target turn in bundle (off) vs opp,
dump bundle's full candidate scoring + score what opp actually played
as a Bundle. Disambiguates the three root-cause hypotheses:

  (a) Scoring weights wrong — bundle CONSIDERS opp's move but scores
      it lower than the one it picked.
  (b) Enumeration wrong — bundle NEVER considers opp's move.
  (c) Search topology wrong — bundle scores it highest but beam prunes
      it.

Usage:
    # First run captures obs to disk (uses default bundle knobs):
    python scripts/diag_single_turn.py [--seed 42] [--turn 20]
    # Subsequent runs with different knobs re-use the pinned obs:
    BUNDLE_HORIZON=30 python scripts/diag_single_turn.py --reuse-obs
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import pickle
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from kaggle_environments import make
from agents.bundle.main import agent as bundle_agent
from lib.trajectory_layer import (
    Bundle, BundleEvaluator, LaunchSpec, World,
)

def obs_cache_path(opp_name: str, seed: int, turn: int) -> Path:
    return REPO / "audit" / f"diag_obs_{opp_name}_seed{seed}_t{turn}.pkl"


def load_opponent(submission_path: Path):
    """Load any submission file's `agent` callable."""
    mod_name = f"_opp_{submission_path.stem}"
    spec = importlib.util.spec_from_file_location(mod_name, submission_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod.agent




def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--turn", type=int, default=20)
    ap.add_argument("--mode", choices=["off", "lite"], default="off")
    ap.add_argument("--opponent", type=str,
                    default="submissions/v7_0_drop_one.py",
                    help="Path to opponent submission file")
    ap.add_argument("--reuse-obs", action="store_true",
                    help="Skip replay, use pinned obs from previous run.")
    args = ap.parse_args()

    opp_path = REPO / args.opponent if not Path(args.opponent).is_absolute() else Path(args.opponent)
    opp_agent = load_opponent(opp_path)
    opp_name = opp_path.stem
    obs_cache = obs_cache_path(opp_name, args.seed, args.turn)
    print(f"Opponent: {opp_name}  Obs cache: {obs_cache.name}")

    if args.reuse_obs and obs_cache.exists():
        with open(obs_cache, "rb") as f:
            cached = pickle.load(f)
        obs_bundle = cached["obs"]
        configuration = cached["configuration"]
        print(f"Reusing pinned obs from {obs_cache}")

        class _StubConfig:
            def __init__(self, d): self.__dict__.update(d)

        class _StubState:
            def __init__(self, obs): self.observation = obs

        # Stub env so the rest of the diagnostic just sees obs.
        cfg_obj = _StubConfig(configuration) if isinstance(configuration, dict) else configuration

        class _StubEnv:
            done = False
            state = [_StubState(obs_bundle)]
            configuration = cfg_obj
        env = _StubEnv()
    else:
        os.environ["BUNDLE_ME_FOLLOWUP"] = args.mode
        env = make("orbit_wars", configuration={"seed": args.seed}, debug=False)
        env.reset(num_agents=2)

        print(f"Replaying to turn {args.turn} (seed={args.seed}, mode={args.mode})...")
        for t in range(args.turn):
            if env.done:
                print(f"Game ended early at turn {t}")
                return
            obs0 = env.state[0].observation
            obs1 = env.state[1].observation
            cfg = env.configuration
            a_bundle = bundle_agent(obs0, cfg)
            a_v7 = opp_agent(obs1, cfg)
            env.step([a_bundle, a_v7])

        if env.done:
            print(f"Game done at turn {args.turn}, can't diagnose")
            return

        obs_bundle = env.state[0].observation
        # Persist obs + config so subsequent --reuse-obs runs see identical state.
        cfg_save = (dict(env.configuration)
                    if hasattr(env.configuration, "keys")
                    else dict(env.configuration.__dict__))
        with open(obs_cache, "wb") as f:
            pickle.dump({"obs": dict(obs_bundle) if isinstance(obs_bundle, dict)
                         else obs_bundle, "configuration": cfg_save}, f)
        print(f"Pinned obs to {obs_cache}")

    print(f"\n=== Turn {args.turn} state (bundle's seat) ===")
    planets = obs_bundle.get("planets", []) if isinstance(obs_bundle, dict) else obs_bundle.planets
    my_p = [p for p in planets if int(p[1]) == 0]
    opp_p = [p for p in planets if int(p[1]) == 1]
    neu_p = [p for p in planets if int(p[1]) == -1]
    print(f"  my={len(my_p)} (ships={sum(int(p[5]) for p in my_p)}, "
          f"prod={sum(int(p[6]) for p in my_p)})")
    print(f"  opp={len(opp_p)} (ships={sum(int(p[5]) for p in opp_p)}, "
          f"prod={sum(int(p[6]) for p in opp_p)})")
    print(f"  neutrals={len(neu_p)}")

    # Instrument BundleEvaluator.score: log every (bundle, score) it computes.
    original_score = BundleEvaluator.score
    score_log: list[tuple[Bundle, float, str]] = []

    def logged_score(self, world, bundle, *, my_id=None, opp_overlays=None):
        result = original_score(
            self, world, bundle, my_id=my_id, opp_overlays=opp_overlays,
        )
        # Tag by my_id so we can filter to our seat's scores.
        seat = "me" if (my_id is None or my_id == world.my_id) else f"opp{my_id}"
        score_log.append((bundle, result.total, seat))
        return result

    BundleEvaluator.score = logged_score

    try:
        print("\n=== Running bundle agent on this turn ===")
        t0 = time.perf_counter()
        chosen_actions = bundle_agent(obs_bundle, env.configuration)
        bundle_ms = (time.perf_counter() - t0) * 1000.0
        print(f"  bundle elapsed: {bundle_ms:.1f}ms")
        print(f"  total score-calls logged: {len(score_log)}")

        me_scores = [(b, s) for (b, s, seat) in score_log if seat == "me"]
        print(f"  from my seat: {len(me_scores)} candidates scored")

        # Dedupe by bundle and sort by score
        unique = {}
        for b, s in me_scores:
            key = b.launches
            if key not in unique or unique[key] < s:
                unique[key] = s
        ranked = sorted(unique.items(), key=lambda kv: -kv[1])

        print(f"\n=== Bundle's top-10 candidates (by score) ===")
        for rank, (launches, score) in enumerate(ranked[:10], 1):
            launches_str = (
                "EMPTY" if not launches else
                ", ".join(f"src{l.src_id}->turn{l.launch_turn}({l.ships}sh)"
                          for l in launches)
            )
            print(f"  {rank:2}. score={score:7.2f}  {launches_str}")

        print(f"\n=== Bundle CHOSE this turn ({len(chosen_actions)} actions) ===")
        for a in chosen_actions:
            src, angle, ships = a
            print(f"  src={src} angle={angle:.3f} ships={ships}")

    finally:
        BundleEvaluator.score = original_score

    # Now: what would opp have played from BUNDLE'S seat?
    print(f"\n=== Running opp from BUNDLE'S seat (same obs) ===")
    # opp expects obs where the planet ownership labels match player perspective.
    # We feed it bundle's obs directly so it gets the same state. But it needs
    # its "player" field correct — bundle's obs has player=0.
    obs_for_v7 = dict(obs_bundle) if isinstance(obs_bundle, dict) else obs_bundle
    v7_actions = opp_agent(obs_for_v7, env.configuration)
    print(f"  opp would play {len(v7_actions)} actions:")
    for a in v7_actions:
        src, angle, ships = a
        print(f"  src={src} angle={angle:.3f} ships={ships}")

    # Convert opp's actions to a Bundle and score it under bundle's evaluator
    # WITH the same opp_overlays bundle used during its search — otherwise the
    # comparison is apples-to-oranges (empty-vs-opp-aware scores vary by ~40 pts).
    v7_bundle = Bundle(launches=tuple(
        LaunchSpec(src_id=int(a[0]), aim_angle=float(a[1]),
                   ships=int(a[2]), owner=0, launch_turn=0)
        for a in v7_actions
    ))
    print(f"\n=== Scoring opp's move WITH opp overlays (apples-to-apples) ===")
    world = World.from_obs(obs_bundle, configuration=env.configuration)
    from agents.bundle.main import _build_searches
    from lib.trajectory_layer import predict_opp_via_event_driven_lite_greedy
    own_search, _, _ = _build_searches()
    ev = own_search.evaluator
    # Bundle agent uses BUNDLE_OPP_MODE=mirror by default but event_driven is
    # also supported. Use event_driven here for determinism.
    opp_overlays = predict_opp_via_event_driven_lite_greedy(
        world, my_id=0, horizon=ev.horizon,
    )
    v7_score = ev.score(world, v7_bundle, my_id=0,
                        opp_overlays=opp_overlays).total
    empty_score = ev.score(world, Bundle(), my_id=0,
                           opp_overlays=opp_overlays).total
    print(f"  empty bundle (do nothing) score: {empty_score:.2f}")
    print(f"  opp's bundle score:             {v7_score:.2f}")

    # Find where v7's bundle ranks among bundle's considered candidates.
    v7_key = v7_bundle.launches
    v7_in_considered = v7_key in unique
    v7_rank = None
    if v7_in_considered:
        for rank, (launches, score) in enumerate(ranked, 1):
            if launches == v7_key:
                v7_rank = rank
                break

    print(f"  opp's-bundle score (re-scored): {v7_score:.2f}")
    print(f"  opp's move appears in bundle's considered set: {v7_in_considered}")
    if v7_in_considered:
        print(f"  rank in considered set: #{v7_rank}")
    else:
        print(f"  → ENUMERATION ROOT-CAUSE: bundle never considered this move")

    bundle_chose = (
        Bundle()  # empty
        if not chosen_actions else
        Bundle(launches=tuple(
            LaunchSpec(src_id=int(a[0]), aim_angle=float(a[1]),
                       ships=int(a[2]), owner=0, launch_turn=0)
            for a in chosen_actions
        ))
    )
    bundle_score = ev.score(world, bundle_chose, my_id=0,
                            opp_overlays=opp_overlays).total
    print(f"\n=== Apples-to-apples (all with same opp_overlays) ===")
    print(f"  bundle's chosen score: {bundle_score:.2f}")
    print(f"  opp's move score:     {v7_score:.2f}")
    print(f"  delta (opp - bundle): {v7_score - bundle_score:+.2f}")
    print(f"  score range across bundle's 72 candidates: "
          f"{min(s for _, s in unique.items()):.2f} to "
          f"{max(s for _, s in unique.items()):.2f}")

    if v7_score > bundle_score and not v7_in_considered:
        print(f"\n  >>> ROOT: ENUMERATION (b) — opp's move would have scored "
              f"{v7_score - bundle_score:+.2f} better but bundle never enumerated it.")
    elif v7_score > bundle_score and v7_in_considered:
        print(f"\n  >>> ROOT: SEARCH TOPOLOGY (c) — bundle saw opp's move "
              f"(rank #{v7_rank}, score {v7_score:.2f}) but its search "
              f"converged on a worse one ({bundle_score:.2f}).")
    elif v7_score <= bundle_score:
        print(f"\n  >>> ROOT: SCORING (a) — bundle's scorer values its own pick "
              f"HIGHER ({bundle_score:.2f}) than opp's ({v7_score:.2f}). "
              f"opp may be winning by something the scorer doesn't measure.")


if __name__ == "__main__":
    main()
