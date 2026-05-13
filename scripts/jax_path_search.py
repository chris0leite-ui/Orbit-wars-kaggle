"""N=2 path-search oracle — measures v7_0 head-room against the depth-2
maximin best move.

For each seed:
1. Make a 2P `orbit_wars` env, advance a few light-play turns so the
   board is non-trivial.
2. From our seat (player 0): build the v7_0_drop_one incumbent +
   drop-one candidate set (~N candidates).
3. Build the same drop-one candidate set for the opponent from their
   POV after we simulate turn 1 (~M candidates per row).
4. For every (our_i, opp_j) pair: simulate the action sequence
   (us=our_i, opp=opp_inc) on turn 1, (us=pass, opp=opp_j) on turn 2,
   then K-2 mirror-mirror follow-up steps. Score with
   `delta_us_minus_them` at the leaf.
5. Compute three first-move choices:
   - v7_0 (single-ply argmax over our_i, opp always plays incumbent)
   - depth-2 maximin (argmax_i of min_j over the cell matrix)
   - depth-2 average (argmax_i of mean_j over the cell matrix)
6. Output one CSV row per seed with all three choices + match flags.

The headline metric is the v7_0-vs-depth-2-maximin **disagreement
rate**:
- If < 60 % match, depth-2 has real head-room and `choose_depth2` is
  worth shipping.
- If > 90 % match, v7_0 already approximates the maximin choice and
  depth-2 is dead weight.

CLI:
    python -m scripts.jax_path_search --seeds 32 --depth 2 [--out PATH]

Output:
    audit/2026-05-13-depth2-oracle-{N}seeds.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import sys
import time
from pathlib import Path

from kaggle_environments import make

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from lib.fast_sim import (  # noqa: E402
    Snapshot,
    clone as fs_clone,
    delta_us_minus_them,
    from_obs as fs_from_obs,
    step as fs_step,
)
from lib.opp_model import top_tier_mirror_policy  # noqa: E402
from lib.v7_search import (  # noqa: E402
    _action_from_intents,
    _build_incumbent_intents,
    _enumerate_drop_one,
    _opp_incumbent_action,
)
from lib.intent import World  # noqa: E402
from lib.world_model import WorldModel  # noqa: E402


def _light_play(env, n_steps: int, rng_seed: int, num_agents: int = 2):
    rng = random.Random(rng_seed)
    for _ in range(n_steps):
        if env.state[0].status != "ACTIVE":
            break
        actions = []
        for ps in range(num_agents):
            mv = []
            for p in env.state[ps].observation.planets:
                if p[1] == ps and p[5] > 5 and rng.random() < 0.2:
                    mv.append([
                        p[0],
                        rng.uniform(0, 2 * math.pi),
                        max(1, int(p[5] * rng.uniform(0.1, 0.3))),
                    ])
            actions.append(mv)
        env.step(actions)


def _rollout_score(
    snap: Snapshot, my_id: int, opp_id: int, K_tail: int,
) -> float:
    """Mirror-mirror rollout from `snap` for K_tail steps; return
    `delta_us_minus_them` from `my_id`'s POV at the leaf."""
    s = snap
    for _ in range(max(0, K_tail)):
        if s.done:
            break
        a0 = top_tier_mirror_policy(s.state[0].observation)
        a1 = top_tier_mirror_policy(s.state[1].observation)
        s = fs_step(s, [a0, a1], in_place=True)
    return delta_us_minus_them(s, my_id)


def _score_path_depth2(
    snap0: Snapshot,
    our_act1: list,
    opp_act1: list,
    opp_act2: list,
    my_id: int,
    opp_id: int,
    K: int,
) -> float:
    """Apply the depth-2 action sequence and return the leaf score.

    Turn 1: us=our_act1, opp=opp_act1.
    Turn 2: us pass (we've committed), opp=opp_act2.
    Turns 3..K: mirror-mirror.
    """
    s = fs_clone(snap0)
    if not s.done:
        actions = [None, None]
        actions[my_id] = our_act1
        actions[opp_id] = opp_act1
        s = fs_step(s, actions, in_place=True)
    if s.done:
        return delta_us_minus_them(s, my_id)
    actions = [None, None]
    actions[my_id] = []
    actions[opp_id] = opp_act2
    s = fs_step(s, actions, in_place=True)
    return _rollout_score(s, my_id, opp_id, K_tail=max(0, K - 2))


def _v7_0_score(
    snap0: Snapshot,
    our_act1: list,
    opp_inc1: list,
    my_id: int,
    opp_id: int,
    K: int,
) -> float:
    """v7_0_drop_one's score for `our_act1`: assume opp plays incumbent
    each turn, then mirror-mirror tail. This is the score v7_0's
    `score_candidate` computes."""
    s = fs_clone(snap0)
    if not s.done:
        actions = [None, None]
        actions[my_id] = our_act1
        actions[opp_id] = opp_inc1
        s = fs_step(s, actions, in_place=True)
    return _rollout_score(s, my_id, opp_id, K_tail=max(0, K - 1))


def _build_incumbent_action_from_obs(obs):
    """v7_0's incumbent (with H11 opening wire + H15 comet reject)
    composed via `_build_incumbent_intents` + `_action_from_intents`."""
    world = World.from_obs(obs)
    if not world.planets_by_id:
        return []
    model = WorldModel.from_world(world)
    intents = _build_incumbent_intents(world, model)
    return _action_from_intents(intents, obs, model)


def run_seed(
    seed: int, *, K: int, light_steps: int, max_our: int, max_opp: int,
) -> dict:
    """Execute the depth-2 oracle for one seed. Returns a dict with the
    headline metrics + per-candidate cell scores (compact)."""
    env = make("orbit_wars", configuration={"seed": seed})
    env.reset(num_agents=2)
    _light_play(env, n_steps=light_steps, rng_seed=seed * 31)
    if env.state[0].status != "ACTIVE":
        return {"seed": seed, "skipped": True, "reason": "terminated_early"}

    obs = env.state[0].observation
    my_id = 0
    opp_id = 1

    # Build our candidates from v7_0's incumbent.
    incumbent_action = _build_incumbent_action_from_obs(obs)
    our_C = _enumerate_drop_one(incumbent_action)[:max_our]
    if len(our_C) <= 1:
        return {"seed": seed, "skipped": True, "reason": "trivial_candidate_set"}

    configuration = env.configuration
    snap0 = fs_from_obs(obs, configuration, episode_seed=env.info.get("seed", 0), num_seats=2)
    opp_inc1 = _opp_incumbent_action(World.from_obs(obs), obs, opp_id)

    # v7_0 single-ply scores: opp always plays incumbent.
    v7_0_scores = [
        _v7_0_score(snap0, our_act1, opp_inc1, my_id, opp_id, K)
        for our_act1 in our_C
    ]

    # Depth-2 payoff matrix: per row i, recompute opp's incumbent after
    # turn 1 (us=our_C[i], opp=opp_inc1), enumerate opp drop-one set.
    matrix: list[list[float]] = []
    for i, our_act1 in enumerate(our_C):
        s_i = fs_clone(snap0)
        if not s_i.done:
            actions = [None, None]
            actions[my_id] = our_act1
            actions[opp_id] = opp_inc1
            s_i = fs_step(s_i, actions, in_place=True)
        if s_i.done:
            # Game ended; payoff is constant for any opp response.
            matrix.append([delta_us_minus_them(s_i, my_id)])
            continue
        opp_obs_after = s_i.state[opp_id].observation
        opp_world = World.from_obs(opp_obs_after)
        opp_model = WorldModel.from_world(opp_world)
        opp_inc_intents = _build_incumbent_intents(opp_world, opp_model)
        opp_inc_action = _action_from_intents(opp_inc_intents, opp_obs_after, opp_model)
        opp_C = _enumerate_drop_one(opp_inc_action)[:max_opp]
        if not opp_C:
            opp_C = [[]]
        row = []
        for opp_act2 in opp_C:
            row.append(_score_path_depth2(
                snap0, our_act1, opp_inc1, opp_act2,
                my_id, opp_id, K,
            ))
        matrix.append(row)

    # Choices.
    v7_0_argmax = int(max(range(len(v7_0_scores)), key=lambda i: v7_0_scores[i]))

    def _row_worst(row): return min(row) if row else float("-inf")
    def _row_mean(row): return sum(row) / len(row) if row else float("-inf")

    maximin_argmax = int(max(range(len(matrix)), key=lambda i: _row_worst(matrix[i])))
    mean_argmax = int(max(range(len(matrix)), key=lambda i: _row_mean(matrix[i])))

    return {
        "seed": seed,
        "skipped": False,
        "our_cands": len(our_C),
        "opp_cands_per_row": [len(r) for r in matrix],
        "v7_0_argmax": v7_0_argmax,
        "v7_0_score": v7_0_scores[v7_0_argmax],
        "depth2_maximin_argmax": maximin_argmax,
        "depth2_maximin_worst": _row_worst(matrix[maximin_argmax]),
        "depth2_mean_argmax": mean_argmax,
        "depth2_mean_score": _row_mean(matrix[mean_argmax]),
        "match_maximin": v7_0_argmax == maximin_argmax,
        "match_mean": v7_0_argmax == mean_argmax,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=32)
    parser.add_argument("--depth", type=int, default=2,
                        help="oracle depth (only 2 implemented)")
    parser.add_argument("--K", type=int, default=6,
                        help="total rollout depth including the 2 forced turns")
    parser.add_argument("--light-steps", type=int, default=20)
    parser.add_argument("--max-our", type=int, default=8)
    parser.add_argument("--max-opp", type=int, default=4)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()
    if args.depth != 2:
        raise NotImplementedError("only depth=2 implemented for this MVP")

    rows: list[dict] = []
    t0 = time.perf_counter()
    for seed in range(args.seeds):
        t_seed = time.perf_counter()
        result = run_seed(
            seed, K=args.K, light_steps=args.light_steps,
            max_our=args.max_our, max_opp=args.max_opp,
        )
        dt = (time.perf_counter() - t_seed) * 1000.0
        rows.append(result)
        flag = "SKIP" if result.get("skipped") else (
            "MATCH" if result.get("match_maximin") else "DIFFER"
        )
        print(
            f"seed={seed:3d} {flag:6s} "
            f"v7_0_i={result.get('v7_0_argmax','-')} "
            f"maximin_i={result.get('depth2_maximin_argmax','-')} "
            f"mean_i={result.get('depth2_mean_argmax','-')} "
            f"wall={dt:6.0f}ms"
        )
    elapsed = time.perf_counter() - t0

    # Aggregate.
    used = [r for r in rows if not r.get("skipped")]
    n = len(used)
    if n == 0:
        print("no usable seeds — all skipped.")
        return
    match_maximin = sum(1 for r in used if r["match_maximin"]) / n
    match_mean = sum(1 for r in used if r["match_mean"]) / n
    print()
    print(f"=== summary ({n} usable, {len(rows)-n} skipped) ===")
    print(f"v7_0 vs depth-2 maximin match-rate: {match_maximin:.1%}")
    print(f"v7_0 vs depth-2 mean    match-rate: {match_mean:.1%}")
    print(f"interpretation: < 60% match → depth-2 has head-room; > 90% → dead.")
    print(f"total wall: {elapsed:.1f} s")

    # Write CSV.
    out = args.out or f"audit/2026-05-13-depth2-oracle-{args.seeds}seeds.csv"
    out_path = REPO / out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "seed", "skipped", "reason", "our_cands",
            "v7_0_argmax", "v7_0_score",
            "depth2_maximin_argmax", "depth2_maximin_worst",
            "depth2_mean_argmax", "depth2_mean_score",
            "match_maximin", "match_mean",
        ])
        for r in rows:
            writer.writerow([
                r.get("seed"),
                r.get("skipped", False),
                r.get("reason", ""),
                r.get("our_cands", 0),
                r.get("v7_0_argmax", ""),
                r.get("v7_0_score", ""),
                r.get("depth2_maximin_argmax", ""),
                r.get("depth2_maximin_worst", ""),
                r.get("depth2_mean_argmax", ""),
                r.get("depth2_mean_score", ""),
                r.get("match_maximin", ""),
                r.get("match_mean", ""),
            ])
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
