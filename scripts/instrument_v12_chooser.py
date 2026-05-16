"""Diagnose v12 chooser under-emission on a specific replay.

For each requested step in a replay JSON, call v12's agent() with monkey
patches that count entries surviving each stage of the funnel:

  enumerate -> cheap>-10 -> per-pair dedup -> N_VALIDATE cap
            -> delta>0 -> non-dogpile dedup -> emit

Prints a compact per-step table.
"""
import argparse, json, sys, types
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

# Default target is v12; can be overridden via --agent.
import importlib
v12 = None  # set in instrument() based on --agent


def instrument(steps_to_probe, replay_path, seat, agent_module="agents.v12.main"):
    global v12
    v12 = importlib.import_module(agent_module)
    r = json.load(open(replay_path))
    rows = []
    for step in steps_to_probe:
        if step >= len(r["steps"]):
            continue
        obs = r["steps"][step][seat]["observation"]
        # Mutate to ensure 'player' is set
        obs = dict(obs)
        obs["player"] = seat

        counts = {
            "n_planets_mine": 0,
            "n_planets_other": 0,
            "n_my_ships": 0,
            "n_enumerated": 0,
            "n_cheap_kept": 0,
            "n_after_pair_dedup": 0,
            "n_validated": 0,
            "n_delta_pos": 0,
            "n_wait_picked": 0,
            "n_emitted": 0,
            "per_step_ms": None,
            "n_affordable": None,
        }

        # Counters via wrapping the candidate-building primitives.
        orig_cheap = v12._cheap_marginal_value
        def cheap_wrap(*a, **kw):
            v = orig_cheap(*a, **kw)
            counts["n_enumerated"] += 1
            return v
        v12._cheap_marginal_value = cheap_wrap

        orig_score = v12._score_action
        validated_deltas = []
        def score_wrap(*a, **kw):
            counts["n_validated"] += 1
            d = orig_score(*a, **kw)
            validated_deltas.append(d)
            return d
        v12._score_action = score_wrap

        # Capture the agent function output and the intermediate state.
        # Re-run agent then inspect emitted moves.
        moves = v12.agent(obs)
        counts["n_emitted"] = len(moves)
        counts["n_delta_pos"] = sum(1 for d in validated_deltas if d > 0)
        counts["n_delta_neg"] = sum(1 for d in validated_deltas if d <= 0)
        counts["n_lost_to_dogpile"] = counts["n_delta_pos"] - counts["n_emitted"]

        # Restore.
        v12._cheap_marginal_value = orig_cheap
        v12._score_action = orig_score

        # Recompute the breakdown without invoking agent twice — easier to
        # just rerun the relevant slice by replicating agent's stage flow.
        # (We accept duplication for clarity.)
        from lib.world_model import WorldModel
        from lib.intent import World
        from kaggle_environments.envs.orbit_wars.orbit_wars import Planet, Fleet

        planets = [Planet(*p) for p in obs["planets"]]
        my_planets = [p for p in planets if int(p.owner) == seat]
        other_planets = [p for p in planets if int(p.owner) != seat]
        counts["n_planets_mine"] = len(my_planets)
        counts["n_planets_other"] = len(other_planets)
        counts["n_my_ships"] = sum(int(p.ships) for p in my_planets)

        # Walk Stage 1 once more without instrumentation to get
        # pair-dedup and cheap-kept counts.
        world = World.from_obs(obs)
        model = WorldModel.from_world(world)
        omega = float(obs.get("angular_velocity", 0.0))
        num_seats = v12._num_seats(planets, [Fleet(*f) for f in obs.get("fleets", []) or []])
        threatened_mine = [
            p for p in my_planets
            if model.time_to_enemy_threat(int(p.id), seat, world) is not None
        ]
        target_pool = other_planets + threatened_mine

        prerank = []
        for src in my_planets:
            if int(src.ships) < v12.MIN_FLEET_SIZE:
                continue
            for tgt in v12._nearest_k(target_pool, src, v12.NUM_TARGETS_PER_SOURCE):
                if int(tgt.id) == int(src.id):
                    continue
                for ships in v12._enumerate_ship_counts_basic(src, tgt, model, omega, seat, world):
                    if ships < v12.MIN_FLEET_SIZE or ships > int(src.ships):
                        continue
                    angle, eta = v12._aim_and_eta(src, tgt, ships, omega)
                    horizon = max(eta + v12.SIM_SETTLE_TURNS, v12.MIN_HORIZON)
                    if horizon >= v12.MAX_HORIZON + 1:
                        horizon = v12.MAX_HORIZON
                    cheap = orig_cheap(src, tgt, ships, eta, world, model, seat, wait_N=0)
                    if cheap > -10.0:
                        prerank.append((cheap, src, tgt, ships, angle, eta, horizon, 0))
                wt = v12._wait_then_fire_candidate(src, tgt, model, omega, seat)
                if wt is not None:
                    w_ships, w_wait_N, w_angle, w_eta = wt
                    w_horizon = max(w_wait_N + w_eta + v12.SIM_SETTLE_TURNS, v12.MIN_HORIZON)
                    if w_horizon < v12.MAX_HORIZON + 1:
                        w_cheap = orig_cheap(src, tgt, w_ships, w_eta, world, model, seat, wait_N=w_wait_N)
                        if w_cheap > -10.0:
                            prerank.append((w_cheap, src, tgt, w_ships, w_angle, w_eta, w_horizon, w_wait_N))

        counts["n_cheap_kept"] = len(prerank)
        best_per_pair = {}
        for entry in prerank:
            key = (int(entry[1].id), int(entry[2].id))
            prev = best_per_pair.get(key)
            if prev is None or entry[0] > prev[0]:
                best_per_pair[key] = entry
        counts["n_after_pair_dedup"] = len(best_per_pair)
        counts["n_wait_in_dedup"] = sum(1 for e in best_per_pair.values() if e[7] > 0)

        rows.append((step, counts))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", required=True)
    ap.add_argument("--seat", type=int, default=0)
    ap.add_argument("--steps", type=str, default="50,80,120,150,170,190")
    ap.add_argument("--agent", default="agents.v12.main",
                    help="Importable agent module (e.g. agents.v15.main)")
    args = ap.parse_args()
    steps = [int(s) for s in args.steps.split(",")]
    rows = instrument(steps, args.replay, args.seat, args.agent)

    print(f"{'step':>4} {'myP':>3} {'otP':>3} {'myS':>5} "
          f"{'enum':>5} {'cheap':>5} {'pairs':>5} {'wait':>4} "
          f"{'val':>4} {'d>0':>4} {'d<=0':>5} {'dogpile':>7} {'emit':>4}")
    for step, c in rows:
        print(f"{step:>4d} {c['n_planets_mine']:>3d} {c['n_planets_other']:>3d} "
              f"{c['n_my_ships']:>5d} "
              f"{c['n_enumerated']:>5d} {c['n_cheap_kept']:>5d} "
              f"{c['n_after_pair_dedup']:>5d} {c['n_wait_in_dedup']:>4d} "
              f"{c['n_validated']:>4d} {c['n_delta_pos']:>4d} "
              f"{c['n_delta_neg']:>5d} {c['n_lost_to_dogpile']:>7d} "
              f"{c['n_emitted']:>4d}")


if __name__ == "__main__":
    main()
