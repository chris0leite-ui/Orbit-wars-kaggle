"""One-game close-read of the Phase 8 chain-bonus mechanism.

Plays one game (focal=chain_on, opp=chain_off, seed configurable) and at
every focal turn dumps:
  - n_candidates / n_chain_candidates from propose() on the obs
  - which (if any) chain-tagged candidates were actually fired by the
    chooser (matched by src + tgt + ships)
  - relay tracking: for each chain launch fired turn T, does the tgt
    become ours by turn T+arrival, and on turn T+arrival+1 do we
    launch from it toward a non-mine planet (relay completion)?

Run:
    python scripts/inspect_chain_game.py --seed 0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--swap", action="store_true", help="focal as P1")
    ap.add_argument("--max-turns", type=int, default=999)
    args = ap.parse_args()

    # Force chain_on for the focal agent's propose() inspection too.
    os.environ["BASELINE_CHAIN_BONUS"] = "1"

    from kaggle_environments import make
    from kaggle_environments.envs.orbit_wars.orbit_wars import Planet, Fleet
    from agents.baseline.main import agent as baseline_agent
    from agents.baseline.proposer import propose, MAX_HORIZON
    from lib.intent import World
    from lib.world_model import WorldModel

    # Wrappers — flip env per call to mirror the A/B harness behaviour.
    def focal_agent(obs, cfg=None):
        os.environ["BASELINE_CHAIN_BONUS"] = "1"
        return baseline_agent(obs, cfg)

    def opp_agent(obs, cfg=None):
        os.environ["BASELINE_CHAIN_BONUS"] = "0"
        return baseline_agent(obs, cfg)

    env = make("orbit_wars", configuration={"seed": args.seed}, debug=False)
    p0, p1 = (opp_agent, focal_agent) if args.swap else (focal_agent, opp_agent)
    me = 1 if args.swap else 0

    # We can't easily inject per-turn probes into env.run, so we step
    # manually: env.reset, then loop until done.
    env.reset(2)
    turn = 0
    history: list[dict] = []
    pending_relays: list[dict] = []  # chain launches awaiting arrival

    while not env.done and turn < args.max_turns:
        states = env.steps[-1]
        obs_focal = states[me].observation
        obs_opp = states[1 - me].observation

        # Reconstruct world from focal's obs to run propose()
        os.environ["BASELINE_CHAIN_BONUS"] = "1"
        obs_d = dict(obs_focal)
        if "comet_planet_ids" not in obs_d:
            obs_d["comet_planet_ids"] = []
        planets = [Planet(*p) for p in obs_d["planets"]]
        my_planets = [p for p in planets if int(p.owner) == me]
        other_planets = [p for p in planets if int(p.owner) != me]
        n_chain_cand = 0
        chain_keys: list[tuple] = []
        if my_planets and other_planets:
            world = World.from_obs(obs_d)
            model = WorldModel.from_world(world)
            omega = float(obs_d.get("angular_velocity", 0.0))
            threatened_mine = [
                p for p in my_planets
                if model.time_to_enemy_threat(int(p.id), me, world) is not None
            ]
            target_pool = other_planets + threatened_mine
            cands = propose(my_planets, target_pool, world, model, me, omega,
                            baseline_len=MAX_HORIZON + 1)
            chain_cands = [c for c in cands if c[8]]
            n_chain_cand = len(chain_cands)
            # Key by (src_id, tgt_id, ships) for emission-matching
            chain_keys = [
                (int(c[1].id), int(c[2].id), int(c[3]),
                 round(float(c[0]), 2))
                for c in chain_cands
            ]
        else:
            cands = []

        # Get focal's actual moves
        focal_action = focal_agent(obs_focal, env.configuration)
        opp_action = opp_agent(obs_opp, env.configuration)
        if args.swap:
            actions = [opp_action, focal_action]
        else:
            actions = [focal_action, opp_action]

        # Match emitted moves to chain candidates (src_id + ships)
        emitted_chain: list[dict] = []
        emit_by_src = {int(m[0]): m for m in focal_action}
        for src_id, tgt_id, ships, cheap in chain_keys:
            m = emit_by_src.get(src_id)
            if m and int(m[2]) == ships:
                emitted_chain.append(
                    {"src": src_id, "tgt": tgt_id, "ships": ships, "cheap": cheap}
                )

        # Relay-completion tracking: for previously-emitted chain launches,
        # check arrival.
        ships_at_step = {int(p.id): (int(p.owner), int(p.ships)) for p in planets}
        relay_events: list[str] = []
        new_pending = []
        for rel in pending_relays:
            if turn < rel["arrival_step"]:
                new_pending.append(rel)
                continue
            tgt = ships_at_step.get(rel["tgt"])
            if tgt is None:
                continue  # planet vanished
            owner, ship_count = tgt
            if owner != me:
                relay_events.append(
                    f"LEG-1 FAILED on T{rel['fire_turn']}→T{turn} src={rel['src']} tgt={rel['tgt']}: tgt owned by p{owner} ({ship_count} ships)"
                )
                continue
            # Captured. Did we launch from it this turn?
            relaunch = any(int(m[0]) == int(rel["tgt"]) for m in focal_action)
            relay_events.append(
                f"LEG-1 OK   on T{rel['fire_turn']}→T{turn} tgt={rel['tgt']} ours={ship_count} ships, relaunch={'yes' if relaunch else 'no'}"
            )

        pending_relays = new_pending
        # Register today's chain firings for future arrival-check
        for ec in emitted_chain:
            # Approximate arrival = current turn + a guess; we don't have
            # easy access to eta here, use 5-10 turns as window.
            pending_relays.append({
                "fire_turn": turn,
                "arrival_step": turn + 8,  # check ~8 turns later
                "src": ec["src"], "tgt": ec["tgt"], "ships": ec["ships"],
            })

        n_my_planets = len(my_planets)
        n_emit = len(focal_action)
        n_chain_emit = len(emitted_chain)

        history.append({
            "turn": turn,
            "my_planets": n_my_planets,
            "n_cand": len(cands),
            "n_chain_cand": n_chain_cand,
            "n_emit": n_emit,
            "n_chain_emit": n_chain_emit,
            "chain_emits": emitted_chain,
            "relay_events": relay_events,
        })

        if relay_events or n_chain_emit > 0:
            ce = ", ".join(
                f"src{e['src']}->tgt{e['tgt']}({e['ships']},Δ{e['cheap']})"
                for e in emitted_chain
            ) or "-"
            print(f"T{turn:3d}  mine={n_my_planets:2d}  cand={len(cands):3d}  chain_cand={n_chain_cand:2d}  emit={n_emit}  chain_emit={n_chain_emit}  | {ce}")
            for ev in relay_events:
                print(f"        relay: {ev}")

        # Advance env one step
        env.step(actions)
        turn += 1

    # Final summary
    total_chain_cand = sum(h["n_chain_cand"] for h in history)
    total_chain_emit = sum(h["n_chain_emit"] for h in history)
    total_emit = sum(h["n_emit"] for h in history)
    n_turns_with_chain_cand = sum(1 for h in history if h["n_chain_cand"] > 0)
    n_turns_with_chain_emit = sum(1 for h in history if h["n_chain_emit"] > 0)
    print()
    print(f"=== SUMMARY  seed={args.seed}  swap={args.swap}  me={me} ===")
    print(f"turns played:                 {turn}")
    print(f"focal launches emitted:       {total_emit}")
    print(f"chain candidates proposed:    {total_chain_cand} across {n_turns_with_chain_cand} turns")
    print(f"chain launches fired:         {total_chain_emit} across {n_turns_with_chain_emit} turns")
    print(f"final rewards:                {[s.reward for s in env.steps[-1]]}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
