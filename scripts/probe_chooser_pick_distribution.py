"""Probe: does the chooser's prerank skew toward incoming-arc captures?

At several snapshot turns of a real game, reconstruct the proposer's
prerank from the observation, then bin each candidate by:
  - SIDE: incoming (target rotating toward source) vs outgoing
  - RANK: top-3, top-10, all (cheap_delta-sorted)
  - cheap_delta sign (the proposer's pre-score; positive => likely to be
    green-lit by the chooser)

If incoming dominates the top of the prerank, the chooser is picking
correctly. If outgoing is overrepresented at the top despite having
2x larger eta, the implicit pv discount isn't enough — there's an
intervention available (explicit asymmetry penalty in
cheap_marginal_value).
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# Use the unmodified production path (env var unset) so the chooser's
# input distribution is what shipping production sees today.
os.environ.pop("BASELINE_OPP_TRAJ_TIER", None)

from kaggle_environments import make

from agents.baseline.main import agent as baseline_agent
from agents.baseline.proposer import propose, MAX_HORIZON
from lib.intent import World
from lib.world_model import WorldModel


CENTER = 50.0


def signed_angular_distance(src_xy, tgt_xy, omega: float) -> float:
    a_src = math.atan2(src_xy[1] - CENTER, src_xy[0] - CENTER)
    a_tgt = math.atan2(tgt_xy[1] - CENTER, tgt_xy[0] - CENTER)
    d = a_tgt - a_src
    while d > math.pi:
        d -= 2 * math.pi
    while d <= -math.pi:
        d += 2 * math.pi
    return d * (1.0 if omega > 0 else -1.0)


def _is_orbiting(planet_xy) -> bool:
    return math.hypot(planet_xy[0] - CENTER, planet_xy[1] - CENTER) > 1.0


def snapshot_chooser_input(obs, me: int) -> dict:
    """Reconstruct the proposer's prerank from this turn's obs."""
    world = World.from_obs(obs)
    model = WorldModel.from_world(world)
    omega = float(world.omega)
    planets = list(world.planets_by_id.values())
    my_planets = [p for p in planets if p.owner == me]
    target_pool = [p for p in planets if p.owner != me]

    prerank = propose(
        my_planets, target_pool, world, model, me, omega,
        baseline_len=MAX_HORIZON + 1,
    )

    rows = []
    for cheap_delta, src, tgt, ships, angle, eta, horizon, wait_N in prerank:
        # Both planets must be orbiting non-comets for the asymmetry
        # framing to apply. Skip comets (they follow polynomial paths,
        # not orbital rotation).
        if int(tgt.id) in world.comet_ids:
            continue
        if int(src.id) in world.comet_ids:
            continue
        if not _is_orbiting((src.x, src.y)) or not _is_orbiting((tgt.x, tgt.y)):
            continue
        sd = signed_angular_distance((src.x, src.y), (tgt.x, tgt.y), omega)
        rows.append({
            "cheap_delta": float(cheap_delta),
            "src": int(src.id),
            "tgt": int(tgt.id),
            "ships": int(ships),
            "eta": int(eta),
            "wait_N": int(wait_N),
            "signed_ang": sd,
            "side": "INCOMING" if sd < 0 else "OUTGOING",
        })
    return {
        "n_my_planets": len(my_planets),
        "n_target_pool": len(target_pool),
        "n_prerank_orbital": len(rows),
        "rows": rows,
        "omega": omega,
    }


def summarize(turn: int, snap: dict) -> None:
    rows = snap["rows"]
    if not rows:
        print(f"  turn {turn:3d}  (no orbital prerank — skipping)")
        return
    # Sort by cheap_delta descending (the proposer's own ordering).
    rows.sort(key=lambda r: -r["cheap_delta"])

    n = len(rows)
    n_pos = sum(1 for r in rows if r["cheap_delta"] > 0)
    n_in = sum(1 for r in rows if r["side"] == "INCOMING")
    n_out = sum(1 for r in rows if r["side"] == "OUTGOING")

    top3 = rows[:3]
    top10 = rows[:10]
    pos = [r for r in rows if r["cheap_delta"] > 0]

    def pct_in(xs):
        if not xs:
            return None
        return 100.0 * sum(1 for r in xs if r["side"] == "INCOMING") / len(xs)

    def avg_eta(xs, side):
        ys = [r["eta"] for r in xs if r["side"] == side]
        return sum(ys) / len(ys) if ys else None

    print(f"  turn {turn:3d}  omega={snap['omega']:+.4f}  "
          f"my={snap['n_my_planets']}  tgts={snap['n_target_pool']}")
    print(f"    prerank rows (orbital only): {n}   delta>0: {n_pos}   "
          f"INCOMING: {n_in}   OUTGOING: {n_out}")
    print(f"    top-3   % incoming: "
          f"{pct_in(top3):>5.1f}%  (delta>0 in top3: "
          f"{sum(1 for r in top3 if r['cheap_delta'] > 0)})")
    print(f"    top-10  % incoming: "
          f"{pct_in(top10):>5.1f}%")
    if pos:
        print(f"    delta>0 % incoming: {pct_in(pos):>5.1f}%   "
              f"(n_pos = {len(pos)})")
    eta_in = avg_eta(rows, "INCOMING")
    eta_out = avg_eta(rows, "OUTGOING")
    if eta_in is not None and eta_out is not None:
        print(f"    avg eta — INCOMING: {eta_in:.1f}   OUTGOING: {eta_out:.1f}   "
              f"ratio: {eta_out/eta_in:.2f}x")
    # Show top 5 picks in detail.
    print(f"    top-5 prerank entries:")
    for r in rows[:5]:
        print(f"      src={r['src']:>2} -> tgt={r['tgt']:>2}  side={r['side']:>8}  "
              f"eta={r['eta']:>3}  ships={r['ships']:>3}  "
              f"cheap_delta={r['cheap_delta']:+.4f}")


def run(seed: int, snapshot_turns: tuple[int, ...]):
    env = make("orbit_wars", configuration={"seed": seed}, debug=False)
    env.reset(num_agents=2)
    print(f"\n========== seed={seed} ==========")

    # Run the game with the production baseline (env var unset). Save
    # obs snapshots from P0's perspective at each requested turn.
    snapshots = {}
    snapshot_set = set(snapshot_turns)

    def wrap(obs, configuration=None):
        turn = int(obs.get("step", 0)) if isinstance(obs, dict) \
            else int(getattr(obs, "step", 0))
        if turn in snapshot_set and turn not in snapshots:
            snapshots[turn] = obs
        return baseline_agent(obs, configuration)

    # Use the same agent at both seats (self-play production).
    env.run([wrap, wrap])

    print(f"game completed in {len(env.steps)} turns")
    for turn in sorted(snapshot_turns):
        if turn not in snapshots:
            print(f"\n  turn {turn:3d}  (game ended before this turn)")
            continue
        # P0 perspective.
        snap = snapshot_chooser_input(snapshots[turn], me=0)
        summarize(turn, snap)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--turns", type=str, default="10,30,60,100",
                    help="comma-separated snapshot turns")
    args = ap.parse_args()
    turns = tuple(int(t.strip()) for t in args.turns.split(","))
    run(args.seed, turns)
