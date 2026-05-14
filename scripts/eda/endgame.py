"""Mine 4 — Endgame anatomy (last 100 turns).

For each replay, sample states at turn fractions {0.80, 0.85, 0.90,
0.95, 0.99} of game length and report winner's ownership share, ship
share, in-flight fleet ships, plus whether the last 50 turns contain a
successful enemy-capture by the winner (offence) vs only reinforcement
(consolidation).

CLI:
    python -m scripts.eda.endgame --replay-dirs audit/external/replays --out audit/<date>-endgame.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def _winner_seat(replay):
    rewards = replay.get("rewards") or []
    if rewards.count(1) == 1:
        return rewards.index(1)
    return None


def _state_at(replay, step_idx):
    obs = replay["steps"][step_idx][0]["observation"]
    return obs


def _shares(obs, seat):
    planets = obs["planets"]
    fleets = obs.get("fleets") or []
    own_planets = sum(1 for p in planets if p[1] == seat)
    n_planets = len(planets)
    own_ships_planet = sum(p[5] for p in planets if p[1] == seat)
    own_ships_fleet = sum(f[6] for f in fleets if f[1] == seat)
    tot_ships = sum(p[5] for p in planets) + sum(f[6] for f in fleets)
    own_prod = sum(p[6] for p in planets if p[1] == seat)
    tot_prod = sum(p[6] for p in planets)
    return {
        "planet_share": own_planets / n_planets if n_planets else 0,
        "ship_share": (own_ships_planet + own_ships_fleet) / tot_ships if tot_ships else 0,
        "prod_share": own_prod / tot_prod if tot_prod else 0,
        "n_fleets_owned": sum(1 for f in fleets if f[1] == seat),
        "ships_in_flight_owned": own_ships_fleet,
    }


def extract(replay_path: Path) -> dict:
    replay = json.load(open(replay_path))
    winner = _winner_seat(replay)
    if winner is None:
        return None
    steps = replay["steps"]
    n = len(steps)
    fractions = [0.80, 0.85, 0.90, 0.95, 0.99]
    sample_idx = [min(int(n * f), n - 1) for f in fractions]
    samples = []
    for idx, frac in zip(sample_idx, fractions):
        obs = _state_at(replay, idx)
        s = _shares(obs, winner)
        s["step"] = idx
        s["frac"] = frac
        samples.append(s)
    # Late-game offence detection: did winner gain ANY planet from an
    # enemy in the last 50 turns?
    last50_start = max(0, n - 50)
    obs_start = _state_at(replay, last50_start)
    obs_end = _state_at(replay, n - 1)
    planet_owners_start = {p[0]: p[1] for p in obs_start["planets"]}
    gained_from_enemy = False
    for p in obs_end["planets"]:
        if p[1] == winner:
            prev = planet_owners_start.get(p[0])
            if prev is not None and prev >= 0 and prev != winner:
                gained_from_enemy = True
                break
    # Ship-share trajectory shape
    ship_share_trajectory = [s["ship_share"] for s in samples]
    # Did ship share grow or contract in last 100 turns?
    trajectory_delta = ship_share_trajectory[-1] - ship_share_trajectory[0]

    return {
        "replay": replay_path.name,
        "winner_seat": winner,
        "n_steps": n,
        "samples": samples,
        "ship_share_trajectory": ship_share_trajectory,
        "ship_share_delta_last100": trajectory_delta,
        "late_offence": gained_from_enemy,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay-dirs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--glob", default="*.json")
    args = ap.parse_args(argv)

    games = []
    for d in args.replay_dirs:
        for fp in sorted(Path(d).glob(args.glob)):
            try:
                g = extract(fp)
                if g:
                    games.append(g)
            except Exception as e:
                print(f"  SKIP {fp.name}: {e}", file=sys.stderr)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"n_games": len(games), "games": games}))
    print(f"wrote {len(games)} games -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
