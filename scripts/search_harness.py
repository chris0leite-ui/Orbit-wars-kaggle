"""Search-wrapper A/B harness.

Runs a challenger agent vs 3 bare producer_plus in 4P, rotating the challenger
across all 4 seats to cancel board asymmetry, and reports 1st-place rate with a
Wilson 95% lower bound. ProducerAgent gives each seat an independent runtime so
four producers can share one game without corrupting each other's memory.
"""
from __future__ import annotations
import sys, math, time
import torch

torch.set_num_threads(2)  # approximate Kaggle's ~1.6 CPU
sys.path.insert(0, "agents/producer")
from kaggle_environments import make
from main import ProducerLiteRuntime, single_obs_to_tensor, sparse_action_row_to_moves  # noqa: E402


class ProducerAgent:
    """Bare producer_plus with its own runtime/memory (one per seat)."""
    def __init__(self):
        self.rt = ProducerLiteRuntime()

    def reset(self):
        self.rt.reset()

    def __call__(self, obs):
        pid = int(obs.get("player", 0) if isinstance(obs, dict) else obs.player)
        ot = single_obs_to_tensor(obs, player_id=pid)
        with torch.no_grad():
            row = self.rt.tensor_action(ot)
        return sparse_action_row_to_moves(row, obs, player_id=pid)


def _winner(env, ns=4):
    o = env.state[0].observation
    sc = [0.0] * ns
    for p in o["planets"]:
        ow = int(p[1])
        if 0 <= ow < ns:
            sc[ow] += float(p[5])
    for f in (o.get("fleets", []) or []):
        ow = int(f[1])
        if 0 <= ow < ns:
            sc[ow] += float(f[6])
    return int(max(range(ns), key=lambda i: sc[i]))


def play_4p(agents, max_steps=500):
    env = make("orbit_wars", configuration={"episodeSteps": 500})
    env.reset(num_agents=4)
    for a in agents:
        if hasattr(a, "reset"):
            a.reset()
    for _ in range(max_steps):
        if env.done:
            break
        acts = [agents[i](env.state[i].observation) for i in range(4)]
        env.step(acts)
    return _winner(env)


def wilson_lo(k, n, z=1.96):
    if n == 0:
        return 0.0
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n)
    return (c - m) / d


def ab_rotate(make_challenger, n_games):
    """challenger in a rotating seat vs 3 bare ProducerAgents. Returns (firsts, n)."""
    firsts = 0
    t0 = time.perf_counter()
    for g in range(n_games):
        seat = g % 4
        agents = [ProducerAgent() for _ in range(4)]
        agents[seat] = make_challenger()
        if play_4p(agents) == seat:
            firsts += 1
        if (g + 1) % 4 == 0:
            print(f"  ...{g+1}/{n_games}  firsts={firsts}  "
                  f"({time.perf_counter()-t0:.0f}s)", flush=True)
    return firsts, n_games


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    print(f"=== NULL sanity: bare producer vs 3 bare producers, {n} games (expect ~25%) ===")
    k, N = ab_rotate(lambda: ProducerAgent(), n)
    print(f"null challenger 1st-place: {k}/{N} = {100*k/N:.0f}%  "
          f"(Wilson-lo {100*wilson_lo(k,N):.0f}%)  [healthy if ~25%, no seat blowout]")
