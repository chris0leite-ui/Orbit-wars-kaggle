"""Does producer convert its ~98% headroom into strength by thinking deeper?

DeepProducerAgent = producer with its conservative 4P config relaxed (3x horizon,
wider shortlists, more waves). A/B vs the default producer answers: is the plateau
a compute/depth problem (deep wins) or a strategy problem (deep ties)?
"""
from __future__ import annotations
import sys, os, time, dataclasses, statistics, torch
torch.set_num_threads(2)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, "agents/producer")
sys.path.insert(0, "scripts")
import main as P
from search_harness import ProducerAgent, play_4p, wilson_lo

DEEP = dataclasses.replace(
    P.CONFIG_4P,
    horizon=39,                       # 13 -> 39 (3x deeper projection)
    max_sources_per_lane=12,          # 6 -> 12
    max_defensive_targets=4,          # 2 -> 4
    max_offensive_targets=12,
    max_waves_per_turn=10,            # 6 -> 10
    max_regroup_targets_per_source=12,
)


class DeepProducerAgent:
    def __init__(self, config=DEEP):
        self.config = config
        self.mem = P.ProducerLiteMemory()

    def reset(self):
        self.mem.reset()

    def __call__(self, obs):
        pid = int(getattr(obs, "player", 0) if not isinstance(obs, dict) else obs.get("player", 0))
        ot = P.single_obs_to_tensor(obs, player_id=pid)
        if bool((ot["step"] == 0).all()):
            self.mem.cached_player_count = None
        if self.mem.cached_player_count is None:
            self.mem.cached_player_count = P.largest_initial_player_count(ot)
        with torch.no_grad():
            row = P.run_turn(ot, config=self.config,
                             player_count=int(self.mem.cached_player_count), memory=self.mem)
        self.mem.last_sparse_action_row = row
        return P.sparse_action_row_to_moves(row, obs, player_id=pid)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "smoke"
    if mode == "smoke":
        from kaggle_environments import make
        env = make("orbit_wars", configuration={"episodeSteps": 500}); env.reset(num_agents=4)
        ag = DeepProducerAgent(); ag.reset()
        from lib.opp_model import lite_greedy_policy
        lat = []
        for step in range(120):
            if env.done: break
            o0 = env.state[0].observation
            t = time.perf_counter(); a0 = ag(o0); dt = (time.perf_counter()-t)*1000
            if step >= 5: lat.append(dt)
            env.step([a0] + [lite_greedy_policy(env.state[i].observation) for i in range(1, 4)])
        print(f"deep latency: median {statistics.median(lat):.1f}ms  max {max(lat):.1f}ms  "
              f"-> {max(lat)/10:.0f}% of budget (worst)")
    else:
        n = int(sys.argv[1])
        print(f"=== DeepProducer vs 3x default producer, {n} games (beat => >25%) ===")
        firsts = 0; t0 = time.perf_counter()
        for g in range(n):
            seat = g % 4
            agents = [ProducerAgent() for _ in range(4)]
            agents[seat] = DeepProducerAgent()
            if play_4p(agents, max_steps=250) == seat:
                firsts += 1
            if (g+1) % 4 == 0:
                print(f"  ...{g+1}/{n} firsts={firsts} ({time.perf_counter()-t0:.0f}s)", flush=True)
        print(f"deep 1st-place: {firsts}/{n} = {100*firsts/n:.0f}%  (Wilson-lo {100*wilson_lo(firsts,n):.0f}%)")
