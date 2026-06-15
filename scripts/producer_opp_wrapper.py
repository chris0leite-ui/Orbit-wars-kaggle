"""Search wrapper with producer as the OPPONENT model (not lite_greedy).

Closes the last gap in the search test: the original wrapper fixed ME (producer
in rollouts) but left opponents weak (lite_greedy), so the rollout mis-predicted
rival responses. Here ALL four seats are producer in the rollout -> a real
opponent model. Necessarily shallow (all-producer rollouts are ~12ms/call):
K=3, so ~3 candidates x 3 steps x 4 seats x 12ms ~= 430ms/turn.
"""
from __future__ import annotations
import sys, os, time, torch
torch.set_num_threads(2)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, "agents/producer")
sys.path.insert(0, "scripts")
import main as P
from lib import fast_sim
from search_wrapper import SearchWrapper, producer_action, _score
from search_harness import ProducerAgent, play_4p, wilson_lo


class ProducerOppWrapper(SearchWrapper):
    def __init__(self, K=3):
        super().__init__(K=K)
        self.opp_rts = [P.ProducerLiteRuntime() for _ in range(4)]

    def _opp(self, i, obs):
        self.opp_rts[i].reset()
        return producer_action(self.opp_rts[i], obs)

    def _eval(self, snap, me, cand):
        s = fast_sim.clone(snap)
        acts = [cand if i == me else self._opp(i, s.state[i].observation) for i in range(4)]
        s = fast_sim.step(s, acts, in_place=True)
        for _ in range(self.K - 1):
            if s.fake_env.done:
                break
            self.roll.reset()
            mine = producer_action(self.roll, s.state[me].observation)
            acts = [mine if i == me else self._opp(i, s.state[i].observation) for i in range(4)]
            s = fast_sim.step(s, acts, in_place=True)
        return _score(s.state[0].observation, me)


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    K = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    ms = int(sys.argv[3]) if len(sys.argv) > 3 else 180
    print(f"=== ProducerOppWrapper(K={K}) vs 3x default producer, {n} games, max_steps={ms} ===")
    firsts = 0; t0 = time.perf_counter()
    for g in range(n):
        seat = g % 4
        agents = [ProducerAgent() for _ in range(4)]
        agents[seat] = ProducerOppWrapper(K=K)
        if play_4p(agents, max_steps=ms) == seat:
            firsts += 1
        print(f"  game {g+1}/{n} seat{seat} firsts={firsts} ({time.perf_counter()-t0:.0f}s)", flush=True)
    print(f"producer-opp wrapper 1st-place: {firsts}/{n} = {100*firsts/n:.0f}%  "
          f"(Wilson-lo {100*wilson_lo(firsts,n):.0f}%)  [vs lite-opp wrapper's 28%]")
