"""MVP search wrapper around producer_plus.

Bare producer plays greedy 1-ply. This wrapper generates a few candidate
first-moves (producer's plan, a more-conservative variant, idle), evaluates
each by a short fast_sim rollout where ME plays producer-strong and opponents
play lite_greedy, and picks the candidate with the best gap-to-strongest leaf.
The key change vs the old (capped) search: ME is producer in the rollout, not
idle — so candidates aren't under-rated by a passive self.
"""
from __future__ import annotations
import sys, os, time, torch
torch.set_num_threads(2)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))  # repo root (lib)
sys.path.insert(0, "agents/producer")
sys.path.insert(0, "scripts")
import main as P
from main import single_obs_to_tensor, sparse_action_row_to_moves
from lib import fast_sim
from lib.opp_model import lite_greedy_policy


def producer_action(rt, o):
    pid = int(getattr(o, "player", 0) if not isinstance(o, dict) else o.get("player", 0))
    ot = single_obs_to_tensor(o, player_id=pid)
    with torch.no_grad():
        row = rt.tensor_action(ot)
    return sparse_action_row_to_moves(row, o, player_id=pid)


def _score(o, me, ns=4):
    sc = [0.0] * ns
    for p in o.planets:
        ow = int(p[1])
        if 0 <= ow < ns:
            sc[ow] += float(p[5])
    for f in (getattr(o, "fleets", []) or []):
        ow = int(f[1])
        if 0 <= ow < ns:
            sc[ow] += float(f[6])
    opp = max(sc[i] for i in range(ns) if i != me)
    return sc[me] - opp


class SearchWrapper:
    def __init__(self, K=4):
        self.rt = P.ProducerLiteRuntime()       # produces the base plan
        self.roll = P.ProducerLiteRuntime()      # me-policy inside rollouts
        self.K = K

    def reset(self):
        self.rt.reset()

    def _candidates(self, base):
        cands = [base, []]                       # full plan, idle
        if base:
            j = max(range(len(base)), key=lambda i: base[i][2])   # drop biggest launch
            cands.append([l for k, l in enumerate(base) if k != j])
        return cands

    def _eval(self, snap, me, cand):
        s = fast_sim.clone(snap)
        acts = [cand if i == me else lite_greedy_policy(s.state[i].observation)
                for i in range(4)]
        s = fast_sim.step(s, acts, in_place=True)
        for _ in range(self.K - 1):
            if s.fake_env.done:
                break
            self.roll.reset()
            mine = producer_action(self.roll, s.state[me].observation)
            acts = [mine if i == me else lite_greedy_policy(s.state[i].observation)
                    for i in range(4)]
            s = fast_sim.step(s, acts, in_place=True)
        return _score(s.state[0].observation, me)

    def __call__(self, obs):
        me = int(getattr(obs, "player", 0))
        base = producer_action(self.rt, obs)
        cands = self._candidates(base)
        snap = fast_sim.from_obs(obs, None, episode_seed=0, num_seats=4)
        best, bestv = base, -1e18
        for c in cands:
            v = self._eval(snap, me, c)
            if v > bestv:
                bestv, best = v, c
        return best


if __name__ == "__main__":
    from search_harness import ProducerAgent, play_4p, wilson_lo
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    K = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    ms = int(sys.argv[3]) if len(sys.argv) > 3 else 220
    print(f"=== SMOKE: SearchWrapper(K={K}) vs 3x bare producer, {n} games, max_steps={ms} ===")
    firsts = 0
    t0 = time.perf_counter()
    for g in range(n):
        seat = g % 4
        agents = [ProducerAgent() for _ in range(4)]
        agents[seat] = SearchWrapper(K=K)
        if play_4p(agents, max_steps=ms) == seat:
            firsts += 1
        print(f"  game {g+1}/{n} seat{seat} -> firsts={firsts}  ({time.perf_counter()-t0:.0f}s)", flush=True)
    print(f"wrapper 1st-place: {firsts}/{n} = {100*firsts/n:.0f}%  (Wilson-lo {100*wilson_lo(firsts,n):.0f}%)  "
          f"[beat bare producer => >25%]")
