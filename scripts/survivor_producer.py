"""Survivor: producer with a defensively-robust OPENING (steps < cutoff).

Hypothesis: the early-death loss cluster (~step 100, ~1/3 of real losses) is
over-extension/fragility, not weak economy. Robust opening = keep garrison
reserves (higher min_ships_to_launch), launch only high-confidence captures
(higher roi_threshold), defend more (more defensive targets). After the rush
window it reverts to default producer so the economic phase is unchanged.
"""
from __future__ import annotations
import sys, os, time, dataclasses, statistics, torch
torch.set_num_threads(2)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, "agents/producer")
sys.path.insert(0, "scripts")
import main as P

EARLY_CUTOFF = 100


def robustify(cfg):
    return dataclasses.replace(
        cfg,
        roi_threshold=max(cfg.roi_threshold, 2.2),       # only clearly-good launches
        min_ships_to_launch=max(cfg.min_ships_to_launch, 8.0),  # keep garrison reserves
        max_defensive_targets=cfg.max_defensive_targets + 2,    # defend more
    )


class SurvivorProducerAgent:
    def __init__(self, cutoff=EARLY_CUTOFF):
        self.mem = P.ProducerLiteMemory()
        self.cutoff = cutoff

    def reset(self):
        self.mem.reset()

    def __call__(self, obs):
        pid = int(getattr(obs, "player", 0) if not isinstance(obs, dict) else obs.get("player", 0))
        ot = P.single_obs_to_tensor(obs, player_id=pid)
        step = int(ot["step"].reshape(-1)[0].item())
        if step == 0:
            self.mem.cached_player_count = None
        if self.mem.cached_player_count is None:
            self.mem.cached_player_count = P.largest_initial_player_count(ot)
        base = P._config_for(int(self.mem.cached_player_count))
        cfg = robustify(base) if step < self.cutoff else base
        with torch.no_grad():
            row = P.run_turn(ot, config=cfg, player_count=int(self.mem.cached_player_count), memory=self.mem)
        self.mem.last_sparse_action_row = row
        return P.sparse_action_row_to_moves(row, obs, player_id=pid)


if __name__ == "__main__":
    from kaggle_environments import make
    from lib.opp_model import lite_greedy_policy
    env = make("orbit_wars", configuration={"episodeSteps": 500}); env.reset(num_agents=4)
    ag = SurvivorProducerAgent(); ag.reset()
    lat, nlaunch_early, nlaunch_late = [], [], []
    for step in range(160):
        if env.done: break
        o0 = env.state[0].observation
        t = time.perf_counter(); a0 = ag(o0); dt = (time.perf_counter()-t)*1000
        if step >= 5: lat.append(dt)
        (nlaunch_early if step < EARLY_CUTOFF else nlaunch_late).append(len(a0))
        env.step([a0] + [lite_greedy_policy(env.state[i].observation) for i in range(1, 4)])
    print(f"survivor latency: median {statistics.median(lat):.1f}ms max {max(lat):.1f}ms ({max(lat)/10:.0f}% budget)")
    print(f"launches/turn  early(<{EARLY_CUTOFF}): {statistics.mean(nlaunch_early):.2f}  "
          f"late: {statistics.mean(nlaunch_late):.2f}  (early should be <= late if more selective)")
