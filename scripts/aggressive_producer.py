"""Test the under-expansion diagnosis: does a more EXPANSIVE producer win more?
Real-game losses show producer stops expanding ~step 40-90 while winners keep
grabbing planets. Lever = launch aggression (opposite of the 'deep' test that hurt)."""
import sys, os, dataclasses, time, torch
torch.set_num_threads(2)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, "agents/producer"); sys.path.insert(0, "scripts")
import main as P
from search_harness import ProducerAgent, play_4p, wilson_lo

AGGR = dataclasses.replace(P.CONFIG_4P,
    roi_threshold=0.8,          # 1.5 -> 0.8 : fire on lower-ROI targets (expand more)
    min_ships_to_launch=3.0,    # 4 -> 3
    max_waves_per_turn=9,       # 6 -> 9 : more launches/turn
    max_offensive_targets=14,   # 12 -> 14
)

class AggressiveProducerAgent:
    def __init__(self, config=AGGR):
        self.config = config; self.mem = P.ProducerLiteMemory()
    def reset(self): self.mem.reset()
    def __call__(self, obs):
        pid = int(getattr(obs, "player", 0) if not isinstance(obs, dict) else obs.get("player", 0))
        ot = P.single_obs_to_tensor(obs, player_id=pid)
        if bool((ot["step"] == 0).all()): self.mem.cached_player_count = None
        if self.mem.cached_player_count is None:
            self.mem.cached_player_count = P.largest_initial_player_count(ot)
        with torch.no_grad():
            row = P.run_turn(ot, config=self.config, player_count=int(self.mem.cached_player_count), memory=self.mem)
        self.mem.last_sparse_action_row = row
        return P.sparse_action_row_to_moves(row, obs, player_id=pid)

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 32
    print(f"=== AggressiveProducer vs 3x default, {n} games (beat => >25%) ===")
    firsts = 0; t0 = time.perf_counter()
    for g in range(n):
        seat = g % 4
        agents = [ProducerAgent() for _ in range(4)]
        agents[seat] = AggressiveProducerAgent()
        if play_4p(agents, max_steps=250) == seat: firsts += 1
        if (g+1) % 4 == 0: print(f"  ...{g+1}/{n} firsts={firsts} ({time.perf_counter()-t0:.0f}s)", flush=True)
    print(f"aggressive 1st-place: {firsts}/{n} = {100*firsts/n:.0f}%  (Wilson-lo {100*wilson_lo(firsts,n):.0f}%)")
