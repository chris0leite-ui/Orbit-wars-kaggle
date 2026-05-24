"""scripts/wave_trace.py — one game baseline_wave (in-process) vs orbitfix,
WAVE_PROPOSER_TRACE on, aggregate wave-spread stats."""
import os, io, sys, collections
from contextlib import redirect_stdout
from pathlib import Path

# Resolve repo root first so `agents.baseline.main` doesn't get shadowed
# by kaggle_environments' lux_ai_s3 env's own `agents.py` (relative import
# break).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.update({
    "BASELINE_JOINT_AGGR": "1", "BASELINE_JOINT_TOP_K": "5",
    "BASELINE_JOINT_MAX_PAIRS": "60",
    "BASELINE_REINFORCE_EMIT": "1", "BASELINE_REINFORCE_ANTICIPATE": "1",
    "BASELINE_NEUTRAL_BONUS": "2.0", "BASELINE_NEUTRAL_EARLY_EXTRA": "1.5",
    "BASELINE_NEUTRAL_EARLY_HORIZON": "50", "BASELINE_ORBITAL_SAFETY": "1",
    "BASELINE_WAVE_PROPOSER": "1", "BASELINE_WAVE_MAX_PER_TURN": "8",
    "BASELINE_WAVE_K": "4", "BASELINE_WAVE_MARGIN": "2",
    "BASELINE_WAVE_TEMPO_GUARD": "8",
    "BASELINE_BLEED_PENALTY": "1", "BASELINE_BLEED_BETA": "0.05",
    "BASELINE_STOCKPILE_PENALTY": "1", "BASELINE_STOCKPILE_EPS": "0.001",
    "BASELINE_STOCKPILE_TARGET": "50",
    "WAVE_PROPOSER_TRACE": "1",
})

from kaggle_environments import make
from agents.baseline.main import agent

env = make("orbit_wars", configuration={"seed": 0, "episodeSteps": 200}, debug=False)
buf = io.StringIO()
with redirect_stdout(buf):
    env.run([lambda o, c=None: agent(o, c),
             "submissions/baseline_joint_aggr_consolidated_orbitfix.py"])
out = buf.getvalue()

# Surface a sample of raw WAVE lines.
wave_lines = [l for l in out.splitlines() if l.startswith("WAVE ")]
print(f"=== first 10 raw WAVE lines (of {len(wave_lines)}) ===")
for l in wave_lines[:10]:
    print(l)
print(f"=== last 5 raw WAVE lines ===")
for l in wave_lines[-5:]:
    print(l)

# Spread distribution.
spreads = [int(l.split("spread=")[1].split()[0]) for l in wave_lines]
zero_spread = sum(1 for s in spreads if s == 0)
print()
print(f"total wave candidates emitted: {len(wave_lines)}")
print(f"spread==0 (true same-step waves): {zero_spread} "
      f"({100 * zero_spread / max(1, len(wave_lines)):.1f}%)")
print(f"mean spread: {sum(spreads) / max(1, len(spreads)):.2f}")
print(f"max spread: {max(spreads) if spreads else 0}")
c = collections.Counter(spreads)
print(f"spread distribution: {dict(sorted(c.items()))}")

# Per-turn wave count.
turn_to_count: dict[int, int] = collections.defaultdict(int)
for l in wave_lines:
    step = int(l.split("step=")[1].split()[0])
    turn_to_count[step] += 1
print(f"\nturns with >=1 wave: {len(turn_to_count)}")
print(f"max waves on any single turn: {max(turn_to_count.values()) if turn_to_count else 0}")

# Final game state.
r = env.steps[-1]
print(f"\ngame end: reward={[s.reward for s in r]} steps={len(env.steps)}")
