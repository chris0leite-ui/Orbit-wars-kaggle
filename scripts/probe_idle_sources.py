"""Idle-source probe — does the champion's OFFENSIVE chooser leave owned
sources uncommitted, in CLOSE games vs a strong opponent?

Re-tests the coordination-axis null (MULTI_BRANCH.md L113-122). That null
("source-saturated => no idle sources for teamwork") was measured only vs weak
opponents the champion blows out (source-saturated by construction). Here we
run the champion vs v7_minimax (a strong maximin opponent that produces CLOSE
games) and measure the idle-source distribution segmented by instantaneous
ship-share, which is where teamwork could actually fire.

Read-only: champion behaviour is byte-identical (probe is gated on
BASELINE_IDLE_PROBE and only appends JSONL). v7_minimax reads no BASELINE_* keys
=> no in-process env contamination; only the champion seat records.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
os.chdir(REPO)
sys.path.insert(0, str(REPO))

# --- champion config (from submissions/baseline_pv_eta_anchor_1163.py header) ---
CHAMP_CFG = {
    "BASELINE_JOINT_AGGR": "1", "BASELINE_JOINT_TOP_K": "5",
    "BASELINE_JOINT_MAX_PAIRS": "60", "BASELINE_REINFORCE_EMIT": "1",
    "BASELINE_REINFORCE_ANTICIPATE": "1", "BASELINE_NEUTRAL_BONUS": "2.0",
    "BASELINE_NEUTRAL_EARLY_EXTRA": "1.5", "BASELINE_NEUTRAL_EARLY_HORIZON": "50",
    "BASELINE_ORBITAL_SAFETY": "1", "BASELINE_PV_ETA": "1",
    "BASELINE_VALUE_HEAD": "hybrid", "BASELINE_CHOOSER": "trajectory",
    "BASELINE_JOINT": "1", "PV_GAMMA": "0.99",
}
for k, v in CHAMP_CFG.items():
    os.environ[k] = v

PROBE_PATH = "/tmp/idle_probe.jsonl"
if os.path.exists(PROBE_PATH):
    os.remove(PROBE_PATH)
os.environ["BASELINE_IDLE_PROBE"] = PROBE_PATH

from kaggle_environments import make  # noqa: E402
from agents.baseline.main import agent as champ  # noqa: E402
from agents.v7_minimax.main import agent as opp   # noqa: E402

N_GAMES = int(os.environ.get("PROBE_GAMES", "3"))
N_MIRROR = int(os.environ.get("PROBE_MIRROR", "3"))
seeds = [101, 202, 303, 404, 505, 606, 707, 808]

print(f"Running {N_GAMES} games champion vs v7_minimax + {N_MIRROR} mirror...",
      flush=True)
for i in range(N_GAMES):
    seed = seeds[i % len(seeds)]
    champ_seat0 = (i % 2 == 0)
    pair = [champ, opp] if champ_seat0 else [opp, champ]
    env = make("orbit_wars", configuration={"seed": seed}, debug=False)
    env.run(pair)
    rew = [s["reward"] for s in env.state]
    champ_idx = 0 if champ_seat0 else 1
    res = "WIN" if rew[champ_idx] == 1 else ("LOSS" if rew[champ_idx] == -1 else "TIE")
    print(f"  vs-minimax {i+1}/{N_GAMES} seed={seed} champ_seat={champ_idx} "
          f"rewards={rew} -> champ {res} ({env.state[0]['observation']['step']} turns)",
          flush=True)

# Champion mirror: identical config both seats => NO env contamination
# (contamination only bites when the two seats want DIFFERENT values for a
# shared key; a mirror wants the same). Generates genuinely CLOSE games that
# stay contested into mid/late game — the regime the coordination null never saw.
for i in range(N_MIRROR):
    seed = seeds[(i + 4) % len(seeds)]
    env = make("orbit_wars", configuration={"seed": seed}, debug=False)
    env.run([champ, champ])
    rew = [s["reward"] for s in env.state]
    print(f"  mirror {i+1}/{N_MIRROR} seed={seed} rewards={rew} "
          f"({env.state[0]['observation']['step']} turns)", flush=True)

# --- aggregate ---
rows = [json.loads(ln) for ln in open(PROBE_PATH) if ln.strip()]
print(f"\nCollected {len(rows)} champion-turn rows.\n")


def share(r):
    tot = r["my_ships"] + r["opp_ships"]
    return r["my_ships"] / tot if tot > 0 else 0.5


def bucket_share(r):
    s = share(r)
    if s > 0.60:
        return "winning(>.60)"
    if s < 0.40:
        return "losing(<.40)"
    return "CLOSE(.40-.60)"


def phase(r):
    st = r["step"]
    if st < 100:
        return "early(<100)"
    if st < 350:
        return "mid(100-350)"
    return "late(>350)"


def summarize(label, rs):
    if not rs:
        print(f"{label:28s} n=0")
        return
    n = len(rs)
    mean_elig = sum(r["elig_src"] for r in rs) / n
    mean_used = sum(r["used_src"] for r in rs) / n
    mean_idle = sum(r["idle_src"] for r in rs) / n
    frac_idle1 = sum(1 for r in rs if r["idle_src"] >= 1) / n
    frac_idle2 = sum(1 for r in rs if r["idle_src"] >= 2) / n
    idle_frac = (sum(r["idle_src"] for r in rs)
                 / max(1, sum(r["elig_src"] for r in rs)))
    print(f"{label:28s} n={n:5d}  elig={mean_elig:4.1f} used={mean_used:4.1f} "
          f"idle={mean_idle:4.1f}  idle/elig={idle_frac:4.0%}  "
          f">=1idle:{frac_idle1:4.0%}  >=2idle:{frac_idle2:4.0%}")


print("=== by instantaneous ship-share ===")
for b in ("winning(>.60)", "CLOSE(.40-.60)", "losing(<.40)"):
    summarize(b, [r for r in rows if bucket_share(r) == b])

print("\n=== CLOSE games only, by phase ===")
close = [r for r in rows if bucket_share(r) == "CLOSE(.40-.60)"]
for p in ("early(<100)", "mid(100-350)", "late(>350)"):
    summarize(p, [r for r in close if phase(r) == p])

print("\n=== ALL turns, by phase ===")
for p in ("early(<100)", "mid(100-350)", "late(>350)"):
    summarize(p, [r for r in rows if phase(r) == p])

print("\n=== overall ===")
summarize("ALL", rows)
