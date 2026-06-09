#!/usr/bin/env bash
# Reproducible build for submissions/baseline_champion_stateDrivenK_tier2.py.
# Live-champion (state_driven_k) full config + Tier-2 opp model (dispatch
# fix from chooser.py 2026-06-02; first agent where trained_logreg_policy
# is actually live in rollouts).
set -euo pipefail
cd /home/user/Orbit-wars-kaggle

LIBS="geometry fleet orbit aim combat world_model intent trajectory kinematic_table mechanism \
mission scoring missions/snipe missions/reinforce missions/recapture \
missions/opening missions/drain missions/gang_up missions/opp_archetypes \
planner lookahead lookahead_planner game/interpreter fast_sim \
_validator_tree_walker opp_features_lite opp_model \
v7_search candidate_portfolios value_heads joint_solver/opening_planner \
joint_solver/columns joint_solver/lp"

export BASELINE_JOINT_AGGR=1 BASELINE_JOINT_TOP_K=5 BASELINE_JOINT_MAX_PAIRS=60 \
  BASELINE_REINFORCE_EMIT=1 BASELINE_REINFORCE_ANTICIPATE=1 \
  BASELINE_NEUTRAL_BONUS=2.0 BASELINE_NEUTRAL_EARLY_EXTRA=1.5 \
  BASELINE_NEUTRAL_EARLY_HORIZON=50 BASELINE_ORBITAL_SAFETY=1 BASELINE_PV_ETA=1 \
  BASELINE_LAUNCH_RULES=1 BASELINE_CAPTURE_HORIZON_K=10 \
  BASELINE_STATE_DRIVEN_K=1 BASELINE_STATE_K_CEIL=30 \
  BASELINE_KINEMATIC_TABLE=1 \
  BASELINE_OPP_TIER=2 BASELINE_OPP_FILTER_THRESHOLD=0.15

python scripts/bundle_agent.py agents/baseline --lib $LIBS --force --skip-parity-gate

python - <<'PY'
src, dst = "submissions/baseline.py", "submissions/baseline_champion_stateDrivenK_tier2.py"
header = '''import os as _sk_os
# Live champion (state_driven_k) full config + Tier-2 opp model.
_sk_os.environ.setdefault("BASELINE_JOINT_AGGR", "1")
_sk_os.environ.setdefault("BASELINE_JOINT_TOP_K", "5")
_sk_os.environ.setdefault("BASELINE_JOINT_MAX_PAIRS", "60")
_sk_os.environ.setdefault("BASELINE_REINFORCE_EMIT", "1")
_sk_os.environ.setdefault("BASELINE_REINFORCE_ANTICIPATE", "1")
_sk_os.environ.setdefault("BASELINE_NEUTRAL_BONUS", "2.0")
_sk_os.environ.setdefault("BASELINE_NEUTRAL_EARLY_EXTRA", "1.5")
_sk_os.environ.setdefault("BASELINE_NEUTRAL_EARLY_HORIZON", "50")
_sk_os.environ.setdefault("BASELINE_ORBITAL_SAFETY", "1")
_sk_os.environ.setdefault("BASELINE_PV_ETA", "1")
_sk_os.environ.setdefault("BASELINE_LAUNCH_RULES", "1")
_sk_os.environ.setdefault("BASELINE_CAPTURE_HORIZON_K", "10")
_sk_os.environ.setdefault("BASELINE_STATE_DRIVEN_K", "1")
_sk_os.environ.setdefault("BASELINE_STATE_K_CEIL", "30")
_sk_os.environ.setdefault("BASELINE_KINEMATIC_TABLE", "1")
_sk_os.environ.setdefault("BASELINE_OPP_TIER", "2")
_sk_os.environ.setdefault("BASELINE_OPP_FILTER_THRESHOLD", "0.15")
'''
with open(src) as f:
    body = f.read()
lines = body.split("\n")
insert_at = 0
for i, ln in enumerate(lines):
    s = ln.strip()
    if s.startswith("from __future__"):
        insert_at = i + 1
    elif s == "" or s.startswith("#"):
        continue
    else:
        break
out = "\n".join(lines[:insert_at]) + "\n" + header + "\n" + "\n".join(lines[insert_at:])
with open(dst, "w") as f:
    f.write(out)
print(f"wrote {dst} ({len(out)} bytes); header injected after line {insert_at} (future-safe)")
PY
echo "BUILD_OK submissions/baseline_champion_stateDrivenK_tier2.py"
