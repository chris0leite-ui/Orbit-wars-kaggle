#!/usr/bin/env bash
# Reproducible build for submissions/baseline_state_driven_k.py.
# Champion (launch_rules_universal) full config, verbatim, + the state-driven
# horizon-K lever baked ON (BASELINE_STATE_DRIVEN_K=1, ceil=30, floor=10).
# Bundle = agents/baseline + the champion lib set MINUS kinematic_table
# (deleted, behaviorally neutral). Config header injected (Kaggle has no env).
set -euo pipefail
cd /home/user/Orbit-wars-kaggle

LIBS="geometry fleet orbit aim combat world_model intent trajectory mechanism \
mission scoring missions/snipe missions/reinforce missions/recapture \
missions/opening missions/drain missions/gang_up missions/opp_archetypes \
planner lookahead lookahead_planner game/interpreter fast_sim opp_model \
v7_search candidate_portfolios value_heads joint_solver/opening_planner \
joint_solver/columns joint_solver/lp"

export BASELINE_JOINT_AGGR=1 BASELINE_JOINT_TOP_K=5 BASELINE_JOINT_MAX_PAIRS=60 \
  BASELINE_REINFORCE_EMIT=1 BASELINE_REINFORCE_ANTICIPATE=1 \
  BASELINE_NEUTRAL_BONUS=2.0 BASELINE_NEUTRAL_EARLY_EXTRA=1.5 \
  BASELINE_NEUTRAL_EARLY_HORIZON=50 BASELINE_ORBITAL_SAFETY=1 BASELINE_PV_ETA=1 \
  BASELINE_LAUNCH_RULES=1 BASELINE_CAPTURE_HORIZON_K=10 \
  BASELINE_STATE_DRIVEN_K=1 BASELINE_STATE_K_CEIL=30

python scripts/bundle_agent.py agents/baseline --lib $LIBS --force --skip-parity-gate

python - <<'PY'
src, dst = "submissions/baseline.py", "submissions/baseline_state_driven_k.py"
header = '''import os as _sk_os
# Champion (launch_rules_universal) full config + STATE-DRIVEN horizon K ON.
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
'''
with open(src) as f:
    body = f.read()
with open(dst, "w") as f:
    f.write(header + "\n" + body)
print(f"wrote {dst} ({len(header)+len(body)} bytes)")
PY
echo "BUILD_OK submissions/baseline_state_driven_k.py"