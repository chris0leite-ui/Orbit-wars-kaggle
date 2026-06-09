#!/usr/bin/env bash
# Reproducible build for submissions/champ_adaptiveK_on.py.
# Champion (launch_rules_universal) full config, verbatim, + the adaptive
# horizon-K lever baked ON (BASELINE_ADAPTIVE_K=1, K_OPEN=20 -> floor 10 by
# step 30). Built from commit 9985e98 (00JzI lineage).
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
  BASELINE_LAUNCH_RULES=1 BASELINE_CAPTURE_HORIZON_K=10 BASELINE_ADAPTIVE_K=1

python scripts/bundle_agent.py agents/baseline --lib $LIBS --force --skip-parity-gate

python - <<'PY'
src, dst = "submissions/baseline.py", "submissions/champ_adaptiveK_on.py"
header = '''import os as _ak_os
# Champion (launch_rules_universal) full config + ADAPTIVE horizon K ON.
_ak_os.environ.setdefault("BASELINE_JOINT_AGGR", "1")
_ak_os.environ.setdefault("BASELINE_JOINT_TOP_K", "5")
_ak_os.environ.setdefault("BASELINE_JOINT_MAX_PAIRS", "60")
_ak_os.environ.setdefault("BASELINE_REINFORCE_EMIT", "1")
_ak_os.environ.setdefault("BASELINE_REINFORCE_ANTICIPATE", "1")
_ak_os.environ.setdefault("BASELINE_NEUTRAL_BONUS", "2.0")
_ak_os.environ.setdefault("BASELINE_NEUTRAL_EARLY_EXTRA", "1.5")
_ak_os.environ.setdefault("BASELINE_NEUTRAL_EARLY_HORIZON", "50")
_ak_os.environ.setdefault("BASELINE_ORBITAL_SAFETY", "1")
_ak_os.environ.setdefault("BASELINE_PV_ETA", "1")
_ak_os.environ.setdefault("BASELINE_LAUNCH_RULES", "1")
_ak_os.environ.setdefault("BASELINE_CAPTURE_HORIZON_K", "10")
_ak_os.environ.setdefault("BASELINE_ADAPTIVE_K", "1")
'''
with open(src) as f:
    body = f.read()
# `from __future__` imports must remain first; inject env header AFTER them.
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

# Structure checks (Rule 46 silent-fail modes).
echo "header baked:                  $(grep -c BASELINE_ADAPTIVE_K submissions/champ_adaptiveK_on.py)"
echo "capture_horizon_k inlined:     $(grep -c 'def capture_horizon_k' submissions/champ_adaptiveK_on.py) / 1"
echo "cross-agent imports (must be 0): $(grep -cE '^\s*(from|import) agents\.' submissions/champ_adaptiveK_on.py)"
python -c "import ast; ast.parse(open('submissions/champ_adaptiveK_on.py').read()); print('parses OK')"
echo "BUILD_OK submissions/champ_adaptiveK_on.py"
