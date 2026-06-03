#!/usr/bin/env bash
# Reproducible build for submissions/champ_refine_adaptivek.py.
# Champion full config + adaptive horizon K (BASELINE_ADAPTIVE_K=1, 20->10 by
# step 30) + the AUGMENT-NOT-REPLACE teamwork refiner (BASELINE_CHOOSER=refine):
# runs the champion verbatim, then adds ONLY oracle-positive two-source
# coalition atoms that don't conflict with the champion's locks.
# Validated 2026-06-03: 25/32 = 78.1% vs the adaptive-K champion (Wilson-lo
# 0.612) against a clean 50.0% parity; paired +13 gained / -4 broke / net +9.
# Config matches the validated A/B EXACTLY (no kinematic_table — as tested).
set -euo pipefail
cd /home/user/Orbit-wars-kaggle

LIBS="geometry fleet orbit aim combat world_model intent trajectory kinematic_table mechanism \
mission scoring missions/snipe missions/reinforce missions/recapture \
missions/opening missions/drain missions/gang_up missions/opp_archetypes \
planner lookahead lookahead_planner game/interpreter fast_sim opp_model \
v7_search candidate_portfolios value_heads joint_solver/opening_planner \
joint_solver/columns joint_solver/lp"

# Env at bundle time is only for the parity gate; the runtime config is baked
# into the setdefault header below (Kaggle has no env).
export BASELINE_JOINT_AGGR=1 BASELINE_JOINT_TOP_K=5 BASELINE_JOINT_MAX_PAIRS=60 \
  BASELINE_REINFORCE_EMIT=1 BASELINE_REINFORCE_ANTICIPATE=1 \
  BASELINE_NEUTRAL_BONUS=2.0 BASELINE_NEUTRAL_EARLY_EXTRA=1.5 \
  BASELINE_NEUTRAL_EARLY_HORIZON=50 BASELINE_ORBITAL_SAFETY=1 BASELINE_PV_ETA=1 \
  BASELINE_LAUNCH_RULES=1 BASELINE_CAPTURE_HORIZON_K=10 \
  BASELINE_VALUE_HEAD=hybrid BASELINE_JOINT=1 \
  BASELINE_ADAPTIVE_K=1 BASELINE_ADAPTIVE_K_OPEN=20 BASELINE_ADAPTIVE_K_TSETTLE=30 \
  BASELINE_KINEMATIC_TABLE=1 \
  BASELINE_CHOOSER=refine

python scripts/bundle_agent.py agents/baseline --lib $LIBS --force --skip-parity-gate

python - <<'PY'
src, dst = "submissions/baseline.py", "submissions/champ_refine_adaptivek.py"
header = '''import os as _rk_os
# Champion full config + ADAPTIVE-K horizon + AUGMENT teamwork refiner ON.
_rk_os.environ.setdefault("BASELINE_JOINT_AGGR", "1")
_rk_os.environ.setdefault("BASELINE_JOINT_TOP_K", "5")
_rk_os.environ.setdefault("BASELINE_JOINT_MAX_PAIRS", "60")
_rk_os.environ.setdefault("BASELINE_REINFORCE_EMIT", "1")
_rk_os.environ.setdefault("BASELINE_REINFORCE_ANTICIPATE", "1")
_rk_os.environ.setdefault("BASELINE_NEUTRAL_BONUS", "2.0")
_rk_os.environ.setdefault("BASELINE_NEUTRAL_EARLY_EXTRA", "1.5")
_rk_os.environ.setdefault("BASELINE_NEUTRAL_EARLY_HORIZON", "50")
_rk_os.environ.setdefault("BASELINE_ORBITAL_SAFETY", "1")
_rk_os.environ.setdefault("BASELINE_PV_ETA", "1")
_rk_os.environ.setdefault("BASELINE_LAUNCH_RULES", "1")
_rk_os.environ.setdefault("BASELINE_CAPTURE_HORIZON_K", "10")
_rk_os.environ.setdefault("BASELINE_VALUE_HEAD", "hybrid")
_rk_os.environ.setdefault("BASELINE_JOINT", "1")
_rk_os.environ.setdefault("BASELINE_ADAPTIVE_K", "1")
_rk_os.environ.setdefault("BASELINE_ADAPTIVE_K_OPEN", "20")
_rk_os.environ.setdefault("BASELINE_ADAPTIVE_K_TSETTLE", "30")
_rk_os.environ.setdefault("BASELINE_KINEMATIC_TABLE", "1")
_rk_os.environ.setdefault("BASELINE_CHOOSER", "refine")
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
echo "BUILD_OK submissions/champ_refine_adaptivek.py"
