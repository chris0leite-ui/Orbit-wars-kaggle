#!/usr/bin/env bash
# Reproducible build for submissions/baseline_size_balance.py (gitignored,
# per the submissions/* convention). Regenerates the EXACT submitted bundle
# from committed source: the current champion config (verbatim from
# baseline_launch_rules_universal) + the size-balance fix (A+D) baked ON.
#
# Bundle = agents/baseline (whose main.py defines `agent`) + the lib set the
# champion used, MINUS kinematic_table (deleted on this branch, commit
# 232307c; it was dead code in the champion — never primed at runtime, always
# fell back to predict_relative, so its removal is behaviorally neutral).
# A config header is injected at the top because Kaggle has no env vars.
set -euo pipefail
cd /home/user/Orbit-wars-kaggle

LIBS="geometry fleet orbit aim combat world_model intent trajectory mechanism \
mission scoring missions/snipe missions/reinforce missions/recapture \
missions/opening missions/drain missions/gang_up missions/opp_archetypes \
planner lookahead lookahead_planner game/interpreter fast_sim opp_model \
v7_search candidate_portfolios value_heads joint_solver/opening_planner \
joint_solver/columns joint_solver/lp"

# Bundle with the shipped config + flag in env so the parity gate exercises
# the real code path. (--skip-parity-gate avoids the kaggle_environments
# lux_ai_s3 import-noise false failure; structure is verified below instead.)
export BASELINE_JOINT_AGGR=1 BASELINE_JOINT_TOP_K=5 BASELINE_JOINT_MAX_PAIRS=60 \
  BASELINE_REINFORCE_EMIT=1 BASELINE_REINFORCE_ANTICIPATE=1 \
  BASELINE_NEUTRAL_BONUS=2.0 BASELINE_NEUTRAL_EARLY_EXTRA=1.5 \
  BASELINE_NEUTRAL_EARLY_HORIZON=50 BASELINE_ORBITAL_SAFETY=1 BASELINE_PV_ETA=1 \
  BASELINE_LAUNCH_RULES=1 BASELINE_CAPTURE_HORIZON_K=10 BASELINE_SIZE_BALANCE=1

python scripts/bundle_agent.py agents/baseline --lib $LIBS --force --skip-parity-gate

python - <<'PY'
src, dst = "submissions/baseline.py", "submissions/baseline_size_balance.py"
header = '''import os as _sb_os
# Champion (launch_rules_universal) full config + size-balance (A+D) baked ON.
_sb_os.environ.setdefault("BASELINE_JOINT_AGGR", "1")
_sb_os.environ.setdefault("BASELINE_JOINT_TOP_K", "5")
_sb_os.environ.setdefault("BASELINE_JOINT_MAX_PAIRS", "60")
_sb_os.environ.setdefault("BASELINE_REINFORCE_EMIT", "1")
_sb_os.environ.setdefault("BASELINE_REINFORCE_ANTICIPATE", "1")
_sb_os.environ.setdefault("BASELINE_NEUTRAL_BONUS", "2.0")
_sb_os.environ.setdefault("BASELINE_NEUTRAL_EARLY_EXTRA", "1.5")
_sb_os.environ.setdefault("BASELINE_NEUTRAL_EARLY_HORIZON", "50")
_sb_os.environ.setdefault("BASELINE_ORBITAL_SAFETY", "1")
_sb_os.environ.setdefault("BASELINE_PV_ETA", "1")
_sb_os.environ.setdefault("BASELINE_LAUNCH_RULES", "1")
_sb_os.environ.setdefault("BASELINE_CAPTURE_HORIZON_K", "10")
_sb_os.environ.setdefault("BASELINE_SIZE_BALANCE", "1")
'''
lines = open(src).read().splitlines(keepends=True)
out, done = [], False
for ln in lines:
    out.append(ln)
    if not done and ln.strip() == "from __future__ import annotations":
        out.append("\n" + header + "\n"); done = True
assert done, "future-import anchor not found"
open(dst, "w").write("".join(out))
print("wrote", dst)
PY

# Restore the force-tracked stale baseline.py the bundler overwrites.
git checkout -- submissions/baseline.py 2>/dev/null || true

# Structure checks (Rule 46 silent-fail modes).
echo "header baked:    $(grep -c BASELINE_SIZE_BALANCE submissions/baseline_size_balance.py)"
echo "fix fns inlined: $(grep -cE 'def (capture_floor_arrival|source_keep_floor|_keep_floor_from_threat)' submissions/baseline_size_balance.py) / 3"
echo "cross-agent imports (must be 0): $(grep -cE '^\s*(from|import) agents\.' submissions/baseline_size_balance.py)"
python -c "import ast; ast.parse(open('submissions/baseline_size_balance.py').read()); print('parses OK')"
