#!/usr/bin/env bash
# Reproducible build for submissions/champ_frontierCirculation_on.py.
# All currently-shipped levers (champion + adaptive_k + compute_by_ships) +
# the new frontier-circulation post-pass (BASELINE_FRONTIER_CIRCULATION=1).
# PI 2026-06-03: geometric rear-to-front flow toward opp centroid -> DAG ->
# loop-proof by construction. Distinct from drain_idle_rear/drain_stagnant_rear
# (state-dependent destination -> loopy) which remain default-OFF.
#
# idle_stockpile_drain is intentionally NOT baked: cleaner A/B isolation
# of circulation as the only delta vs the current live champion
# (sub 53332500 == champ_computeByShips_on).
set -euo pipefail
cd /home/user/Orbit-wars-kaggle

export BASELINE_JOINT_AGGR=1 BASELINE_JOINT_TOP_K=5 BASELINE_JOINT_MAX_PAIRS=60 \
  BASELINE_REINFORCE_EMIT=1 BASELINE_REINFORCE_ANTICIPATE=1 \
  BASELINE_NEUTRAL_BONUS=2.0 BASELINE_NEUTRAL_EARLY_EXTRA=1.5 \
  BASELINE_NEUTRAL_EARLY_HORIZON=50 BASELINE_ORBITAL_SAFETY=1 BASELINE_PV_ETA=1 \
  BASELINE_LAUNCH_RULES=1 BASELINE_CAPTURE_HORIZON_K=10 BASELINE_ADAPTIVE_K=1 \
  BASELINE_COMPUTE_BY_SHIPS=1 BASELINE_FRONTIER_CIRCULATION=1

python scripts/bundle_agent.py agents/baseline --force --skip-parity-gate

python - <<'PY'
src, dst = "submissions/baseline.py", "submissions/champ_frontierCirculation_on.py"
header = '''import os as _fc_os
# Champion + adaptive K + compute_by_ships + frontier_circulation ON.
_fc_os.environ.setdefault("BASELINE_JOINT_AGGR", "1")
_fc_os.environ.setdefault("BASELINE_JOINT_TOP_K", "5")
_fc_os.environ.setdefault("BASELINE_JOINT_MAX_PAIRS", "60")
_fc_os.environ.setdefault("BASELINE_REINFORCE_EMIT", "1")
_fc_os.environ.setdefault("BASELINE_REINFORCE_ANTICIPATE", "1")
_fc_os.environ.setdefault("BASELINE_NEUTRAL_BONUS", "2.0")
_fc_os.environ.setdefault("BASELINE_NEUTRAL_EARLY_EXTRA", "1.5")
_fc_os.environ.setdefault("BASELINE_NEUTRAL_EARLY_HORIZON", "50")
_fc_os.environ.setdefault("BASELINE_ORBITAL_SAFETY", "1")
_fc_os.environ.setdefault("BASELINE_PV_ETA", "1")
_fc_os.environ.setdefault("BASELINE_LAUNCH_RULES", "1")
_fc_os.environ.setdefault("BASELINE_CAPTURE_HORIZON_K", "10")
_fc_os.environ.setdefault("BASELINE_ADAPTIVE_K", "1")
_fc_os.environ.setdefault("BASELINE_COMPUTE_BY_SHIPS", "1")
_fc_os.environ.setdefault("BASELINE_FRONTIER_CIRCULATION", "1")
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
print(f"wrote {dst} ({len(out)} bytes); header injected after line {insert_at}")
PY

echo "frontier_circulation baked:        $(grep -c BASELINE_FRONTIER_CIRCULATION submissions/champ_frontierCirculation_on.py)"
echo "compute_by_ships baked:            $(grep -c BASELINE_COMPUTE_BY_SHIPS submissions/champ_frontierCirculation_on.py)"
echo "adaptive_k baked:                  $(grep -c BASELINE_ADAPTIVE_K submissions/champ_frontierCirculation_on.py)"
echo "emit_frontier_circulation inlined: $(grep -c 'def emit_frontier_circulation' submissions/champ_frontierCirculation_on.py) / 1"
echo "cross-agent imports (must be 0):   $(grep -cE '^\s*(from|import) agents\.' submissions/champ_frontierCirculation_on.py)"
python -c "import ast; ast.parse(open('submissions/champ_frontierCirculation_on.py').read()); print('parses OK')"
echo "BUILD_OK submissions/champ_frontierCirculation_on.py"

# AB sibling: identical config WITHOUT the new lever.
echo
echo "--- AB sibling (lever OFF) ---"
python - <<'PY'
src, dst = "submissions/baseline.py", "submissions/champ_frontierCirculation_AB_baseline.py"
header = '''import os as _fc_os
# Sibling AB baseline: matches the currently-shipped champ_computeByShips_on.py.
# frontier_circulation INTENTIONALLY NOT BAKED.
_fc_os.environ.setdefault("BASELINE_JOINT_AGGR", "1")
_fc_os.environ.setdefault("BASELINE_JOINT_TOP_K", "5")
_fc_os.environ.setdefault("BASELINE_JOINT_MAX_PAIRS", "60")
_fc_os.environ.setdefault("BASELINE_REINFORCE_EMIT", "1")
_fc_os.environ.setdefault("BASELINE_REINFORCE_ANTICIPATE", "1")
_fc_os.environ.setdefault("BASELINE_NEUTRAL_BONUS", "2.0")
_fc_os.environ.setdefault("BASELINE_NEUTRAL_EARLY_EXTRA", "1.5")
_fc_os.environ.setdefault("BASELINE_NEUTRAL_EARLY_HORIZON", "50")
_fc_os.environ.setdefault("BASELINE_ORBITAL_SAFETY", "1")
_fc_os.environ.setdefault("BASELINE_PV_ETA", "1")
_fc_os.environ.setdefault("BASELINE_LAUNCH_RULES", "1")
_fc_os.environ.setdefault("BASELINE_CAPTURE_HORIZON_K", "10")
_fc_os.environ.setdefault("BASELINE_ADAPTIVE_K", "1")
_fc_os.environ.setdefault("BASELINE_COMPUTE_BY_SHIPS", "1")
# BASELINE_FRONTIER_CIRCULATION intentionally NOT baked.
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
print(f"wrote {dst} ({len(out)} bytes)")
PY
echo "frontier_circulation baked (must be 0): $(grep -c '\"BASELINE_FRONTIER_CIRCULATION\", \"1\"' submissions/champ_frontierCirculation_AB_baseline.py)"
echo "compute_by_ships baked:            $(grep -c BASELINE_COMPUTE_BY_SHIPS submissions/champ_frontierCirculation_AB_baseline.py)"
python -c "import ast; ast.parse(open('submissions/champ_frontierCirculation_AB_baseline.py').read()); print('parses OK')"
echo "BUILD_OK submissions/champ_frontierCirculation_AB_baseline.py"
