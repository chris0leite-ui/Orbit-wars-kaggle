#!/usr/bin/env bash
# Reproducible build for submissions/champ_idleStockpile_on.py.
# All currently-shipped levers (champion + adaptive_k + compute_by_ships) +
# the new large-idle-stockpile spend-down post-pass
# (BASELINE_IDLE_STOCKPILE_DRAIN=1). PI 2026-06-03: any planet that holds
# 3x the average ship count (OR > 25% of total fleet) AND is not under
# threat AND hasn't fired this turn emits one forced launch at an opponent
# (positive-EV preferred, nearest-opp fallback). K-eta cap bypassed by
# post-enforce slotting.
set -euo pipefail
cd /home/user/Orbit-wars-kaggle

export BASELINE_JOINT_AGGR=1 BASELINE_JOINT_TOP_K=5 BASELINE_JOINT_MAX_PAIRS=60 \
  BASELINE_REINFORCE_EMIT=1 BASELINE_REINFORCE_ANTICIPATE=1 \
  BASELINE_NEUTRAL_BONUS=2.0 BASELINE_NEUTRAL_EARLY_EXTRA=1.5 \
  BASELINE_NEUTRAL_EARLY_HORIZON=50 BASELINE_ORBITAL_SAFETY=1 BASELINE_PV_ETA=1 \
  BASELINE_LAUNCH_RULES=1 BASELINE_CAPTURE_HORIZON_K=10 BASELINE_ADAPTIVE_K=1 \
  BASELINE_COMPUTE_BY_SHIPS=1 BASELINE_IDLE_STOCKPILE_DRAIN=1

python scripts/bundle_agent.py agents/baseline --force --skip-parity-gate

python - <<'PY'
src, dst = "submissions/baseline.py", "submissions/champ_idleStockpile_on.py"
header = '''import os as _isp_os
# Champion + adaptive K + compute_by_ships + idle-stockpile spend-down ON.
_isp_os.environ.setdefault("BASELINE_JOINT_AGGR", "1")
_isp_os.environ.setdefault("BASELINE_JOINT_TOP_K", "5")
_isp_os.environ.setdefault("BASELINE_JOINT_MAX_PAIRS", "60")
_isp_os.environ.setdefault("BASELINE_REINFORCE_EMIT", "1")
_isp_os.environ.setdefault("BASELINE_REINFORCE_ANTICIPATE", "1")
_isp_os.environ.setdefault("BASELINE_NEUTRAL_BONUS", "2.0")
_isp_os.environ.setdefault("BASELINE_NEUTRAL_EARLY_EXTRA", "1.5")
_isp_os.environ.setdefault("BASELINE_NEUTRAL_EARLY_HORIZON", "50")
_isp_os.environ.setdefault("BASELINE_ORBITAL_SAFETY", "1")
_isp_os.environ.setdefault("BASELINE_PV_ETA", "1")
_isp_os.environ.setdefault("BASELINE_LAUNCH_RULES", "1")
_isp_os.environ.setdefault("BASELINE_CAPTURE_HORIZON_K", "10")
_isp_os.environ.setdefault("BASELINE_ADAPTIVE_K", "1")
_isp_os.environ.setdefault("BASELINE_COMPUTE_BY_SHIPS", "1")
_isp_os.environ.setdefault("BASELINE_IDLE_STOCKPILE_DRAIN", "1")
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

# Structure checks (Rule 46 silent-fail modes).
echo "idle_stockpile baked:              $(grep -c BASELINE_IDLE_STOCKPILE_DRAIN submissions/champ_idleStockpile_on.py)"
echo "compute_by_ships baked:            $(grep -c BASELINE_COMPUTE_BY_SHIPS submissions/champ_idleStockpile_on.py)"
echo "adaptive_k baked:                  $(grep -c BASELINE_ADAPTIVE_K submissions/champ_idleStockpile_on.py)"
echo "drain_idle_stockpile_to_opp inlined: $(grep -c 'def drain_idle_stockpile_to_opp' submissions/champ_idleStockpile_on.py) / 1"
echo "_pick_idle_stockpile_target inlined: $(grep -c 'def _pick_idle_stockpile_target' submissions/champ_idleStockpile_on.py) / 1"
echo "predict_garrison_at inlined:       $(grep -c 'def predict_garrison_at' submissions/champ_idleStockpile_on.py) / 1"
echo "cross-agent imports (must be 0):   $(grep -cE '^\s*(from|import) agents\.' submissions/champ_idleStockpile_on.py)"
python -c "import ast; ast.parse(open('submissions/champ_idleStockpile_on.py').read()); print('parses OK')"
echo "BUILD_OK submissions/champ_idleStockpile_on.py"

# Now build the AB-baseline sibling: identical config WITHOUT the new lever,
# so the only delta is BASELINE_IDLE_STOCKPILE_DRAIN.
echo
echo "--- AB sibling (lever OFF) ---"
python - <<'PY'
src, dst = "submissions/baseline.py", "submissions/champ_idleStockpile_AB_baseline.py"
header = '''import os as _isp_os
# Sibling AB baseline: champion + adaptive K + compute_by_ships ON. The new
# idle-stockpile spend-down lever is INTENTIONALLY NOT BAKED so this bundle
# matches the currently-shipped champ_computeByShips_on.py behaviour.
_isp_os.environ.setdefault("BASELINE_JOINT_AGGR", "1")
_isp_os.environ.setdefault("BASELINE_JOINT_TOP_K", "5")
_isp_os.environ.setdefault("BASELINE_JOINT_MAX_PAIRS", "60")
_isp_os.environ.setdefault("BASELINE_REINFORCE_EMIT", "1")
_isp_os.environ.setdefault("BASELINE_REINFORCE_ANTICIPATE", "1")
_isp_os.environ.setdefault("BASELINE_NEUTRAL_BONUS", "2.0")
_isp_os.environ.setdefault("BASELINE_NEUTRAL_EARLY_EXTRA", "1.5")
_isp_os.environ.setdefault("BASELINE_NEUTRAL_EARLY_HORIZON", "50")
_isp_os.environ.setdefault("BASELINE_ORBITAL_SAFETY", "1")
_isp_os.environ.setdefault("BASELINE_PV_ETA", "1")
_isp_os.environ.setdefault("BASELINE_LAUNCH_RULES", "1")
_isp_os.environ.setdefault("BASELINE_CAPTURE_HORIZON_K", "10")
_isp_os.environ.setdefault("BASELINE_ADAPTIVE_K", "1")
_isp_os.environ.setdefault("BASELINE_COMPUTE_BY_SHIPS", "1")
# BASELINE_IDLE_STOCKPILE_DRAIN intentionally NOT baked.
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
echo "idle_stockpile baked (must be 0):  $(grep -c BASELINE_IDLE_STOCKPILE_DRAIN submissions/champ_idleStockpile_AB_baseline.py)"
echo "compute_by_ships baked:            $(grep -c BASELINE_COMPUTE_BY_SHIPS submissions/champ_idleStockpile_AB_baseline.py)"
python -c "import ast; ast.parse(open('submissions/champ_idleStockpile_AB_baseline.py').read()); print('parses OK')"
echo "BUILD_OK submissions/champ_idleStockpile_AB_baseline.py"
