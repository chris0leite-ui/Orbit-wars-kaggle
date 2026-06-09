#!/usr/bin/env bash
# Reproducible build for submissions/champ_computeByShips_on.py.
# Champion (launch_rules_universal) full config + adaptive horizon K
# (BASELINE_ADAPTIVE_K=1, K_OPEN=20 -> floor 10 by step 30) + per-source
# compute_by_ships lever (BASELINE_COMPUTE_BY_SHIPS=1, log-scaled per-source
# enumeration breadth in [4,16] + per-source K bonus capped at +50%).
# PI observation 2026-06-03: high-ship rear planets idle when far from
# opponents — the compound lever surfaces those launches.
set -euo pipefail
cd /home/user/Orbit-wars-kaggle

# Use DEFAULT_LIB_ORDER from scripts/bundle_agent.py — kept in sync with
# agents/baseline imports (older hand-rolled LIBS lists go stale when new
# lib modules land; pulling from default catches drift automatically).

export BASELINE_JOINT_AGGR=1 BASELINE_JOINT_TOP_K=5 BASELINE_JOINT_MAX_PAIRS=60 \
  BASELINE_REINFORCE_EMIT=1 BASELINE_REINFORCE_ANTICIPATE=1 \
  BASELINE_NEUTRAL_BONUS=2.0 BASELINE_NEUTRAL_EARLY_EXTRA=1.5 \
  BASELINE_NEUTRAL_EARLY_HORIZON=50 BASELINE_ORBITAL_SAFETY=1 BASELINE_PV_ETA=1 \
  BASELINE_LAUNCH_RULES=1 BASELINE_CAPTURE_HORIZON_K=10 BASELINE_ADAPTIVE_K=1 \
  BASELINE_COMPUTE_BY_SHIPS=1

python scripts/bundle_agent.py agents/baseline --force --skip-parity-gate

python - <<'PY'
src, dst = "submissions/baseline.py", "submissions/champ_computeByShips_on.py"
header = '''import os as _cbs_os
# Champion (launch_rules_universal) full config + ADAPTIVE horizon K + compute_by_ships ON.
_cbs_os.environ.setdefault("BASELINE_JOINT_AGGR", "1")
_cbs_os.environ.setdefault("BASELINE_JOINT_TOP_K", "5")
_cbs_os.environ.setdefault("BASELINE_JOINT_MAX_PAIRS", "60")
_cbs_os.environ.setdefault("BASELINE_REINFORCE_EMIT", "1")
_cbs_os.environ.setdefault("BASELINE_REINFORCE_ANTICIPATE", "1")
_cbs_os.environ.setdefault("BASELINE_NEUTRAL_BONUS", "2.0")
_cbs_os.environ.setdefault("BASELINE_NEUTRAL_EARLY_EXTRA", "1.5")
_cbs_os.environ.setdefault("BASELINE_NEUTRAL_EARLY_HORIZON", "50")
_cbs_os.environ.setdefault("BASELINE_ORBITAL_SAFETY", "1")
_cbs_os.environ.setdefault("BASELINE_PV_ETA", "1")
_cbs_os.environ.setdefault("BASELINE_LAUNCH_RULES", "1")
_cbs_os.environ.setdefault("BASELINE_CAPTURE_HORIZON_K", "10")
_cbs_os.environ.setdefault("BASELINE_ADAPTIVE_K", "1")
_cbs_os.environ.setdefault("BASELINE_COMPUTE_BY_SHIPS", "1")
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
echo "header baked:                  $(grep -c BASELINE_COMPUTE_BY_SHIPS submissions/champ_computeByShips_on.py)"
echo "adaptive_k baked:              $(grep -c BASELINE_ADAPTIVE_K submissions/champ_computeByShips_on.py)"
echo "capture_horizon_k inlined:     $(grep -c 'def capture_horizon_k' submissions/champ_computeByShips_on.py) / 1"
echo "_apply_src_ratio_bonus inlined: $(grep -c 'def _apply_src_ratio_bonus' submissions/champ_computeByShips_on.py) / 1"
echo "_targets_for_src inlined:      $(grep -c 'def _targets_for_src' submissions/champ_computeByShips_on.py) / 1"
echo "cross-agent imports (must be 0): $(grep -cE '^\s*(from|import) agents\.' submissions/champ_computeByShips_on.py)"
python -c "import ast; ast.parse(open('submissions/champ_computeByShips_on.py').read()); print('parses OK')"
echo "BUILD_OK submissions/champ_computeByShips_on.py"
