#!/usr/bin/env bash
# Reproducible build for submissions/baseline_champion_stateDrivenK_vh.py.
# Live champion (state_driven_k) config + BASELINE_VH_LAMBDA=1.0 + the
# trained value-head model inlined as gzip+base64 into _VH_MODEL_B64
# (so the bundle is self-contained on Kaggle — no data/ files needed).
#
# Build pipeline:
#   1. bundle_agent inlines baseline + champion LIBS into submissions/baseline.py
#   2. python post-step copies → submissions/baseline_champion_stateDrivenK_vh.py
#      with: (a) state-K env header injected, (b) BASELINE_VH_LAMBDA=1.0
#      added, (c) _VH_MODEL_B64 patched from data/value_head/value_head_model.txt
set -euo pipefail
cd /home/user/Orbit-wars-kaggle

LIBS="geometry fleet orbit aim combat world_model intent trajectory kinematic_table mechanism \
mission scoring missions/snipe missions/reinforce missions/recapture \
missions/opening missions/drain missions/gang_up missions/opp_archetypes \
planner lookahead lookahead_planner game/interpreter fast_sim \
_validator_tree_walker opp_features_lite opp_model \
v7_search candidate_portfolios value_heads joint_solver/opening_planner \
joint_solver/columns joint_solver/lp \
value_head_features"

export BASELINE_JOINT_AGGR=1 BASELINE_JOINT_TOP_K=5 BASELINE_JOINT_MAX_PAIRS=60 \
  BASELINE_REINFORCE_EMIT=1 BASELINE_REINFORCE_ANTICIPATE=1 \
  BASELINE_NEUTRAL_BONUS=2.0 BASELINE_NEUTRAL_EARLY_EXTRA=1.5 \
  BASELINE_NEUTRAL_EARLY_HORIZON=50 BASELINE_ORBITAL_SAFETY=1 BASELINE_PV_ETA=1 \
  BASELINE_LAUNCH_RULES=1 BASELINE_CAPTURE_HORIZON_K=10 \
  BASELINE_STATE_DRIVEN_K=1 BASELINE_STATE_K_CEIL=30 \
  BASELINE_KINEMATIC_TABLE=1 BASELINE_VH_LAMBDA=1.0

python scripts/bundle_agent.py agents/baseline --lib $LIBS --force --skip-parity-gate

python - <<'PY'
import base64, gzip, re
from pathlib import Path

src = Path("submissions/baseline.py")
dst = Path("submissions/baseline_champion_stateDrivenK_vh.py")
vh_model = Path("data/value_head/value_head_model.txt")

if not vh_model.is_file():
    raise FileNotFoundError(f"VH model not found: {vh_model}")

header = '''import os as _sk_os
# Live champion (state_driven_k) config + value head ON.
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
_sk_os.environ.setdefault("BASELINE_VH_LAMBDA", "1.0")
'''
body = src.read_text()
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

# Patch _VH_MODEL_B64.
b64 = base64.b64encode(gzip.compress(vh_model.read_text().encode())).decode()
patch_re = re.compile(r'^_VH_MODEL_B64: str = ""\s*$', re.MULTILINE)
out_patched = patch_re.sub(f'_VH_MODEL_B64: str = "{b64}"', out, count=1)
if out_patched == out:
    raise RuntimeError("failed to patch _VH_MODEL_B64 in bundled output")

dst.write_text(out_patched)
print(f"wrote {dst} ({len(out_patched):,} bytes); VH model inlined ({len(b64):,} b64-chars)")
PY
echo "BUILD_OK submissions/baseline_champion_stateDrivenK_vh.py"
