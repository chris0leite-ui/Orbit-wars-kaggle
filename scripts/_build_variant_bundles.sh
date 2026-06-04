#!/usr/bin/env bash
# Reproducible build for the opponent-agnostic variants, off the SAME baseline
# bundle the champion uses (scripts/_build_refine_adaptivek_bundle.sh). Emits:
#   submissions/champ_passive.py   = champion config + BASELINE_OPP_PASSIVE=1
#                                    (frozen-opponent rollout; the spike).
#   submissions/champ_netswing.py  = champ_passive + BASELINE_VALUE_HEAD=net_swing
#                                    (Producer's exact lens: passive opponent +
#                                    net-ship-swing leaf — Option B leaf-swap).
# The champion control is submissions/champ_refine_adaptivek.py (built by the
# sister script). All three share one baseline.py bundle, differing only in the
# setdefault header, so any A/B isolates exactly the flag(s) under test.
set -euo pipefail
cd /home/user/Orbit-wars-kaggle

LIBS="geometry fleet orbit aim combat world_model intent trajectory kinematic_table mechanism \
mission scoring missions/snipe missions/reinforce missions/recapture \
missions/opening missions/drain missions/gang_up missions/opp_archetypes \
planner lookahead lookahead_planner game/interpreter fast_sim opp_model \
v7_search candidate_portfolios value_heads joint_solver/opening_planner \
joint_solver/columns joint_solver/lp"

# Env at bundle time is only for the parity gate; runtime config is baked into
# the setdefault header below.
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
src = "submissions/baseline.py"
# Champion header (identical to _build_refine_adaptivek_bundle.sh).
champ = '''import os as _rk_os
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
# Variant header lines appended after the champion block.
passive_extra = '_rk_os.environ.setdefault("BASELINE_OPP_PASSIVE", "1")\n'
# net_swing REPLACES the champion's hybrid head (swap the setdefault line) and
# adds the passive-opponent flag — together they are Producer's exact lens.
netswing_header = champ.replace(
    '_rk_os.environ.setdefault("BASELINE_VALUE_HEAD", "hybrid")\n',
    '_rk_os.environ.setdefault("BASELINE_VALUE_HEAD", "net_swing")\n',
) + passive_extra

variants = {
    "submissions/champ_passive.py": champ + passive_extra,
    "submissions/champ_netswing.py": netswing_header,
}

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
for dst, header in variants.items():
    out = "\n".join(lines[:insert_at]) + "\n" + header + "\n" + "\n".join(lines[insert_at:])
    with open(dst, "w") as f:
        f.write(out)
    print(f"wrote {dst} ({len(out)} bytes)")
PY
echo "BUILD_OK submissions/champ_passive.py submissions/champ_netswing.py"
