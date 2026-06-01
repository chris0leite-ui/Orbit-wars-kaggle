"""baseline + pv_eta + distilled opp + B.3 head + per-class slot reservation.

Layers slot reservation on top of the composite (pv_eta + distilled-Tier-2
opp + B.3 head). Origin: 2026-06-01 sub 53239342 (composite at μ=460)
diagnosed via per-state probe — `time_to_enemy_threat` flagged 11/11 of
our planets as threatened, `cheap_marginal_value` ranked defenses +12
median vs attacks -1 median, and the wallclock cap let the chooser score
only ~5 candidates — all defenses. Attack/expansion candidates with
positive leaf_delta existed in the prerank but never got scored.

The fix (BASELINE_SLOT_RESERVATION) lives in chooser_trajectory.py;
this wrapper sets the env vars to activate it.

Probe evidence at ep 78367540 step 100 with this config:
- Bare composite (no slot res): 5 defenses scored, all leaf_delta=0,
  agent emits 2 small defenses.
- With slot res 3/2/2: 2 defenses + 2 expansions + 1 attack scored.
  Attack leaf_delta=+40.3, expansions +29.4 and +20.7.
  Agent emits attack(823 ships toward enemy planet 29) + expansion
  (706 ships toward neutral 15) + small defense.

Env vars set:
- `BASELINE_OPP_TIER=2`: distilled-ladder opp predictor.
- `BASELINE_VH_LAMBDA=1.0`: B.3 head additive term ON.
- `BASELINE_PV_ETA=1`: foundation lock.
- `BASELINE_SLOT_RESERVATION=3/2/2`: 3 attacks + 2 expansions + 2 defenses
  scoring slots per turn (vs the current ~5 monopolised by defenses).
- `BASELINE_WALLCLOCK_MS=800`: bump from 600. Composite single-game smoke
  ran max=986ms; 200ms headroom is unused under the 1000ms env cap.
- Standard peak-orbitfix preamble.
- `KINEMATIC_TABLE_ENABLED=0`: forced (mutable-singleton bug fix).

Bundler patches BOTH `_OPP_BOOSTER_B64` (in inlined lib/opp_model.py)
AND `_VH_MODEL_B64` (in inlined agents/baseline/_value_head.py).

At BASELINE_SLOT_RESERVATION unset, equivalent to the composite bundle.
"""

from __future__ import annotations

import os as _os

# peak orbitfix preamble
_os.environ.setdefault("BASELINE_JOINT_AGGR", "1")
_os.environ.setdefault("BASELINE_JOINT_TOP_K", "5")
_os.environ.setdefault("BASELINE_JOINT_MAX_PAIRS", "60")
_os.environ.setdefault("BASELINE_REINFORCE_EMIT", "1")
_os.environ.setdefault("BASELINE_REINFORCE_ANTICIPATE", "1")
_os.environ.setdefault("BASELINE_NEUTRAL_BONUS", "2.0")
_os.environ.setdefault("BASELINE_NEUTRAL_EARLY_EXTRA", "1.5")
_os.environ.setdefault("BASELINE_NEUTRAL_EARLY_HORIZON", "50")
_os.environ.setdefault("BASELINE_ORBITAL_SAFETY", "1")

# Foundation: pv_eta time-discount.
_os.environ.setdefault("BASELINE_PV_ETA", "1")

# Composite layer 1: distilled-ladder opp model (Tier 2 v2).
_os.environ.setdefault("BASELINE_OPP_TIER", "2")
_os.environ.setdefault("BASELINE_OPP_FILTER_THRESHOLD", "0.15")

# Composite layer 2: B.3 CRN-advantage value head.
_os.environ.setdefault("BASELINE_VH_LAMBDA", "1.0")

# New 2026-06-01: per-class scoring slots. Wallclock kept at the 600ms
# default — single-game smoke with BASELINE_WALLCLOCK_MS=800 showed
# max=1195ms (over the 1000ms env cap), p95=1025ms (also over). The
# slot reservation doesn't change wallclock; it changes WHICH
# candidates get scored within the existing budget. The probe showed
# attacks + expansions get scored even at default wallclock — the
# bump was secondary and not load-bearing.
_os.environ.setdefault("BASELINE_SLOT_RESERVATION", "3/2/2")

# Kinematic-table bug fix (May-30; non-negotiable).
_os.environ["KINEMATIC_TABLE_ENABLED"] = "0"

from agents.baseline.main import agent  # noqa: E402,F401
