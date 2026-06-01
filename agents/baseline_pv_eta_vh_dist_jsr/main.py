"""baseline + pv_eta + distilled opp + B.3 head + slot res + joint sync + size balance.

Integration of three orthogonal attack-strategy fixes on top of the
composite foundation (pv_eta + distilled-Tier-2 opp + B.3 head). Origin:
2026-06-01 — slotres alone (sub 53243763) settled mu=695.7 because the
chooser scored attacks but the attacks themselves were under-sized,
under-coordinated, or wasted by capture mis-sizing.

The three fixes address distinct attack-pipeline bottlenecks:

1. BASELINE_SLOT_RESERVATION=3/2/2 (chooser pre-loop): partitions the
   prerank by target class so attack candidates aren't starved out by
   defenses that flood the score budget. (Origin: sub 53239342, our
   earlier probe.)

2. BASELINE_JOINT_SYNC=1 (chooser post-loop): synchronized-arrival
   multi-source attacks. The fire-now joint launches both legs at
   wait_N=0, so sources at different distances arrive on different
   ticks and never STACK. JOINT_SYNC waits the closer source so all
   legs land the same tick and combine forces — capturing planets
   neither source could take alone. (Origin: 00JzI branch, lived as
   sub 53223160 at mu=1147 before eviction.)

3. BASELINE_SIZE_BALANCE=1 (proposer fire-now sizing): unified
   arrival-correct + source-safe sizing for non-owned targets.
   Failure D (under-delivery): capture_size sized the defender garrison
   at the eta of the slowest probe fleet, so the real launched count
   flew faster and the garrison was mis-sampled — emitted cap could
   bounce. Failure A (source over-drain): full-budget sends from
   threatened sources stripped them below their residue floor.
   (Origin: 00JzI branch, n=16 75% A/B vs OFF.)

These three operate at different pipeline stages — proposer (sizing) →
prerank assembly → chooser pre-loop (slot res) → main loop scoring →
chooser post-loop (joint sync). No semantic overlap; expected to
compose orthogonally.

Env vars set:
- Standard peak-orbitfix preamble (joint aggr, neutral bonus, orbital safety).
- `BASELINE_PV_ETA=1`: foundation lock.
- `BASELINE_OPP_TIER=2`, `BASELINE_OPP_FILTER_THRESHOLD=0.15`: distilled opp.
- `BASELINE_VH_LAMBDA=1.0`: B.3 head additive term.
- `BASELINE_SLOT_RESERVATION=3/2/2`: per-class scoring slots.
- `BASELINE_JOINT_SYNC=1` + tuning: synchronized multi-source attacks.
- `BASELINE_SIZE_BALANCE=1`: arrival-correct + source-safe fire-now sizing.
- `KINEMATIC_TABLE_ENABLED=0`: forced (mutable-singleton bug fix).

Bundler patches `_OPP_BOOSTER_B64` (lib/opp_model.py) AND `_VH_MODEL_B64`
(agents/baseline/_value_head.py).

Default-OFF parity: with all env vars unset, byte-identical to bare baseline.
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

# Attack fix 1: per-class scoring slots (attack starvation fix).
_os.environ.setdefault("BASELINE_SLOT_RESERVATION", "3/2/2")

# Attack fix 2: synchronized multi-source coalitions.
_os.environ.setdefault("BASELINE_JOINT_SYNC", "1")
_os.environ.setdefault("BASELINE_JOINT_SYNC_MAX_PAIRS", "30")
_os.environ.setdefault("BASELINE_JOINT_SYNC_SRC_K", "3")

# Attack fix 3: arrival-correct + source-safe sizing (proposer).
_os.environ.setdefault("BASELINE_SIZE_BALANCE", "1")

# Kinematic-table bug fix (May-30; non-negotiable).
_os.environ["KINEMATIC_TABLE_ENABLED"] = "0"

from agents.baseline.main import agent  # noqa: E402,F401
