"""baseline_region — champion config + region/chunk-aware augmentation ON.

Identical to the live champion (`baseline_pv_eta_anchor_1163`) 13-variable
config, plus `BASELINE_REGION=1`, which turns on the region layer in
`agents/baseline/main.py`: cluster planets into orbital regions, bias the
existing per-launch candidates toward high-value *predictable* contested
regions (hold / skip-the-unpredictable), and advance idle rear mass toward the
frontier region. The region layer only re-weights/appends candidates the
K-step rollout chooser still validates — it never emits a move on its own.

GAIN(region) stays OFF (BASELINE_REGION_TAKE unset, and its body is an empty
stub) so the competitive floor is champion-parity.

Bundle this for the contamination-safe A/B and (after PI approval) submission:
    python scripts/bundle_agent.py agents/baseline_region
A bundle is self-contained, so region-bundle vs champion-bundle share no
module state — no in-process os.environ contamination.
"""
from __future__ import annotations
import os

# --- champion 13-var config (matches submissions/baseline_pv_eta_anchor_1163) ---
os.environ.setdefault("BASELINE_JOINT_AGGR", "1")
os.environ.setdefault("BASELINE_JOINT_TOP_K", "5")
os.environ.setdefault("BASELINE_JOINT_MAX_PAIRS", "60")
os.environ.setdefault("BASELINE_REINFORCE_EMIT", "1")
os.environ.setdefault("BASELINE_REINFORCE_ANTICIPATE", "1")
os.environ.setdefault("BASELINE_NEUTRAL_BONUS", "2.0")
os.environ.setdefault("BASELINE_NEUTRAL_EARLY_EXTRA", "1.5")
os.environ.setdefault("BASELINE_NEUTRAL_EARLY_HORIZON", "50")
os.environ.setdefault("BASELINE_ORBITAL_SAFETY", "1")
os.environ.setdefault("BASELINE_PV_ETA", "1")
os.environ.setdefault("BASELINE_VALUE_HEAD", "hybrid")
os.environ.setdefault("BASELINE_CHOOSER", "trajectory")
os.environ.setdefault("BASELINE_JOINT", "1")
os.environ.setdefault("PV_GAMMA", "0.99")

# --- region augmentation ON (hold + advance; gain stays off) ---
os.environ.setdefault("BASELINE_REGION", "1")

from agents.baseline.main import agent  # noqa: E402
