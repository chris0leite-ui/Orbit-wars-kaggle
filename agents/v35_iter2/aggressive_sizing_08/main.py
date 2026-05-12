"""aggressive_sizing variant with SHIP_FRACTION=0.8 (closer to top-10 empirical ~0.78)."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import agents.v35_iter2.aggressive_sizing.main as base


def agent(obs):
    base.SHIP_FRACTION = 0.8
    return base.agent(obs)
