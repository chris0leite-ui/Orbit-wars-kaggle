"""baseline_full — coherent all-features stack vs consolidated.

Consolidated base (sub 52882014):
- JOINT_AGGR + TOP_K=5 + MAX_PAIRS=60 — multi-source same-target
- JOINT-in-4P fix (lift used_tgts double-count)
- REINFORCE_EMIT — defends predicted-to-fall friendlies
- REINFORCE_ANTICIPATE — defends inbound-enemy-thinned friendlies
- NEUTRAL_BONUS=2.0 — early-game expansion tilt

Added this session (2026-05-21):
- ORBITAL_SAFETY — correctness fix for time_to_enemy_threat: post-arrival
  hold of orbiting targets uses predicted position at our arrival.
- STAGNANT_DRAIN — dynamic-reserve drain of true-rear sources toward
  closer-to-front friendlies. n=16 alone showed 37.5% (Wilson [0.185,0.614]).
- COMBAT_STACK — drain excess directly onto NON-OUR planets we're
  already attacking (cluster at combat).
- SNIPER — eta-sorted decisive cross-map strikes at enemy >=+4 prod
  planets when total reserve > 300 (close + big = fast).
"""
from __future__ import annotations
import os
os.environ.setdefault("BASELINE_JOINT_AGGR", "1")
os.environ.setdefault("BASELINE_JOINT_TOP_K", "5")
os.environ.setdefault("BASELINE_JOINT_MAX_PAIRS", "60")
os.environ.setdefault("BASELINE_REINFORCE_EMIT", "1")
os.environ.setdefault("BASELINE_REINFORCE_ANTICIPATE", "1")
os.environ.setdefault("BASELINE_NEUTRAL_BONUS", "2.0")
os.environ.setdefault("BASELINE_NEUTRAL_EARLY_EXTRA", "1.5")
os.environ.setdefault("BASELINE_NEUTRAL_EARLY_HORIZON", "50")
os.environ.setdefault("BASELINE_ORBITAL_SAFETY", "1")
os.environ.setdefault("BASELINE_STAGNANT_DRAIN", "1")
os.environ.setdefault("BASELINE_COMBAT_STACK", "1")
os.environ.setdefault("BASELINE_SNIPER", "1")
from agents.baseline.main import agent  # noqa: E402
