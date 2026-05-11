"""Shared agent-name → file-path resolver for the 2P / 4P panel scripts.

Lifted out of `scripts/strategy_panel.py::_resolve_agent_path` so the
`scripts/ffa_panel.py` 4P harness uses the same name-resolution rules.
Source of truth for which short names map where.
"""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# Builtin kaggle_environments agents (no file path needed).
BUILTIN_AGENTS = {"random", "starter"}


def resolve_agent_path(name: str) -> str:
    """Map a panel-name to the file path the tournament fixture loads.

    Resolution order:
    1. Builtin name (random / starter) — returned as-is.
    2. `baseline` -> `data/main.py` (comp-shipped Nearest Planet Sniper).
    3. `agents/<name>/main.py` (directory-style agents like v1_orbitfix,
       v2, v3_snipe).
    4. `agents/simple/<name>.py` (flat-file strategies: roi, nearest, …).
    5. Literal path (e.g. `submissions/v2.py`, `agents/v2/main.py`).

    Raises ValueError if no match.
    """
    if name in BUILTIN_AGENTS:
        return name
    if name == "baseline":
        return str(REPO / "data" / "main.py")
    # agents/<name>/main.py — directory-style entry point.
    dir_main = REPO / "agents" / name / "main.py"
    if dir_main.is_file():
        return str(dir_main)
    # agents/simple/<name>.py — flat-file strategy.
    simple = REPO / "agents" / "simple" / f"{name}.py"
    if simple.is_file():
        return str(simple)
    # Literal path.
    p = Path(name)
    if p.is_file():
        return str(p)
    raise ValueError(f"unknown strategy / agent: {name}")
