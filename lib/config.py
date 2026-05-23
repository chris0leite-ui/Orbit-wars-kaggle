"""Per-call env-var helpers for chooser/proposer/value-head constants.

Pre-2026-05-23 several tunable constants were captured at module-import
time (`MAX_HORIZON = int(os.environ.get(..., "40"))`). That made tests
and A/B harnesses that monkey-patch env vars BETWEEN fixtures in the
same Python process silently use the import-time defaults. The
existing `_select_opp_policy` (chooser.py) already documented the
per-call pattern as the correct one ("Per-call selection so env-var
overrides inside test fixtures take effect without re-importing the
module"); this module makes that the convention across the stack.

API:
    env_int(name, default) -> int
    env_float(name, default) -> float
    env_bool(name, default=False) -> bool

All three:
- Read os.environ fresh on every call (no caching).
- Wrap parsing in try/except — malformed env content falls back to
  default rather than crashing at agent import. This closes the
  "kaggle returns ERROR submission, evicts previous good submit" risk
  for env vars set by external tooling (Rule 12).
- env_bool: accepts "1", "true", "on", "yes" (case-insensitive) as
  True; everything else False. Picks ONE convention to replace the
  three pre-existing inconsistent parse rules in the codebase.
"""
from __future__ import annotations

import os


def env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw.strip())
    except (TypeError, ValueError):
        return default


def env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw.strip())
    except (TypeError, ValueError):
        return default


_TRUTHY = frozenset(("1", "true", "on", "yes"))


def env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in _TRUTHY
