"""Rebuild `submissions/baseline_joint_aggr_consolidated_orbitfix.py` as a
proper agent file.

Workaround for friction `bundle-agent-doesnt-inline-from-baseline-main`:
`scripts/bundle_agent.py` cannot produce a working bundle from
`agents/baseline_joint_aggr_consolidated_orbitfix/main.py` because that
main.py is just `from agents.baseline.main import agent` plus env-var
setdefaults — the bundler comments the line out without inlining the body.
The bundler now refuses to leave the broken file (see scripts/bundle_agent.py
sanity-check added 2026-05-21), but it still cannot PRODUCE a working one.

This script does the manual inlining: take `submissions/baseline.py`
(which is bundled from `agents/baseline/main.py` and contains a proper
`def agent` at line ~13527) and prepend the consolidated_orbitfix env-var
block. The env vars are read at module import time by baseline.main, so
prepending works correctly.

Usage:
    python scripts/build_orbitfix_workaround.py

Prereq: `submissions/baseline.py` must already exist (rebuild via
`python scripts/bundle_agent.py agents/baseline ...` if missing).
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SUBMISSIONS = REPO / "submissions"


def main() -> int:
    baseline_path = SUBMISSIONS / "baseline.py"
    if not baseline_path.is_file():
        print(f"missing prereq: {baseline_path}; rebuild via "
              f"`python scripts/bundle_agent.py agents/baseline --lib ...`",
              file=sys.stderr)
        return 1

    baseline = baseline_path.read_text()
    prepend = (
        "# Built by scripts/build_orbitfix_workaround.py — see file header.\n"
        "import os as _orbitfix_os\n"
        '_orbitfix_os.environ.setdefault("BASELINE_JOINT_AGGR", "1")\n'
        '_orbitfix_os.environ.setdefault("BASELINE_JOINT_TOP_K", "5")\n'
        '_orbitfix_os.environ.setdefault("BASELINE_JOINT_MAX_PAIRS", "60")\n'
        '_orbitfix_os.environ.setdefault("BASELINE_REINFORCE_EMIT", "1")\n'
        '_orbitfix_os.environ.setdefault("BASELINE_REINFORCE_ANTICIPATE", "1")\n'
        '_orbitfix_os.environ.setdefault("BASELINE_NEUTRAL_BONUS", "2.0")\n'
        '_orbitfix_os.environ.setdefault("BASELINE_NEUTRAL_EARLY_EXTRA", "1.5")\n'
        '_orbitfix_os.environ.setdefault("BASELINE_NEUTRAL_EARLY_HORIZON", "50")\n'
        '_orbitfix_os.environ.setdefault("BASELINE_ORBITAL_SAFETY", "1")\n'
    )
    out_path = SUBMISSIONS / "baseline_joint_aggr_consolidated_orbitfix.py"
    lines = baseline.splitlines(keepends=True)
    insert_idx = None
    for i, ln in enumerate(lines):
        if ln.startswith("from __future__"):
            insert_idx = i + 1
            break
    if insert_idx is None:
        print(f"{baseline_path} has no `from __future__` line — aborting",
              file=sys.stderr)
        return 1
    new_lines = lines[:insert_idx] + ["\n", prepend, "\n"] + lines[insert_idx:]
    out_path.write_text("".join(new_lines))
    print(f"wrote {out_path} ({out_path.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
