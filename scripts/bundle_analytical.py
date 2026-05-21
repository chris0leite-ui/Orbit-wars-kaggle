"""Bundle the analytical agent on top of the baseline bundle.

The analytical agent depends on `lib.joint_solver.*` modules AND on
`agents.baseline.*` siblings (proposer, migration_solver, predicates,
chooser_trajectory). The general-purpose `scripts/bundle_agent.py`
inlines an agent's own siblings but not cross-agent files, so this
script implements the workaround documented in
audit/friction.md::cross-agent-imports-not-bundled:

  1. Run `bundle_agent.py agents/baseline` (helpers already inlined
     into submissions/baseline.py).
  2. Append the joint_solver modules in dependency order with
     intra-package imports stripped.
  3. Append agents/analytical/main.py (its `agent()` overrides the
     baseline `agent()` defined earlier in the file).

Output: submissions/analytical.py

Usage:
    python scripts/bundle_analytical.py
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# Dependency order: each entry must be importable given everything above
# it. predicate has no joint_solver deps; portfolio uses predicate;
# outcome_table is standalone; columns/lp/value/opp_projection chain;
# opening_planner uses columns + value; lp_outcome uses outcome_table;
# mpc is the orchestrator and imports nearly everything else.
JOINT_SOLVER_ORDER = [
    "lib/joint_solver/predicate.py",
    "lib/joint_solver/portfolio.py",
    "lib/joint_solver/outcome_table.py",
    "lib/joint_solver/columns.py",
    "lib/joint_solver/lp.py",
    "lib/joint_solver/value.py",
    "lib/joint_solver/opp_projection.py",
    "lib/joint_solver/opening_planner.py",
    "lib/joint_solver/lp_outcome.py",
    "lib/joint_solver/mpc.py",
]

ANALYTICAL_MAIN = "agents/analytical/main.py"

# Strip patterns: same set as scripts/bundle_agent.py — intra-package
# imports (lib.*, agents.<x>.*, relative) get removed because the
# referenced symbols are inlined.
_INTRA_IMPORT_RE = re.compile(
    r"^\s*from (lib|\.|agents\.[\w]+)[\w.]*\s+import\b.*$"
)
_FUTURE_IMPORT_RE = re.compile(r"^\s*from __future__\s+import\b.*$")


def _strip_imports(src: str) -> str:
    """Strip intra-package and __future__ imports.

    Multi-line `from X import ( ... )` are kept single-line in this
    codebase (Rule: bundle-multi-line-imports-broken friction); we
    don't try to handle them.
    """
    out_lines: list[str] = []
    for line in src.splitlines():
        if _INTRA_IMPORT_RE.match(line):
            continue
        if _FUTURE_IMPORT_RE.match(line):
            continue
        out_lines.append(line)
    return "\n".join(out_lines)


def _inline_block(rel_path: str, sub_marker: bool = True) -> str:
    """Read REPO/<rel_path>, strip imports, wrap in a marker comment."""
    src = (REPO / rel_path).read_text(encoding="utf-8")
    stripped = _strip_imports(src)
    marker = "---" if sub_marker else "==="
    header = f"# {marker} inlined: {rel_path} {marker}"
    return f"\n\n{header}\n\n{stripped}\n"


def main() -> int:
    # 1. Build baseline bundle (idempotent; bundle_agent.py refuses to
    # overwrite git-tracked files but submissions/* is .gitignored).
    print("== rebuilding submissions/baseline.py via bundle_agent.py ==")
    subprocess.check_call([
        sys.executable, str(REPO / "scripts" / "bundle_agent.py"),
        "agents/baseline",
        "--skip-parity-gate",
        "--force",
    ], cwd=str(REPO))

    baseline_bundle = (REPO / "submissions" / "baseline.py").read_text(
        encoding="utf-8",
    )

    # 2. Append Phase 5 section.
    parts: list[str] = [baseline_bundle.rstrip()]
    parts.append("\n\n# === Phase 5 analytical agent (joint_solver) ===\n")
    for rel in JOINT_SOLVER_ORDER:
        parts.append(_inline_block(rel, sub_marker=True))

    # 3. Append agents/analytical/main.py. Its `agent()` overrides the
    # baseline `agent()` already in the file because Python definitions
    # are processed top-to-bottom and the last `def agent` wins.
    parts.append(_inline_block(ANALYTICAL_MAIN, sub_marker=True))

    out = REPO / "submissions" / "analytical.py"
    out.write_text("".join(parts), encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
