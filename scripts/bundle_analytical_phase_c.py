"""Bundle the analytical Phase C agent.

Phase C extends the analytical agent's seven-stage pipeline (see
lib/pipeline/) with two stage swaps:
  Stage 3: prerank_passthrough (closes pre-filter amputation)
  Stage 7: commit_persistent   (closes wait_N evaporation)

This script extends `bundle_analytical.py` by ALSO inlining the
`lib/pipeline/` modules in dependency order, then bundling the
Phase C entry point.

Dependency order matters because intra-package imports get stripped:
  - types.py        (no internal deps)
  - perception.py, candidates.py, prerank.py, prerank_passthrough.py,
    opp_model.py, decision.py, leaf_outcome_table.py
  - pending_schedule.py
  - commit.py, commit_persistent.py
  - opening.py
  - compose.py     (imports default-stage references)

Output: submissions/analytical_phase_c.py

Usage:
    python scripts/bundle_analytical_phase_c.py
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# Joint-solver dependency order (same as bundle_analytical.py).
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

# Pipeline modules in dependency order.
PIPELINE_ORDER = [
    "lib/pipeline/types.py",
    "lib/pipeline/perception.py",
    "lib/pipeline/candidates.py",
    "lib/pipeline/prerank.py",
    "lib/pipeline/prerank_passthrough.py",
    "lib/pipeline/opp_model.py",
    "lib/pipeline/decision.py",
    "lib/pipeline/leaf_outcome_table.py",
    "lib/pipeline/pending_schedule.py",
    "lib/pipeline/commit.py",
    "lib/pipeline/commit_persistent.py",
    "lib/pipeline/opening.py",
    "lib/pipeline/compose.py",
]

ANALYTICAL_MAIN = "agents/analytical_phase_c/main.py"

# Same regex as bundle_analytical.py.
_INTRA_IMPORT_RE = re.compile(
    r"^\s*from (lib|\.|agents\.[\w]+)[\w.]*\s+import\b.*$"
)
_INTRA_IMPORT_OPEN_RE = re.compile(
    r"^\s*from (lib|\.|agents\.[\w]+)[\w.]*\s+import\s*\(\s*$"
)
_FUTURE_IMPORT_RE = re.compile(r"^\s*from __future__\s+import\b.*$")


def _strip_imports(src: str) -> str:
    """Strip intra-package imports.

    Handles both single-line `from lib.X import a, b` and multi-line
    `from lib.X import (\n  a,\n  b,\n)` (closing paren on its own line
    or with content). When a multi-line open is seen, skip until the
    closing paren line is consumed too.

    NOTE: `from __future__ import annotations` is KEPT. Without it,
    pipeline dataclasses (e.g. TurnContext with `world: World` /
    `list[Column]` type hints) trigger
    `dataclasses._is_type` → `sys.modules.get(cls.__module__).__dict__`
    which fails when the bundled module isn't registered in
    sys.modules (Kaggle's loader path). Keeping future-annotations
    defers all type-hint evaluation to strings, avoiding the lookup.
    De-duped at the top of the bundle by `_dedupe_future_imports`.
    """
    out_lines: list[str] = []
    skip_until_close_paren = False
    for line in src.splitlines():
        if skip_until_close_paren:
            stripped = line.strip()
            if stripped.endswith(")"):
                skip_until_close_paren = False
            continue
        if _INTRA_IMPORT_OPEN_RE.match(line):
            skip_until_close_paren = True
            continue
        if _INTRA_IMPORT_RE.match(line):
            continue
        # Drop __future__ imports from the BODY (they get hoisted to a
        # single canonical line at the top of the bundle).
        if _FUTURE_IMPORT_RE.match(line):
            continue
        out_lines.append(line)
    return "\n".join(out_lines)


def _inline_block(rel_path: str, sub_marker: bool = True) -> str:
    src = (REPO / rel_path).read_text(encoding="utf-8")
    stripped = _strip_imports(src)
    marker = "---" if sub_marker else "==="
    header = f"# {marker} inlined: {rel_path} {marker}"
    return f"\n\n{header}\n\n{stripped}\n"


_SYSMOD_SHIM = """
# Bundle-self-registration shim. Required because Kaggle's loader calls
# importlib.util.exec_module without first registering the module in
# sys.modules; the dataclasses module then crashes when it does
# sys.modules.get(cls.__module__).__dict__ during KW_ONLY detection
# (Python 3.11 dataclasses.py:712). Registering ourselves as a stub
# pointing at __main__ short-circuits the lookup harmlessly.
import sys as _bundle_sys
_bundle_sys.modules.setdefault(__name__, _bundle_sys.modules.get("__main__"))
"""


def main() -> int:
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

    # Inject sys.modules self-registration shim immediately after the
    # `from __future__ import annotations` line at the top of the
    # baseline bundle. Pipeline dataclasses defined later in the bundle
    # need this to load on Kaggle.
    import re as _re
    baseline_bundle = _re.sub(
        r"(from __future__ import annotations\n)",
        r"\1" + _SYSMOD_SHIM,
        baseline_bundle, count=1,
    )

    parts: list[str] = [baseline_bundle.rstrip()]

    parts.append("\n\n# === joint_solver (Phase 5 analytical primitives) ===\n")
    for rel in JOINT_SOLVER_ORDER:
        parts.append(_inline_block(rel, sub_marker=True))

    parts.append("\n\n# === lib/pipeline (Phase A scaffold + Phase C swaps) ===\n")
    for rel in PIPELINE_ORDER:
        parts.append(_inline_block(rel, sub_marker=True))

    parts.append("\n\n# === Phase C analytical entry point ===\n")
    parts.append(_inline_block(ANALYTICAL_MAIN, sub_marker=True))

    out = REPO / "submissions" / "analytical_phase_c.py"
    out.write_text("".join(parts), encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
