"""Regression check: every file in `submissions/` must expose a callable
`agent` at module top level.

History — 2026-05-21: `scripts/bundle_agent.py` silently produced a copy of
`baseline_joint_aggr_consolidated.py` with no `agent` symbol (it stripped
`from agents.baseline.main import agent` without inlining the function body).
kaggle_environments fell back to the last callable (`opening_plan`), wrong
signature, every game ERROR'd at step 0. The sibling branch's "n=8 vs LATEST
8W/0L" A/B was Phase 4 beating an ERROR-on-step-0 file and was used to ship
sub 52894340. This test catches that class of bug before it leaves the local
tree. Cherry-picked from `claude/strategy-axis-decision-3437` c25a329.

Bundles whose top-of-file comment marks them BROKEN-AS-AGENT-FILE are skipped
— that marker is a deliberate keep-for-history flag, not a pass.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SUBMISSIONS = REPO / "submissions"


def _is_marked_broken(path: Path) -> bool:
    """Files with the BROKEN-AS-AGENT-FILE marker in the first 50 lines are
    intentionally kept for history; skip them."""
    try:
        head = "\n".join(path.read_text().splitlines()[:50])
    except Exception:
        return False
    return "BROKEN-AS-AGENT-FILE" in head


def _python_submission_files() -> list[Path]:
    if not SUBMISSIONS.exists():
        return []
    files = []
    for p in sorted(SUBMISSIONS.glob("*.py")):
        if p.name.startswith("_bundle_"):
            continue
        files.append(p)
    return files


@pytest.mark.parametrize("path", _python_submission_files(),
                         ids=lambda p: p.name)
def test_submission_exposes_agent_callable(path: Path) -> None:
    if _is_marked_broken(path):
        pytest.skip(f"{path.name}: marked BROKEN-AS-AGENT-FILE in header")

    spec = importlib.util.spec_from_file_location(
        f"_submcheck_{path.stem}", path)
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules[f"_submcheck_{path.stem}"] = mod
    sys.modules.setdefault("__main__", mod)
    try:
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
    except Exception as e:
        pytest.fail(f"{path.name}: import failed — {type(e).__name__}: {e}")

    assert callable(getattr(mod, "agent", None)), (
        f"{path.name}: no callable `agent` at module top level. "
        f"kaggle_environments will fall back to last callable in the "
        f"module namespace (often the wrong signature → game ERROR at step 0). "
        f"If this bundle was built from a wrapper like "
        f"`from agents.baseline.main import agent`, the bundler likely "
        f"stripped the import without inlining the body — see friction tag "
        f"`bundle-agent-doesnt-inline-from-baseline-main`."
    )
