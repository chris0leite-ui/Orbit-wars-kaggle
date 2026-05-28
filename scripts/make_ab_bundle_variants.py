"""Post-process the baseline bundle into A/B variants with baked-in
value-head selection.

`scripts/bundle_agent.py` doesn't inline cross-agent imports (Rule 46
silent-fail mode #3), so the simple "wrapper agent dir" pattern fails
at bundle time. Instead, we bundle `agents/baseline/` once, then
generate two derived submission files by patching the canonical bundle:

  submissions/baseline_learned.py:  wraps agent() with
    `os.environ["BASELINE_VALUE_HEAD"] = "learned"` (per-call override).
  submissions/baseline_favor.py:    wraps agent() with
    `os.environ["BASELINE_VALUE_HEAD"] = ""`     (default favor head).

Per-call (not module-level) override is critical: `fast.py:play_one`
loads BOTH agents into the same worker process, so module-level
mutation leaks across the A/B.

Usage:
  python scripts/bundle_agent.py agents/baseline --force --skip-parity-gate
  python scripts/make_ab_bundle_variants.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "submissions" / "baseline.py"
VARIANTS = {
    "baseline_learned": "learned",
    "baseline_favor": "",
    # `baseline_hybrid` — A/B reference for Phase A distillation. Bakes
    # in `BASELINE_VALUE_HEAD=hybrid` (the mu=1149 head from
    # kaggle-baseline-strategy-lO4mm: composite waste-aware in 2P + A2
    # weakness exploit in 4P; see agents/baseline/value.py:favor_hybrid).
    "baseline_hybrid": "hybrid",
}


def _make_wrapper(source: str, head: str) -> str:
    """Rename the canonical `def agent` -> `def _canonical_agent`, then
    append a thin `def agent` that sets the env var before delegating.
    """
    needle = "\n# === agent ===\n"
    if needle not in source:
        raise RuntimeError(
            f"source missing the '# === agent ===' marker — was the source "
            f"bundled by scripts/bundle_agent.py? "
        )
    # Locate the `def agent(` line after the marker and rename it.
    idx = source.index(needle) + len(needle)
    rest = source[idx:]
    # Replace ONLY the first `def agent(` after the marker (the canonical
    # entry point). Subsequent `agent(` calls inside the body stay intact.
    if "def agent(" not in rest:
        raise RuntimeError("no 'def agent(' after the agent marker")
    rest_renamed = rest.replace("def agent(", "def _canonical_agent(", 1)
    head_repr = repr(head)
    wrapper = (
        "\n\n"
        "# === A/B wrapper appended by scripts/make_ab_bundle_variants.py ===\n"
        "import os as _os_for_ab\n"
        "def agent(obs, configuration=None):\n"
        f"    _os_for_ab.environ['BASELINE_VALUE_HEAD'] = {head_repr}\n"
        "    return _canonical_agent(obs, configuration)\n"
    )
    return source[:idx] + rest_renamed + wrapper


def main() -> int:
    if not SOURCE.exists():
        print(f"error: {SOURCE} not found — bundle agents/baseline first",
              file=sys.stderr)
        return 1
    source = SOURCE.read_text()
    for name, head in VARIANTS.items():
        out = ROOT / "submissions" / f"{name}.py"
        out.write_text(_make_wrapper(source, head))
        print(
            f"wrote {out.relative_to(ROOT)} "
            f"(head={head!r}, {len(out.read_text())} chars)"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
