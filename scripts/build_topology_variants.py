"""Build two hardcoded topology-variant bundles for the Phase β A/B.

Source: `submissions/analytical_phase_c.py` (must exist; run
`python scripts/bundle_analytical_phase_c.py` first).

Output bundles (overwrites if present):
  - submissions/topology_on.py   — topology features hardcoded TRUE
  - submissions/topology_off.py  — topology features hardcoded FALSE

Why hardcoded: the existing build_fnd_baseline.sh approach (replace the
`os.environ.setdefault("LP_TOPOLOGY_FEATURES", "1")` line) does NOT
isolate topology in env.run, because kaggle_environments loads both
agents in the SAME Python process. The first-loaded agent's setdefault
sets the env-var; the second-loaded agent's setdefault is a no-op for
already-set keys; both then read env at call time via the lazy
`_topology_features_enabled()` and end up with the FIRST-loaded agent's
intended config. The 2026-05-22 audit (audit/2026-05-22/SESSION_SUMMARY.md)
documented this. The Phase β A/B (n=8, perfect 4W/4L per-seat mirror,
Wilson[0.215, 0.785]) is consistent with both agents running the same
config in each game.

Fix: replace the four lazy `_*_enabled()` function bodies with literal
`return True` / `return False`. The bundle then has no env-var read for
topology and cannot be contaminated by the other agent's setdefault.

Usage:
    python scripts/build_topology_variants.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "submissions" / "analytical_phase_c.py"
OUT_ON = REPO / "submissions" / "topology_on.py"
OUT_OFF = REPO / "submissions" / "topology_off.py"


# Match each lazy function block (def line + 1-3 body lines until next blank).
LAZY_BLOCK_RE = re.compile(
    r"(def _(?:topology_features|reach_bonus|defense_bonus|front_penalty)_enabled\(\) -> bool:\n)"
    r"((?:    .*\n)+?)"
    r"(\n)",
    re.MULTILINE,
)


def _replace_with(src: str, retval: str) -> str:
    """Replace each lazy `_*_enabled()` body with `return <retval>`."""
    def repl(m: re.Match) -> str:
        signature = m.group(1)
        trailing = m.group(3)
        return f"{signature}    return {retval}  # hardcoded by build_topology_variants.py\n{trailing}"
    return LAZY_BLOCK_RE.sub(repl, src)


def main() -> int:
    if not SRC.exists():
        print(f"ERROR: {SRC.relative_to(REPO)} missing. "
              f"Run `python scripts/bundle_analytical_phase_c.py` first.",
              file=sys.stderr)
        return 1

    src = SRC.read_text()
    # Sanity: confirm we see all 4 lazy functions.
    matches = LAZY_BLOCK_RE.findall(src)
    if len(matches) != 4:
        print(f"ERROR: expected 4 lazy `_*_enabled()` blocks; found {len(matches)}.",
              file=sys.stderr)
        return 1

    on_src = _replace_with(src, "True")
    off_src = _replace_with(src, "False")

    OUT_ON.write_text(on_src)
    OUT_OFF.write_text(off_src)

    # Verify both load and report the right hardcoded value.
    for path, expected in [(OUT_ON, True), (OUT_OFF, False)]:
        import importlib.util
        spec = importlib.util.spec_from_file_location("m", str(path))
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        flag = m.__dict__.get("_topology_features_enabled")
        if flag is None:
            print(f"FAIL: {path.name} missing _topology_features_enabled",
                  file=sys.stderr)
            return 1
        actual = flag()
        if actual != expected:
            print(f"FAIL: {path.name} _topology_features_enabled() = {actual}, expected {expected}",
                  file=sys.stderr)
            return 1
        if not callable(getattr(m, "agent", None)):
            print(f"FAIL: {path.name} no callable agent symbol", file=sys.stderr)
            return 1
        print(f"OK: {path.name} loads, agent callable, "
              f"_topology_features_enabled() = {actual} (expected {expected}), "
              f"{path.stat().st_size} bytes")

    return 0


if __name__ == "__main__":
    sys.exit(main())
