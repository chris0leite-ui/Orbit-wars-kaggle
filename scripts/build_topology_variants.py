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

# Phase α: same idea for the smooth-ΔW lazy gate.
SMOOTH_DELTA_W_RE = re.compile(
    r"(def _smooth_delta_w_enabled\(\) -> bool:\n)"
    r"((?:    .*\n)+?)"
    r"(\n)",
    re.MULTILINE,
)


def _replace_topology(src: str, retval: str) -> str:
    """Replace each topology lazy `_*_enabled()` body with `return <retval>`."""
    def repl(m: re.Match) -> str:
        signature = m.group(1)
        trailing = m.group(3)
        return f"{signature}    return {retval}  # hardcoded by build_topology_variants.py\n{trailing}"
    return LAZY_BLOCK_RE.sub(repl, src)


def _replace_smooth_delta_w(src: str, retval: str) -> str:
    """Replace `_smooth_delta_w_enabled()` body."""
    def repl(m: re.Match) -> str:
        signature = m.group(1)
        trailing = m.group(3)
        return f"{signature}    return {retval}  # hardcoded by build_topology_variants.py\n{trailing}"
    return SMOOTH_DELTA_W_RE.sub(repl, src)


def main() -> int:
    if not SRC.exists():
        print(f"ERROR: {SRC.relative_to(REPO)} missing. "
              f"Run `python scripts/bundle_analytical_phase_c.py` first.",
              file=sys.stderr)
        return 1

    src = SRC.read_text()
    # Sanity: confirm we see all 4 topology lazy fns + 1 smooth-ΔW lazy fn.
    topo_matches = LAZY_BLOCK_RE.findall(src)
    if len(topo_matches) != 4:
        print(f"ERROR: expected 4 lazy `_*_enabled()` topology blocks; "
              f"found {len(topo_matches)}.", file=sys.stderr)
        return 1
    smooth_matches = SMOOTH_DELTA_W_RE.findall(src)
    if len(smooth_matches) != 1:
        print(f"ERROR: expected 1 `_smooth_delta_w_enabled()` block; "
              f"found {len(smooth_matches)}. (Did Phase α land in tree?)",
              file=sys.stderr)
        return 1

    # Two baseline variants (topology on/off, smooth-ΔW off — Phase β isolation):
    on_src = _replace_smooth_delta_w(_replace_topology(src, "True"), "False")
    off_src = _replace_smooth_delta_w(_replace_topology(src, "False"), "False")

    OUT_ON.write_text(on_src)
    OUT_OFF.write_text(off_src)

    # Phase α variants (smooth-ΔW on/off, topology off — Phase α isolation):
    alpha_on = OUT_ON.parent / "smooth_dw_on.py"
    alpha_off = OUT_ON.parent / "smooth_dw_off.py"
    alpha_on.write_text(
        _replace_smooth_delta_w(_replace_topology(src, "False"), "True")
    )
    alpha_off.write_text(
        _replace_smooth_delta_w(_replace_topology(src, "False"), "False")
    )

    # Stacked variants (both α + β features ON or both OFF — for the
    # combined-features A/B once α-alone and β-alone null):
    stacked_on = OUT_ON.parent / "alpha_beta_on.py"
    stacked_off = OUT_ON.parent / "alpha_beta_off.py"
    stacked_on.write_text(
        _replace_smooth_delta_w(_replace_topology(src, "True"), "True")
    )
    stacked_off.write_text(
        _replace_smooth_delta_w(_replace_topology(src, "False"), "False")
    )

    # Verify all six bundles load and report the right hardcoded value.
    for path, topo, smooth in [
        (OUT_ON, True, False),
        (OUT_OFF, False, False),
        (alpha_on, False, True),
        (alpha_off, False, False),
        (stacked_on, True, True),
        (stacked_off, False, False),
    ]:
        import importlib.util
        spec = importlib.util.spec_from_file_location(path.stem, str(path))
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        topo_fn = m.__dict__.get("_topology_features_enabled")
        smooth_fn = m.__dict__.get("_smooth_delta_w_enabled")
        if topo_fn is None or smooth_fn is None:
            print(f"FAIL: {path.name} missing gate fn", file=sys.stderr)
            return 1
        a_topo = topo_fn()
        a_smooth = smooth_fn()
        if a_topo != topo or a_smooth != smooth:
            print(f"FAIL: {path.name} topo={a_topo}/{topo}, "
                  f"smooth={a_smooth}/{smooth}", file=sys.stderr)
            return 1
        if not callable(getattr(m, "agent", None)):
            print(f"FAIL: {path.name} no callable agent symbol",
                  file=sys.stderr)
            return 1
        print(f"OK: {path.name} loads, agent callable, "
              f"topology={a_topo}, smooth_ΔW={a_smooth}, "
              f"{path.stat().st_size} bytes")

    return 0


if __name__ == "__main__":
    sys.exit(main())
