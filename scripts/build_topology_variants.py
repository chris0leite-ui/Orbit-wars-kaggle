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

# Phase ε.1: maximin search gate.
MAXIMIN_RE = re.compile(
    r"(def _maximin_enabled\(\) -> bool:\n)"
    r"((?:    .*\n)+?)"
    r"(\n)",
    re.MULTILINE,
)

# Phase 3 (λ_W sweep): the `LAMBDA_W_DEFAULT = X.Y` assignment in the
# inlined `lp_outcome.py` section. Read at call time by `_lambda_w()`
# (lazy) when `LP_LAMBDA_W` env var is unset — so rewriting this
# constant in the bundle is the cleanest way to produce per-λ_W
# variants without touching env-var pollution.
LAMBDA_W_DEFAULT_RE = re.compile(
    r"^LAMBDA_W_DEFAULT\s*=\s*[0-9.]+\s*$",
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


def _replace_maximin(src: str, retval: str) -> str:
    """Replace `_maximin_enabled()` body."""
    def repl(m: re.Match) -> str:
        signature = m.group(1)
        trailing = m.group(3)
        return f"{signature}    return {retval}  # hardcoded by build_topology_variants.py\n{trailing}"
    return MAXIMIN_RE.sub(repl, src)


def _replace_lambda_w(src: str, value: float) -> str:
    """Replace `LAMBDA_W_DEFAULT = X.Y` with `LAMBDA_W_DEFAULT = <value>`."""
    return LAMBDA_W_DEFAULT_RE.sub(
        f"LAMBDA_W_DEFAULT = {value}  # hardcoded by build_topology_variants.py",
        src,
    )


def _apply_all(src: str, *, topo: str, smooth: str, maximin: str,
               lambda_w: float | None = None) -> str:
    """Compose all rewrites — order independent."""
    out = _replace_maximin(
        _replace_smooth_delta_w(_replace_topology(src, topo), smooth),
        maximin,
    )
    if lambda_w is not None:
        out = _replace_lambda_w(out, lambda_w)
    return out


def main() -> int:
    if not SRC.exists():
        print(f"ERROR: {SRC.relative_to(REPO)} missing. "
              f"Run `python scripts/bundle_analytical_phase_c.py` first.",
              file=sys.stderr)
        return 1

    src = SRC.read_text()
    # Sanity: confirm we see all 4 topology lazy fns + 1 smooth-ΔW + 1 maximin.
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
    maximin_matches = MAXIMIN_RE.findall(src)
    if len(maximin_matches) != 1:
        print(f"ERROR: expected 1 `_maximin_enabled()` block; "
              f"found {len(maximin_matches)}. (Did Phase ε.1 land?)",
              file=sys.stderr)
        return 1

    # Phase β isolation: topology on/off, smooth-ΔW off, maximin off.
    OUT_ON.write_text(_apply_all(src, topo="True",  smooth="False", maximin="False"))
    OUT_OFF.write_text(_apply_all(src, topo="False", smooth="False", maximin="False"))

    # Phase α isolation: smooth-ΔW on/off, topology off, maximin off.
    alpha_on = OUT_ON.parent / "smooth_dw_on.py"
    alpha_off = OUT_ON.parent / "smooth_dw_off.py"
    alpha_on.write_text(_apply_all(src, topo="False", smooth="True",  maximin="False"))
    alpha_off.write_text(_apply_all(src, topo="False", smooth="False", maximin="False"))

    # α+β stacked: both on/off, maximin off.
    stacked_on = OUT_ON.parent / "alpha_beta_on.py"
    stacked_off = OUT_ON.parent / "alpha_beta_off.py"
    stacked_on.write_text(_apply_all(src, topo="True",  smooth="True",  maximin="False"))
    stacked_off.write_text(_apply_all(src, topo="False", smooth="False", maximin="False"))

    # Phase ε.1 isolation: α+β baseline (both ON) + maximin on/off.
    # focal = α+β+maximin ALL ON; opp = α+β ON, maximin OFF.
    maximin_on = OUT_ON.parent / "maximin_on.py"
    maximin_off = OUT_ON.parent / "maximin_off.py"
    maximin_on.write_text(_apply_all(src, topo="True",  smooth="True",  maximin="True"))
    maximin_off.write_text(_apply_all(src, topo="True",  smooth="True",  maximin="False"))

    # Phase 3 (λ_W sweep): α+β stacked variants at different λ_W
    # default values. All share topo=True smooth=True maximin=False.
    # File naming: alpha_beta_lambda_<int>_<frac>.py — e.g. _0_1, _1_0.
    lambda_paths: list[tuple[Path, float]] = []
    for lw in (0.1, 0.3, 1.0, 3.0):
        # Normalize 0.1 → "0_1", 1.0 → "1_0", 3.0 → "3_0", 0.3 → "0_3"
        i, frac = divmod(lw, 1)
        name = f"alpha_beta_lambda_{int(i)}_{int(round(frac * 10))}.py"
        p = OUT_ON.parent / name
        p.write_text(_apply_all(src, topo="True", smooth="True",
                                maximin="False", lambda_w=lw))
        lambda_paths.append((p, lw))

    # Verify all bundles load and report the right hardcoded value.
    base_checks = [
        (OUT_ON,      True,  False, False, None),
        (OUT_OFF,     False, False, False, None),
        (alpha_on,    False, True,  False, None),
        (alpha_off,   False, False, False, None),
        (stacked_on,  True,  True,  False, None),
        (stacked_off, False, False, False, None),
        (maximin_on,  True,  True,  True,  None),
        (maximin_off, True,  True,  False, None),
    ]
    lambda_checks = [(p, True, True, False, lw) for p, lw in lambda_paths]
    for path, topo, smooth, maximin, expected_lambda in base_checks + lambda_checks:
        import importlib.util
        spec = importlib.util.spec_from_file_location(path.stem, str(path))
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        topo_fn = m.__dict__.get("_topology_features_enabled")
        smooth_fn = m.__dict__.get("_smooth_delta_w_enabled")
        maximin_fn = m.__dict__.get("_maximin_enabled")
        if topo_fn is None or smooth_fn is None or maximin_fn is None:
            print(f"FAIL: {path.name} missing gate fn", file=sys.stderr)
            return 1
        a_topo, a_smooth, a_maximin = topo_fn(), smooth_fn(), maximin_fn()
        if a_topo != topo or a_smooth != smooth or a_maximin != maximin:
            print(f"FAIL: {path.name} topo={a_topo}/{topo}, "
                  f"smooth={a_smooth}/{smooth}, maximin={a_maximin}/{maximin}",
                  file=sys.stderr)
            return 1
        if not callable(getattr(m, "agent", None)):
            print(f"FAIL: {path.name} no callable agent symbol",
                  file=sys.stderr)
            return 1
        # If a specific λ_W was baked in, verify _lambda_w() returns it
        # (the lazy fn reads `LP_LAMBDA_W` env first, then falls back to
        # LAMBDA_W_DEFAULT). With LP_LAMBDA_W unset, the bundle constant
        # is the source of truth.
        lambda_str = ""
        if expected_lambda is not None:
            lambda_fn = m.__dict__.get("_lambda_w")
            assert lambda_fn is not None, "_lambda_w missing from bundle"
            actual_lambda = float(lambda_fn())
            if abs(actual_lambda - expected_lambda) > 1e-9:
                print(f"FAIL: {path.name} _lambda_w()={actual_lambda} "
                      f"≠ {expected_lambda}", file=sys.stderr)
                return 1
            lambda_str = f", λ_W={actual_lambda}"
        print(f"OK: {path.name} loads, agent callable, "
              f"topology={a_topo}, smooth_ΔW={a_smooth}, maximin={a_maximin}"
              f"{lambda_str}, {path.stat().st_size} bytes")

    return 0


if __name__ == "__main__":
    sys.exit(main())
