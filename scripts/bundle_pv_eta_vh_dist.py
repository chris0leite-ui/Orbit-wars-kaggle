"""Bundler for `agents/baseline_pv_eta_vh_dist` — Tier 2 v2 distilled-ladder
opp-model wrapper.

Mirrors `scripts/bundle_pv_eta_vh_opp.py`. The wrapper is env-var-only
(no Python wrapping code). Bundle structure:

    1. Wrapper env-var preamble (from agents/baseline_pv_eta_vh_dist/main.py).
    2. Full `agents/baseline` bundle (lib + agent submodules + main with
       `def agent(...)` at the bottom).
    3. `_OPP_BOOSTER_B64` patched into the inlined `lib/opp_model.py`
       body so the distilled-ladder booster is available at submit time
       without lightgbm. Source: `data/opp_distill/distill_booster.txt`.

At `BASELINE_OPP_TIER=0`, this submission is byte-equivalent to bare
pv_eta. The Tier-2 v2 policy itself falls back to lite_greedy_policy if
the booster fails to load — never silent garbage launches.
"""

from __future__ import annotations

import argparse
import base64
import gzip
import importlib.util
import re
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.bundle_agent import (  # noqa: E402
    DEFAULT_LIB_ORDER,
    bundle as _bundle_agent,
)

DEFAULT_INNER = REPO / "agents" / "baseline"
DEFAULT_WRAPPER = REPO / "agents" / "baseline_pv_eta_vh_dist"
DEFAULT_OUT = REPO / "submissions" / "baseline_pv_eta_vh_dist.py"
DEFAULT_BOOSTER = REPO / "data" / "opp_distill" / "distill_booster.txt"

_WRAPPER_IMPORT_RE = re.compile(
    r"^\s*from\s+agents\.baseline\.main\s+import\s+agent\b.*$",
    re.MULTILINE,
)

_FUTURE_IMPORT_RE = re.compile(
    r"^\s*from __future__\s+import\b.*$",
    re.MULTILINE,
)

_OPP_BOOSTER_B64_PATCH_RE = re.compile(
    r'^_OPP_BOOSTER_B64: str = ""\s*$',
    re.MULTILINE,
)


def _booster_to_b64(booster_path: Path) -> str:
    text = booster_path.read_text()
    return base64.b64encode(gzip.compress(text.encode())).decode()


def _strip_wrapper_import(text: str) -> str:
    new = _WRAPPER_IMPORT_RE.sub("", text, count=1)
    if new == text:
        raise RuntimeError(
            "wrapper bundler: expected to strip "
            "`from agents.baseline.main import agent` but found none."
        )
    return new


def _bundle_inner_to_text(inner_dir: Path, lib_modules: list[str]) -> str:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        out = _bundle_agent(inner_dir, lib_modules, out_dir=tmp_path, force=True)
        return out.read_text()


def _patch_opp_booster_b64(text: str, b64: str) -> str:
    new = _OPP_BOOSTER_B64_PATCH_RE.sub(
        f'_OPP_BOOSTER_B64: str = "{b64}"', text, count=1,
    )
    if new == text:
        raise RuntimeError(
            "bundler: failed to patch `_OPP_BOOSTER_B64: str = \"\"` in the "
            "bundled lib/opp_model body. Has the literal changed?"
        )
    return new


def _smoke_import(bundle_path: Path) -> None:
    spec = importlib.util.spec_from_file_location(
        "_bundle_pv_eta_vh_dist_smoke", bundle_path,
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    fn = getattr(mod, "agent", None)
    if not callable(fn):
        raise RuntimeError(
            f"bundle {bundle_path} has no top-level callable `agent`."
        )


def build(
    inner_dir: Path = DEFAULT_INNER,
    wrapper_dir: Path = DEFAULT_WRAPPER,
    booster_path: Path = DEFAULT_BOOSTER,
    lib_modules: list[str] | None = None,
    out_path: Path = DEFAULT_OUT,
) -> Path:
    lib_modules = lib_modules or DEFAULT_LIB_ORDER
    for required in ("shot_features", "_validator_tree_walker",
                     "value_head_features"):
        if required not in lib_modules:
            lib_modules = list(lib_modules) + [required]

    if not booster_path.is_file():
        raise FileNotFoundError(
            f"booster artifact missing: {booster_path}. "
            f"Run `python scripts/train_opp_distill.py` first."
        )

    inner_text = _bundle_inner_to_text(inner_dir, lib_modules)
    b64 = _booster_to_b64(booster_path)
    inner_patched = _patch_opp_booster_b64(inner_text, b64)
    inner_stripped = _FUTURE_IMPORT_RE.sub("", inner_patched, count=1)

    wrapper_main = wrapper_dir / "main.py"
    wrapper_body = _strip_wrapper_import(wrapper_main.read_text())
    wrapper_body = _FUTURE_IMPORT_RE.sub("", wrapper_body, count=1)

    header = (
        f"# Bundled by scripts/bundle_pv_eta_vh_dist.py — "
        f"{inner_dir.relative_to(REPO)} (inner) + "
        f"{wrapper_dir.relative_to(REPO)} (wrapper) + "
        f"{booster_path.relative_to(REPO)} (distill-booster B64).\n"
        f"# Single-file Kaggle submission for Orbit Wars.\n\n"
    )

    parts = [
        header,
        "from __future__ import annotations\n\n",
        "# === wrapper preamble (env-var defaults) ===\n",
        wrapper_body,
        "\n# === inner bundle ===\n",
        inner_stripped,
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(parts))

    _smoke_import(out_path)
    return out_path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--inner", default=str(DEFAULT_INNER))
    p.add_argument("--wrapper", default=str(DEFAULT_WRAPPER))
    p.add_argument("--booster", default=str(DEFAULT_BOOSTER))
    p.add_argument("--out", default=str(DEFAULT_OUT))
    args = p.parse_args(argv)

    out = build(
        inner_dir=Path(args.inner),
        wrapper_dir=Path(args.wrapper),
        booster_path=Path(args.booster),
        out_path=Path(args.out),
    )
    print(f"wrote {out} ({out.stat().st_size:,} bytes)")
    print(f"  smoke import OK; top-level `agent` callable")
    print(f"  next: `python fast.py play {out.relative_to(REPO)}` (Rule 46)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
