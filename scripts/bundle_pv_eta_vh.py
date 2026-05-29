"""Bundler for `agents/baseline_pv_eta_vh` — Reframe B.2 value-head wrapper.

Mirrors `scripts/bundle_pv_eta_ml.py`. The wrapper is env-var-only —
no Python wrapping code. Bundle structure:

    1. Wrapper env-var preamble (from agents/baseline_pv_eta_vh/main.py).
    2. Full `agents/baseline` bundle (lib + agent submodules + main with
       `def agent(...)` at the bottom).
    3. `_VH_MODEL_B64` patched into the inlined `_value_head.py` body so
       the regressor is available at submit time without lightgbm.

At `BASELINE_VH_LAMBDA=0`, this submission is byte-equivalent to bare
pv_eta. λ is overridden by the calling shell or by editing the wrapper's
setdefault before bundling.
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
DEFAULT_WRAPPER = REPO / "agents" / "baseline_pv_eta_vh"
DEFAULT_OUT = REPO / "submissions" / "baseline_pv_eta_vh.py"
DEFAULT_MODEL = REPO / "data" / "value_head" / "value_head_model.txt"

# Wrapper preamble = everything BEFORE the `from agents.baseline.main
# import agent` line. The import line itself is stripped (the inner
# bundle provides `agent` at top level).
_WRAPPER_IMPORT_RE = re.compile(
    r"^\s*from\s+agents\.baseline\.main\s+import\s+agent\b.*$",
    re.MULTILINE,
)

# `from __future__` imports must be the first executable statement.
_FUTURE_IMPORT_RE = re.compile(
    r"^\s*from __future__\s+import\b.*$",
    re.MULTILINE,
)

# Patch target inside the inlined _value_head body. Must match the
# literal in agents/baseline/_value_head.py exactly.
_VH_MODEL_B64_PATCH_RE = re.compile(
    r'^_VH_MODEL_B64: str = ""\s*$',
    re.MULTILINE,
)


def _model_to_b64(model_path: Path) -> str:
    text = model_path.read_text()
    return base64.b64encode(gzip.compress(text.encode())).decode()


def _strip_wrapper_import(text: str) -> str:
    new = _WRAPPER_IMPORT_RE.sub("", text, count=1)
    if new == text:
        raise RuntimeError(
            "wrapper bundler: expected to strip "
            "`from agents.baseline.main import agent` but found none. "
            "Has the wrapper been refactored?"
        )
    return new


def _bundle_inner_to_text(inner_dir: Path, lib_modules: list[str]) -> str:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        out = _bundle_agent(inner_dir, lib_modules, out_dir=tmp_path, force=True)
        return out.read_text()


def _patch_vh_model_b64(text: str, b64: str) -> str:
    new = _VH_MODEL_B64_PATCH_RE.sub(
        f'_VH_MODEL_B64: str = "{b64}"', text, count=1,
    )
    if new == text:
        raise RuntimeError(
            "bundler: failed to patch `_VH_MODEL_B64: str = \"\"` in the "
            "bundled _value_head body. Has the literal changed?"
        )
    return new


def _smoke_import(bundle_path: Path) -> None:
    spec = importlib.util.spec_from_file_location(
        "_bundle_pv_eta_vh_smoke", bundle_path,
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
    model_path: Path = DEFAULT_MODEL,
    lib_modules: list[str] | None = None,
    out_path: Path = DEFAULT_OUT,
) -> Path:
    # Ensure `value_head_features` lives in the bundled lib set.
    lib_modules = lib_modules or DEFAULT_LIB_ORDER
    if "value_head_features" not in lib_modules:
        lib_modules = list(lib_modules) + ["value_head_features"]

    inner_text = _bundle_inner_to_text(inner_dir, lib_modules)
    b64 = _model_to_b64(model_path)
    inner_patched = _patch_vh_model_b64(inner_text, b64)
    inner_stripped = _FUTURE_IMPORT_RE.sub("", inner_patched, count=1)

    wrapper_main = wrapper_dir / "main.py"
    wrapper_body = _strip_wrapper_import(wrapper_main.read_text())
    wrapper_body = _FUTURE_IMPORT_RE.sub("", wrapper_body, count=1)

    header = (
        f"# Bundled by scripts/bundle_pv_eta_vh.py — "
        f"{inner_dir.relative_to(REPO)} (inner) + "
        f"{wrapper_dir.relative_to(REPO)} (wrapper) + "
        f"{model_path.relative_to(REPO)} (value-head B64).\n"
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
    p.add_argument("--model", default=str(DEFAULT_MODEL))
    p.add_argument("--out", default=str(DEFAULT_OUT))
    args = p.parse_args(argv)

    out = build(
        inner_dir=Path(args.inner),
        wrapper_dir=Path(args.wrapper),
        model_path=Path(args.model),
        out_path=Path(args.out),
    )
    print(f"wrote {out} ({out.stat().st_size:,} bytes)")
    print(f"  smoke import OK; top-level `agent` callable")
    print(f"  next: `python fast.py play {out.relative_to(REPO)}` (Rule 46)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
