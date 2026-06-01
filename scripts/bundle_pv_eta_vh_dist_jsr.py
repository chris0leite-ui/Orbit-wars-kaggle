"""Bundler for `agents/baseline_pv_eta_vh_dist_jsr`.

Wrapper that sets slot reservation + joint sync + size balance on top
of the composite (pv_eta + distilled opp Tier 2 v2 + B.3 head λ=1.0).
The three feature mechanisms themselves live in:
  - `agents/baseline/chooser_trajectory.py` (slot reservation + joint_sync)
  - `agents/baseline/proposer.py` (size_balance)
All env-var gated, default OFF — bundle parity preserved.

Patches BOTH blobs into the inlined bundle:
  - `_OPP_BOOSTER_B64`  (in inlined `lib/opp_model.py`)
       <- `data/opp_distill/distill_booster.txt`
  - `_VH_MODEL_B64`     (in inlined `agents/baseline/_value_head.py`)
       <- `data/value_head/value_head_model.txt`
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
DEFAULT_WRAPPER = REPO / "agents" / "baseline_pv_eta_vh_dist_jsr"
DEFAULT_OUT = REPO / "submissions" / "baseline_pv_eta_vh_dist_jsr.py"
DEFAULT_OPP_BOOSTER = REPO / "data" / "opp_distill" / "distill_booster.txt"
DEFAULT_VH_MODEL = REPO / "data" / "value_head" / "value_head_model.txt"

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
_VH_MODEL_B64_PATCH_RE = re.compile(
    r'^_VH_MODEL_B64: str = ""\s*$',
    re.MULTILINE,
)


def _to_b64(path: Path) -> str:
    text = path.read_text()
    return base64.b64encode(gzip.compress(text.encode())).decode()


def _strip_wrapper_import(text: str) -> str:
    new = _WRAPPER_IMPORT_RE.sub("", text, count=1)
    if new == text:
        raise RuntimeError("wrapper bundler: missing inner-agent import")
    return new


def _bundle_inner_to_text(inner_dir: Path, lib_modules: list[str]) -> str:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        out = _bundle_agent(inner_dir, lib_modules, out_dir=tmp_path, force=True)
        return out.read_text()


def _patch_opp_booster(text: str, b64: str) -> str:
    new = _OPP_BOOSTER_B64_PATCH_RE.sub(
        f'_OPP_BOOSTER_B64: str = "{b64}"', text, count=1,
    )
    if new == text:
        raise RuntimeError("failed to patch _OPP_BOOSTER_B64 in inlined opp_model")
    return new


def _patch_vh_model(text: str, b64: str) -> str:
    new = _VH_MODEL_B64_PATCH_RE.sub(
        f'_VH_MODEL_B64: str = "{b64}"', text, count=1,
    )
    if new == text:
        raise RuntimeError("failed to patch _VH_MODEL_B64 in inlined _value_head")
    return new


def _smoke_import(bundle_path: Path) -> None:
    spec = importlib.util.spec_from_file_location(
        "_bundle_pv_eta_vh_dist_jsr_smoke", bundle_path,
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    fn = getattr(mod, "agent", None)
    if not callable(fn):
        raise RuntimeError(f"bundle {bundle_path} has no top-level callable `agent`")


def build(
    inner_dir: Path = DEFAULT_INNER,
    wrapper_dir: Path = DEFAULT_WRAPPER,
    opp_booster_path: Path = DEFAULT_OPP_BOOSTER,
    vh_model_path: Path = DEFAULT_VH_MODEL,
    lib_modules: list[str] | None = None,
    out_path: Path = DEFAULT_OUT,
) -> Path:
    lib_modules = lib_modules or DEFAULT_LIB_ORDER
    for required in ("shot_features", "_validator_tree_walker",
                     "value_head_features", "opp_features_lite"):
        if required not in lib_modules:
            lib_modules = list(lib_modules) + [required]

    for src, label in [(opp_booster_path, "distilled opp booster"),
                       (vh_model_path, "B.3 value-head model")]:
        if not src.is_file():
            raise FileNotFoundError(f"{label} missing: {src}")

    inner_text = _bundle_inner_to_text(inner_dir, lib_modules)
    opp_b64 = _to_b64(opp_booster_path)
    vh_b64 = _to_b64(vh_model_path)
    inner_patched = _patch_opp_booster(inner_text, opp_b64)
    inner_patched = _patch_vh_model(inner_patched, vh_b64)
    inner_stripped = _FUTURE_IMPORT_RE.sub("", inner_patched, count=1)

    wrapper_main = wrapper_dir / "main.py"
    wrapper_body = _strip_wrapper_import(wrapper_main.read_text())
    wrapper_body = _FUTURE_IMPORT_RE.sub("", wrapper_body, count=1)

    header = (
        f"# Bundled by scripts/bundle_pv_eta_vh_dist_jsr.py — composite + jsr:\n"
        f"#   inner: {inner_dir.relative_to(REPO)}\n"
        f"#   wrapper: {wrapper_dir.relative_to(REPO)}\n"
        f"#   opp booster: {opp_booster_path.relative_to(REPO)}\n"
        f"#   vh model:    {vh_model_path.relative_to(REPO)}\n"
        f"# slot res 3/2/2 + joint_sync (max 30 pairs, k=3) + size_balance\n"
        f"# Single-file Kaggle submission for Orbit Wars.\n\n"
    )

    parts = [
        header,
        "from __future__ import annotations\n\n",
        "# === wrapper preamble (env-var defaults) ===\n",
        wrapper_body,
        "\n# === inner bundle (with both blobs patched) ===\n",
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
    p.add_argument("--opp-booster", default=str(DEFAULT_OPP_BOOSTER))
    p.add_argument("--vh-model", default=str(DEFAULT_VH_MODEL))
    p.add_argument("--out", default=str(DEFAULT_OUT))
    args = p.parse_args(argv)

    out = build(
        inner_dir=Path(args.inner),
        wrapper_dir=Path(args.wrapper),
        opp_booster_path=Path(args.opp_booster),
        vh_model_path=Path(args.vh_model),
        out_path=Path(args.out),
    )
    print(f"wrote {out} ({out.stat().st_size:,} bytes)")
    print(f"  smoke import OK; top-level `agent` callable")
    print(f"  composite + slot res 3/2/2 + joint_sync (max=30,k=3) + size_balance")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
