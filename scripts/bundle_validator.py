"""Wrapper bundler for `agents/baseline_validated/main.py`.

The wrapper agent imports an inner agent (`from agents.baseline.main
import agent as _inner_agent`) and feature-encoder symbols from
`lib.shot_features`. The standard `scripts/bundle_agent.py` cannot
handle this because:

  - `from agents.baseline.main import ...` is stripped by the intra-
    package import regex, but the baseline body isn't inlined → the
    bundle has no `_inner_agent` symbol at runtime.
  - The strip emits an alias rebind (`_inner_agent = agent`) which
    would NameError because `agent` doesn't exist as a top-level name
    in the wrapper alone.

This bundler:

  1. Calls `bundle_agent.bundle()` on `agents/baseline` to get a
     fully-inlined baseline (lib modules + submodules + `def agent(...)`).
  2. Renames `def agent(` → `def _inner_agent(` in the baseline bundle.
  3. Reads `agents/baseline_validated/main.py` and strips three
     specific imports — they're already in scope from the inner bundle:
       - `from __future__ import annotations`
       - `from agents.baseline.main import agent as _inner_agent`
       - `from lib.shot_features import FEATURE_DIM, encode_shot_features, target_owned_by`
  4. Concatenates `[inner_renamed] + [wrapper_body_stripped]` and writes
     to `submissions/baseline_validated.py`.
  5. Verifies the bundle imports cleanly and exposes a top-level
     callable `agent` (Rule 46 smoke; the caller is expected to also
     run `python fast.py play <bundle>` separately).
"""

from __future__ import annotations

import argparse
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
DEFAULT_WRAPPER = REPO / "agents" / "baseline_validated"
DEFAULT_OUT = REPO / "submissions" / "baseline_validated.py"

# Top-level `def agent(`. Anchored at column 0 so we don't accidentally
# rename a nested helper named `agent`.
_DEF_AGENT_TOPLEVEL_RE = re.compile(r"^def agent\(", re.MULTILINE)

# Three lines to strip from the wrapper body wholesale.
_STRIP_PATTERNS = (
    re.compile(r"^\s*from __future__\s+import\b.*$", re.MULTILINE),
    re.compile(
        r"^\s*from agents\.baseline\.main\s+import\s+agent as _inner_agent\s*$",
        re.MULTILINE,
    ),
    re.compile(
        r"^\s*from lib\.shot_features\s+import\s+FEATURE_DIM,\s*encode_shot_features,\s*target_owned_by\s*$",
        re.MULTILINE,
    ),
    re.compile(
        r"^\s*from lib\._validator_tree_walker\s+import\s+parse_booster_text,\s*predict_proba\s*$",
        re.MULTILINE,
    ),
)


def _bundle_inner_to_text(inner_dir: Path, lib_modules: list[str]) -> str:
    """Bundle the inner agent to a temp path, read the text, and remove
    the temp file. Returns the bundled source string."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        out = _bundle_agent(inner_dir, lib_modules, out_dir=tmp_path, force=True)
        text = out.read_text()
        # bundle() writes inside tmp; TemporaryDirectory cleanup removes it.
    return text


def _rename_agent_to_inner(text: str) -> str:
    """Rename the top-level `def agent(` → `def _inner_agent(`.

    Asserts exactly one top-level match; the inner bundle is expected
    to contain a single canonical entry point.
    """
    matches = _DEF_AGENT_TOPLEVEL_RE.findall(text)
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one top-level `def agent(` in inner bundle, "
            f"found {len(matches)}. Bundler can't safely rename."
        )
    return _DEF_AGENT_TOPLEVEL_RE.sub("def _inner_agent(", text, count=1)


def _strip_wrapper_imports(text: str) -> str:
    """Remove the three import lines the inner bundle already provides."""
    out = text
    for pat in _STRIP_PATTERNS:
        new = pat.sub("", out, count=1)
        if new == out:
            raise RuntimeError(
                f"wrapper bundler: expected to strip a line matching "
                f"{pat.pattern!r} but found none. Has the wrapper been "
                f"refactored? Update the strip patterns to match."
            )
        out = new
    return out


def _smoke_import(bundle_path: Path) -> None:
    """Import the bundle as a module; assert top-level `agent` is callable.
    Mirrors the equivalent check in `scripts/bundle_agent.py:main()`."""
    spec = importlib.util.spec_from_file_location(
        "_bundle_validator_smoke", bundle_path
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    fn = getattr(mod, "agent", None)
    if not callable(fn):
        raise RuntimeError(
            f"bundle {bundle_path} has no top-level callable `agent`."
        )
    inner = getattr(mod, "_inner_agent", None)
    if not callable(inner):
        raise RuntimeError(
            f"bundle {bundle_path} has no `_inner_agent` (rename failed)."
        )


def build(
    inner_dir: Path = DEFAULT_INNER,
    wrapper_dir: Path = DEFAULT_WRAPPER,
    lib_modules: list[str] | None = None,
    out_path: Path = DEFAULT_OUT,
) -> Path:
    """End-to-end build. Returns the path of the written bundle."""
    lib_modules = lib_modules or DEFAULT_LIB_ORDER

    inner_text = _bundle_inner_to_text(inner_dir, lib_modules)
    inner_renamed = _rename_agent_to_inner(inner_text)

    wrapper_main = wrapper_dir / "main.py"
    wrapper_text = wrapper_main.read_text()
    wrapper_body = _strip_wrapper_imports(wrapper_text)

    header = (
        f"# Bundled by scripts/bundle_validator.py — "
        f"{inner_dir.relative_to(REPO)} (inner) + "
        f"{wrapper_dir.relative_to(REPO)} (wrapper).\n"
        f"# Single-file Kaggle submission for Orbit Wars.\n\n"
    )

    parts = [
        header,
        inner_renamed,
        "\n# === wrapper ===\n",
        wrapper_body,
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(parts))

    _smoke_import(out_path)

    return out_path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--inner", default=str(DEFAULT_INNER),
                   help=f"inner agent dir (default: {DEFAULT_INNER})")
    p.add_argument("--wrapper", default=str(DEFAULT_WRAPPER),
                   help=f"wrapper agent dir (default: {DEFAULT_WRAPPER})")
    p.add_argument("--out", default=str(DEFAULT_OUT),
                   help=f"output bundle path (default: {DEFAULT_OUT})")
    args = p.parse_args(argv)

    out = build(
        inner_dir=Path(args.inner),
        wrapper_dir=Path(args.wrapper),
        out_path=Path(args.out),
    )
    print(f"wrote {out} ({out.stat().st_size:,} bytes)")
    print(f"  smoke import OK; top-level `agent` callable; `_inner_agent` callable")
    print(f"  next: `python fast.py play {out.relative_to(REPO)}` to verify "
          f"crash-free game (Rule 46)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
