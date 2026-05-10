"""Bundle an agent + its `lib/` dependencies into a single submission file.

Kaggle accepts a single `main.py` or a tar.gz of flat files at the root.
This script takes the simpler path: concatenate the relevant `lib/*.py`
modules ahead of the agent's `main.py`, with all intra-package imports
(`from .other import X`, `from lib.other import X`) stripped because the
referenced symbols are already in scope after concatenation.

CLI:
    python scripts/bundle_agent.py <agent_dir> [--lib mod1 mod2 ...]
        e.g. python scripts/bundle_agent.py agents/v1_orbitfix --lib geometry fleet orbit

Output: submissions/<basename(agent_dir)>.py
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_LIB_ORDER = ["geometry", "fleet", "orbit"]
SUBMISSIONS = REPO / "submissions"


_INTRA_IMPORT_RE = re.compile(r"^\s*from (lib|\.)[\w.]*\s+import\b.*$")
_FUTURE_IMPORT_RE = re.compile(r"^\s*from __future__\s+import\b.*$")
_DOCSTRING_OPENER_RE = re.compile(r'^\s*("""|\'\'\')')


def _strip_module_docstring(src: str) -> str:
    """Drop the leading triple-quoted module docstring if present.

    Bundled lib modules don't need their own docstrings in the submission;
    keeping them just inflates the artifact.
    """
    lines = src.splitlines(keepends=True)
    i = 0
    # Skip blank + comment lines until we find code.
    while i < len(lines) and (lines[i].strip() == "" or lines[i].lstrip().startswith("#")):
        i += 1
    if i >= len(lines):
        return src
    m = _DOCSTRING_OPENER_RE.match(lines[i])
    if not m:
        return src
    quote = m.group(1)
    # Single-line docstring?
    if lines[i].count(quote) >= 2 and len(lines[i].strip()) > len(quote):
        return "".join(lines[:i] + lines[i + 1 :])
    # Multi-line: scan forward for closing quote.
    j = i + 1
    while j < len(lines) and quote not in lines[j]:
        j += 1
    return "".join(lines[:i] + lines[j + 1 :])


def _clean_lib_source(src: str) -> str:
    """Drop intra-package imports and `from __future__` lines from a lib module."""
    out: list[str] = []
    for line in src.splitlines(keepends=True):
        if _INTRA_IMPORT_RE.match(line) or _FUTURE_IMPORT_RE.match(line):
            continue
        out.append(line)
    return _strip_module_docstring("".join(out))


def _extract_aliases(line: str) -> list[tuple[str, str]]:
    """For a `from X import a as b, c as d` line, return [(b, a), (d, c)].

    If the line doesn't parse or has no aliases, returns []. Used to emit
    rebinding statements after the import is commented out, since the
    inlined module exports the original names — e.g. `from lib.fleet
    import speed as fleet_speed` becomes `fleet_speed = speed` in the bundle.
    """
    try:
        tree = ast.parse(line.strip())
    except SyntaxError:
        return []
    if not tree.body:
        return []
    node = tree.body[0]
    if not isinstance(node, ast.ImportFrom):
        return []
    return [(n.asname, n.name) for n in node.names if n.asname is not None]


def _clean_agent_source(src: str) -> str:
    """Strip `from __future__` (already emitted at the bundle top) and rewrite
    intra-lib imports — comment out the import and emit alias rebindings so
    `from lib.fleet import speed as fleet_speed` keeps working.
    """
    out: list[str] = []
    for line in src.splitlines(keepends=True):
        if _FUTURE_IMPORT_RE.match(line):
            continue
        if _INTRA_IMPORT_RE.match(line):
            stripped = line.rstrip("\n")
            out.append(f"# {stripped}  # inlined by bundle_agent.py\n")
            for asname, original in _extract_aliases(line):
                out.append(f"{asname} = {original}\n")
        else:
            out.append(line)
    return "".join(out)


def bundle(agent_dir: Path, lib_modules: list[str], out_dir: Path = SUBMISSIONS) -> Path:
    """Produce `<out_dir>/<basename(agent_dir)>.py` and return its path."""
    main = agent_dir / "main.py"
    if not main.is_file():
        raise FileNotFoundError(f"agent has no main.py: {agent_dir}")

    parts: list[str] = []
    parts.append(
        f"# Bundled by scripts/bundle_agent.py from {agent_dir.relative_to(REPO)} + "
        f"lib/{{{','.join(lib_modules)}}}.\n"
        f"# Single-file Kaggle submission for Orbit Wars.\n\n"
    )
    parts.append("from __future__ import annotations\n")

    for mod in lib_modules:
        path = REPO / "lib" / f"{mod}.py"
        if not path.is_file():
            raise FileNotFoundError(f"lib module missing: {path}")
        parts.append(f"\n# === inlined: lib/{mod}.py ===\n")
        parts.append(_clean_lib_source(path.read_text()))

    parts.append("\n# === agent ===\n")
    parts.append(_clean_agent_source(main.read_text()))

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{agent_dir.name}.py"
    out_path.write_text("".join(parts))
    return out_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("agent_dir", type=Path, help="path to agents/<name>/")
    parser.add_argument(
        "--lib",
        nargs="*",
        default=DEFAULT_LIB_ORDER,
        help=f"lib modules to inline, in order (default: {' '.join(DEFAULT_LIB_ORDER)})",
    )
    parser.add_argument("--out-dir", type=Path, default=SUBMISSIONS)
    args = parser.parse_args(argv)
    out = bundle(args.agent_dir.resolve(), args.lib, out_dir=args.out_dir.resolve())
    print(f"wrote {out} ({out.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
