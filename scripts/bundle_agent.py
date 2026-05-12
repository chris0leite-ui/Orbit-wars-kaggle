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
# Order matters: each module must come AFTER its lib-internal dependencies.
# fleet/orbit use geometry; aim uses fleet/orbit; trajectory uses aim/fleet/
# geometry/orbit; mechanism uses fleet/orbit/aim/intent/trajectory.
# Block E mission framework (2026-05-11):
#   mission (dataclass) needs intent;
#   missions/snipe + missions/reinforce need fleet/intent/mission/world_model;
#   planner needs intent/mission/world_model.
# Subpackage paths like "missions/snipe" resolve to lib/missions/snipe.py
# via pathlib's `/` operator transparently.
DEFAULT_LIB_ORDER = [
    "geometry",
    "fleet",
    "orbit",
    "aim",
    "combat",
    "world_model",
    "intent",
    "trajectory",
    "mechanism",
    "mission",
    "missions/snipe",
    "missions/reinforce",
    "planner",
    # v7 lookahead substrate (2026-05-12). Order matters:
    # fast_sim is foundational; opp_model uses missions/* + planner +
    # intent + mechanism + world_model; v7_search uses everything.
    # Inlining these is a no-op for non-v7 agents (their agent() never
    # imports from them) — they bloat the bundle by ~35 KB. Acceptable.
    "fast_sim",
    "opp_model",
    "v7_search",
]
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
    """Drop intra-package imports and `from __future__` lines from a lib module,
    but emit alias rebindings for any aliased intra-imports.

    Without the alias rebind, `from lib.fleet import speed as fleet_speed` in
    a lib file would silently leave `fleet_speed` undefined in the bundle —
    NameError at runtime, swallowed by kaggle_environments' try/except. The
    parity gate catches it but only on integration; cheaper to rebind here.
    """
    out: list[str] = []
    for line in src.splitlines(keepends=True):
        if _FUTURE_IMPORT_RE.match(line):
            continue
        if _INTRA_IMPORT_RE.match(line):
            for asname, original in _extract_aliases(line):
                out.append(f"{asname} = {original}\n")
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
    """Produce `<out_dir>/<name>.py` and return its path.

    `agent_dir` may be either:
      - a directory containing `main.py` (canonical multi-file agent shape;
        `<name>` is the directory name); or
      - a path to a single `.py` file (flat agent shape used by
        `agents/simple/<n>.py`; `<name>` is the file stem).
    """
    if agent_dir.is_file() and agent_dir.suffix == ".py":
        main = agent_dir
        name = agent_dir.stem
        source_label = agent_dir.relative_to(REPO)
    elif agent_dir.is_dir():
        main = agent_dir / "main.py"
        if not main.is_file():
            raise FileNotFoundError(f"agent has no main.py: {agent_dir}")
        name = agent_dir.name
        source_label = agent_dir.relative_to(REPO)
    else:
        raise FileNotFoundError(f"agent path not found: {agent_dir}")

    parts: list[str] = []
    parts.append(
        f"# Bundled by scripts/bundle_agent.py from {source_label} + "
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
    out_path = out_dir / f"{name}.py"
    out_path.write_text("".join(parts))
    return out_path


def _bundle_hash(path: Path) -> str:
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _parity_gate(bundle_path: Path, agent_dir: Path, seeds=(0,)) -> bool:
    """Compare source-agent vs bundle-agent on self-play obs streams.

    For each seed, generate a self-play game using the SOURCE agent, then
    feed each turn's observation to BOTH the source and the bundle and
    assert they emit identical actions. The bundle is meant to be a
    syntactic restamp of the source — if they diverge, the bundler has
    a bug (silent intra-import strip, name collision, ordering).

    Returns True if all seeds match perfectly.
    """
    import importlib.util
    import sys
    repo = Path(__file__).resolve().parents[1]
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from kaggle_environments import make

    # Load source agent
    if agent_dir.is_file():
        agent_module_name = agent_dir.stem
        source_agent = _load_module_from_file(agent_dir, agent_module_name).agent
    else:
        source_agent = _load_module_from_file(agent_dir / "main.py", agent_dir.name + "_source").agent

    # Load bundle (registered in sys.modules first so dataclasses resolve)
    bundle_module = _load_module_from_file(bundle_path, "_bundle_" + bundle_path.stem)
    bundle_agent = bundle_module.agent

    mismatches = 0
    compared = 0
    for seed in seeds:
        env = make("orbit_wars", configuration={"seed": seed})
        env.run([source_agent, source_agent])
        steps = env.toJSON()["steps"]
        for t in range(len(steps) - 1):
            for seat in (0, 1):
                if steps[t][seat]["status"] != "ACTIVE":
                    continue
                obs = steps[t][seat]["observation"]
                if obs.get("step") is None:
                    obs = dict(obs)
                    obs["step"] = t
                src_out = source_agent(obs)
                bnd_out = bundle_agent(obs)
                compared += 1
                if src_out != bnd_out:
                    mismatches += 1
                    if mismatches <= 3:
                        print(f"  MISMATCH seed={seed} t={t} seat={seat}: src={src_out} bundle={bnd_out}",
                              file=sys.stderr)
    if mismatches:
        print(f"  PARITY FAIL: {mismatches}/{compared} mismatched turns", file=sys.stderr)
        return False
    print(f"  parity OK: {compared} turns matched across {len(seeds)} self-play seed(s)")
    return True


def _load_module_from_file(path: Path, name: str):
    import importlib.util
    import sys
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


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
    parser.add_argument(
        "--skip-parity-gate", action="store_true",
        help="skip the post-bundle self-play parity check (NOT recommended)",
    )
    args = parser.parse_args(argv)
    out = bundle(args.agent_dir.resolve(), args.lib, out_dir=args.out_dir.resolve())
    h = _bundle_hash(out)
    print(f"wrote {out} ({out.stat().st_size} bytes) sha256:{h}")
    if not args.skip_parity_gate:
        ok = _parity_gate(out, args.agent_dir.resolve())
        if not ok:
            print(f"REFUSING TO LEAVE BUNDLE: removing {out}", file=sys.stderr)
            out.unlink()
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
