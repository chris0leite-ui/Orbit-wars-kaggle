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

Parity gate: post-bundle self-play comparison between the source agent
and the bundled file across seeds=(0,). Results are cached by full
sha256 in ``audit/bundle-parity-cache.json``; a cache hit skips the
~30-60s gate on repeat builds of an unchanged bundle. Use
``--ignore-parity-cache`` to force re-verification, or
``--skip-parity-gate`` to bypass both gate and cache (not recommended).
"""

from __future__ import annotations

import argparse
import ast
import os
import re
import subprocess
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
    # kinematic_table imports geometry + orbit; trajectory imports
    # kinematic_table (opt-in via KINEMATIC_TABLE_ENABLED). Must precede
    # trajectory so the inlined module is in scope when trajectory's
    # `from lib.kinematic_table import ...` line is stripped.
    "kinematic_table",
    "aim",
    "combat",
    "world_model",
    "intent",
    "trajectory",
    "mechanism",
    "mission",
    # `scoring` exposes `pv_horizon` + `PV_GAMMA` used by missions/snipe
    # and missions/reinforce since H16 (2026-05-13). Must precede the
    # mission modules so the inlined symbols are visible at parse time.
    "scoring",
    # Layer R reliability multiplier (2026-05-25). Imported by
    # opening_planner + proposer; module-level constants read env vars
    # at import time. Default OFF preserves behavior.
    "reliability",
    # Layer D plan-level drop-one validator (2026-05-25). Imported by
    # agents/baseline/main.py at the post-chooser emit site. Closed-form
    # plan_production_advantage uses predict_fleet_fate. Default OFF.
    "drop_one",
    "missions/snipe",
    "missions/reinforce",
    "missions/recapture",
    # Mission proposers wired into v7_search (lib/v7_search.py imports
    # all three). Opening: wired live by main's cb02fd9 (H11). Drain
    # and gang_up: wired this session (Mission Renaissance) behind
    # USE_*_MISSION flags (drain/gang_up default 0 — both falsified;
    # opening default 1 to preserve main's intent). Bundles built
    # without flag overrides keep main's v7_1 behaviour.
    "missions/opening",
    "missions/drain",
    "missions/gang_up",
    # v7.3 (2026-05-14): hand-crafted opp archetypes for min-regret /
    # maximin depth-2 search. Replaces the v3.5.1-mirror opp model.
    "missions/opp_archetypes",
    "planner",
    # lib/lookahead.py — env-clone-based forward sim used by
    # v7_minimax (env_from_obs, score_joint_action_symmetric). Distinct
    # from lookahead_planner.py. Latent bug surfaced by the 2026-05-14
    # loud-error guard: the existing tracked v7_minimax bundle has these
    # symbols undefined; agent's wallclock fallback masked the NameError
    # on the live ladder. Future bundles get them inlined correctly.
    "lookahead",
    "lookahead_planner",
    # Pure-Python rebuild of the orbit_wars game engine (Phase 2 of the
    # consolidate-fast-simulation work; 2026-05-12). fast_sim imports
    # `interpreter` from here, so it must precede fast_sim in the bundle.
    # Byte-exact parity is enforced by tests/test_game_parity.py.
    "game/interpreter",
    # v7 lookahead substrate (2026-05-12). Order matters:
    # fast_sim is foundational; opp_model uses missions/* + planner +
    # intent + mechanism + world_model; v7_search uses everything.
    # Inlining these is a no-op for non-v7 agents (their agent() never
    # imports from them) — they bloat the bundle by ~35 KB. Acceptable.
    "fast_sim",
    "opp_model",
    "v7_search",
    # v4_planner brain (2026-05-12 evening): candidate portfolios +
    # adaptive K + production-share value head. Inlined for v8_*
    # agents that consume them; harmless for older agents that don't
    # import (just adds ~10 KB to the bundle).
    "candidate_portfolios",
    # v9 super-version (2026-05-12 evening): composite value heads
    # for receding-horizon-pathology fix. Used by v9_inflight + v9_combined.
    "value_heads",
    # 2026-05-21: opening planner cherry-picked from analytical track.
    # Multi-turn MILP for steps 0..OPENING_HORIZON-1. Used by AGGR when
    # BASELINE_OPENING_MILP=1. Imports from agents.baseline.proposer +
    # lib.fleet/trajectory/world_model — all already in DEFAULT_LIB_ORDER
    # above.
    "joint_solver/opening_planner",
    # 2026-05-23: predicate from analytical track. `is_winning_state_if_owned`
    # is the closed-form 2P winning-state gate used by buildup_planner's
    # STRIKE predicate, FINISHER, and DOGPILE. Pure closed-form — depends
    # only on world + production accounting. Must be in the bundle so the
    # `from lib.joint_solver.predicate import is_winning_state_if_owned`
    # lines in agents/buildup_planner/{predicates,endgame}.py can be stripped
    # and the symbol stays in scope.
    "joint_solver/predicate",
]
SUBMISSIONS = REPO / "submissions"


# Strip these intra-package import patterns from the bundle (the referenced
# symbols are inlined into the concatenated file, so the import lines would
# fail at runtime in Kaggle's flat-filesystem sandbox):
#   - `from lib.X import Y`         (lib modules, the long-established case)
#   - `from .X import Y`            (relative imports within an agent package)
#   - `from agents.<name>.X import` (modular agent pattern; the agents/baseline
#     split + 2026-05-17 friction `bundler-ships-with-wrong-default-env-var`
#     plus a sibling-class bug — bundle ran locally because agents.<name> was
#     importable from cwd, then ERRORED on the Kaggle ladder because Kaggle's
#     sandbox doesn't have the `agents` package on its filesystem).
_INTRA_IMPORT_RE = re.compile(
    r"^\s*from (lib|\.|agents\.[\w]+)[\w.]*\s+import\b.*$"
)
# Captures the lib-relative module path so we can verify it's in the bundle
# order list. Friction: `bundler-missing-block-e-modules`,
# `new-lib-module-silently-broken-bundle`, `bundle-default-lib-order-stale-...`
# — three classes of the same silent failure mode.
_LIB_IMPORT_MODULE_RE = re.compile(r"^\s*from\s+lib\.([\w.]+)\s+import\b")
_FUTURE_IMPORT_RE = re.compile(r"^\s*from __future__\s+import\b.*$")
_DOCSTRING_OPENER_RE = re.compile(r'^\s*("""|\'\'\')')


def _extract_lib_module(line: str) -> str | None:
    """Return the `lib.X.Y` module name from a `from lib.X.Y import ...` line.

    Returns the bundle-order form (e.g. `missions/snipe` for `lib.missions.snipe`),
    or None if the line doesn't match the lib-import pattern.
    """
    m = _LIB_IMPORT_MODULE_RE.match(line)
    if not m:
        return None
    return m.group(1).replace(".", "/")


def _assert_lib_imports_resolved(src: str, lib_modules: list[str], source_label: str) -> None:
    """Loud-error if a `from lib.X import Y` line strips to a module not in lib_modules.

    The bundler's `_INTRA_IMPORT_RE` strips intra-package imports and assumes
    the referenced symbols are already in scope from a preceding inlined
    module. When `lib/X.py` isn't in `lib_modules`, the symbols are never
    defined — the bundle runs but blows up with NameError on the first call
    that hits them. Tests using a non-exercising agent (e.g. v1_orbitfix
    against the snipe-stack mission framework) don't catch this.

    Friction recurrences this comp: 2026-05-11 (block E), 2026-05-12 (v7.5),
    2026-05-13 (H16 PV target valuation — silent draws in 8-seed A/B).
    """
    lib_set = set(lib_modules)
    missing: list[tuple[int, str]] = []
    for lineno, line in enumerate(src.splitlines(), start=1):
        mod = _extract_lib_module(line)
        if mod is None:
            continue
        if mod not in lib_set:
            missing.append((lineno, mod))
    if missing:
        lines = "\n".join(
            f"  {source_label}:{lineno}: from lib.{mod.replace('/', '.')} import ..."
            f"  (lib/{mod}.py not in --lib order)"
            for lineno, mod in missing
        )
        raise RuntimeError(
            "bundler: lib import(s) without a corresponding module in --lib order:\n"
            f"{lines}\n"
            "Add the module(s) to DEFAULT_LIB_ORDER in scripts/bundle_agent.py "
            "(or pass --lib explicitly). Stripping these imports without an "
            "inlined source would NameError at runtime."
        )


def _is_tracked(path: Path) -> bool:
    """True iff `path` is tracked by git in the repo. False on git failure."""
    try:
        result = subprocess.run(
            ["git", "ls-files", "--error-unmatch", str(path)],
            cwd=REPO,
            capture_output=True,
            check=False,
        )
        return result.returncode == 0
    except (FileNotFoundError, OSError):
        return False


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


def _consume_multiline_import(
    lines: list[str], start_idx: int,
) -> tuple[int, str]:
    """If the import line at `start_idx` opens with `(`, consume continuation
    lines until the matching `)`. Returns `(end_idx_exclusive, joined_block)`.

    Single-line imports return `(start_idx + 1, lines[start_idx])`. Closes
    the bundler's known multi-line-imports silent-fail mode (Rule 46):
    `from X import (\n  a,\n  b,\n)` was previously stripping ONLY the
    first line, leaving the continuation lines as bare-indented expressions
    that IndentationError on bundle import.
    """
    first = lines[start_idx]
    # No open paren → single-line, common case.
    if "(" not in first:
        return start_idx + 1, first
    end = start_idx
    while end < len(lines):
        if ")" in lines[end]:
            break
        end += 1
    joined = "".join(lines[start_idx:end + 1])
    return end + 1, joined


def _clean_lib_source(src: str) -> str:
    """Drop intra-package imports and `from __future__` lines from a lib module,
    but emit alias rebindings for any aliased intra-imports.

    Without the alias rebind, `from lib.fleet import speed as fleet_speed` in
    a lib file would silently leave `fleet_speed` undefined in the bundle —
    NameError at runtime, swallowed by kaggle_environments' try/except. The
    parity gate catches it but only on integration; cheaper to rebind here.
    """
    lines = src.splitlines(keepends=True)
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if _FUTURE_IMPORT_RE.match(line):
            i += 1
            continue
        if _INTRA_IMPORT_RE.match(line):
            next_i, block = _consume_multiline_import(lines, i)
            for asname, original in _extract_aliases(block):
                out.append(f"{asname} = {original}\n")
            i = next_i
            continue
        out.append(line)
        i += 1
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
    `from lib.fleet import speed as fleet_speed` keeps working. Handles
    multi-line `from X import (a, b, c)` blocks via `_consume_multiline_import`.
    """
    lines = src.splitlines(keepends=True)
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if _FUTURE_IMPORT_RE.match(line):
            i += 1
            continue
        if _INTRA_IMPORT_RE.match(line):
            next_i, block = _consume_multiline_import(lines, i)
            indent = line[: len(line) - len(line.lstrip())]
            # Comment-only the FIRST line (the import directive); drop
            # the continuation lines entirely. Then emit alias rebinds.
            out.append(f"{indent}# {line.strip()}  # inlined by bundle_agent.py\n")
            for asname, original in _extract_aliases(block):
                out.append(f"{indent}{asname} = {original}\n")
            i = next_i
            continue
        out.append(line)
        i += 1
    return "".join(out)


def bundle(
    agent_dir: Path,
    lib_modules: list[str],
    out_dir: Path = SUBMISSIONS,
    force: bool = False,
) -> Path:
    """Produce `<out_dir>/<name>.py` and return its path.

    `agent_dir` may be either:
      - a directory containing `main.py` (canonical multi-file agent shape;
        `<name>` is the directory name); or
      - a path to a single `.py` file (flat agent shape used by
        `agents/simple/<n>.py`; `<name>` is the file stem).

    Raises:
      RuntimeError if any `from lib.X import ...` strips to an X not in
        `lib_modules` (would NameError silently at runtime).
      FileExistsError if `<out_dir>/<name>.py` is tracked by git and
        `force` is False (would silently clobber a known-good submission).
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

    # Pre-flight: verify every stripped lib import has an inlined target.
    # Check the agent first (most common failure mode), then each lib module
    # in turn (a new lib module may itself import another new lib module).
    agent_src = main.read_text()
    _assert_lib_imports_resolved(agent_src, lib_modules, str(source_label))
    lib_srcs: list[tuple[str, str]] = []
    for mod in lib_modules:
        path = REPO / "lib" / f"{mod}.py"
        if not path.is_file():
            raise FileNotFoundError(f"lib module missing: {path}")
        src = path.read_text()
        _assert_lib_imports_resolved(src, lib_modules, f"lib/{mod}.py")
        lib_srcs.append((mod, src))

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{name}.py"

    # Overwrite protection: a tracked submission file is a known-good
    # bundle for a live Kaggle entry. Silent overwrite has clobbered live
    # references twice (`bundle-output-clobbers-prior-bundles` 2026-05-10,
    # `bundler-overwrites-tracked-submission` 2026-05-13).
    if out_path.exists() and _is_tracked(out_path) and not force:
        raise FileExistsError(
            f"refusing to overwrite tracked file {out_path.relative_to(REPO)} "
            f"(it backs a live submission). Pass --force to override, or write "
            f"the new bundle to a different name."
        )

    parts: list[str] = []
    parts.append(
        f"# Bundled by scripts/bundle_agent.py from {source_label} + "
        f"lib/{{{','.join(lib_modules)}}}.\n"
        f"# Single-file Kaggle submission for Orbit Wars.\n\n"
    )
    parts.append("from __future__ import annotations\n")

    for mod, src in lib_srcs:
        parts.append(f"\n# === inlined: lib/{mod}.py ===\n")
        parts.append(_clean_lib_source(src))

    # Inline agent submodules (e.g. agents/baseline/{value,proposer,chooser}.py)
    # before main.py. Modular agent pattern (2026-05-17): main.py imports symbols
    # from sibling files; without inlining the bundle would NameError after the
    # intra-package import lines are stripped. Topological order by
    # `from agents.<name>.X import` references.
    agent_submodules: list[tuple[str, str]] = []
    if agent_dir.is_dir():
        agent_submodules = _topo_sort_agent_submodules(agent_dir)

    # Inline cross-agent dependencies (e.g. buildup_planner pulls in
    # agents/baseline/* and agents/precision/* via direct attribute
    # imports). Must precede the primary agent's submodules so the
    # primary's stripped `from agents.X.Y import Z` lines leave `Z` in
    # scope. Discovery walks both main.py and every primary submodule.
    cross_packages = _discover_cross_agent_packages(
        agent_src, name, agent_submodules,
    )
    for dep_pkg in cross_packages:
        dep_dir = REPO / "agents" / dep_pkg
        dep_submodules = _topo_sort_agent_submodules(
            dep_dir, include_main=True,
        )
        for sub_name, sub_src in dep_submodules:
            parts.append(
                f"\n# === inlined: agents/{dep_pkg}/{sub_name}.py ===\n"
            )
            parts.append(_clean_agent_source(sub_src))
            # Namespace alias: `from agents.<pkg> import <sub>` is a
            # module-as-namespace import (callers use `<sub>.foo()`),
            # but the bundler inlines `<sub>`'s symbols at top level.
            # Emit a `<sub> = SimpleNamespace(foo=foo, …)` block so
            # both `<sub>.foo` and bare `foo` resolve. Public names
            # are top-level defs / classes / non-_underscore assigns.
            parts.append(
                _namespace_alias_block(dep_pkg, sub_name, sub_src)
            )

    for sub_name, sub_src in agent_submodules:
        parts.append(f"\n# === inlined: agents/{name}/{sub_name}.py ===\n")
        parts.append(_clean_agent_source(sub_src))
        # Primary-agent submodules also need a namespace alias for
        # callers that do `from agents.<name> import <sub>` (most modular
        # main.py files do this and reference `<sub>.foo`).
        parts.append(_namespace_alias_block(name, sub_name, sub_src))

    parts.append("\n# === agent ===\n")
    parts.append(_clean_agent_source(agent_src))

    # === kaggle entrypoint trailer ===
    # kaggle_environments/agent.py:get_last_callable picks the LAST
    # callable in env.values() insertion order. When a cross-agent
    # dependency (e.g. baseline/main.py) is inlined and defines its
    # own `def agent`, the primary's later `def agent` overwrites the
    # VALUE but Python's dict preserves the original insertion POSITION
    # of the key. Any helper defined AFTER baseline's `agent` insert
    # but BEFORE the primary's `agent` redef (e.g. buildup_planner's
    # `_reset_if_new_game`) becomes the last fresh callable, and
    # kaggle calls IT with (obs, configuration) — exploding on the
    # first arg type-mismatch. Symptom on the LB: ERROR with traceback
    # in some helper function that never expected to be the entry.
    # Origin: 2026-05-23, sub 52968305 (buildup_planner) ERROR'd.
    # The trailer below introduces a NEW symbol guaranteed to be last
    # in env.values(), forwarding to the real `agent`.
    parts.append(
        "\n# === kaggle entrypoint trailer (bundle_agent.py) ===\n"
        "def _kaggle_orbit_wars_entrypoint(observation, configuration=None):\n"
        "    return agent(observation, configuration)\n"
    )

    out_path.write_text("".join(parts))
    return out_path


def _topo_sort_agent_submodules(
    agent_dir: Path, *, include_main: bool = False,
) -> list[tuple[str, str]]:
    """Discover and topologically order the agent's sibling .py modules.

    Returns `[(name, source), ...]` ordered so that any module X appears
    before any module Y that does `from agents.<pkg>.X import ...`. Skips
    `__init__.py` (irrelevant after inlining). By default skips
    `main.py` too (the primary-agent caller emits its main last); pass
    `include_main=True` for cross-agent dependency packages where main.py
    must be inlined alongside its siblings. Cycles → ValueError; in
    practice they indicate a layering bug.
    """
    pkg = agent_dir.name
    submodules: dict[str, str] = {}
    deps: dict[str, set[str]] = {}
    # Two patterns: `from agents.pkg.X import a` (attribute import) AND
    # `from agents.pkg import X[, Y]` (module-as-namespace import). Both
    # establish X as a dep that must be inlined first. Origin: cherry-
    # picked `agents/precision/intercept.py` uses the namespace form
    # `from agents.precision import sim` and `sim` was being placed
    # AFTER `intercept` in the inlined bundle → NameError.
    attr_re = re.compile(rf"^\s*from\s+agents\.{re.escape(pkg)}\.([\w]+)\s+import\b")
    mod_re = re.compile(rf"^\s*from\s+agents\.{re.escape(pkg)}\s+import\s+(.+)$")

    for path in sorted(agent_dir.glob("*.py")):
        if path.name == "__init__.py":
            continue
        if path.name == "main.py" and not include_main:
            continue
        mod_name = path.stem
        src = path.read_text()
        submodules[mod_name] = src
        deps[mod_name] = set()
        for line in src.splitlines():
            m_attr = attr_re.match(line)
            if m_attr and m_attr.group(1) != mod_name:
                deps[mod_name].add(m_attr.group(1))
                continue
            m_mod = mod_re.match(line)
            if m_mod:
                # `from agents.pkg import a, b as c` — strip aliases and
                # commas, drop the trailing comment if any.
                rhs = m_mod.group(1).split("#")[0]
                for token in rhs.replace("(", "").replace(")", "").split(","):
                    token = token.strip()
                    if not token:
                        continue
                    # Handle `X as Y` — the dep is X, not Y.
                    name = token.split(" as ")[0].strip()
                    if name and name != mod_name:
                        deps[mod_name].add(name)

    # Kahn's algorithm.
    ordered: list[tuple[str, str]] = []
    pending = {k: set(v) for k, v in deps.items()}
    while pending:
        ready = sorted(n for n, ds in pending.items() if not (ds & set(pending)))
        if not ready:
            raise ValueError(
                f"cycle in agent submodule deps: {pending}"
            )
        for n in ready:
            ordered.append((n, submodules[n]))
            del pending[n]
    return ordered


# Pattern for any `from agents.<pkg>...` import line, capturing <pkg>.
# Used to discover cross-agent dependencies (the bundler strips intra-
# agent imports; without inlining, the stripped line leaves NameErrors).
_CROSS_AGENT_RE = re.compile(r"^\s*from\s+agents\.(\w+)(?:\.\w+)*\s+import\b")


def _public_top_level_names(src: str) -> list[str]:
    """Return public top-level names defined by `src`.

    Includes: function defs, class defs (incl. @dataclass), simple
    assignments to UPPER_CASE or plain identifiers. Excludes names
    starting with `_`. Used to build namespace-alias blocks for
    `from agents.<pkg> import <sub>` callers that use `<sub>.foo()`.
    """
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return []
    names: list[str] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if not node.name.startswith("_"):
                names.append(node.name)
        elif isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and not tgt.id.startswith("_"):
                    names.append(tgt.id)
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and not node.target.id.startswith("_"):
                names.append(node.target.id)
    # Preserve definition order; dedupe.
    seen: set[str] = set()
    return [n for n in names if not (n in seen or seen.add(n))]


def _namespace_alias_block(pkg: str, sub: str, src: str) -> str:
    """Emit a `<sub> = SimpleNamespace(name=name, …)` block for the
    inlined module, so `<pkg>` package callers using `from agents.<pkg>
    import <sub>` + `<sub>.foo()` still resolve after stripping.

    Falls back to an empty block when the module exports nothing
    public (e.g., constants-only header or empty file).
    """
    names = _public_top_level_names(src)
    if not names:
        return ""
    pairs = ",\n    ".join(f"{n}={n}" for n in names)
    return (
        f"\n# Namespace alias so `from agents.{pkg} import {sub}` callers "
        f"still see `{sub}.<name>`.\n"
        f"from types import SimpleNamespace as _SimpleNamespace_{sub}\n"
        f"{sub} = _SimpleNamespace_{sub}(\n    {pairs},\n)\n"
    )


def _discover_cross_agent_packages(
    main_src: str, primary_pkg: str, primary_subs: list[tuple[str, str]],
) -> list[str]:
    """Return a list of agent package names referenced by `from agents.X.*`
    imports, where X != primary_pkg. Recursively walks each discovered
    package's own files so transitive deps surface too.

    Order: discovery order (BFS). Callers should inline in order so that
    each package's content is in scope when later content references it.
    Cross-package dependency cycles are NOT supported (rare and indicate
    a layering bug); they raise via the topological sort downstream.
    """
    discovered: list[str] = []
    seen: set[str] = set()

    def _scan(src: str) -> list[str]:
        out = []
        for line in src.splitlines():
            m = _CROSS_AGENT_RE.match(line)
            if m and m.group(1) != primary_pkg and m.group(1) not in seen:
                out.append(m.group(1))
        return out

    worklist: list[str] = list(_scan(main_src))
    for _, sub_src in primary_subs:
        worklist.extend(_scan(sub_src))

    while worklist:
        pkg = worklist.pop(0)
        if pkg in seen:
            continue
        seen.add(pkg)
        pkg_dir = REPO / "agents" / pkg
        if not pkg_dir.is_dir():
            raise RuntimeError(
                f"cross-agent dep `agents/{pkg}` referenced but missing on disk"
            )
        # Recurse: scan every .py file in this package for further deps.
        for path in sorted(pkg_dir.rglob("*.py")):
            if path.name == "__init__.py":
                continue
            for dep in _scan(path.read_text()):
                if dep not in seen:
                    worklist.append(dep)
        discovered.append(pkg)

    return discovered


def _bundle_hash(path: Path) -> str:
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _bundle_full_hash(path: Path) -> str:
    """Full sha256 used as the parity-cache key (not truncated)."""
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


_PARITY_CACHE_PATH = REPO / "audit" / "bundle-parity-cache.json"


def _parity_cache_load() -> dict:
    import json
    if not _PARITY_CACHE_PATH.exists():
        return {}
    try:
        return json.loads(_PARITY_CACHE_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def _parity_cache_hit(full_sha: str) -> bool:
    entry = _parity_cache_load().get(full_sha)
    return bool(entry and entry.get("passed"))


def _parity_cache_record(full_sha: str, turns: int, seeds: tuple[int, ...]) -> None:
    import json
    import time
    cache = _parity_cache_load()
    cache[full_sha] = {
        "passed": True,
        "turns": int(turns),
        "seeds": list(seeds),
        "recorded_at": time.time(),
    }
    _PARITY_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _PARITY_CACHE_PATH.write_text(json.dumps(cache, indent=2, sort_keys=True) + "\n")


def _parity_gate(bundle_path: Path, agent_dir: Path, seeds=(0,)) -> tuple[bool, int]:
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

    # Override the chooser wallclock for the duration of the parity check.
    # Without this, `lib.v7_search.choose*` bails its candidate-scoring
    # loop on the 700 ms watchdog mid-list, so argmax picks over a subset
    # that depends on CPU jitter — source and bundle then disagree on
    # ~0.2 % of turns purely from timing noise. Setting the budget
    # effectively unbounded makes each call a pure function of `obs`,
    # which is what the parity contract should be testing.
    parity_env_var = "ORBIT_WARS_PARITY_WALLCLOCK_MS"
    prior = os.environ.get(parity_env_var)
    os.environ[parity_env_var] = "60000"
    try:
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
    finally:
        if prior is None:
            os.environ.pop(parity_env_var, None)
        else:
            os.environ[parity_env_var] = prior

    if mismatches:
        print(f"  PARITY FAIL: {mismatches}/{compared} mismatched turns", file=sys.stderr)
        return False, compared
    print(f"  parity OK: {compared} turns matched across {len(seeds)} self-play seed(s)")
    return True, compared


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
        help="skip the post-bundle self-play parity check (NOT recommended; "
             "prefer relying on the sha256 parity cache at "
             "audit/bundle-parity-cache.json — a cache hit auto-skips)",
    )
    parser.add_argument(
        "--ignore-parity-cache", action="store_true",
        help="run the parity gate even if this bundle's sha256 is in "
             "audit/bundle-parity-cache.json (use to force re-verification)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="overwrite an existing git-tracked submission file (DEFAULT: refuse)",
    )
    args = parser.parse_args(argv)
    out = bundle(
        args.agent_dir.resolve(),
        args.lib,
        out_dir=args.out_dir.resolve(),
        force=args.force,
    )
    h = _bundle_hash(out)
    full_sha = _bundle_full_hash(out)
    print(f"wrote {out} ({out.stat().st_size} bytes) sha256:{h}")

    # Sanity check: the bundle must expose an `agent` callable at module
    # top level. The bundler comments out `from agents.<name>.main import
    # agent` for wrapper-style entries without inlining the body, leaving
    # bundles with no `agent` symbol; kaggle_environments falls back to
    # the last callable (wrong signature), every game ERRORs at step 0.
    # Catch this here before the parity gate's AttributeError leaks out
    # AND leaves the broken bundle in submissions/. Cherry-picked from
    # `claude/strategy-axis-decision-3437` c25a329 (2026-05-21 PM) which
    # root-caused the friction tag `bundle-agent-doesnt-inline-from-
    # baseline-main` — silent broken-bundle invalidated their n=8 A/B
    # and falsified a submit decision.
    try:
        import importlib.util as _ilu
        _spec = _ilu.spec_from_file_location("_bundle_smoke_" + out.stem, out)
        _mod = _ilu.module_from_spec(_spec)  # type: ignore[arg-type]
        sys.modules[_spec.name] = _mod
        _spec.loader.exec_module(_mod)  # type: ignore[union-attr]
    except Exception as e:
        print(f"REFUSING TO LEAVE BUNDLE: import failed — "
              f"{type(e).__name__}: {e}", file=sys.stderr)
        out.unlink()
        return 1
    if not callable(getattr(_mod, "agent", None)):
        print(f"REFUSING TO LEAVE BUNDLE: bundle has no callable `agent` "
              f"at module top level. The entry main.py likely uses "
              f"`from agents.<x>.main import agent` and the bundler "
              f"stripped the line without inlining the body. Either inline "
              f"`agent` manually or have the entry main.py define `agent` "
              f"directly. Removing {out}.", file=sys.stderr)
        out.unlink()
        return 1

    if args.skip_parity_gate:
        return 0
    if not args.ignore_parity_cache and _parity_cache_hit(full_sha):
        print(f"  parity cache HIT for sha256:{h} — skipping self-play gate")
        return 0
    ok, turns = _parity_gate(out, args.agent_dir.resolve())
    if not ok:
        print(f"REFUSING TO LEAVE BUNDLE: removing {out}", file=sys.stderr)
        out.unlink()
        return 1
    _parity_cache_record(full_sha, turns, (0,))
    return 0


if __name__ == "__main__":
    sys.exit(main())
