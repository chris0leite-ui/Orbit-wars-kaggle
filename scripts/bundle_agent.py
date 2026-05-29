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
    # 2026-05-29: mirror — 180-degree bijection helpers used by
    # missions/macro (and reusable for any 2P symmetric-board logic).
    # Pure Python, depends only on geometry.
    "mirror",
    "fleet",
    "orbit",
    "aim",
    "combat",
    "world_model",
    "intent",
    # Lazy-imported from lib/trajectory.py inside `_table_window_or_none`
    # when `_kinematic_table_enabled()`. Static import-scan in the
    # bundler treats it as required. Must precede `trajectory` so its
    # symbols are in scope before trajectory references it.
    "kinematic_table",
    "trajectory",
    "mechanism",
    "mission",
    # `scoring` exposes `pv_horizon` + `PV_GAMMA` used by missions/snipe
    # and missions/reinforce since H16 (2026-05-13). Must precede the
    # mission modules so the inlined symbols are visible at parse time.
    "scoring",
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
    # 2026-05-29: macro mission planner (2P EXPAND/STOCKPILE/STRIKE/DEFEND).
    # Pure-lib; depends only on geometry + mirror (both already inlined
    # above). Wired by agents/baseline/main.py behind BASELINE_MACRO=1.
    "missions/macro",
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
    # 2026-05-29: MLP-as-opp-model substrate (sub 53131296's 3-MLP shot
    # validator, repurposed as a learned opponent policy via
    # lib.opp_model.mlp_validated_policy). Inlined unconditionally — bundle
    # cost ~70 KB (almost all in `_validator_weights`) and the chooser's
    # opp-policy selector tries to import them lazily when
    # `BASELINE_OPP_MODEL=mlp`. Order matters: `_validator_weights` ships
    # the base64 blob; `_validator_mlp` parses it on first call;
    # `shot_features` encodes the 25-d feature vector. `opp_model` then
    # imports `shot_features` + `_validator_mlp` inside the new policy.
    "_validator_weights",
    "_validator_mlp",
    "shot_features",
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
            indent = line[: len(line) - len(line.lstrip())]
            stripped = line.strip()
            out.append(f"{indent}# {stripped}  # inlined by bundle_agent.py\n")
            for asname, original in _extract_aliases(line):
                out.append(f"{indent}{asname} = {original}\n")
        else:
            out.append(line)
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
    if agent_dir.is_dir():
        agent_submodules = _topo_sort_agent_submodules(agent_dir)
        for sub_name, sub_src in agent_submodules:
            parts.append(f"\n# === inlined: agents/{name}/{sub_name}.py ===\n")
            parts.append(_clean_agent_source(sub_src))

    parts.append("\n# === agent ===\n")
    parts.append(_clean_agent_source(agent_src))

    out_path.write_text("".join(parts))
    return out_path


def _topo_sort_agent_submodules(agent_dir: Path) -> list[tuple[str, str]]:
    """Discover and topologically order the agent's sibling .py modules.

    Returns `[(name, source), ...]` ordered so that any module X appears
    before any module Y that does `from agents.<pkg>.X import ...`. Skips
    `main.py` (caller emits it last) and `__init__.py` (irrelevant after
    inlining). Cycles → ValueError; in practice they indicate a layering bug.
    """
    pkg = agent_dir.name
    submodules: dict[str, str] = {}
    deps: dict[str, set[str]] = {}
    dep_re = re.compile(rf"^\s*from\s+agents\.{re.escape(pkg)}\.([\w]+)\s+import\b")

    for path in sorted(agent_dir.glob("*.py")):
        if path.name in ("main.py", "__init__.py"):
            continue
        mod_name = path.stem
        src = path.read_text()
        submodules[mod_name] = src
        deps[mod_name] = set()
        for line in src.splitlines():
            m = dep_re.match(line)
            if m and m.group(1) in submodules or (m and m.group(1) != mod_name):
                deps[mod_name].add(m.group(1)) if m else None

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
