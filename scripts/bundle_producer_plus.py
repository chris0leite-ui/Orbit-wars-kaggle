"""Bundle producer_plus into a single Kaggle-submittable .py file.

Producer_plus depends on agents/producer/orbit_lite/* (the vendored
Producer engine). The bundler concatenates all orbit_lite modules in
topological order, strips intra-package imports, appends
agents/producer_plus/main.py with its `from orbit_lite.X` imports
stripped, and bakes the env vars (PRODUCER_PLUS_ADAPTIVE_K=1,
PRODUCER_PLUS_MULTI_SIZE=1) at the top.

Output: submissions/producer_plus_multi_size_on.py

CLI:
    python scripts/bundle_producer_plus.py [--out PATH] [--variant adaptive_k|multi_size]
"""
from __future__ import annotations

import argparse
import ast
import os
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# Topological order: each module must come after its orbit_lite-internal deps.
ORBIT_LITE_ORDER = [
    "constants",
    "aiming",
    "geometry",
    "obs",
    "movement_aiming",
    "movement",
    "distance_cache",
    "garrison_launch",
    "intercept_aim",
    "movement_step",
    "adapter",
    "planner_core",
]

ORBIT_LITE_DIR = REPO / "agents" / "producer" / "orbit_lite"
PRODUCER_PLUS_MAIN = REPO / "agents" / "producer_plus" / "main.py"

ENV_VARIANTS = {
    "adaptive_k": {
        "PRODUCER_PLUS_ADAPTIVE_K": "1",
    },
    "multi_size": {
        # Adaptive_K (Step 2) deliberately OFF: 16-game seat-alt A/B
        # 2026-06-04 showed Step 2+4 regressed to 5/16 vs producer while
        # Step 4 alone hit 10/16. Adaptive_K is preserved in main.py as
        # a gated path for future tuning.
        "PRODUCER_PLUS_MULTI_SIZE": "1",
    },
    "coalitions": {
        # Step 5: L=2 multi-source coalitions packed alongside single-
        # source candidates. Multi_size (Step 4) deliberately OFF to
        # avoid the 3-size × C(K,2)-pair candidate explosion — compose
        # as Step 5b later only if both lift independently.
        "PRODUCER_PLUS_COALITIONS": "1",
    },
}


# Strip `from .X import ...` (orbit_lite intra-package).
RE_INTRA_IMPORT_SINGLE = re.compile(r"^from \.[a-z_][a-z_0-9]* import [^()\n]+$", re.MULTILINE)
# Strip `from orbit_lite.X import ...` (producer_plus -> orbit_lite).
RE_OL_IMPORT_SINGLE = re.compile(r"^from orbit_lite\.[a-z_][a-z_0-9]* import [^()\n]+$", re.MULTILINE)
# Multi-line versions: `from .X import (a, b, c)` may span lines.
RE_INTRA_IMPORT_MULTI = re.compile(r"^from \.[a-z_][a-z_0-9]* import \([^)]*\)", re.MULTILINE | re.DOTALL)
RE_OL_IMPORT_MULTI = re.compile(r"^from orbit_lite\.[a-z_][a-z_0-9]* import \([^)]*\)", re.MULTILINE | re.DOTALL)
# producer_plus/main.py has sys.path injection that's not needed in a bundle.
RE_SYS_PATH_BLOCK = re.compile(
    r"# Make the sibling[\s\S]+?if _HERE not in sys\.path:\n\s+sys\.path\.insert\(0, _HERE\)\n",
    re.MULTILINE,
)


def strip_imports(text: str, kind: str) -> str:
    if kind == "orbit_lite":
        text = RE_INTRA_IMPORT_MULTI.sub("", text)
        text = RE_INTRA_IMPORT_SINGLE.sub("", text)
    elif kind == "producer_plus":
        text = RE_OL_IMPORT_MULTI.sub("", text)
        text = RE_OL_IMPORT_SINGLE.sub("", text)
        text = RE_SYS_PATH_BLOCK.sub("", text)
    return text


def strip_future_imports(text: str) -> tuple[str, str]:
    """Aggregate ALL `from __future__` lines (anywhere in module) for the
    bundle header and remove them from the body — `from __future__` must
    appear before any other statement in the final file, but module
    docstrings can hide them in the middle of source.
    """
    futures: list[str] = []
    rest_lines: list[str] = []
    for ln in text.split("\n"):
        if ln.strip().startswith("from __future__"):
            futures.append(ln.strip())
        else:
            rest_lines.append(ln)
    return "\n".join(futures), "\n".join(rest_lines)


def build(env_vars: dict, out_path: Path) -> None:
    parts: list[str] = []
    all_futures: set[str] = set()

    for mod_name in ORBIT_LITE_ORDER:
        src = (ORBIT_LITE_DIR / f"{mod_name}.py").read_text()
        src = strip_imports(src, "orbit_lite")
        futures, body = strip_future_imports(src)
        for ln in futures.splitlines():
            if ln.strip():
                all_futures.add(ln.strip())
        parts.append(f"\n# === orbit_lite.{mod_name} ===\n{body}\n")

    main_src = PRODUCER_PLUS_MAIN.read_text()
    main_src = strip_imports(main_src, "producer_plus")
    futures, body = strip_future_imports(main_src)
    for ln in futures.splitlines():
        if ln.strip():
            all_futures.add(ln.strip())
    parts.append(f"\n# === producer_plus.main ===\n{body}\n")

    env_header = ""
    if env_vars:
        env_header = "import os as _pp_os\n"
        for k, v in env_vars.items():
            env_header += f'_pp_os.environ.setdefault({k!r}, {v!r})\n'

    futures_header = "\n".join(sorted(all_futures)) + "\n" if all_futures else ""

    bundle = futures_header + env_header + "".join(parts) + "\n"
    # Kaggle entry point: a top-level `agent(obs, configuration=None)`.
    # producer_plus.main defines `def agent(obs):` — wrap it to add the
    # configuration arg the harness expects.
    bundle += (
        "\n# === bundle entry point (Kaggle expects 2-arg agent) ===\n"
        "_pp_inner_agent = agent\n"
        "def agent(obs, configuration=None):  # noqa: F811  shadow for harness\n"
        "    return _pp_inner_agent(obs)\n"
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(bundle)
    # Verify the bundle parses.
    ast.parse(bundle)
    print(f"wrote {out_path}  ({len(bundle):_} bytes)")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=list(ENV_VARIANTS), default="multi_size")
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output path; default submissions/producer_plus_<variant>_on.py",
    )
    args = p.parse_args()
    out = args.out or REPO / "submissions" / f"producer_plus_{args.variant}_on.py"
    build(ENV_VARIANTS[args.variant], out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
