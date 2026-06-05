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
    # opp_projection depends on garrison_launch, intercept_aim, movement,
    # obs, planner_core — all above. Must come AFTER planner_core.
    "opp_projection",
    # recapture imports DistanceCache, fleet_speed, PlanetGarrisonStatus —
    # all from modules above. No interdependence with opp_projection.
    "recapture",
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
    "composed": {
        # Step 4 + Step 5: BOTH multi_size and coalitions ON. The plan.py
        # `plan_lite_waves` composed branch packs S*T*N + T*C(K,2) candidates
        # at L=2. Tests the hypothesis that coalitions only lift when paired
        # with multi-size single-source variants.
        "PRODUCER_PLUS_MULTI_SIZE": "1",
        "PRODUCER_PLUS_COALITIONS": "1",
    },
    "opp_proj": {
        # Step 3 redux: per-turn opp multi-launch projection injected as
        # background LaunchSet slots in the scorer. Multi_size and coalitions
        # deliberately OFF — this is the standalone variant testing the
        # opp-projection mechanism in isolation.
        "PRODUCER_PLUS_OPP_PROJECTION": "1",
    },
    "multi_opp_def": {
        # Step 4 (multi_size) + opp_projection (Producer-mirror) + the
        # opp-aware defensive shortlist augmentation in friendly_flip_targets
        # (which activates unconditionally when background is non-empty).
        # Coalitions deliberately OFF — diagnostic at seed 7 showed they
        # barely fire and actively hurt the kitchen-sink variant (-1 win).
        # Local n=16: 12/16 wins vs producer (Wilson [50%, 90%]).
        "PRODUCER_PLUS_MULTI_SIZE": "1",
        "PRODUCER_PLUS_OPP_PROJECTION": "1",
    },
    "multi_tick_opp_K3": {
        # Multi_opp_def + K-round opp projection. Opp's planner runs K
        # successive rounds, each round seeing prior rounds' projected
        # launches as `background`; per-round launches are eta-shifted by
        # +k before merging. Addresses the cycle stalemate diagnosis
        # (knowledge-base/thoughts/2026-06-05-cycle-stalemate-and-horizon-
        # scaling.md): scorer was blind past tick ~8 because opp_proj only
        # projected one tick. 4P stalemate is the target pathology; 2P
        # gets K=2 so the 2P A/B harness can still detect breakage.
        # Horizon bump intentionally NOT baked here — separate A/B once
        # this variant lands.
        "PRODUCER_PLUS_MULTI_SIZE": "1",
        "PRODUCER_PLUS_OPP_PROJECTION": "1",
        "PRODUCER_PLUS_MULTI_TICK_OPP_K_4P": "3",
        "PRODUCER_PLUS_MULTI_TICK_OPP_K_2P": "2",
    },
    "recapture_penalty": {
        # Standalone recapture penalty: per-candidate leaf-scorer discount
        # for thin captures opp can recapture. Tests the mechanism in
        # isolation (no multi_size, no opp_proj, no multi-tick). See
        # agents/producer/orbit_lite/recapture.py for the math.
        "PRODUCER_PLUS_RECAPTURE_PENALTY": "1",
    },
    "multi_tick_recap": {
        # Composed: multi_size + opp_proj + multi-tick + recapture penalty.
        # The path that ships if the standalone A/B clears. Recapture
        # penalty's K_recap_eff = max(1, K_recap - K_opp) clips the
        # window to past what multi-tick already modeled, avoiding
        # double-counting.
        "PRODUCER_PLUS_MULTI_SIZE": "1",
        "PRODUCER_PLUS_OPP_PROJECTION": "1",
        "PRODUCER_PLUS_MULTI_TICK_OPP_K_4P": "3",
        "PRODUCER_PLUS_MULTI_TICK_OPP_K_2P": "2",
        "PRODUCER_PLUS_RECAPTURE_PENALTY": "1",
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
