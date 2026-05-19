"""Run the Phase-1b scenario suite against an agent and print PASS/FAIL.

Usage:
    python scripts/run_scenarios.py --agent agents.baseline.main
    python scripts/run_scenarios.py --agent agents.bundle.main

Exit code 0 only if every registered scenario passes. The same suite is
also invokable as pytest tests under `tests/scenarios/`; the standalone
runner is for quick local iteration during ROI development.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# Register all scenario classes via the imports below.
import tests.scenarios.test_observed  # noqa: F401, E402
from tests.scenarios.base import all_scenarios  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--agent", required=True,
        help="Python module path of the agent under test "
             "(must expose `agent(obs, configuration)`).",
    )
    ap.add_argument(
        "--names", default="",
        help="Comma-separated subset of scenario names to run "
             "(default: run all).",
    )
    args = ap.parse_args(argv)

    # Fail-fast on agent import — otherwise every scenario fails with
    # the same opaque error.
    try:
        importlib.import_module(args.agent)
    except Exception as exc:  # noqa: BLE001
        print(f"FATAL: cannot import agent module {args.agent!r}: {exc}",
              file=sys.stderr)
        return 2

    name_filter = {n.strip() for n in args.names.split(",") if n.strip()}

    scenarios = all_scenarios()
    if name_filter:
        scenarios = [s for s in scenarios if s.name in name_filter]
        if not scenarios:
            print(f"FATAL: no scenarios match --names={args.names!r}; "
                  f"available: {[s.name for s in all_scenarios()]}",
                  file=sys.stderr)
            return 2

    failures = 0
    print(f"=== run_scenarios — agent={args.agent} "
          f"({len(scenarios)} scenarios) ===")
    for sc in scenarios:
        result = sc.run(args.agent)
        mark = "PASS" if result.passed else "FAIL"
        print(f"  [{mark}] {sc.name}  ({sc.flavour})")
        print(f"         rationale: {sc.rationale}")
        print(f"         source:    {sc.source}")
        print(f"         outcome:   {result.explanation}")
        if not result.passed:
            failures += 1
    print(f"=== summary: {len(scenarios) - failures}/{len(scenarios)} pass ===")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
