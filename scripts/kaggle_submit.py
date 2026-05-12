"""Submit agent.py to the Orbit Wars Kaggle competition.

Wraps `kaggle competitions submit` with a pre-submit gate:
1. Print sha256 of agent.py.
2. Run a single 4P self-play game with `kaggle_environments.make("orbit_wars")`
   to confirm the agent doesn't crash on a fresh env.
3. Invoke the Kaggle CLI.

Rule 1 (CLAUDE.md): single-shot, PI-authorized. This script does NOT
loop or retry. If the submit fails, surface the error and stop.

Usage:
    python scripts/kaggle_submit.py "<message>"
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
AGENT_FILE = REPO / "agent.py"
COMP_SLUG = "orbit-wars"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _load_agent(path: Path):
    spec = importlib.util.spec_from_file_location("submission_agent", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["submission_agent"] = mod
    spec.loader.exec_module(mod)
    return mod


def _selfplay_smoke() -> None:
    """One 4P self-play episode; assert no exceptions and at least one action."""
    from kaggle_environments import make

    mod = _load_agent(AGENT_FILE)
    env = make("orbit_wars", debug=True)
    env.reset(num_agents=4)
    env.run([mod.agent, mod.agent, mod.agent, mod.agent])
    total_actions = sum(
        len(step[0].get("action") or [])
        for step in env.steps[1:]
        if step[0].get("status") in ("ACTIVE", "DONE")
    )
    if total_actions == 0:
        raise RuntimeError("agent emitted zero actions across the 4P self-play episode")
    print(f"selfplay smoke OK: {len(env.steps)} steps, {total_actions} total actions")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("message", help="submission message (kaggle -m)")
    parser.add_argument(
        "--skip-smoke", action="store_true",
        help="skip the self-play parity check (debug only)",
    )
    args = parser.parse_args(argv)

    if not AGENT_FILE.is_file():
        raise SystemExit(f"agent.py not found at {AGENT_FILE}")
    print(f"agent.py sha256: {_sha256(AGENT_FILE)}  ({AGENT_FILE.stat().st_size} bytes)")

    if not args.skip_smoke:
        _selfplay_smoke()

    cmd = [
        "kaggle", "competitions", "submit",
        "-c", COMP_SLUG,
        "-f", str(AGENT_FILE),
        "-m", args.message,
    ]
    print("$", " ".join(cmd))
    res = subprocess.run(cmd, cwd=REPO)
    return res.returncode


if __name__ == "__main__":
    raise SystemExit(main())
