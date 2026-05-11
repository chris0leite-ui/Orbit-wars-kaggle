"""Verify the packaged submission tarball works when imported from a clean cwd.

Simulates what Kaggle does: extract the tarball, cd into the extracted dir,
import main, run a short episode.
"""
from __future__ import annotations

import importlib
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import pathlib


def _ensure_packed():
    """Run the pack script if the tarball is missing or older than source."""
    repo = pathlib.Path(__file__).resolve().parents[3]
    tarball = repo / "submissions" / "precision_v2.tar.gz"
    src_dir = repo / "agents" / "precision"
    if tarball.exists():
        tar_mtime = tarball.stat().st_mtime
        src_mtime = max(p.stat().st_mtime for p in src_dir.glob("*.py"))
        if tar_mtime >= src_mtime:
            return tarball
    script = repo / "scripts" / "pack_precision.sh"
    subprocess.run(["bash", str(script)], check=True, cwd=repo)
    return tarball


def test_packaged_submission_runs_in_isolation():
    tarball = _ensure_packed()

    workdir = tempfile.mkdtemp(prefix="precision_v2_test_")
    try:
        with tarfile.open(tarball, "r:gz") as tf:
            tf.extractall(workdir)

        # Drop any inherited sys.path entries that might let it find the source
        # via package-style imports — simulate Kaggle's "main.py at root" runtime.
        old_path = list(sys.path)
        old_modules = set(sys.modules.keys())
        try:
            sys.path[:] = [workdir] + [p for p in sys.path if "precision" not in p]
            # Force-reimport: drop any cached precision modules.
            for mod in list(sys.modules.keys()):
                if mod in (
                    "main", "intercept", "planner", "sim", "prediction",
                    "scoring", "enemy_model", "bundling",
                ):
                    sys.modules.pop(mod, None)
            mod = importlib.import_module("main")
            assert hasattr(mod, "agent"), "packaged main has no agent()"
            # Run a tiny episode through the actual engine
            from kaggle_environments import make
            env = make("orbit_wars", configuration={"seed": 0, "episodeSteps": 30})
            env.run([mod.agent, mod.agent])
            assert len(env.steps) >= 2, "episode didn't run"
            # No exception during the episode.
            print(f"  packaged agent ran {len(env.steps)} steps cleanly")
        finally:
            sys.path[:] = old_path
            # Clear the modules we imported so the rest of the test process
            # picks up the unpacked vs source distinction correctly.
            for mod_name in set(sys.modules.keys()) - old_modules:
                sys.modules.pop(mod_name, None)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    test_packaged_submission_runs_in_isolation()
    print("\nPackaged-submission test passed.")
