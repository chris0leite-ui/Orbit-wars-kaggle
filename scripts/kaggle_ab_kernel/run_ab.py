"""Orbit Wars A/B Kernel — sub-phase 8 deploy harness.

Runs an A/B between two Kaggle Simulations bundle files using
Kaggle Kernel's CPU pool. This is the **scalar** path (no JAX agent
needed) — it gets us to ~5 min A/B on Kaggle's 4-9 vCPU kernels even
before the agent port (sub-phases 3-7) completes.

For now we only have a HELLO-WORLD body that probes the kernel
environment: Python version, available CPU count, kaggle_environments
version. Once we confirm the deploy/run/output flow works, we'll wire
in the A/B harness (run two bundles head-to-head over N seeds).

Output written to `/kaggle/working/result.json`.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import platform
import subprocess
import sys
import time


def _ensure_orbit_wars():
    """Kaggle Kernels ship kaggle_environments==1.27.3 which doesn't
    include orbit_wars. Pip-install the version our local code targets."""
    try:
        from kaggle_environments import make
        make("orbit_wars", configuration={"seed": 0})
        return None  # already works
    except Exception:
        pass
    print("Installing kaggle_environments==1.29.1 ...")
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q",
         "kaggle_environments==1.29.1"],
    )
    # Re-import in a fresh process is cleanest, but a module reload
    # generally works for top-level packages.
    import importlib, kaggle_environments
    importlib.reload(kaggle_environments)
    return "installed 1.29.1"


def main():
    install_msg = _ensure_orbit_wars()
    info = {
        "install_msg": install_msg,
        "python_version": platform.python_version(),
        "machine": platform.machine(),
        "cpu_count_logical": os.cpu_count(),
        "cpu_count_mp": mp.cpu_count(),
        "sys_path_first": sys.path[:5],
        "env_pwd": os.getcwd(),
        "kaggle_input_exists": os.path.isdir("/kaggle/input"),
        "kaggle_working_exists": os.path.isdir("/kaggle/working"),
        "files_in_cwd": sorted(os.listdir("."))[:30],
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    # Probe kaggle_environments availability + version.
    try:
        import kaggle_environments
        info["kaggle_environments_version"] = getattr(kaggle_environments, "__version__", "?")
        from kaggle_environments import make
        env = make("orbit_wars", configuration={"seed": 0})
        info["orbit_wars_make_ok"] = True
    except Exception as e:
        info["kaggle_environments_error"] = repr(e)

    out_path = "/kaggle/working/result.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(info, f, indent=2)
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
