"""Kaggle GPU training kernel for the Orbit Wars RL agent.

Unpacks the code tarball from the input dataset, then runs rl.train.
Tuning knobs are set here (edit + `kaggle kernels push` for each run).
"""
import glob
import os
import sys

DS = "/kaggle/input/orbitwars-rl-code"
print("input tree (2 levels):", flush=True)
for p in sorted(glob.glob("/kaggle/input/*/*"))[:40]:
    print("  ", p, flush=True)

# Self-locate the code root (Kaggle's archive auto-decompression layout
# has varied): find rl/__init__.py and add its parent to sys.path.
hits = glob.glob("/kaggle/input/**/rl/__init__.py", recursive=True)
if not hits:
    raise RuntimeError("rl package not found under /kaggle/input")
CODE_DIR = os.path.dirname(os.path.dirname(hits[0]))
print("code root:", CODE_DIR, flush=True)
sys.path.insert(0, CODE_DIR)

import jax  # noqa: E402

print("jax", jax.__version__, "devices:", jax.devices(), flush=True)
if not any(d.platform == "gpu" for d in jax.devices()):
    print("WARNING: no GPU visible to JAX — training on CPU", flush=True)

# Resume from a prior checkpoint if the dataset ships one.
resume = ""
for cand in glob.glob("/kaggle/input/**/ckpt_latest.pkl", recursive=True):
    resume = cand
    print("resuming from", cand, flush=True)
    break

pool_hits = glob.glob("/kaggle/input/**/rl_pool_train.npz", recursive=True)
if not pool_hits:
    raise RuntimeError("rl_pool_train.npz not found under /kaggle/input")

# --- NIGHT-3 RUN: league + producer behavior cloning ---
bc_hits = glob.glob("/kaggle/input/**/bc_samples.npz", recursive=True)
RUN_ARGS = [
    "rl.train",
    "--pool", pool_hits[0],
    "--out-dir", "/kaggle/working",
    "--batch", "256",
    "--rollout-steps", "32",
    "--minibatches", "16",
    "--epochs", "2",
    "--lr", "3e-4",
    "--hours", "8.2",
    "--eval-every", "25",
    "--eval-envs", "64",
    "--eval-opp", "rusher",
    "--ckpt-every-min", "20",
    "--league",
    "--snapshot-every", "150",
    "--snapshot-cap", "12",
    "--greedy-frac", "0.4",
]
if bc_hits:
    RUN_ARGS += ["--bc-npz", bc_hits[0], "--bc-coef", "0.3",
                 "--bc-batch", "192"]
    print("BC samples:", bc_hits[0], flush=True)
if resume:
    RUN_ARGS += ["--resume", resume]

sys.argv = RUN_ARGS
from rl.train import main  # noqa: E402

main()
