"""Kaggle GPU kernel: pretrain the producer-CLONE (fast frozen league
opponent) by pure behavior cloning on the producer replay corpus.

Output: /kaggle/working/bc_net.pkl — ship this into the code dataset,
then the RL kernel adds it to the league via --bc-opponent.
"""
import glob
import os
import sys

hits = glob.glob("/kaggle/input/**/rl/__init__.py", recursive=True)
if not hits:
    raise RuntimeError("rl package not found under /kaggle/input")
CODE_DIR = os.path.dirname(os.path.dirname(hits[0]))
sys.path.insert(0, CODE_DIR)
print("code root:", CODE_DIR, flush=True)

import jax  # noqa: E402
print("jax", jax.__version__, "devices:", jax.devices(), flush=True)

bc_hits = glob.glob("/kaggle/input/**/bc_samples.npz", recursive=True)
pool_hits = glob.glob("/kaggle/input/**/rl_pool_train.npz", recursive=True)
if not bc_hits or not pool_hits:
    raise RuntimeError("bc_samples.npz / rl_pool_train.npz not found")

sys.argv = [
    "rl.train_bc",
    "--bc-npz", bc_hits[0],
    "--pool", pool_hits[0],
    "--steps", "8000",
    "--batch", "256",
    "--lr", "5e-4",
    "--eval-every", "1000",
    "--out-dir", "/kaggle/working",
]
from rl.train_bc import main  # noqa: E402
main()
