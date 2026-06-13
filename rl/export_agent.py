"""Export a training checkpoint to a single-file Kaggle submission.

Usage:
  python -m rl.export_agent /path/to/ckpt.pkl submissions/rl_v1.py

The output file = rl/numpy_infer.py source (standalone numpy) +
base64(zlib(fp16 params npz)) + `agent(obs)` entry point.
"""
from __future__ import annotations

import base64
import io
import pickle
import sys
import zlib

import numpy as np

TEMPLATE = '''

# ===================== embedded model parameters =====================
import base64 as _b64
import io as _io
import zlib as _zlib

_PARAMS_B64 = (
{params_b64}
)


def _load_params():
    raw = _zlib.decompress(_b64.b64decode(_PARAMS_B64))
    z = np.load(_io.BytesIO(raw))
    return {{k: z[k].astype(np.float64) for k in z.files}}


_AGENT = None


def agent(obs):
    global _AGENT
    if _AGENT is None:
        _AGENT = RLAgent(_load_params())
    try:
        return _AGENT.act(obs)
    except Exception:
        return []
'''


def export(ckpt_path: str, out_path: str):
    with open(ckpt_path, "rb") as f:
        d = pickle.load(f)
    params = {k: np.asarray(v) for k, v in d["params"].items()}

    # fp32, not fp16: a prior review measured fp16 round-trip flipping
    # the argmax target on ~0.8% of turns (borderline logit gaps) — the
    # fp64 parity tests never see it. fp32 keeps the exported agent
    # decision-identical to the trained policy; zlib still gets the file
    # to ~0.6 MB, far under the Kaggle limit.
    buf = io.BytesIO()
    np.savez(buf, **{k: v.astype(np.float32) for k, v in params.items()})
    blob = zlib.compress(buf.getvalue(), 9)
    b64 = base64.b64encode(blob).decode()
    lines = [f'    "{b64[i:i + 96]}"' for i in range(0, len(b64), 96)]
    params_b64 = "\n".join(lines)

    with open("rl/numpy_infer.py") as f:
        src = f.read()

    with open(out_path, "w") as f:
        f.write(src)
        f.write(TEMPLATE.format(params_b64=params_b64))
    n_params = sum(v.size for v in params.values())
    import os
    print(f"exported {n_params:,} params -> {out_path} "
          f"({os.path.getsize(out_path):,} bytes)")


if __name__ == "__main__":
    export(sys.argv[1], sys.argv[2])
