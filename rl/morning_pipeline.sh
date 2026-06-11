#!/bin/bash
# Morning-after pipeline: pull overnight kernel output, summarize the
# learning curve, eval the final checkpoint vs the local panel.
# Usage: bash rl/morning_pipeline.sh [output_dir]
set -euo pipefail
cd "$(dirname "$0")/.."

OUT="${1:-/tmp/kernel_overnight}"
rm -rf "$OUT" && mkdir -p "$OUT"

echo "=== downloading kernel output ==="
kaggle kernels output chrisleitescha/orbitwars-rl-train -p "$OUT"
ls -la "$OUT"

echo "=== learning curve (eval probes + entropy) ==="
python3 - "$OUT/train_log.jsonl" <<'EOF'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1])]
probes = [r for r in rows if "wr_vs_greedy" in r]
print(f"iters: {rows[-1]['iter']}  env_steps: {rows[-1]['env_steps']:,}  "
      f"wall_min: {rows[-1]['wall_min']}")
print("iter | env_steps | wr_vs_greedy | entropy | v_loss")
for r in probes:
    print(f"{r['iter']:>5} | {r['env_steps']:>10,} | {r['wr_vs_greedy']:>5} "
          f"| {r['entropy']:.3f} | {r['v_loss']:.4f}")
EOF

CKPT="$OUT/ckpt_final.pkl"
[ -f "$CKPT" ] || CKPT="$OUT/ckpt_latest.pkl"
echo "=== evaluating $CKPT vs panel ==="
bash rl/eval_ckpt.sh "$CKPT" "${2:-32}"
