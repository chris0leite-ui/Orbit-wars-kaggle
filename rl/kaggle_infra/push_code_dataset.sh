#!/bin/bash
# Build the code tarball + pool into a private Kaggle dataset.
# Usage: bash rl/kaggle_infra/push_code_dataset.sh [create|version] ["message"]
set -euo pipefail
cd "$(dirname "$0")/../.."

MODE="${1:-version}"
MSG="${2:-update}"
STAGE=$(mktemp -d)

tar czf "$STAGE/code.tar.gz" \
    --exclude='__pycache__' --exclude='*.pyc' \
    lib rl tests/test_rl_aim.py
cp data/rl_pool_train.npz "$STAGE/" 2>/dev/null || echo "WARN: no train pool"
# Optional: ship a resume checkpoint into the next dataset version.
if [ -n "${RESUME_CKPT:-}" ] && [ -f "${RESUME_CKPT}" ]; then
    cp "${RESUME_CKPT}" "$STAGE/ckpt_latest.pkl"
    echo "including resume checkpoint: ${RESUME_CKPT}"
fi

cat > "$STAGE/dataset-metadata.json" <<'EOF'
{
  "title": "orbitwars-rl-code",
  "id": "chrisleitescha/orbitwars-rl-code",
  "licenses": [{"name": "CC0-1.0"}]
}
EOF

ls -la "$STAGE"
if [ "$MODE" = "create" ]; then
    kaggle datasets create -p "$STAGE" --dir-mode skip
else
    kaggle datasets version -p "$STAGE" -m "$MSG" --dir-mode skip
fi
rm -rf "$STAGE"
