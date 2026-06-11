#!/bin/bash
# Preserve the oracle track's data assets on a private Kaggle dataset.
# - replays are tarred in 4 compressed shards (parallel-friendly, resumable)
# - the built training datasets + episode catalog ride along uncompressed
# Re-running creates a new dataset VERSION (kaggle datasets version).
#
# Usage: bash scripts/upload_oracle_data.sh [dataset-slug]
set -euo pipefail
cd "$(dirname "$0")/.."

SLUG="${1:-orbit-wars-oracle-data}"
USER=$(kaggle config view 2>/dev/null | grep -oP 'username: \K\S+' || true)
[ -z "$USER" ] && USER=$(python3 -c "
import json,urllib.request
req=urllib.request.Request('https://www.kaggle.com/api/v1/datasets/list?user=self')
" 2>/dev/null || echo "")
STAGE=/tmp/oracle_data_upload
mkdir -p "$STAGE"

echo "--- staging built datasets + catalog"
cp -f data/external/oracle_policy_ds.npz "$STAGE/" 2>/dev/null || echo "  (policy ds missing)"
cp -f data/external/oracle_ds.npz "$STAGE/" 2>/dev/null || echo "  (value ds missing)"
cp -f data/external/episodes.jsonl "$STAGE/"
cp -f data/external/crawl_state.json "$STAGE/"

echo "--- sharding replays (skip if shards exist)"
cd data/external/replays
ls episode-*.json | sort > /tmp/replay_list.txt
N=$(wc -l < /tmp/replay_list.txt)
PER=$(( (N + 3) / 4 ))
for i in 0 1 2 3; do
  SHARD="$STAGE/replays_shard_$i.tar.gz"
  [ -s "$SHARD" ] && { echo "  shard $i exists, skipping"; continue; }
  sed -n "$((i*PER+1)),$(((i+1)*PER))p" /tmp/replay_list.txt > /tmp/shard_$i.txt
  [ -s /tmp/shard_$i.txt ] || continue
  tar -czf "$SHARD" -T /tmp/shard_$i.txt
  echo "  shard $i: $(du -h "$SHARD" | cut -f1)"
done
cd ../../..

echo "--- dataset metadata"
KUSER=$(kaggle competitions submissions orbit-wars 2>/dev/null >/dev/null; kaggle config view 2>/dev/null | sed -n 's/.*username: //p')
if [ -z "$KUSER" ]; then
  # fall back: whoami via API
  KUSER=$(python3 - <<'PY'
import os, json, urllib.request
tok = os.environ.get("KAGGLE_API_TOKEN", "")
req = urllib.request.Request("https://www.kaggle.com/api/v1/users/me",
                             headers={"Authorization": f"Bearer {tok}"})
try:
    print(json.load(urllib.request.urlopen(req)).get("userName", ""))
except Exception:
    print("")
PY
)
fi
echo "  kaggle user: ${KUSER:-UNKNOWN}"
cat > "$STAGE/dataset-metadata.json" <<EOF
{
  "title": "Orbit Wars oracle data (private)",
  "id": "${KUSER}/${SLUG}",
  "licenses": [{"name": "CC0-1.0"}]
}
EOF

echo "--- create or version"
if kaggle datasets status "${KUSER}/${SLUG}" >/dev/null 2>&1; then
  kaggle datasets version -p "$STAGE" -m "refresh $(date -u +%F_%H%M)" --dir-mode skip
else
  kaggle datasets create -p "$STAGE" --private --dir-mode skip
fi
echo "--- done: https://www.kaggle.com/datasets/${KUSER}/${SLUG}"
