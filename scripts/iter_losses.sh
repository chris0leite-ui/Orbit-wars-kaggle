#!/usr/bin/env bash
# iter_losses.sh — pull live replays for <submission_id>, classify losses,
# print top loss buckets. Used after a submitted iter variant settles
# (~24 h post-submit when TrueSkill sigma tightens).
#
# Usage:
#     bash scripts/iter_losses.sh 52630118
#
# Output:
#     audit/live-episodes/<sid>/summary.json     (from live_episode_summary)
#     audit/loss-modes-<sid>.csv                 (from classify_losses)
#     stdout: Counter of bucket -> count
set -euo pipefail

SUB=${1:?usage: bash scripts/iter_losses.sh <submission_id>}
OUT_CSV="audit/loss-modes-${SUB}.csv"

# Auto-route harness-named credentials (KaggleAPIToke / KaggleUserName) to the
# canonical kaggle CLI env vars. Mirrors bootstrap.sh; needed because subshell
# exports there don't persist into ad-hoc script invocations.
if [[ -z "${KAGGLE_API_TOKEN:-}" && -n "${KaggleAPIToke:-}" ]]; then
    export KAGGLE_API_TOKEN="$KaggleAPIToke"
fi
if [[ -z "${KAGGLE_USERNAME:-}" && -n "${KaggleUserName:-}" ]]; then
    export KAGGLE_USERNAME="$KaggleUserName"
fi

python -m scripts.live_episode_summary "$SUB" --pull
python -m scripts.classify_losses "$SUB" --out "$OUT_CSV"

python - <<EOF
import collections, csv
rows = list(csv.DictReader(open("${OUT_CSV}")))
print()
print(f"=== Loss buckets for submission ${SUB} (n={len(rows)}) ===")
for bucket, n in collections.Counter(r["bucket"] for r in rows).most_common():
    print(f"  {bucket:24s} {n}")
print(f"\nCSV: ${OUT_CSV}")
EOF
