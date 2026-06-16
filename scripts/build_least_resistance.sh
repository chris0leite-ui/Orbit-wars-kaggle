#!/usr/bin/env bash
# Build the least_resistance submission tarball (flat layout: main.py at root,
# alongside the producer engine it reuses).
#
# Kaggle runs a tar.gz agent by extracting it and selecting the LAST top-level
# callable in main.py as the entry point — so the agent file must end with the
# `agent` def (see agents/least_resistance/main.py header).
#
# Flat layout produced (mirrors how Kaggle sees it):
#   ./main.py            <- agents/least_resistance/main.py
#   ./producer_main.py   <- agents/producer/main.py (the orbit_lite runtime)
#   ./orbit_lite/        <- agents/producer/orbit_lite/
#   ./lib/               <- repo lib/ (physics + fallback evaluator)
#
# Usage: bash scripts/build_least_resistance.sh [OUT_TARBALL]
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-/tmp/lr_submission.tar.gz}"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

cp "$REPO/agents/least_resistance/main.py" "$STAGE/main.py"
cp "$REPO/agents/producer/main.py"         "$STAGE/producer_main.py"
cp -r "$REPO/agents/producer/orbit_lite"   "$STAGE/orbit_lite"
cp -r "$REPO/lib"                          "$STAGE/lib"

# Drop caches so the tarball is deterministic.
find "$STAGE" -name __pycache__ -type d -prune -exec rm -rf {} +
find "$STAGE" -name '*.pyc' -delete

tar -czf "$OUT" -C "$STAGE" .
echo "built $OUT"
echo "  bytes : $(stat -c%s "$OUT")"
echo "  sha256: $(sha256sum "$OUT" | cut -d' ' -f1)"
echo "  root  : $(tar -tzf "$OUT" | sed 's#^\./##' | awk -F/ 'NF&&$1{print $1}' | sort -u | tr '\n' ' ')"
