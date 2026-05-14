#!/usr/bin/env bash
# submit.sh — submit main.py to Orbit Wars with a message.
#
# Usage:   ./submit.sh "v0: fresh start; nearest sniper baseline"
#
# Pre-flight:
#   - main.py exists at repo root
#   - kaggle CLI authenticated (bootstrap.sh smoke passed)
#
# Rule 1 (CLAUDE.md): every submit is single-shot, PI-approved. No
#   retry/until/while loops. If this fails, fix the issue and rerun once.
# Rule 12: rolling-last-2 — Kaggle auto-keeps your last 2 submits for
#   final evaluation. Don't push a speculative variant after a known-good
#   one unless you're willing to lose the good one's ladder spot.

set -euo pipefail
cd "$(dirname "$0")"

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 \"message describing the submission\"" >&2
    exit 2
fi

MSG="$1"
FILE="main.py"

if [[ ! -f "$FILE" ]]; then
    echo "ERROR: $FILE not found at repo root." >&2
    exit 2
fi

echo "--- submitting $FILE to orbit-wars ---"
echo "    message: $MSG"
kaggle competitions submit orbit-wars -f "$FILE" -m "$MSG"

echo "--- last 5 submissions ---"
kaggle competitions submissions orbit-wars | head -8
