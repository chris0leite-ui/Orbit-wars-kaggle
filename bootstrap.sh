#!/usr/bin/env bash
# bootstrap.sh — set up a fresh container or laptop for Orbit Wars.
# Reads COMP from .comp.env (defaults to "orbit-wars" if absent).
#
# What it does:
#   1. Source .comp.env (COMP).
#   2. Resolve Kaggle credentials (three accepted forms).
#   3. Create ~/.kaggle/kaggle.json from KAGGLE_USERNAME + KAGGLE_KEY if needed.
#   4. Install Python requirements.
#   5. Download competition data into data/ if not present.
#   6. (Optional) Pull reference notebooks into external/kernels/.

set -euo pipefail
cd "$(dirname "$0")"

# ---------------------------------------------------------------------------
# Step 0 — load per-comp config
# ---------------------------------------------------------------------------
if [[ -f .comp.env ]]; then
    # shellcheck disable=SC1091
    source .comp.env
fi

# Default for Orbit Wars; .comp.env can override.
COMP="${COMP:-orbit-wars}"

echo "--- comp: $COMP"

# ---------------------------------------------------------------------------
# Step 1 — credentials
# ---------------------------------------------------------------------------
KAGGLE_JSON="$HOME/.kaggle/kaggle.json"

# Harness env-var name fall-through: $KaggleUserName / $KaggleAPIToke (sic)
# come from the Claude Code harness; map them to the canonical names so
# downstream code stays single-sourced.
if [[ -z "${KAGGLE_USERNAME:-}" && -n "${KaggleUserName:-}" ]]; then
    export KAGGLE_USERNAME="$KaggleUserName"
fi
if [[ -z "${KAGGLE_KEY:-}" && -n "${KaggleAPIToke:-}" ]]; then
    export KAGGLE_KEY="$KaggleAPIToke"
fi

if [[ -z "${KAGGLE_API_TOKEN:-}" && -n "${KAGGLE_KEY:-}" ]]; then
    export KAGGLE_API_TOKEN="$KAGGLE_KEY"
fi

# KGAT_-prefixed tokens are the new Kaggle personal-access-token format
# and must be read from $KAGGLE_API_TOKEN, NOT from the `key` field of
# kaggle.json (which routes through the legacy 32-hex auth path and
# returns 401). When we see a KGAT_ token, skip the kaggle.json write
# and use the env-var path unconditionally.
if [[ "${KAGGLE_API_TOKEN:-}" == KGAT_* || "${KAGGLE_KEY:-}" == KGAT_* ]]; then
    echo "--- credentials: KGAT_-prefix token; using KAGGLE_API_TOKEN env path ---"
elif [[ -f "$KAGGLE_JSON" ]]; then
    echo "--- credentials: $KAGGLE_JSON found, using it ---"
elif [[ -n "${KAGGLE_USERNAME:-}" && -n "${KAGGLE_KEY:-}" ]]; then
    echo "--- credentials: writing $KAGGLE_JSON from KAGGLE_USERNAME + KAGGLE_KEY ---"
    mkdir -p "$(dirname "$KAGGLE_JSON")"
    umask 077
    printf '{"username":"%s","key":"%s"}\n' \
        "$KAGGLE_USERNAME" "$KAGGLE_KEY" > "$KAGGLE_JSON"
    chmod 600 "$KAGGLE_JSON"
elif [[ -n "${KAGGLE_API_TOKEN:-}" ]]; then
    echo "--- credentials: KAGGLE_API_TOKEN set; works with the harness CLI ---"
else
    echo "ERROR: no Kaggle credentials found."
    echo "  Set up ~/.kaggle/kaggle.json (mode 600) — see SETUP.md."
    exit 2
fi

# Cred smoke: surface 401s in the first 5 minutes, not at submit time.
# NOTE: `set -o pipefail` makes `cmd | head -3` propagate SIGPIPE from the
# upstream when head closes the pipe early — looks like a CLI failure when
# it's actually success-with-trimmed-output. Capture full output to a tmp,
# then head-display, and check the CLI's actual exit code separately.
echo "--- credentials smoke: kaggle competitions list -s orbit ---"
_cred_tmp=$(mktemp)
if kaggle competitions list -s orbit > "$_cred_tmp" 2>&1; then
    head -3 "$_cred_tmp"
else
    echo "WARNING: kaggle CLI smoke failed; submit-time will also fail."
    echo "        Full error below — do not truncate; Python tracebacks"
    echo "        run 6-20 lines and head -5 hid real auth diagnoses in"
    echo "        prior sessions."
    cat "$_cred_tmp"
fi
rm -f "$_cred_tmp"

# ---------------------------------------------------------------------------
# Step 2 — Python deps
# ---------------------------------------------------------------------------
if [[ -f requirements.txt ]]; then
    echo "--- installing requirements ---"
    # On Debian/Ubuntu base images, pip can't uninstall the OS-installed
    # `python3-blinker` (no RECORD metadata) and aborts the whole install.
    # Pre-replace it with a pip-managed copy so `pip install -r ...` succeeds.
    pip install -q --ignore-installed blinker 2>/dev/null || true
    pip install -q -r requirements.txt
else
    echo "--- (no requirements.txt; skipping pip install) ---"
fi

# torch (CPU) is NOT in requirements.txt (large; provided by Kaggle's eval
# runtime) but the least_resistance agent's orbit_lite leaf scorer + producer
# mirror need it for local dev/eval — without it the agent silently falls back
# to a weaker pure-Python path and the deep-search code never runs. Missing
# torch has bitten multiple sessions; install it here (idempotent, non-fatal so
# a network-restricted policy doesn't abort the hook).
if python -c "import torch" 2>/dev/null; then
    echo "--- torch: already installed ($(python -c 'import torch; print(torch.__version__)' 2>/dev/null)) ---"
else
    echo "--- torch: installing CPU build from download.pytorch.org ---"
    pip install -q torch --index-url https://download.pytorch.org/whl/cpu || {
        echo "WARNING: torch install failed (network policy?). The agent will run"
        echo "  in its degraded no-torch fallback; deep-search eval is unavailable."
    }
fi

# ---------------------------------------------------------------------------
# Step 3 — competition data
# ---------------------------------------------------------------------------
mkdir -p data
# Canonical-file check: the comp ships data/main.py (the baseline opponent
# every tournament uses). The repo also tracks data/shot_validator/ (the
# IL-pipeline spec dir), so the older "any non-gitkeep file" heuristic
# always evaluated true on fresh clones and silently skipped the download.
# Friction recurrences: 2026-05-10, 2026-05-12, 2026-05-13.
if [[ -f data/main.py ]]; then
    echo "--- data: data/main.py present, skipping download ---"
else
    echo "--- data: downloading $COMP ---"
    kaggle competitions download -c "$COMP" -p data/ || {
        echo "WARNING: kaggle competitions download failed."
        echo "  Possible causes: comp slug wrong, you haven't accepted the rules"
        echo "  on the comp page yet, or your token doesn't have permission."
    }
    if [[ -f "data/${COMP}.zip" ]]; then
        unzip -qo "data/${COMP}.zip" -d data/
        rm -f "data/${COMP}.zip"
    fi
fi
ls -lh data/ 2>/dev/null | head -10

# ---------------------------------------------------------------------------
# Step 4 — local simulator smoke check
# ---------------------------------------------------------------------------
echo "--- smoke: random-vs-random in kaggle_environments ---"
python - <<'PY' || echo "WARNING: simulator smoke failed; check kaggle-environments install."
from kaggle_environments import make
env = make("orbit_wars", configuration={"seed": 42}, debug=False)
env.run(["random", "random"])
final = env.steps[-1]
print("smoke ok:", [(i, s.reward, s.status) for i, s in enumerate(final)])
PY

# ---------------------------------------------------------------------------
# Step 5 — reference notebooks (optional; deferred)
# ---------------------------------------------------------------------------
# The comp ships its own working baseline (data/main.py — Nearest Planet
# Sniper). Skip external notebook pulls on Day 1; only pull if you hit a
# plateau and want cross-reference. To pull later:
#   kaggle kernels list -s orbit-wars --sort-by voteCount
#   kaggle kernels pull <user>/<slug> -p external/kernels/<slug>/

echo "--- bootstrap done ---"
