#!/bin/bash
# session-start.sh — Orbit-wars-kaggle session readiness gate.
#
# Origin: claude/audit-workflow-friction-XD56a (2026-05-14). Three same-class
# friction incidents in two days where a written rule was skipped:
#   - agent-introspection-skipped-bootstrap     (5/13)
#   - handover-stale-at-session-start-no-git-log-check  (5/13)
#   - fix-not-validated-against-real-failing-state      (5/14)
#
# Behavioural rules (CLAUDE.md Rule 32 + Rule 38) cover the *what*; this hook
# enforces the *when* so the next session can't start work on a half-set-up
# repo and dismiss the resulting test failures as "pre-existing."
#
# Steps (in order, sync):
#   1. git fetch origin + log -5 HEAD     (Rule 32)
#   2. bash bootstrap.sh                  (closes the data-main-py-missing
#                                          + kaggle-cli-401 + pip-blinker
#                                          friction tags)
#   3. simulator import smoke             (proves the env loads)
#
# Pytest is NOT in this hook (~12 min full run). The hook prints a hint
# banner; the agent runs pytest on demand per Rule 38.

set +e  # diagnostic prints, but never block session start
cd "${CLAUDE_PROJECT_DIR:-$(git rev-parse --show-toplevel 2>/dev/null)}" || exit 0

# Only run in the web/remote sandbox by default. Local dev gets a no-op.
# Override with FORCE_LOCAL_HOOK=1 to run locally for testing.
if [[ "${CLAUDE_CODE_REMOTE:-}" != "true" && "${FORCE_LOCAL_HOOK:-}" != "1" ]]; then
    exit 0
fi

echo "=== Orbit Wars — session-start hook ==="
echo

# --- Step 1: git state (Rule 32) ----------------------------------------
echo "--- git: fetch origin + HEAD log ---"
if git fetch origin --quiet 2>&1; then
    branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")
    ahead=$(git rev-list --count origin/main..HEAD 2>/dev/null || echo "?")
    behind=$(git rev-list --count HEAD..origin/main 2>/dev/null || echo "?")
    echo "  branch: $branch  (ahead $ahead / behind $behind  origin/main)"
    git log -5 --oneline HEAD 2>/dev/null | sed 's/^/  /'
else
    echo "  WARN: git fetch failed (offline?). Branch state may be stale."
fi
echo

# --- Step 1b: kaggle CLI wrapper (auth persists in every Bash shell) ----
# Friction: 2026-05-17 — bootstrap.sh exports KAGGLE_API_TOKEN in its own
# shell only; subsequent Bash tool calls come up with no token and the
# real CLI 401s. KGAT_-prefix tokens can't live in ~/.kaggle/kaggle.json
# (legacy 32-hex auth path returns 401). Install a thin shim on PATH that
# derives KAGGLE_API_TOKEN from the harness env vars on every invocation.
#
# 2026-05-24 update: the harness may surface the KGAT_ token under any of
# {KAGGLE_API_TOKEN, KaggleAPIToke, KAGGLE_KEY}. Earlier guard only matched
# KaggleAPIToke; sessions where the token came in via KAGGLE_KEY skipped
# the install and 401'd. Treat any KGAT_-prefix sighting as a trigger,
# and (re)write the shim each run so body fixes propagate to old containers.
_KAGGLE_SHIM="$HOME/.local/bin/kaggle"
_KAGGLE_TOKEN_RAW="${KAGGLE_API_TOKEN:-${KaggleAPIToke:-${KAGGLE_KEY:-}}}"
if [[ "$_KAGGLE_TOKEN_RAW" == KGAT_* ]]; then
    echo "--- kaggle CLI shim: writing $_KAGGLE_SHIM ---"
    mkdir -p "$(dirname "$_KAGGLE_SHIM")"
    cat > "$_KAGGLE_SHIM" <<'SHIM'
#!/bin/bash
# Auto-installed by .claude/hooks/session-start.sh — see hook for context.
# KGAT_-prefix tokens must route through KAGGLE_API_TOKEN, not the legacy
# KAGGLE_KEY env path; the CLI 401s otherwise. Derive the token from
# whichever env var the harness populated this session.
_t="${KAGGLE_API_TOKEN:-${KaggleAPIToke:-${KAGGLE_KEY:-}}}"
if [[ "$_t" == KGAT_* ]]; then
    export KAGGLE_API_TOKEN="$_t"
    unset KAGGLE_KEY  # belt-and-braces: kill legacy-auth fallback path
fi
export KAGGLE_USERNAME="${KAGGLE_USERNAME:-${KaggleUserName:-}}"
exec /usr/local/bin/kaggle "$@"
SHIM
    chmod +x "$_KAGGLE_SHIM"
fi

# --- Step 2: bootstrap (data + creds + deps) ----------------------------
echo "--- bootstrap.sh ---"
if [[ -x bootstrap.sh ]]; then
    # Filter the kaggle_environments OpenSpiel banner (23+ lines of noise on
    # every import). Do NOT pipe through tail — truncation has hidden real
    # errors (kaggle auth tracebacks, pip install failures) in past sessions.
    # The filtered bootstrap output is ~12-15 lines normally, ~30 with errors;
    # both are fine to show in full.
    if bash bootstrap.sh 2>&1 | grep -v "open_spiel_env"; then
        :
    else
        echo "  WARN: bootstrap.sh exited non-zero. Investigate before"
        echo "        running tests or building agents."
    fi
else
    echo "  WARN: bootstrap.sh missing or not executable."
fi
echo

# --- Step 3: simulator import smoke (1-2s) ------------------------------
echo "--- simulator smoke (import only) ---"
python - <<'PY' 2>&1 | grep -v "open_spiel_env" | tail -5
import sys
try:
    from kaggle_environments import make  # noqa
    import pytest  # noqa
    from pathlib import Path
    data_main = Path("data/main.py")
    print(f"  kaggle_environments + pytest: import OK")
    print(f"  data/main.py present: {data_main.is_file()}")
except Exception as e:
    print(f"  IMPORT FAILED: {e}", file=sys.stderr)
    sys.exit(1)
PY
echo

# --- Hint banner for the agent ------------------------------------------
echo "=== session ready ==="
echo "  Rule 38 (CLAUDE.md): fix-verification reproduces failure state."
echo "  Full test baseline (12 min):  python -m pytest tests/ -q --tb=line"
echo "  Fast test sample (<10s):       python -m pytest tests/test_geometry.py -q"
echo

exit 0
