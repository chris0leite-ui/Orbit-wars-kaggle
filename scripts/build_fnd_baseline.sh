#!/usr/bin/env bash
#
# build_fnd_baseline.sh — produce submissions/_phase4_step1_FND.py as the
# no-topology comparison baseline for the topology A/B (Phase β).
#
# Source: submissions/analytical_phase_c.py (must be built first via
# `python scripts/bundle_analytical_phase_c.py`).
#
# Transformation: replace the `LP_TOPOLOGY_FEATURES=1` default with
# `LP_TOPOLOGY_FEATURES=0`. Every other code path is identical, so the
# two bundles isolate exactly the topology axis when run head-to-head.
#
# Re-run this each fresh container.

set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$REPO/submissions/analytical_phase_c.py"
DST="$REPO/submissions/_phase4_step1_FND.py"

if [ ! -f "$SRC" ]; then
    echo "ERROR: $SRC not found. Run 'python scripts/bundle_analytical_phase_c.py' first." >&2
    exit 1
fi

# Replace the setdefault for LP_TOPOLOGY_FEATURES from "1" to "0".
# Leave LP_REACH_BONUS / LP_DEFENSE_BONUS / LP_FRONT_PENALTY at their
# defaults — they're gated by _TOPOLOGY_FEATURES_ENABLED anyway, so
# disabling the top-level toggle disables all three.
sed 's|os.environ.setdefault("LP_TOPOLOGY_FEATURES", "1")|os.environ.setdefault("LP_TOPOLOGY_FEATURES", "0")|' \
    "$SRC" > "$DST"

# Verify the substitution actually happened (sed silently produces a
# byte-identical copy if the pattern doesn't match — guard against that).
if ! grep -q 'os.environ.setdefault("LP_TOPOLOGY_FEATURES", "0")' "$DST"; then
    echo "ERROR: substitution failed — $DST still has LP_TOPOLOGY_FEATURES=1" >&2
    rm -f "$DST"
    exit 1
fi

# Verify loadable + agent symbol present.
python - <<EOF
import importlib.util
spec = importlib.util.spec_from_file_location("m", "$DST")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
assert callable(getattr(m, "agent", None)), "no callable agent in $DST"
print(f"OK: $DST loads, agent callable, $(wc -c < "$DST") bytes")
EOF
