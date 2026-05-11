#!/usr/bin/env bash
# Pack agents/precision/ as a Kaggle submission tar.gz.
#
# Output:
#   submissions/precision_v2/                 — flattened, sibling-relative imports
#   submissions/precision_v2.tar.gz           — tarball ready for `kaggle competitions submit`
#
# Idempotent. Re-run after any change to agents/precision/.

set -euo pipefail

REPO="$(cd "$(dirname "$0")"/.. && pwd)"
SRC="$REPO/agents/precision"
DST="$REPO/submissions/precision_v2"
TAR="$REPO/submissions/precision_v2.tar.gz"

mkdir -p "$DST"
rm -f "$DST"/*.py "$TAR"

MODULES=(main sim intercept prediction planner scoring enemy_model bundling)

for m in "${MODULES[@]}"; do
    cp "$SRC/$m.py" "$DST/$m.py"
done

# Rewrite package-style imports to absolute-sibling.
# `from agents.precision import X, Y` -> `import X` and `import Y` (one per name).
python3 - "$DST" <<'PY'
import os, re, sys
dst = sys.argv[1]
pat = re.compile(r'^from agents\.precision import (.+)$', re.MULTILINE)
def replace(m):
    names = [n.strip() for n in m.group(1).split(',')]
    return '\n'.join(f'import {n}' for n in names)
for f in os.listdir(dst):
    if not f.endswith('.py'): continue
    p = os.path.join(dst, f)
    s = open(p).read()
    s = pat.sub(replace, s)
    open(p, 'w').write(s)
PY

# Rewrite main.py wholesale: Kaggle imports this as a top-level module, so we
# need sys.path setup + absolute sibling imports.
cat > "$DST/main.py" <<'EOF'
"""Precision agent entry point (Kaggle submission)."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import intercept
import planner


def agent(obs):
    """Return a list of [src_id, angle, ships] launches."""
    t0 = time.perf_counter()
    try:
        world = intercept.parse_world(obs)
    except Exception:
        return []
    if world["step"] == 0:
        return []
    deadline = t0 + 0.85
    try:
        plan = planner.plan_turn(world, deadline=deadline)
    except Exception:
        plan = []
    return planner.emit_actions(plan)
EOF

# Build the tarball with main.py at archive root alongside siblings.
( cd "$DST" && tar czf "$TAR" --exclude='__pycache__' --exclude='__init__.py' *.py )

echo "Built: $TAR"
ls -la "$TAR"
