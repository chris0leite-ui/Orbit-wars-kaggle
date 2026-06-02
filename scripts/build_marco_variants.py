"""Generate the two phase-4 A/B bundles by patching env vars into the
existing `submissions/baseline.py`:

  - submissions/baseline_marco_default_off.py  → flags explicitly OFF
  - submissions/baseline_marco_on.py            → flags ON

The "default OFF" file is the byte-behaviour anchor: with the new env
vars EXPLICITLY set to "0", the bundle must run identically to
`submissions/baseline.py` (which is itself the default-OFF behaviour
via the chooser_trajectory module reading os.environ.get(...,"0")).

The "marco_on" file is the candidate for the Rule 43/45 gates.

Usage:
    python scripts/build_marco_variants.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

INJECTION_OFF = """\
# === marco-EAM opp model + adversarial rerank — default OFF override ===
import os as _os
_os.environ.setdefault("BASELINE_OPP_MARCO", "0")
_os.environ.setdefault("BASELINE_ADVERSARIAL_RERANK", "0")
# === end marco override ===
"""

INJECTION_ON = """\
# === marco-EAM opp model + adversarial rerank — flags ON ===
import os as _os
_os.environ["BASELINE_OPP_MARCO"] = "1"
_os.environ["BASELINE_ADVERSARIAL_RERANK"] = "1"
# Faster planner budget keeps per-turn wallclock under control. 100ms
# is still well above the 50ms parity-failure floor (84.6% match at
# 250-800ms; partial-plan returns below 50ms).
_os.environ["BASELINE_ADV_RERANK_MARCO_BUDGET_MS"] = "100.0"
# === end marco override ===
"""


def _patch(src: Path, dst: Path, injection: str) -> None:
    text = src.read_text()
    # Inject AFTER the `from __future__ import annotations` line so it's
    # at module top but doesn't clobber the future import.
    marker = "from __future__ import annotations"
    if marker not in text:
        raise RuntimeError(f"{src}: missing __future__ marker")
    idx = text.index(marker) + len(marker)
    out = text[:idx] + "\n\n" + injection + text[idx:]
    dst.write_text(out)


def main() -> int:
    src = REPO / "submissions" / "baseline.py"
    if not src.exists():
        raise FileNotFoundError(f"build the canonical bundle first: {src}")
    off = REPO / "submissions" / "baseline_marco_default_off.py"
    on = REPO / "submissions" / "baseline_marco_on.py"
    _patch(src, off, INJECTION_OFF)
    _patch(src, on, INJECTION_ON)
    print(f"wrote {off.relative_to(REPO)}  ({off.stat().st_size} bytes)")
    print(f"wrote {on.relative_to(REPO)}  ({on.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
