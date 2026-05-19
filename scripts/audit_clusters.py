"""Audit: compare `trajectory_roi`'s first-move pick against the cluster
solver's first-move pick on every cluster in
`audit/tablebase/clusters.jsonl`.

Output: `audit/2026-05-19-tablebase-audit.md` with PASS/PARTIAL/DISAGREE
breakdown and concrete examples of each.

Discrepancy classes:
  - AGREE: heuristic and solver both IDLE, OR both launch at the same
    target with similar sizing (±10% on total_ships).
  - PARTIAL: same target, different sizing (heuristic over- or
    under-allocates).
  - DISAGREE-OVER: heuristic launches; solver IDLE.
  - DISAGREE-UNDER: solver launches; heuristic IDLE.
  - DISAGREE-TARGET: both launch but at different targets.

Caveats logged in the doc:
  - Solver depth is bounded; deep payoffs (production accrual over the
    full K_HORIZON=30) may not be visible. Disagreements should be
    read as "heuristic and bounded-depth solver differ," not
    "heuristic is wrong."
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from agents.trajectory_roi.main import agent as troi_agent          # noqa: E402


JSONL = _REPO / "audit/tablebase/clusters.jsonl"
AUDIT_DOC = _REPO / "audit/2026-05-19-tablebase-audit.md"


def _move_target(move: list, obs: dict) -> int | None:
    """Best-effort: pick the first emit and raycast to a target planet."""
    if not move:
        return None
    src_id, angle, _ships = move[0]
    src = next((p for p in obs["planets"] if p[0] == src_id), None)
    if src is None:
        return None
    sx, sy = src[2], src[3]
    dx, dy = math.cos(angle), math.sin(angle)
    best_id, best_d = None, float("inf")
    for p in obs["planets"]:
        if p[0] == src_id:
            continue
        ex, ey, pr = p[2] - sx, p[3] - sy, p[4]
        proj = ex * dx + ey * dy
        if proj < 0:
            continue
        perp_sq = ex * ex + ey * ey - proj * proj
        if perp_sq >= pr * pr:
            continue
        hit = proj - math.sqrt(pr * pr - perp_sq)
        if hit < best_d:
            best_d, best_id = hit, p[0]
    return best_id


def _classify(heuristic: list, solver: list, obs: dict) -> str:
    h_target = _move_target(heuristic, obs)
    s_target = _move_target(solver, obs)
    if not heuristic and not solver:
        return "AGREE-IDLE"
    if not heuristic and solver:
        return "DISAGREE-UNDER"
    if heuristic and not solver:
        return "DISAGREE-OVER"
    # both launch
    if h_target != s_target:
        return "DISAGREE-TARGET"
    h_ships = sum(m[2] for m in heuristic)
    s_ships = sum(m[2] for m in solver)
    rel_diff = abs(h_ships - s_ships) / max(s_ships, 1)
    if rel_diff <= 0.10:
        return "AGREE-LAUNCH"
    return "PARTIAL-SIZING"


def _render_audit(records: list[dict], buckets: Counter,
                   examples: dict[str, list]) -> str:
    total = sum(buckets.values())
    lines: list[str] = []
    lines.append("# 2026-05-19 — Tablebase Audit (trajectory_roi v3.1 vs cluster solver)")
    lines.append("")
    lines.append("> Phase A.5 deliverable. See "
                 "`/root/.claude/plans/optimized-questing-shell.md`.")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"Total clusters audited: **{total}**")
    lines.append("")
    lines.append("| Class | Count | Rate |")
    lines.append("|---|---|---|")
    for cls in ("AGREE-IDLE", "AGREE-LAUNCH", "PARTIAL-SIZING",
                "DISAGREE-OVER", "DISAGREE-UNDER", "DISAGREE-TARGET"):
        n = buckets.get(cls, 0)
        rate = n / total if total else 0.0
        lines.append(f"| {cls} | {n} | {rate:.0%} |")
    lines.append("")
    lines.append("## What each class means")
    lines.append("")
    lines.append("- **AGREE-IDLE**: both heuristic and solver choose no launch.")
    lines.append("- **AGREE-LAUNCH**: both launch at the same target, total ship count within ±10%.")
    lines.append("- **PARTIAL-SIZING**: both launch at the same target but ship-count differs > 10%.")
    lines.append("- **DISAGREE-OVER**: heuristic launches; solver chooses IDLE.")
    lines.append("- **DISAGREE-UNDER**: solver launches; heuristic chooses IDLE.")
    lines.append("- **DISAGREE-TARGET**: both launch but at different targets.")
    lines.append("")
    lines.append("## Caveats")
    lines.append("")
    lines.append("Solver depth is bounded (typical search depth 6-8). Trajectory_roi's "
                 "value function uses K_HORIZON=30 forward projection. Disagreements may "
                 "reflect bounded-depth solver missing deep payoffs (e.g. production "
                 "accumulated over 20+ post-capture turns) rather than heuristic bugs. "
                 "Higher-confidence audits require either (a) deeper search with shaped "
                 "leaf rewards, or (b) target-specific deeper probes for any "
                 "DISAGREE-OVER cluster the heuristic relies on.")
    lines.append("")
    lines.append("## Examples per class")
    lines.append("")
    for cls in ("DISAGREE-OVER", "DISAGREE-UNDER", "DISAGREE-TARGET",
                "PARTIAL-SIZING", "AGREE-LAUNCH"):
        cases = examples.get(cls, [])
        if not cases:
            continue
        lines.append(f"### {cls}")
        lines.append("")
        for ex in cases[:3]:
            lines.append(f"- replay `{ex['source_replay']}` step={ex['source_step']} "
                         f"seat={ex['source_seat']} planet_ids={ex['planet_ids']} "
                         f"depth={ex['depth_reached']}")
            lines.append(f"  - heuristic: `{ex['heuristic_action']}`")
            lines.append(f"  - solver:    `{ex['best_action']}`  value={ex['value']:.1f}")
        lines.append("")
    return "\n".join(lines)


def audit(jsonl_path: Path, out_path: Path) -> int:
    if not jsonl_path.exists():
        print(f"ERROR: {jsonl_path} not found. Run scripts/mine_clusters.py first.",
              file=sys.stderr)
        return 1
    buckets: Counter = Counter()
    examples: dict[str, list] = {}
    with open(jsonl_path) as f:
        for line in f:
            r = json.loads(line)
            heur = troi_agent(r["isolated_obs"], None)
            cls = _classify(heur, r["best_action"], r["isolated_obs"])
            buckets[cls] += 1
            examples.setdefault(cls, []).append({
                **r,
                "heuristic_action": heur,
            })
    doc = _render_audit([], buckets, examples)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(doc)
    total = sum(buckets.values())
    print(f"Audited {total} clusters → {out_path}")
    for cls, n in buckets.most_common():
        print(f"  {cls}: {n} ({n/total:.0%})")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", type=Path, default=JSONL)
    parser.add_argument("--out", type=Path, default=AUDIT_DOC)
    args = parser.parse_args()
    return audit(args.jsonl, args.out)


if __name__ == "__main__":
    sys.exit(main())
