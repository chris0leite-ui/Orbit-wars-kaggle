"""Stage 4 — opp portfolio set via perturbations of greedy-ROI baseline.

Phase D MVP: instead of building a true mirror-analytical opp (recursive,
expensive), we derive a small set of opp portfolio variants from
`predict_opp_multi_launch`'s output. The variants give the maximin
decision rule a non-trivial opp strategy space without the recursion
cost.

The variants (default 4):
  1. greedy_baseline:  the full greedy-ROI projection (current default)
  2. drop_largest:     remove the single largest-ship launch
  3. drop_smallest:    remove the single smallest-ship launch
  4. idle:             no opp launches (lower-bound on opp aggression)

For maximin's purposes, this gives the decision rule a small but
diverse opp space. Drop-largest models "opp underestimates this turn";
drop-smallest models "opp is more aggressive"; idle is the floor.

If the greedy baseline produces < 2 arrivals, the perturbations
collapse and we just return [greedy_baseline, idle].

Future: replace with true mirror-analytical opp (opp solves its own
analytical pipeline from its POV given my candidate action). Will
register here under a different name; this MVP stays as a fallback.
"""

from __future__ import annotations

from lib.joint_solver.opp_projection import predict_opp_multi_launch

from lib.pipeline.types import TurnContext


def opp_portfolios_perturbations(
    ctx: TurnContext, *, max_portfolios: int = 4,
) -> list[list[tuple[int, int, int, int]]]:
    """Return a list of opp arrival portfolios.

    Each portfolio is `list[(target_pid, eta_absolute, opp_owner, ships)]`
    matching `predict_opp_multi_launch`'s output shape.

    Always includes the idle (empty) portfolio as a baseline.
    """
    try:
        greedy = list(predict_opp_multi_launch(
            ctx.world, int(ctx.me), int(ctx.num_seats),
        ))
    except Exception:
        greedy = []

    portfolios: list[list[tuple]] = [list(greedy)]   # 1: greedy baseline (may be empty)
    seen_keys = {tuple(sorted((int(p), int(e), int(o), int(s)) for (p, e, o, s) in greedy))}

    def _add(p: list[tuple], _label: str) -> None:
        key = tuple(sorted((int(pp), int(ee), int(oo), int(ss)) for (pp, ee, oo, ss) in p))
        if key in seen_keys:
            return
        if len(portfolios) >= max_portfolios:
            return
        seen_keys.add(key)
        portfolios.append(p)

    # 2: drop_largest — remove the single launch with the most ships
    if greedy:
        idx_largest = max(range(len(greedy)), key=lambda i: int(greedy[i][3]))
        _add(
            [a for i, a in enumerate(greedy) if i != idx_largest],
            "drop_largest",
        )

    # 3: drop_smallest — remove the single launch with the fewest ships
    if len(greedy) >= 2:
        idx_smallest = min(range(len(greedy)), key=lambda i: int(greedy[i][3]))
        _add(
            [a for i, a in enumerate(greedy) if i != idx_smallest],
            "drop_smallest",
        )

    # 4: idle — no opp launches at all
    _add([], "idle")

    return portfolios[:max_portfolios]
