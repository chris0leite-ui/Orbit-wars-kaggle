"""Measure source diversity in top-K portfolios across many turns.

ITEM 4a step 1 (composed-noodling-riddle.md): before adding a
`min_distinct_primary_sources` constraint to
`lib.pipeline.portfolio_enum_lp_seeded.enumerate_top_k_portfolios_lp_seeded`,
empirically check whether top-K portfolios already have distinct
primary sources. If they do (≥80% of turns), the constraint is a
no-op.

Methodology:
  - Play `--n-seeds` games of focal=alpha_beta_on vs opp=alpha_beta_off
    (the current best-known configuration).
  - Instrument the agent's `decision_lagrangian_maximin` to capture
    the top-K portfolios at each turn — or, if maximin is OFF in the
    bundle, instrument `enumerate_top_k_portfolios_lp_seeded` directly.
  - For each turn's K=4 portfolio set, count distinct (src_id of column
    with max ships) — the "primary source" — across the K portfolios.
  - Report a histogram: how often do we see 1, 2, 3, 4 distinct primary
    sources?

Usage:
    python scripts/measure_portfolio_diversity.py --n-seeds 4

Output: per-turn histogram + summary statistics. If ≥80% of turns
have ≥2 distinct primary sources, the constraint is unnecessary.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _primary_source(portfolio: list) -> int | None:
    """The source contributing the most ships in this portfolio.

    Ties broken by min src_id. Empty portfolio returns None.
    """
    if not portfolio:
        return None
    src_ships: dict[int, int] = {}
    for c in portfolio:
        src_ships[int(c.src_id)] = src_ships.get(int(c.src_id), 0) + int(c.ships)
    if not src_ships:
        return None
    max_ships = max(src_ships.values())
    candidates = sorted(s for s, v in src_ships.items() if v == max_ships)
    return candidates[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-seeds", type=int, default=2)
    ap.add_argument("--seed-offset", type=int, default=0)
    ap.add_argument("--k", type=int, default=4,
                    help="K for enumerate_top_k_portfolios_lp_seeded")
    ap.add_argument("--episode-steps", type=int, default=300,
                    help="cap episode length to limit wallclock")
    args = ap.parse_args()

    # Enable α+β features so the portfolio enumeration sees the
    # production decision context.
    import os
    os.environ["LP_TOPOLOGY_FEATURES"] = "1"
    os.environ["LP_SMOOTH_DELTA_W"] = "1"
    os.environ["LP_TOPOLOGY_4P"] = "0"  # 2P-only

    from kaggle_environments import make
    from lib.pipeline import compose
    from lib.pipeline.candidates import candidates_default
    from lib.pipeline.commit_persistent import commit_persistent
    from lib.pipeline.opening import opening_default
    from lib.pipeline.opp_model import opp_greedy_roi
    from lib.pipeline.perception import perception_default
    from lib.pipeline.prerank_passthrough import prerank_passthrough
    from lib.pipeline.decision import decision_outcome_aware_milp
    from lib.pipeline import portfolio_enum_lp_seeded as plps

    # Captured top-K portfolios per turn.
    captures: list[list[list]] = []

    orig_fn = plps.enumerate_top_k_portfolios_lp_seeded

    def _wrap(cols, ctx, opp, **kw):
        kw["k"] = args.k
        portfolios = orig_fn(cols, ctx, opp, **kw)
        captures.append(portfolios)
        return portfolios

    plps.enumerate_top_k_portfolios_lp_seeded = _wrap

    # Build a decision stage that always invokes the portfolio enum
    # (regardless of maximin gate). This is a measurement-only path.
    def _measurement_decision(cols, opp, ctx, **kw):
        # Call the wrapped enum directly (captures the result), then
        # call the actual LP decision so the game proceeds normally.
        if cols.columns:
            try:
                _wrap(cols, ctx, opp)
            except Exception:
                pass
        return decision_outcome_aware_milp(cols, opp, ctx, **kw)

    agent_fn = compose(
        perception=perception_default,
        opening_override=opening_default,
        candidates=candidates_default,
        opp_model=opp_greedy_roi,
        prerank=prerank_passthrough,
        decision=_measurement_decision,
        commit=commit_persistent,
    )

    # Play n_seeds games against itself (self-play; we just want
    # representative game states, not an A/B).
    for s in range(args.seed_offset, args.seed_offset + args.n_seeds):
        env = make("orbit_wars", configuration={"seed": s,
                                                 "episodeSteps": args.episode_steps},
                   debug=False)
        env.run([agent_fn, agent_fn])
        print(f"  seed={s} done — total captures so far: {len(captures)}")

    # Analyze: per-turn count of distinct primary sources.
    histo: Counter[int] = Counter()
    n_turns_with_data = 0
    for turn_portfolios in captures:
        non_empty = [p for p in turn_portfolios if p]
        if not non_empty:
            continue
        n_turns_with_data += 1
        distinct = len({_primary_source(p) for p in non_empty})
        histo[distinct] += 1

    print()
    print(f"=== portfolio diversity ({args.n_seeds} seeds, "
          f"{n_turns_with_data} turns with ≥1 non-empty portfolio) ===")
    if n_turns_with_data == 0:
        print("  no turns had portfolios — measurement failed")
        return 1
    cumul_ge2 = 0
    for k in range(1, 5):
        count = histo.get(k, 0)
        pct = 100.0 * count / n_turns_with_data
        bar = "█" * int(pct / 2.5)
        print(f"  {k} distinct primary sources: {count:4d} turns "
              f"({pct:5.1f}%) {bar}")
        if k >= 2:
            cumul_ge2 += count
    pct_ge2 = 100.0 * cumul_ge2 / n_turns_with_data
    print(f"  ≥2 distinct sources: {cumul_ge2}/{n_turns_with_data} ({pct_ge2:.1f}%)")
    print()
    if pct_ge2 >= 80:
        print("  → Existing diversity ≥80%. The min_distinct_primary_sources "
              "constraint would be a NO-OP. SKIP Item 4a implementation.")
    elif pct_ge2 >= 50:
        print(f"  → Existing diversity {pct_ge2:.0f}% (50-80% band). "
              "Constraint may help on the lower-diversity turns; implement "
              "with default off, measure A/B.")
    else:
        print(f"  → Existing diversity {pct_ge2:.0f}% < 50%. "
              "Constraint is well-motivated; implement.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
