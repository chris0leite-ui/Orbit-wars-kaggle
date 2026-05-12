"""Candidate mission portfolios for v4_planner's lookahead loop.

Each portfolio is a list of `Mission` objects fed through the standard
`settle_plan` → `realize` pipeline. v3_lookahead's drop-one operated on
realized actions and yielded only subsets of v3_snipe's choices — the
post-mortem (audit/2026-05-11-v3-lookahead-mvp-parity.md) identified
that narrowness as the reason it tied at 50% Wilson. Portfolios let us
propose *different* mission compositions per turn, which the Sim<K>
scorer then ranks via the goal-shaped value function.

Five portfolios, in priority order so a time-budget abort still falls
back to the v3.5.1 incumbent (always scored first):

1. **incumbent**: snipe(aggressive=True) + reinforce — the v3.5.1 stack.
2. **conservative**: snipe(aggressive=False) + reinforce — re-decides
   per turn whether fat (top-10 fingerprint) or minimum-viable sizing
   is right for *this* state. v3.5.1 always-aggressive may be wrong on
   some boards where conservative wins; lookahead can pick per-turn.
3. **per_source_swap**: incumbent missions with the top-1 dropped for
   the source whose top-1 vs top-2 score gap is smallest — the closest
   "next best" alternative for v3_snipe's per-source greedy.
4. **drop_weakest_source**: incumbent missions with ALL missions from
   the source whose best score was the lowest filtered out — that
   source idles this turn. Tests "is this source's launch actually
   net-positive in the rollout?" (Phase 2 audit's drop-one logic.)
5. **noop**: empty mission list. Always safe; useful when every
   alternative looks net-negative in the rollout.

Duplicates are de-duplicated by the realized-action signature in the
agent loop — not here — so this module stays pure (no env touched).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from lib.intent import World
from lib.mission import Mission
from lib.missions.reinforce import propose_reinforce_missions
from lib.missions.snipe import propose_snipe_missions
from lib.world_model import WorldModel


@dataclass
class Portfolio:
    """A labelled mission list to be ranked by the Sim<K> scorer."""

    label: str
    missions: list[Mission]


def _incumbent_missions(world: World, model: WorldModel) -> list[Mission]:
    """v3.5.1's mission set — aggressive snipe + reinforce."""
    return (
        propose_snipe_missions(world, model, aggressive=True)
        + propose_reinforce_missions(world, model)
    )


def _conservative_missions(world: World, model: WorldModel) -> list[Mission]:
    """v3_snipe's mission set — minimum-viable snipe + reinforce."""
    return (
        propose_snipe_missions(world, model, aggressive=False)
        + propose_reinforce_missions(world, model)
    )


def _per_source_swap(missions: list[Mission]) -> list[Mission] | None:
    """Drop top-1 for the source with the smallest top-1 / top-2 score gap.

    Returns None if no source has at least 2 missions (no swap possible)
    or if there are no missions at all.
    """
    if not missions:
        return None
    by_src: dict[int, list[Mission]] = defaultdict(list)
    for m in missions:
        by_src[m.src_id].append(m)
    # Sort each bucket high → low by score, find the source with
    # the smallest (top1 - top2) gap.
    candidates = []
    for src_id, bucket in by_src.items():
        if len(bucket) < 2:
            continue
        bucket.sort(key=lambda m: -m.score)
        gap = bucket[0].score - bucket[1].score
        candidates.append((gap, src_id, bucket[0]))
    if not candidates:
        return None
    # Smallest gap = most-marginal greedy decision.
    _gap, swap_src, top1 = min(candidates, key=lambda t: t[0])
    return [m for m in missions if not (m.src_id == swap_src and m is top1)]


def _drop_weakest_source(missions: list[Mission]) -> list[Mission] | None:
    """Drop ALL missions from the source whose best score is the lowest.

    Equivalent to "idle the weakest source this turn." Returns None if
    there are fewer than 2 sources (dropping the only source = no-op
    duplicate, the noop portfolio covers it).
    """
    if not missions:
        return None
    by_src: dict[int, list[Mission]] = defaultdict(list)
    for m in missions:
        by_src[m.src_id].append(m)
    if len(by_src) < 2:
        return None
    # Best score per source; weakest is the one whose best is lowest.
    weakest_src = min(
        by_src.keys(), key=lambda s: max(m.score for m in by_src[s])
    )
    return [m for m in missions if m.src_id != weakest_src]


def _drop_smallest_launch(missions: list[Mission]) -> list[Mission] | None:
    """Drop the single mission with the fewest ships (across all sources).

    Different from `_drop_weakest_source` (which drops ALL missions from
    one source). This drops exactly one mission — the one with the
    smallest ship count — wherever it sits. Captures "hold the most
    marginal ship contribution back" — a defensive perturbation that
    v4.5_robust's opp-model O1 used. Ties broken by lowest src_id then
    target_id for σ-determinism.

    Returns None if there are fewer than 2 missions (dropping the only
    one = noop, the noop portfolio covers it).
    """
    if not missions or len(missions) < 2:
        return None
    # Sort by (ships, src_id, target_id) ascending; first is smallest.
    sorted_idx = sorted(
        range(len(missions)),
        key=lambda i: (missions[i].ships, missions[i].src_id, missions[i].target_id),
    )
    drop_idx = sorted_idx[0]
    return [m for i, m in enumerate(missions) if i != drop_idx]


def generate_portfolios(
    world: World,
    model: WorldModel,
    incumbent_missions: list[Mission] | None = None,
    *,
    include_drop_smallest: bool = False,
) -> list[Portfolio]:
    """Build ≤ 5 (or ≤ 6 with `include_drop_smallest`) mission portfolios.

    `incumbent_missions` may be passed in if the caller already built
    them (avoiding a duplicate proposer call); otherwise this rebuilds
    them. The incumbent is always portfolios[0] so the scorer's "score
    incumbent first" loop has a safe fallback.

    `include_drop_smallest` (default False; bit-identical to v4_planner)
    adds a 6th portfolio that drops the single mission with the fewest
    ships. Lesson 1 from the v4.5_robust postmortem: a fine-grained
    "drop the most marginal launch" perturbation is different from
    `drop_weakest_source` and meaningfully expands portfolio diversity
    at one extra Sim call per turn.
    """
    incumbent = (
        incumbent_missions
        if incumbent_missions is not None
        else _incumbent_missions(world, model)
    )
    portfolios: list[Portfolio] = [Portfolio("incumbent", incumbent)]

    conservative = _conservative_missions(world, model)
    # Only add conservative if it differs in at least one ship count from
    # incumbent — otherwise it would settle to the same action.
    if _missions_differ(conservative, incumbent):
        portfolios.append(Portfolio("conservative", conservative))

    swap = _per_source_swap(incumbent)
    if swap is not None and swap != incumbent:
        portfolios.append(Portfolio("per_source_swap", swap))

    drop_weak = _drop_weakest_source(incumbent)
    if drop_weak is not None and drop_weak != incumbent:
        portfolios.append(Portfolio("drop_weakest_source", drop_weak))

    if include_drop_smallest:
        drop_smallest = _drop_smallest_launch(incumbent)
        if drop_smallest is not None and drop_smallest != incumbent:
            # Also dedupe against any prior portfolio that happens to
            # produce the same mission set (e.g. per_source_swap on a
            # 2-mission source could collide).
            sigs = {
                tuple(sorted((m.src_id, m.target_id, m.ships) for m in p.missions))
                for p in portfolios
            }
            ds_sig = tuple(sorted(
                (m.src_id, m.target_id, m.ships) for m in drop_smallest
            ))
            if ds_sig not in sigs:
                portfolios.append(Portfolio("drop_smallest_launch", drop_smallest))

    portfolios.append(Portfolio("noop", []))
    return portfolios


def _missions_differ(a: list[Mission], b: list[Mission]) -> bool:
    """Cheap structural compare on (src, target, ships) keys.

    score / mission_class metadata is irrelevant for whether settle_plan
    would emit a different action — only the launch tuple matters.
    """
    def key(ms: list[Mission]):
        return sorted((m.src_id, m.target_id, m.ships) for m in ms)
    return key(a) != key(b)
