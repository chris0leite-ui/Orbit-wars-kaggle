"""settle_plan — per-source greedy with a same-turn arrival ledger.

v3.1 solver. Each source picks the highest-score mission whose target
isn't already over-committed by earlier-this-turn picks. Two key
invariants:

1. **Allow gang-up when needed.** Multiple sources MAY pick the same
   target if a single source's contribution is insufficient (e.g. when
   the defender's predicted garrison exceeds one source's affordable
   fleet). The first attempt at v3.0 used a blanket no-double-commit
   rule and regressed on dense boards (`audit/2026-05-11-block-e-
   snipe-mvp.md`).
2. **Prevent overcommit when one source is enough.** The earlier v2
   strategy-level greedy couldn't see this-turn launches by other
   sources, so two sources could both pile on a lightly-defended
   target and waste the surplus. The same-turn ledger fixes this:
   after each pick, the mission's `(eta, ships)` is registered, and
   subsequent sources see the cumulative pending arrivals at each
   target. A mission is skipped only if the cumulative arrivals by
   its eta already exceed the predicted enemy garrison + 1 buffer.

The ledger logic is class-agnostic — it works for snipe, reinforce,
and any future mission class that targets a planet. For reinforce
missions, the "target" is OUR planet and the env's combat resolver
adds surplus to garrison (same-owner arrivals don't fight).

Pure function of (missions, world, model). The ledger is rebuilt per
call; no state leaks across turns.
"""

from __future__ import annotations

from collections import defaultdict

from lib.intent import Intent, World
from lib.mission import Mission
from lib.world_model import WorldModel


def settle_plan(
    missions: list[Mission],
    world: World,
    model: WorldModel,
) -> list[Intent]:
    """Pick at most one mission per source under a same-turn ledger.

    Algorithm:
    1. Bucket missions by source; sort each bucket by score descending.
    2. Order sources by their top-mission score (highest first).
    3. For each source in order, walk its ranked candidates and accept
       the first one whose target isn't already over-committed by
       prior this-turn picks.
    4. After accepting a mission, register its arrival in the ledger.
    """
    if not missions:
        return []

    by_src: dict[int, list[Mission]] = defaultdict(list)
    for m in missions:
        by_src[m.src_id].append(m)

    # σ-equivariant tie-break on equal-score targets. Without this, ties
    # default to insertion order (= target.id ascending), which makes
    # σ-paired sources pick the SAME target instead of σ-paired targets.
    # That single-turn asymmetry cascades to elimination over 500 steps;
    # it's the cause of the 19% non-draws in v3-vs-v3 self-play
    # (audit/2026-05-11-cannot-lose-final-finding.md).
    #
    # Key: -(src.x - CENTER) * (target.x - CENTER). σ negates both
    # factors → product invariant → σ-paired (src, target) get the
    # same key. Within a source's tied targets, T and σ(T) get
    # opposite-sign keys → consistent σ-equivariant choice.
    # Falls back to y-axis product when degenerate (planets on x=50 axis),
    # then target.id for full determinism.
    def _tb(m: Mission):
        src = world.planets_by_id.get(m.src_id)
        tgt = world.planets_by_id.get(m.target_id)
        if src is None or tgt is None:
            return (0.0, 0.0, m.target_id)
        kx = (src.x - 50.0) * (tgt.x - 50.0)
        ky = (src.y - 50.0) * (tgt.y - 50.0)
        return (-kx, -ky, m.target_id)

    for src_id in by_src:
        by_src[src_id].sort(key=lambda m: (-m.score, _tb(m)))

    source_order = sorted(
        by_src.keys(),
        key=lambda s: (-by_src[s][0].score, _tb(by_src[s][0])),
    )

    # target_id -> list of (eta, ships) for this-turn pending arrivals.
    pending: dict[int, list[tuple[int, int]]] = defaultdict(list)
    chosen: list[Mission] = []
    for src_id in source_order:
        for m in by_src[src_id]:
            # Ships our prior this-turn picks have committed to land at
            # m.target_id by step m.eta (or earlier). A defender at
            # T_loss < m.eta will already be neutralised by earlier
            # arrivals; we only need to add to that pool.
            already = sum(
                s for (e, s) in pending[m.target_id] if e <= m.eta
            )
            pred_enemy = model.ships_at(m.target_id, m.eta)
            # For our own planets (reinforce target), the planet may
            # never be "enemy-held" at our arrival; pred_enemy is just
            # the garrison total. We use it as the "size needed to
            # contest" — if our prior picks already supply that much,
            # any additional ships are surplus.
            if pred_enemy is None:
                pred_enemy = 0.0
            # Skip when our prior this-turn picks already exceed
            # (enemy garrison + 1 buffer). The +1 matches the snipe /
            # reinforce ship-sizing convention.
            if already >= pred_enemy + 1:
                continue
            chosen.append(m)
            pending[m.target_id].append((m.eta, m.ships))
            break

    return [m.to_intent() for m in chosen]
