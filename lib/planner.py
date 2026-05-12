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
    reasons: dict[int, str] | None = None,
) -> list[Intent]:
    """Pick at most one mission per source under a same-turn ledger.

    Algorithm:
    1. Bucket missions by source; sort each bucket by score descending.
    2. Order sources by their top-mission score (highest first).
    3. For each source in order, walk its ranked candidates and accept
       the first one whose target isn't already over-committed by
       prior this-turn picks.
    4. After accepting a mission, register its arrival in the ledger.

    Idle-source tracing (opt-in): pass a `reasons` dict to receive a
    classification of why each non-emitting owned-and-shipped source
    went idle this turn. Keys are planet ids; values are one of:

    - `"NO_PROPOSALS"` — no proposer emitted a Mission for this source.
    - `"LEDGER_LOSS"` — proposer(s) emitted Mission(s) but all were
      skipped because earlier this-turn picks already covered every
      candidate target.

    `MECHANISM_DROP` (intent built but dropped by the realize pipeline)
    is set by `lib.intent.realize`, not here, since this function returns
    before mechanisms run.
    """
    if reasons is None and not missions:
        return []

    by_src: dict[int, list[Mission]] = defaultdict(list)
    for m in missions:
        by_src[m.src_id].append(m)
    # σ-equivariant tie-break + SCORE_ROUND REVERTED for v9 (2026-05-12).
    # v7.6 bisect confirmed: σ-equiv layer regresses v7_0 architecture
    # by ~54pp (6/24 vs v7_0). The layer helps v7_minimax (K=3 maximin)
    # and v4_planner (portfolio) — both have receding-horizon-pathology
    # bugs that σ-equiv's deterministic tie-break partially masks — but
    # in v7_0's drop-one + ship-delta regime it disrupts whatever
    # diversity the rollout selector exploits. See
    # audit/2026-05-12-v4-planner-receding-horizon-pathology.md.
    for src_id in by_src:
        by_src[src_id].sort(key=lambda m: -m.score)

    source_order = sorted(
        by_src.keys(),
        key=lambda s: -by_src[s][0].score,
    )

    # target_id -> list of (eta, ships) for this-turn pending arrivals.
    pending: dict[int, list[tuple[int, int]]] = defaultdict(list)
    chosen: list[Mission] = []
    for src_id in source_order:
        selected = False
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
            selected = True
            break
        if reasons is not None and not selected and by_src[src_id]:
            reasons[src_id] = "LEDGER_LOSS"

    if reasons is not None:
        chosen_srcs = {m.src_id for m in chosen}
        for p in world.planets_by_id.values():
            if p.owner != world.my_id or p.ships <= 0:
                continue
            if p.id in chosen_srcs or p.id in reasons:
                continue
            reasons[p.id] = "NO_PROPOSALS"

    return [m.to_intent() for m in chosen]
