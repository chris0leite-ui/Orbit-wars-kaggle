"""Guard: wider candidate generation (LR_WIDE_CANDIDATES) — the breadth lever.

These exercise the score-free plan builders directly on synthetic candidate dicts /
planet rows, so they're torch-AGNOSTIC (no orbit_lite leaf needed). The deep search
does the real scoring elsewhere; here we only check that generation produces a
diverse, de-duped, bounded pool and that holdability ordering / reach are correct.
"""
import importlib.util
import os

_MAIN = os.path.join(os.path.dirname(__file__), "..", "agents",
                     "least_resistance", "main.py")


def _load_main():
    spec = importlib.util.spec_from_file_location("lr_main_wide_test", _MAIN)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _p(pid, owner, x, ships, y=0.0, radius=0.0, prod=1):
    return [pid, owner, float(x), float(y), float(radius), float(ships), prod]


def _cand(tid, srcs, emit, rank, front=0.0, kind="neutral"):
    return {"emit": emit, "units": [], "srcs": srcs, "rank": rank,
            "front": front, "kind": kind, "tid": tid}


# ---- _reachable_rival_mass ------------------------------------------------

def test_reachable_rival_mass_max_not_sum():
    mod = _load_main()
    os.environ.pop("LR_CONTAGION_REACH_TICKS", None)
    target = _p(9, -1, x=0.0, ships=5)
    planets = [
        target,
        _p(0, 1, x=-2.0, ships=30),   # rival A reachable
        _p(1, 2, x=2.0, ships=40),    # rival B reachable, stronger
    ]
    # strongest single reachable rival = 40 (NOT 70 = the sum)
    assert mod._reachable_rival_mass(target, planets, me=0) == 40.0


def test_reachable_rival_mass_none_in_reach():
    mod = _load_main()
    target = _p(9, 0, x=0.0, ships=5)          # mine
    planets = [target, _p(0, 1, x=100000.0, ships=999)]
    assert mod._reachable_rival_mass(target, planets, me=0) == 0.0


# ---- _greedy_commit_cheap -------------------------------------------------

def test_greedy_commit_cheap_respects_budget_and_order():
    mod = _load_main()
    available = {0: 10}
    # two captures from the same source, each needs 8 -> only the first fits.
    c_hi = _cand(1, {0: 8}, [[0, 0.0, 8]], rank=9.0)
    c_lo = _cand(2, {0: 8}, [[0, 1.0, 8]], rank=1.0)
    order = lambda c: -c["rank"]               # highest rank first
    emit = mod._greedy_commit_cheap([c_lo, c_hi], order, available)
    assert emit == [[0, 0.0, 8]], "should commit the high-rank capture, source then exhausted"


def test_greedy_commit_cheap_max_captures():
    mod = _load_main()
    available = {0: 100, 1: 100, 2: 100}
    cands = [_cand(i, {i: 5}, [[i, 0.0, 5]], rank=float(10 - i)) for i in range(3)]
    order = lambda c: -c["rank"]
    emit = mod._greedy_commit_cheap(cands, order, available, max_captures=2)
    assert len(emit) == 2, "max_captures must bound the plan size"


# ---- _wide_candidate_plans ------------------------------------------------

def _scenario(mod):
    # me=0. Two affordable neutrals + one enemy, from distinct sources.
    planets = [
        _p(0, 0, x=0.0, ships=100),    # my source A
        _p(1, 0, x=1.0, ships=100),    # my source B
        _p(2, -1, x=2.0, ships=3),     # neutral near
        _p(3, -1, x=40.0, ships=3),    # neutral far (holdable)
        _p(4, 1, x=2.5, ships=3),      # enemy near my sources
        _p(5, 1, x=2.4, ships=200),    # strong rival near the near targets
    ]
    by_id = {row[0]: row for row in planets}
    cands = [
        _cand(2, {0: 5}, [[0, 0.0, 5]], rank=5.0, kind="neutral"),
        _cand(3, {1: 5}, [[1, 0.1, 5]], rank=4.0, kind="neutral"),
        _cand(4, {0: 6}, [[0, 0.2, 6]], rank=6.0, kind="enemy"),
    ]
    available = {0: 100, 1: 100}
    return cands, planets, by_id, available


def test_wide_pool_diverse_deduped_capped():
    mod = _load_main()
    cands, planets, by_id, available = _scenario(mod)
    producer_me = [[0, 0.9, 7]]
    committed = [[0, 0.0, 5], [1, 0.1, 5]]
    pool = mod._wide_candidate_plans(cands, producer_me, committed, available,
                                     planets, by_id, me=0, max_plans=12)
    assert producer_me in pool, "producer floor must be in the pool"
    assert [] in pool, "hold-everything must be in the pool"
    assert len(pool) >= 3, "pool should offer several distinct plans"
    assert len(pool) <= 12
    # no duplicate canonical reprs
    keys = [repr(sorted(p, key=lambda e: (int(e[0]), round(float(e[1]), 3), int(e[2]))))
            for p in pool]
    assert len(keys) == len(set(keys)), "pool must be de-duped by canonical plan"


def test_wide_pool_canonical_dedup_collapses_reordered():
    mod = _load_main()
    cands, planets, by_id, available = _scenario(mod)
    # producer_me equal to committed but launches in a different order -> must collapse.
    committed = [[0, 0.0, 5], [1, 0.1, 5]]
    producer_me = [[1, 0.1, 5], [0, 0.0, 5]]
    pool = mod._wide_candidate_plans(cands, producer_me, committed, available,
                                     planets, by_id, me=0)
    canon = lambda p: repr(sorted(p, key=lambda e: (int(e[0]), round(float(e[1]), 3), int(e[2]))))
    assert canon(producer_me) == canon(committed)
    assert sum(1 for p in pool if canon(p) == canon(committed)) == 1, \
        "reorder-equivalent plans must appear once"


def test_holdability_order_prefers_unthreatened_target():
    mod = _load_main()
    cands, planets, by_id, available = _scenario(mod)
    # tid 3 (far neutral) is unthreatened; tid 2 (near neutral) sits next to a 200-ship rival.
    near = mod._reachable_rival_mass(by_id[2], planets, me=0)
    far = mod._reachable_rival_mass(by_id[3], planets, me=0)
    assert near > far, "the near target must read as more threatened than the far one"
