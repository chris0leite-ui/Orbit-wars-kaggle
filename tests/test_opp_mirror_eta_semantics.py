"""Bug #2 — opp_mirror_analytical eta-semantics mismatch.

`lib/pipeline/opp_mirror_analytical.py:_columns_to_arrivals` returns
arrivals with absolute etas (`step_now + wait_N + eta`). The same
arrivals are then merged into the ledger via `merge_ledgers` and passed
to `simulate_planet_timeline`, which expects **relative** etas (per
`lib/world_model.py:171-188` and the pin at
`tests/test_world_model.py:125-134`: `bucket = max(1, int(ceil(eta)))`).

Effect: with step_now=50, an opp's planned arrival at relative step 8
(5 wait + 3 flight) is bucketed at absolute step 58 of the timeline.
If horizon=50 the arrival is silently dropped; otherwise it's recorded
at the wrong time. Stackelberg-leader's "opp best response" is therefore
computed against a timeline that doesn't reflect opp's projected
arrivals correctly.

Pin (Rule 38): pre-fix this test asserts the buggy absolute value; after
the fix it asserts the correct relative value. The test name says what
the CORRECT behaviour is; pre-fix it fails.
"""

from __future__ import annotations


def _make_synthetic_column(*, src_id, tgt_id, ships, wait_N, eta, angle=0.0,
                           owner=0, value=1.0, column_id=0):
    from lib.joint_solver.columns import Column
    return Column(
        column_id=column_id, src_id=src_id, tgt_id=tgt_id,
        ships=ships, wait_N=wait_N, angle=angle, eta=eta,
        owner=owner, value=value,
    )


def test_columns_to_arrivals_emits_relative_etas():
    """`_columns_to_arrivals` must produce relative etas (wait_N + eta),
    not absolute (step_now + wait_N + eta), so downstream
    `simulate_planet_timeline` buckets the arrival at the intended tick.

    Failure mode pre-fix: with step_now=50, the function returns
    eta_out=58 (absolute). simulate_planet_timeline would then bucket
    the arrival at step 58 of the timeline instead of step 8.
    """
    from lib.pipeline.opp_mirror_analytical import _columns_to_arrivals

    col = _make_synthetic_column(
        src_id=0, tgt_id=5, ships=10, wait_N=5, eta=3,
    )
    step_now = 50
    arrivals = _columns_to_arrivals([col], step_now=step_now, my_id=0)

    assert len(arrivals) == 1
    pid, eta_out, owner, ships = arrivals[0]
    assert pid == 5
    assert owner == 0
    assert ships == 10
    # Contract: eta must be RELATIVE-from-step_now. The expected value
    # is wait_N + eta = 8. Buggy value (absolute) is step_now + wait_N
    # + eta = 58.
    assert eta_out == 8, (
        f"_columns_to_arrivals returned eta_out={eta_out}; expected 8 "
        f"(relative). If this is 58, the contract is absolute-etas — "
        f"merge_ledgers passes them to simulate_planet_timeline which "
        f"will bucket at the wrong (off-by-step_now) tick or drop them "
        f"entirely if past horizon."
    )


def test_columns_to_arrivals_zero_ships_filtered():
    """Sanity guard: zero-ship columns are skipped (pre-existing logic;
    the fix must not regress this)."""
    from lib.pipeline.opp_mirror_analytical import _columns_to_arrivals

    cols = [
        _make_synthetic_column(src_id=0, tgt_id=5, ships=0, wait_N=2, eta=3),
        _make_synthetic_column(src_id=0, tgt_id=6, ships=7, wait_N=1, eta=2),
    ]
    arrivals = _columns_to_arrivals(cols, step_now=10, my_id=0)
    assert len(arrivals) == 1
    assert arrivals[0][0] == 6  # only the non-zero-ship column survives


def test_columns_to_arrivals_zero_wait_relative():
    """wait_N=0 should yield eta_out == eta (relative-from-step_now)."""
    from lib.pipeline.opp_mirror_analytical import _columns_to_arrivals

    col = _make_synthetic_column(src_id=0, tgt_id=5, ships=10, wait_N=0, eta=7)
    arrivals = _columns_to_arrivals([col], step_now=42, my_id=0)
    _pid, eta_out, _owner, _ships = arrivals[0]
    assert eta_out == 7, (
        f"wait_N=0 + eta=7 → expected eta_out=7 (relative). Got {eta_out}. "
        f"Buggy absolute would be 42+0+7 = 49."
    )
