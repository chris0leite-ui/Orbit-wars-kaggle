"""Dropout-native mean-field flip-hazard forward model (Phase A heart).

Pins the load-bearing claims of agents/dropout_native/forward_model.py:
  1. flip_prob is a steep-near-parity contest curve (monotone right ways).
  2. reachable_enemy_mass is cumulative in k and counts only enemy mass.
  3. build_candidate_trajectories reduces EXACTLY to the trusted engine
     recurrence when no launches are applied (construction parity).
  4. THE THESIS: a holdable capture (thick garrison vs reachable enemy mass)
     scores strictly higher than a thin, recapturable one — the property the
     2-point bolt-on could only approximate.
  5. Determinism: identical inputs -> identical output (no RNG).
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "agents" / "producer"))
sys.path.insert(0, str(REPO))

from orbit_lite.garrison_launch import _run_exact_recurrence  # noqa: E402
from orbit_lite.native_forward import (  # noqa: E402
    build_candidate_trajectories,
    flip_prob,
    hazard_ownership_value,
    reachable_enemy_mass,
    score_candidates_native,
)


def test_flip_prob_monotone_and_parity():
    deff = torch.tensor([10.0, 10.0, 10.0])
    atk = torch.tensor([0.0, 10.0, 100.0])
    f = flip_prob(atk, deff, steepness=5.0)
    # out-massed -> high flip; parity -> ~0.5; dominant defence -> low flip.
    assert f[0] < 0.5 < f[2]
    assert abs(float(f[1]) - 0.5) < 0.05
    # monotone increasing in attacker mass
    assert float(f[0]) < float(f[1]) < float(f[2])


def test_reachable_mass_cumulative_and_enemy_only():
    # 3 planets; planet 2 is the only enemy, adjacent to planet 1.
    P, H = 3, 6
    big = 1e6  # unreachable
    # cross_dist[k, s, t]; make planet 2 -> planet 1 reachable from k=1, all else far.
    cross = torch.full((H + 1, P, P), big)
    for k in range(H + 1):
        cross[k, 2, 1] = 0.5  # enemy(2) reaches target(1) immediately
    ships = torch.tensor([100.0, 5.0, 40.0])
    is_enemy = torch.tensor([False, False, True])
    reach = reachable_enemy_mass(cross_dist=cross, ships=ships, is_enemy=is_enemy, H=H)
    assert reach.shape == (P, H + 1)
    # planet 1 sees the enemy's 40 from step 1 on; cumulative (non-decreasing).
    assert float(reach[1, 0]) == 0.0
    assert float(reach[1, 1]) == 40.0
    assert torch.all(reach[1, 1:] >= reach[1, :-1] - 1e-6)
    # planet 0 is unreachable -> zero throughout; my own mass never counts.
    assert float(reach[0].max()) == 0.0


def test_max_threat_concentrates_where_allocate_dilutes():
    """The LR_NATIVE_THREAT_MAX fix (PI 2026-06-21): one enemy stronghold within
    reach of SEVERAL of our planets must threaten EACH with its full mass under
    `max` (worst-case single attacker), whereas `allocate` SPLITS that mass across
    them so each sees only a fraction -- the dilution that hid the incoming attack
    and let the search drain a defended source."""
    # planets 0..3 are ours; planet 4 is the lone enemy stronghold reachable to
    # all of ours from step 1.
    P, H = 5, 6
    big = 1e6
    cross = torch.full((H + 1, P, P), big)
    for k in range(H + 1):
        for t in range(4):
            cross[k, 4, t] = 0.5            # enemy(4) reaches each of our planets
    ships = torch.tensor([90.0, 90.0, 90.0, 90.0, 95.0])   # stronghold = 95
    prod = torch.tensor([5.0, 5.0, 5.0, 5.0, 5.0])
    is_enemy = torch.tensor([False, False, False, False, True])

    mx = reachable_enemy_mass(cross_dist=cross, ships=ships, is_enemy=is_enemy,
                              H=H, aggregate="max")
    al = reachable_enemy_mass(cross_dist=cross, ships=ships, is_enemy=is_enemy,
                              H=H, aggregate="allocate", prod=prod,
                              our_garrison=ships)
    # MAX: every one of our planets sees the FULL 95 (worst-case concentrated hit).
    for t in range(4):
        assert abs(float(mx[t, H]) - 95.0) < 1e-4
    # ALLOCATE: the 95 is CONSERVED across our planets -> each sees only a share,
    # and the shares sum to ~the stronghold's mass (not 4x it).
    shares = [float(al[t, H]) for t in range(4)]
    assert all(s < 95.0 for s in shares)
    assert abs(sum(shares) - 95.0) < 1.0
    # The diluted share is far below the garrison (90) -> looks safe; the
    # concentrated view (95) does not. This is exactly the decision the fix flips.
    assert max(shares) < 90.0 < 95.0


def test_threat_growth_adds_prod_times_k_alpha():
    """Anticipatory growth: the reachable enemy reservoir rises by prod*alpha*k."""
    P, H = 3, 6
    cross = torch.full((H + 1, P, P), 1e6)
    for k in range(H + 1):
        cross[k, 2, 1] = 0.5
    ships = torch.tensor([100.0, 5.0, 40.0])
    prod = torch.tensor([1.0, 3.0, 2.0])          # enemy p2 prod = 2.0
    is_enemy = torch.tensor([False, False, True])
    alpha = 0.5
    static = reachable_enemy_mass(cross_dist=cross, ships=ships, is_enemy=is_enemy, H=H)
    grown = reachable_enemy_mass(cross_dist=cross, ships=ships, is_enemy=is_enemy, H=H,
                                 prod=prod, growth_alpha=alpha)
    for k in range(1, H + 1):
        assert abs(float(static[1, k]) - 40.0) < 1e-5
        assert abs(float(grown[1, k]) - (40.0 + 2.0 * alpha * k)) < 1e-5
    assert torch.all(grown[1, 1:] > static[1, 1:])
    assert torch.all(grown[1, 1:] >= grown[1, :-1] - 1e-6)   # still non-decreasing


def test_threat_growth_alpha_zero_is_static():
    """alpha=0 (or prod=None) is byte-identical to the reactive reservoir."""
    P, H = 3, 6
    cross = torch.full((H + 1, P, P), 1e6)
    for k in range(H + 1):
        cross[k, 2, 1] = 0.5
    ships = torch.tensor([100.0, 5.0, 40.0])
    prod = torch.tensor([1.0, 3.0, 2.0])
    is_enemy = torch.tensor([False, False, True])
    a = reachable_enemy_mass(cross_dist=cross, ships=ships, is_enemy=is_enemy, H=H)
    b = reachable_enemy_mass(cross_dist=cross, ships=ships, is_enemy=is_enemy, H=H,
                             prod=prod, growth_alpha=0.0)
    assert torch.equal(a, b)


def test_growing_threat_lowers_frontier_capture_value():
    """Anticipating the enemy's production growth makes a thin frontier capture
    less attractive (rising leak prices more of its ships as at-risk)."""
    init_owner, init_ships, prod, alive, background, cross = _board()
    prod = torch.tensor([1.0, 3.0, 4.0])          # bump enemy p2 prod so growth bites
    common = dict(
        init_owner=init_owner, init_ships=init_ships, prod=prod,
        alive_by_step=alive, background_arrivals=background,
        src=torch.tensor([[0]]), tgt=torch.tensor([[1]]),
        ships=torch.tensor([[60.0]]), eta=torch.tensor([[2.0]]),
        owner=torch.tensor([[0]]), valid=torch.tensor([[True]]),
        cross_dist=cross, cur_ships=init_ships,
        is_enemy=(init_owner == 1), me=0, value_mode="ships",
    )
    static = score_candidates_native(prod_growth_alpha=0.0, **common)
    grown = score_candidates_native(prod_growth_alpha=0.5, **common)
    assert grown[0] < static[0]


def _board():
    """p0 mine (huge), p1 neutral target, p2 enemy adjacent to p1."""
    P, H, A = 3, 8, 2
    init_owner = torch.tensor([0, -1, 1], dtype=torch.long)
    init_ships = torch.tensor([500.0, 4.0, 50.0])
    prod = torch.tensor([1.0, 3.0, 1.0])  # target is productive (worth holding)
    alive = torch.ones(H + 1, P, dtype=torch.bool)
    background = torch.zeros(P, H, A)
    cross = torch.full((H + 1, P, P), 1e6)
    for k in range(H + 1):
        cross[k, 2, 1] = 0.5  # enemy(2) can reach target(1)
    return init_owner, init_ships, prod, alive, background, cross


def test_trajectory_parity_with_no_launches():
    init_owner, init_ships, prod, alive, background, _ = _board()
    P, H, A = 3, 8, 2
    # reference: the engine recurrence on the background board directly.
    ref_owner, ref_ships, _, _ = _run_exact_recurrence(
        init_owner=init_owner.view(1, P), init_ships=init_ships.view(1, P),
        prod=prod.view(1, P), alive=alive.permute(1, 0).view(1, P, H + 1),
        arrivals=background.view(1, P, H, A),
    )
    # one all-invalid candidate -> must reproduce the background exactly.
    z = torch.zeros(1, 1)
    owner_traj, ships_traj, _ = build_candidate_trajectories(
        init_owner=init_owner, init_ships=init_ships, prod=prod,
        alive_by_step=alive, background_arrivals=background,
        src=torch.tensor([[-1]]), tgt=torch.tensor([[-1]]), ships=z,
        eta=torch.ones(1, 1), owner=torch.zeros(1, 1, dtype=torch.long),
        valid=torch.tensor([[False]]),
    )
    assert torch.equal(owner_traj, ref_owner)
    assert torch.allclose(ships_traj, ref_ships)


def test_holdable_capture_beats_thin_capture():
    init_owner, init_ships, prod, alive, background, cross = _board()
    me = 0
    # Two candidates capture the SAME neutral target p1 from p0:
    #   thin    -> send just enough to flip (5 ships; out-massed by the enemy's 50)
    #   holdable-> send a thick garrison (200 ships; out-masses the enemy)
    src = torch.tensor([[0], [0]])
    tgt = torch.tensor([[1], [1]])
    ships = torch.tensor([[5.0], [200.0]])
    eta = torch.tensor([[2.0], [2.0]])
    owner = torch.tensor([[0], [0]])
    valid = torch.tensor([[True], [True]])

    val = score_candidates_native(
        init_owner=init_owner, init_ships=init_ships, prod=prod,
        alive_by_step=alive, background_arrivals=background,
        src=src, tgt=tgt, ships=ships, eta=eta, owner=owner, valid=valid,
        cross_dist=cross, cur_ships=init_ships, is_enemy=(init_owner == 1),
        me=me, steepness=5.0,
    )
    # both deterministically capture p1, but the thick garrison is HELD (low
    # flip) while the thin one leaks to the reachable enemy mass -> lower value.
    assert val[1] > val[0]


def test_opp_expansion_rewards_grabbing_contested_neutral():
    """With opponent-expansion modeling, capturing a neutral the opponent can
    reach out-scores leaving it (which cedes it to the opponent). Without the
    term, doing nothing is costless and the passivity returns."""
    init_owner, init_ships, prod, alive, background, cross = _board()
    me = 0
    # cand A: capture the contested neutral p1; cand B: no-op (launch nothing).
    src = torch.tensor([[0], [-1]])
    tgt = torch.tensor([[1], [-1]])
    ships = torch.tensor([[60.0], [0.0]])
    eta = torch.tensor([[2.0], [1.0]])
    owner = torch.tensor([[0], [0]])
    valid = torch.tensor([[True], [False]])
    common = dict(
        init_owner=init_owner, init_ships=init_ships, prod=prod,
        alive_by_step=alive, background_arrivals=background,
        src=src, tgt=tgt, ships=ships, eta=eta, owner=owner, valid=valid,
        cross_dist=cross, cur_ships=init_ships, is_enemy=(init_owner == 1), me=me,
    )
    with_exp = score_candidates_native(model_opp_expansion=True, **common)
    # capturing (A) beats the no-op (B) once the opponent would take the neutral.
    assert with_exp[0] > with_exp[1]
    # without the term the no-op is not penalised for ceding the neutral, so the
    # gap shrinks (the expansion incentive is what the term adds).
    without = score_candidates_native(model_opp_expansion=False, **common)
    assert (with_exp[0] - with_exp[1]) > (without[0] - without[1])


def test_no_threat_means_full_survival():
    """A planet I own with ZERO reachable enemy mass must keep ownership
    probability 1 over the whole horizon (no spurious hazard haircut), and the
    present (step 0) must be certain."""
    P, H = 2, 6
    owner = torch.zeros(1, P, H + 1, dtype=torch.long)   # I own both, all steps
    ships = torch.full((1, P, H + 1), 10.0)
    prod = torch.tensor([1.0, 1.0])
    atk_reach = torch.zeros(P, H + 1)                     # nobody can reach me
    # ownership mode: value = Σ_k Σ_p prod·(1 - 0) = (H+1)·P·1 = 14, undiscounted.
    val_own = hazard_ownership_value(owner=owner, ships=ships, prod=prod,
                                     atk_reach=atk_reach, me=0, steepness=5.0,
                                     value_mode="ownership")
    assert abs(float(val_own[0]) - (H + 1) * P) < 1e-4
    # ships mode (terminal=0 to isolate the garrison scale): discounted MEAN of
    # the per-step ship margin; with garrison 10 on each of P planets fully held,
    # every step's margin is P·10, so the mean is P·10 (= 20), NOT (H+1)·P·10 —
    # pins the /Σdisc normalization to ship units.
    val_ships = hazard_ownership_value(owner=owner, ships=ships, prod=prod,
                                       atk_reach=atk_reach, me=0, steepness=5.0,
                                       value_mode="ships", terminal=0.0)
    assert abs(float(val_ships[0]) - P * 10.0) < 1e-4


def test_over_committed_source_no_negative_garrison():
    """Two launch slots from one source summing past its garrison must not feed
    a negative garrison into the recurrence (clamped like the trusted path)."""
    P, H, A = 2, 6, 2
    init_owner = torch.tensor([0, -1], dtype=torch.long)
    init_ships = torch.tensor([10.0, 3.0])               # source has only 10
    prod = torch.tensor([1.0, 1.0])
    alive = torch.ones(H + 1, P, dtype=torch.bool)
    background = torch.zeros(P, H, A)
    # two slots from planet 0 sending 8 + 8 = 16 > 10 garrison.
    owner_traj, ships_traj, _ = build_candidate_trajectories(
        init_owner=init_owner, init_ships=init_ships, prod=prod,
        alive_by_step=alive, background_arrivals=background,
        src=torch.tensor([[0, 0]]), tgt=torch.tensor([[1, 1]]),
        ships=torch.tensor([[8.0, 8.0]]), eta=torch.tensor([[2.0, 2.0]]),
        owner=torch.tensor([[0, 0]]), valid=torch.tensor([[True, True]]),
    )
    # source garrison never goes negative at any step.
    assert float(ships_traj[0, 0].min()) >= 0.0


def _churn_board(enemy_ships):
    """p0 mine (big source), p1 neutral prod-3 target, p2 enemy adjacent to p1."""
    P, H, A = 3, 10, 2
    init_owner = torch.tensor([0, -1, 1], dtype=torch.long)
    init_ships = torch.tensor([500.0, 4.0, float(enemy_ships)])
    prod = torch.tensor([1.0, 3.0, 1.0])
    alive = torch.ones(H + 1, P, dtype=torch.bool)
    background = torch.zeros(P, H, A)
    cross = torch.full((H + 1, P, P), 1e6)
    for k in range(H + 1):
        cross[k, 2, 1] = 0.5   # enemy(2) threatens p1
    return init_owner, init_ships, prod, alive, background, cross


def test_decisive_capture_nets_strongly_positive_ships():
    """A thick HOLDABLE capture clears the ~1.5-ship roi floor by a wide margin."""
    io, ish, prod, alive, bg, cross = _churn_board(50)
    val = score_candidates_native(
        init_owner=io, init_ships=ish, prod=prod, alive_by_step=alive,
        background_arrivals=bg,
        src=torch.tensor([[0]]), tgt=torch.tensor([[1]]),
        ships=torch.tensor([[200.0]]), eta=torch.tensor([[2.0]]),
        owner=torch.tensor([[0]]), valid=torch.tensor([[True]]),
        cross_dist=cross, cur_ships=ish, is_enemy=(io == 1), me=0,
        value_mode="ships",
    )
    assert float(val[0]) > 1.5


def test_churn_capture_nets_negative_marginal_ships():
    """A thin capture into a huge enemy reservoir bleeds its ships to the opponent
    on the reflip: NEGATIVE in ship-mode, but ~non-negative in ownership-mode
    (which can't see the ship economy) — the exact divergence the change fixes."""
    io, ish, prod, alive, bg, cross = _churn_board(500)
    common = dict(
        init_owner=io, init_ships=ish, prod=prod, alive_by_step=alive,
        background_arrivals=bg,
        src=torch.tensor([[0]]), tgt=torch.tensor([[1]]),
        ships=torch.tensor([[60.0]]), eta=torch.tensor([[2.0]]),
        owner=torch.tensor([[0]]), valid=torch.tensor([[True]]),
        cross_dist=cross, cur_ships=ish, is_enemy=(io == 1), me=0,
    )
    ships_val = float(score_candidates_native(value_mode="ships", **common)[0])
    own_val = float(score_candidates_native(value_mode="ownership", **common)[0])
    assert ships_val < 0.0
    assert ships_val < own_val


def test_inflight_not_penalized_during_flight():
    """A long-flight launch to a planet I will clearly hold (no enemy reach) must
    NOT net negative in ship-mode — the in-flight term covers the flight window
    where the ships have left the source but not yet landed."""
    io, ish, prod, alive, bg, _ = _churn_board(50)
    no_reach = torch.full((11, 3, 3), 1e6)   # nobody can reach anything
    val = float(score_candidates_native(
        init_owner=io, init_ships=ish, prod=prod, alive_by_step=alive,
        background_arrivals=bg,
        src=torch.tensor([[0]]), tgt=torch.tensor([[1]]),
        ships=torch.tensor([[60.0]]), eta=torch.tensor([[5.0]]),  # long flight
        owner=torch.tensor([[0]]), valid=torch.tensor([[True]]),
        cross_dist=no_reach, cur_ships=ish, is_enemy=(io == 1), me=0,
        value_mode="ships",
    )[0])
    assert val >= -1e-6


def test_deterministic_repeat():
    init_owner, init_ships, prod, alive, background, cross = _board()
    args = dict(
        init_owner=init_owner, init_ships=init_ships, prod=prod,
        alive_by_step=alive, background_arrivals=background,
        src=torch.tensor([[0]]), tgt=torch.tensor([[1]]), ships=torch.tensor([[200.0]]),
        eta=torch.tensor([[2.0]]), owner=torch.tensor([[0]]), valid=torch.tensor([[True]]),
        cross_dist=cross, cur_ships=init_ships, is_enemy=(init_owner == 1), me=0,
    )
    a = score_candidates_native(**args)
    b = score_candidates_native(**args)
    assert torch.equal(a, b)


def test_concentrated_adversary_reorders_toward_defense():
    """The uniform leak gives the same penalty to all candidates (cancels in the
    argmax); the concentrated adversary penalises whoever leaves a valuable
    planet thin. Here a candidate that REINFORCES a threatened high-prod planet
    must out-score one that ignores it under `concentrate=True`."""
    P, H, A = 3, 8, 2
    # p0 mine (source), p1 mine + HIGH prod + thin garrison + enemy adjacent
    # (the threatened planet), p2 enemy adjacent to p1.
    init_owner = torch.tensor([0, 0, 1], dtype=torch.long)
    init_ships = torch.tensor([300.0, 3.0, 80.0])
    prod = torch.tensor([1.0, 9.0, 1.0])
    alive = torch.ones(H + 1, P, dtype=torch.bool)
    background = torch.zeros(P, H, A)
    cross = torch.full((H + 1, P, P), 1e6)
    for k in range(H + 1):
        cross[k, 2, 1] = 0.5  # enemy(2) threatens my high-prod p1

    # cand A: reinforce the threatened p1 from p0 (defend).
    # cand B: send the same mass to capture neutral-ish elsewhere (ignore p1).
    #   (model "ignore" as launching from p0 to p2's far side = no help to p1).
    src = torch.tensor([[0], [0]])
    tgt = torch.tensor([[1], [0]])          # B: tgt=0 (self, no-op-ish)
    ships = torch.tensor([[150.0], [150.0]])
    eta = torch.tensor([[2.0], [2.0]])
    owner = torch.tensor([[0], [0]])
    valid = torch.tensor([[True], [True]])

    common = dict(
        init_owner=init_owner, init_ships=init_ships, prod=prod,
        alive_by_step=alive, background_arrivals=background,
        src=src, tgt=tgt, ships=ships, eta=eta, owner=owner, valid=valid,
        cross_dist=cross, cur_ships=init_ships, is_enemy=(init_owner == 1),
        me=0, steepness=5.0,
    )
    # Tests the concentrate mechanism in its native (ownership) semantics. (In
    # ship-mode, piling ships onto a not-fully-securable planet raises ships-at-
    # risk, so "reinforce > ignore" is not a ship-space invariant.)
    conc = score_candidates_native(concentrate=True, value_mode="ownership", **common)
    # Reinforcing the threatened high-prod planet (A) beats ignoring it (B).
    assert conc[0] > conc[1]


def test_value_drops_as_enemy_mass_grows():
    """Same holdable capture, but a bigger reachable enemy reservoir lowers it."""
    init_owner, init_ships, prod, alive, background, cross = _board()
    base = dict(
        init_owner=init_owner, init_ships=init_ships, prod=prod,
        alive_by_step=alive, background_arrivals=background,
        src=torch.tensor([[0]]), tgt=torch.tensor([[1]]), ships=torch.tensor([[60.0]]),
        eta=torch.tensor([[2.0]]), owner=torch.tensor([[0]]), valid=torch.tensor([[True]]),
        cross_dist=cross, me=0,
    )
    weak = score_candidates_native(cur_ships=torch.tensor([500.0, 4.0, 20.0]),
                                   is_enemy=(init_owner == 1), **base)
    strong = score_candidates_native(cur_ships=torch.tensor([500.0, 4.0, 300.0]),
                                     is_enemy=(init_owner == 1), **base)
    assert weak[0] > strong[0]
