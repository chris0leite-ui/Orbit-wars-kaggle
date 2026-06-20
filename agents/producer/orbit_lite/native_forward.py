"""Dropout-NATIVE forward model — mean-field flip-hazard ownership (Phase A).

The dropout BOLT-ON (producer_plus) values a capture by a 2-point blend (clean
score vs one reflipped score) over a static one-ply scorer. The continuous-score
A/B (state/DROPOUT_PLAN.md) showed that graft is SATURATED: its perturbation has
no distribution to refine. This module is the design's answer
(state/DROPOUT_NATIVE_DESIGN.md, Phase A): the forward model itself is a
distribution over ownership futures.

THE MODEL (mean-field, deterministic, NO RNG)
---------------------------------------------
1. Deterministic dynamics, unchanged: production accrues, our launches land,
   in-flight fleets resolve, combat at arrivals — exactly the engine recurrence
   (`_run_exact_recurrence`, reused). This yields, per candidate, the
   owner/garrison trajectory [C, P, H+1].
2. Flip HAZARD overlay: at each (planet p, step k) a planet I own is held with
   probability `1 - flip(atk_reach, garrison)`, where `atk_reach(p,k)` is the
   enemy's physically-routable mass that can reach p by step k and `garrison` is
   my projected ship count there. `flip = sigmoid(steepness·(atk-def)/(atk+def))`
   is steep near parity. This is the per-step Markov-ownership probability
   `P(I own p at k)` — a genuine continuous distribution over futures, not a
   2-point blend. (Phase B will calibrate `flip` to observed flip rates.)
3. Value functional = expected production-weighted ownership margin over the
   horizon:  `Σ_k discount^k Σ_p prod_p · (P_mine(p,k) − P_opp(p,k))`.
   A capture I cannot hold (high atk_reach, thin garrison) leaks its ownership
   probability to the opponent and earns little; a holdable capture earns the
   full production stream. This subsumes the bolt-on's reflip and the
   terminal-prod / hold-value hacks.

Everything is one deterministic pass over the batch axis — CPU/GPU bit-identical,
the codebase's hard requirement.
"""
from __future__ import annotations

import torch
from torch import Tensor

from .garrison_launch import _run_exact_recurrence
from .geometry import fleet_speed


def build_candidate_trajectories(
    *,
    init_owner: Tensor,          # [P] long
    init_ships: Tensor,          # [P] float (current garrison)
    prod: Tensor,                # [P] float
    alive_by_step: Tensor,       # [H+1, P] bool
    background_arrivals: Tensor,  # [P, H, A] float (do-nothing arrivals)
    src: Tensor,                 # [C, L] long  source slot (-1 = none)
    tgt: Tensor,                 # [C, L] long  target slot
    ships: Tensor,               # [C, L] float
    eta: Tensor,                 # [C, L] float
    owner: Tensor,               # [C, L] long  arriving owner
    valid: Tensor,               # [C, L] bool
) -> tuple[Tensor, Tensor, Tensor]:
    """Per-candidate post-combat owner/garrison trajectories + the arrivals tensor.

    Returns ``(owner_out, ships_out, arrivals_c)`` — owner/ships are ``[C,P,H+1]``
    post-combat garrison; ``arrivals_c`` is ``[C,P,H,A]`` the per-candidate
    per-owner arrival buckets (background + this candidate's credits), needed to
    attribute IN-FLIGHT ship mass to owners in the ship-margin value.

    Applies each candidate's launches (source debit + target credit at the eta
    bucket) on top of the shared background arrivals, then walks the SAME engine
    recurrence the producer trusts. Dense across candidates (the shortlist is
    small; the freed opponent-mirror budget covers it).
    """
    P = int(init_owner.shape[0])
    H = int(background_arrivals.shape[1])
    A = int(background_arrivals.shape[2])
    C = int(src.shape[0])
    device = init_ships.device
    fdtype = init_ships.dtype

    h_idx = torch.ceil(eta.to(fdtype)).to(torch.long) - 1                 # [C, L]
    valid_t = (valid & (ships > 0) & (tgt >= 0) & (tgt < P)
               & (owner >= 0) & (owner < A) & (h_idx >= 0) & (h_idx < H))
    valid_s = valid & (ships > 0) & (src >= 0) & (src < P)
    src_safe = src.clamp(0, max(P - 1, 0))
    tgt_safe = tgt.clamp(0, max(P - 1, 0))

    # Source debit -> per-candidate init ships.
    init_ships_c = init_ships.view(1, P).expand(C, P).clone()
    debit = torch.zeros(C, P, dtype=fdtype, device=device)
    debit.scatter_add_(1, src_safe, torch.where(valid_s, ships, torch.zeros_like(ships)))
    # Clamp at 0 exactly as the trusted production path does (garrison_launch
    # source-debit): an over-committed source (multiple launch slots summing past
    # the current garrison) must not feed a NEGATIVE garrison into the recurrence,
    # which never clamps and would spuriously flip ownership in combat.
    init_ships_c = (init_ships_c - debit).clamp(min=0.0)

    # Per-candidate arrivals = background + candidate credits.
    arrivals_c = background_arrivals.view(1, P, H, A).expand(C, P, H, A).clone()
    c_idx = torch.arange(C, device=device).view(C, 1).expand_as(tgt)
    m = valid_t
    if bool(m.any()):
        arrivals_c.index_put_(
            (c_idx[m], tgt_safe[m], h_idx[m], owner[m]),
            ships[m], accumulate=True,
        )

    init_owner_c = init_owner.view(1, P).expand(C, P).contiguous()
    prod_c = prod.view(1, P).expand(C, P).contiguous()
    alive_c = alive_by_step.permute(1, 0).view(1, P, H + 1).expand(C, P, H + 1).contiguous()

    owner_out, ships_out, _, _ = _run_exact_recurrence(
        init_owner=init_owner_c, init_ships=init_ships_c, prod=prod_c,
        alive=alive_c, arrivals=arrivals_c,
    )
    return owner_out, ships_out, arrivals_c


def reachable_enemy_mass(
    *,
    cross_dist: Tensor,   # [K+1, P, P]  cross_dist[k, s, t] = dist(s@0, t@k)
    ships: Tensor,        # [P] float (current garrison; enemy mass + target defense)
    is_enemy: Tensor,     # [P] bool
    H: int,
    aggregate: str = "max",
    prod: Tensor | None = None,   # [P] enemy production rate (None => no growth)
    growth_alpha: float = 0.0,    # damp the over-horizon production growth; 0 = static
    our_garrison: Tensor | None = None,  # [P] target defense (allocate; default=ships)
    alloc_w_prox: float = 1.0,    # allocate: proximity weight
    alloc_w_val: float = 1.0,     # allocate: planet-value (production) weight
    alloc_w_def: float = 0.5,     # allocate: inverse-garrison (weak-spot) weight
    alloc_eps: float = 1.0,       # allocate: numerical floor for logs / prox denom
    target_mask: Tensor | None = None,  # [P] bool valid targets (allocate; default=~is_enemy)
    exclude_self: bool = False,   # allocate: mask the source==target diagonal
) -> Tensor:
    """Enemy mass physically routable to each planet by step k -> ``[P, H+1]``.

    Enemy planet q can reach target p arriving at step k by launching now if
    ``dist(q@0, p@k) <= k · speed(q)``. Once reachable it stays reachable, so the
    result is cumulative (non-decreasing) in k.

    ``aggregate`` decides how multiple reachable enemy sources combine into the
    threat on one target:
    - ``"max"`` (default): the STRONGEST single reachable enemy planet. This is
      the realistic concentrated threat — one enemy launches a decisive fleet.
      SUM (below) over-counts catastrophically: it assumes the WHOLE enemy army
      hits every one of my planets at once, so each frontier planet looks ~doomed
      (garrison 50 vs "threat" 340), a partial reinforcement can never help, and
      the agent abandons its planets -> the observed mid-game passivity/collapse.
    - ``"sum"``: total reachable enemy mass (the old over-pessimistic behaviour).
    - ``"allocate"`` (PI 2026-06-20, conserved): each enemy source q splits its
      OWN mass over exactly the NON-enemy planets it can reach by step k, via a
      softmax weighted toward our close / big / weakly-defended planets. Summing
      within a source spends q exactly ONCE (no over-count across our planets);
      summing across sources gives the grand total = total reachable enemy mass
      (conserved / "integrates to enemy strength"); a planet no enemy can reach
      gets 0 (proximity honored). The per-planet share is intuitive: it rises with
      opponent strength (mass_k), with proximity (reach slack), with planet value
      (log prod), and falls with our garrison (-log defense). This is the
      principled middle between ``max`` (one fleet, ignores the army) and ``sum``
      (whole army on every planet). ``alloc_w_*`` tune the softmax peakedness.

    ANTICIPATORY growth (``prod`` + ``growth_alpha`` > 0): the reservoir grows by
    the enemy's production accrued by step k — ``mass_k = ships + prod·k·alpha``.
    A purely-current reservoir is REACTIVE: a frontier planet's leak never rises
    over the horizon even as the opponent's production visibly accrues into the
    army that flips it mid-game, so the value over-expands into a thin frontier it
    cannot hold (the observed turn 60-80 collapse). Growth makes the threat rise
    EARLIER so ships are held/reinforced there. ``alpha`` damps the rate (the
    opponent cannot route ALL accrued production into one fleet); 0 = static."""
    K = int(cross_dist.shape[0]) - 1
    P = int(ships.shape[0])
    device = ships.device
    fdtype = ships.dtype
    Hc = min(H, K)

    speed = fleet_speed(ships).to(fdtype)                       # [P] per source
    enemy_mass = torch.where(is_enemy, ships, torch.zeros_like(ships))  # [P]
    grow = prod is not None and growth_alpha != 0.0
    enemy_prod = (torch.where(is_enemy, prod.to(fdtype), torch.zeros_like(ships))
                  if grow else None)                            # [P]

    allocate = aggregate == "allocate"
    if allocate:
        # Per-TARGET bias for the allocation softmax (constant in k): planet value
        # (concave log prod, so a big planet doesn't dominate) minus a gentle
        # inverse-garrison term (weak spots draw more of the pot). Garrison is
        # deliberately ALSO in flip_prob downstream — distinct jobs (target
        # selection here vs combat resistance there).
        tgt_def = (our_garrison if our_garrison is not None else ships).to(fdtype)
        val_vec = (prod.to(fdtype) if prod is not None
                   else torch.ones_like(ships))                 # [P]
        log_val = alloc_w_val * torch.log(val_vec.clamp(min=0.0) + alloc_eps)
        log_def = alloc_w_def * torch.log(tgt_def.clamp(min=0.0) + alloc_eps)
        target_bias = (log_val - log_def).view(1, P)            # [1, P]
        # Valid targets: default = non-enemy (mine + neutral). `target_mask` lets the
        # same primitive be pointed a third way -- our planets reinforcing OUR planets
        # (sources=is_mine, target_mask=is_mine) for the reinforcement-race holdability.
        tmask = (~is_enemy) if target_mask is None else target_mask
        not_enemy = tmask.view(1, P)                            # [1, P]
        self_mask = (~torch.eye(P, dtype=torch.bool, device=device)
                     if exclude_self else None)                 # [P, P] off-diagonal
        NEG = float(torch.finfo(fdtype).min) / 4.0

    out = torch.zeros(P, H + 1, dtype=fdtype, device=device)
    reachable_any = torch.zeros(P, P, dtype=torch.bool, device=device)  # [src, tgt]
    for k in range(1, Hc + 1):
        d_k = cross_dist[k]                                    # [src, tgt] dist(s@0, t@k)
        reach_k = d_k <= (float(k) * speed.view(P, 1))         # [src, tgt]
        reachable_any = reachable_any | reach_k
        # per-source mass at step k: current garrison + production accrued (damped).
        mass_k = (enemy_mass + enemy_prod * (float(growth_alpha) * float(k))
                  if grow else enemy_mass)                     # [P]
        if allocate:
            # Reach slack at step k (>=0 where reachable). Softmax per SOURCE over
            # the non-enemy planets it can reach; invalid targets masked to -inf.
            prox = (float(k) * speed.view(P, 1)) - d_k         # [src, tgt]
            denom = (float(k) * speed.view(P, 1) + alloc_eps)  # [src, 1]
            logit = alloc_w_prox * (prox / denom) + target_bias  # [src, tgt]
            valid = reachable_any & not_enemy                  # [src, tgt]
            if self_mask is not None:
                valid = valid & self_mask                      # a planet doesn't reinforce itself
            logit = torch.where(valid, logit,
                                torch.full_like(logit, NEG))
            m = logit.max(dim=1, keepdim=True).values          # [src, 1]
            ex = torch.exp(logit - m)                          # invalid -> ~0
            w = ex / ex.sum(dim=1, keepdim=True).clamp(min=1e-30)  # [src, tgt]
            has_tgt = valid.any(dim=1, keepdim=True).to(fdtype)   # [src, 1]
            share = (mass_k.view(P, 1) * w) * has_tgt          # [src, tgt]
            out[:, k] = share.sum(dim=0)                       # per-target share sum
            continue
        masked = reachable_any.to(fdtype) * mass_k.view(P, 1)  # [src, tgt]
        if aggregate == "sum":
            out[:, k] = masked.sum(dim=0)
        else:
            out[:, k] = masked.max(dim=0).values              # strongest single source
    for k in range(Hc + 1, H + 1):
        out[:, k] = out[:, Hc]  # horizon beyond the distance cache: hold last
    return out


def flip_prob(atk: Tensor, deff: Tensor, *, steepness: float, eps: float = 1.0
              ) -> Tensor:
    """Probability the opponent flips a planet I hold, from local force balance.
    Steep near parity; -> 1 when out-massed, -> 0 when dominant. In [0, 1)."""
    bal = (atk - deff) / (atk + deff + eps)
    return torch.sigmoid(steepness * bal)


def hazard_ownership_value(
    *,
    owner: Tensor,         # [C, P, H+1] long  (post-combat deterministic owner)
    ships: Tensor,         # [C, P, H+1] float (my projected garrison)
    prod: Tensor,          # [P] float
    atk_reach: Tensor,     # [P, H+1] float (enemy routable mass by step k)
    me: int,
    steepness: float = 5.0,
    discount: float = 1.0,
    concentrate: bool = False,
    model_opp_expansion: bool = True,
    def_reach: Tensor | None = None,   # [P, H+1] OUR reinforcement mass onto each planet
    def_weight: float = 1.0,           # weight of reinforcement in the leak denominator
    value_mode: str = "ships",
    inflight: Tensor | None = None,
    terminal: float = 12.0,
    off_reach: Tensor | None = None,   # [P, H+1] OUR routable mass onto each planet
    off_weight: float = 0.0,           # offensive-potential weight (0 => term OFF)
    off_steepness: float = 5.0,        # capture-prob steepness (mirror of `steepness`)
) -> Tensor:
    """Per-candidate value over the horizon -> ``[C]``.

    `P_mine(p,k) = [owner==me] · surv(p,k)` with cumulative survival
    `surv = Π_{j<=k} (1 - leak_j)` applied while I hold the planet; the leaked
    share, opponent-held planets, and opponent-expansion onto neutrals go to
    `P_opp`.

    ``value_mode``:
    - ``"ships"`` (default, engine-aligned): EXPECTED SHIP-MARGIN. Weight the
      ownership probabilities by the per-planet ship count ``s(p,k)`` (post-combat
      garrison) instead of production, add the in-flight ship mass per owner, and
      take the discounted MEAN over the horizon (``/Σ_k disc^k``). This matches the
      engine's win condition (total ships, planets + fleets) and prices the ships
      a churning capture bleeds to the opponent on a reflip.
    - ``"ownership"`` (ablation): the legacy production-weighted ownership margin
      (byte-identical to the pre-reformulation behaviour).

    ``inflight`` (``[C, H+1, A]``, ships mode only): per-owner ship mass still in
    transit after each step, so a launch isn't penalised during its flight window.

    ``concentrate`` (self-consistency / 1-round fictitious play): instead of the
    opponent leaking from EVERY planet at once, the opponent commits its routable
    mass to the SINGLE most damaging of my planets per candidate."""
    C, P, H1 = owner.shape
    device = ships.device
    fdtype = ships.dtype

    atk = atk_reach.view(1, P, H1).to(fdtype)
    garrison = ships.clamp(min=0.0)
    # Defender of the flip contest. By default the planet's own garrison; with
    # `def_reach` (PI 2026-06-20, the reinforcement race) it ALSO counts how much our
    # OTHER planets can route to support it -> holdability becomes "can we reinforce
    # faster than the enemy attacks", so far/unsupportable captures flip (high leak)
    # and close/supported ones stick. `def_reach` is conserved + ~0 off our planets,
    # so neutrals/enemy are unaffected. None => byte-identical (garrison only).
    deff = garrison
    if def_reach is not None and def_weight != 0.0:
        deff = garrison + float(def_weight) * def_reach.view(1, P, H1).to(fdtype)
    # Zero the hazard where NO enemy mass can reach (atk==0): a planet under no
    # physical threat must have flip probability 0, not the sigmoid's nonzero
    # floor. This also pins present-certainty: atk_reach[:,0]==0 by construction,
    # so step-0 survival is exactly 1 (the present is known, not discounted).
    leak = flip_prob(atk, deff, steepness=steepness) * (atk > 0).to(fdtype)

    is_mine = (owner == me).to(fdtype)
    is_opp = ((owner >= 0) & (owner != me)).to(fdtype)
    is_neutral = (owner < 0).to(fdtype)

    k = torch.arange(H1, device=device, dtype=fdtype)
    disc = torch.pow(torch.tensor(float(discount), dtype=fdtype, device=device), k)

    # CUMULATIVE survival under the per-planet hazard (my planets flipping away).
    keep = 1.0 - leak * is_mine                                 # [C, P, H+1]
    surv = torch.cumprod(keep.clamp(0.0, 1.0), dim=2)          # [C, P, H+1]

    # Opponent EXPANSION onto neutrals: a reachable neutral I leave unclaimed is
    # progressively taken by the opponent (same routable-mass hazard). Without
    # this term, doing nothing is costless (neutrals stay neutral, P_opp=0) and
    # the agent idles — the observed passivity. With it, NOT grabbing a contested
    # neutral cedes it to the opponent (P_opp rises), creating the opportunity
    # cost that drives continued expansion. (Capturing it removes the term: the
    # planet is then `is_mine`, not `is_neutral`, in this candidate's trajectory.)
    if model_opp_expansion:
        keep_n = 1.0 - leak * is_neutral                       # [C, P, H+1]
        surv_n = torch.cumprod(keep_n.clamp(0.0, 1.0), dim=2)
        p_opp_neutral = is_neutral * (1.0 - surv_n)
    else:
        p_opp_neutral = torch.zeros_like(is_neutral)

    p_mine = is_mine * surv
    p_opp = is_opp + is_mine * (1.0 - surv) + p_opp_neutral

    if value_mode == "ownership":
        # ---- LEGACY production-weighted ownership margin (byte-identical) ----
        if not concentrate:
            margin = (p_mine - p_opp) * prod.view(1, P, 1)
            return (margin.sum(dim=1) * disc.view(1, H1)).sum(dim=1)
        det_margin = ((is_mine - is_opp) * prod.view(1, P, 1)
                      * disc.view(1, 1, H1)).sum(dim=(1, 2))
        loss_pk = is_mine * (1.0 - surv) * prod.view(1, P, 1)
        loss_p = (loss_pk * disc.view(1, 1, H1)).sum(dim=2)
        worst = loss_p.max(dim=1).values
        neutral_cost = (p_opp_neutral * prod.view(1, P, 1)
                        * disc.view(1, 1, H1)).sum(dim=(1, 2))
        return det_margin - worst - neutral_cost

    # ---- SHIPS: expected ship-margin, discounted MEAN, with in-flight mass ----
    # Ship-weighting uses the INSTANTANEOUS leak (prob the planet is overwhelmed
    # at step k given reachable mass vs garrison), NOT the cumulative product.
    # The cumulative product compounds a STATIC enemy reservoir into near-certain
    # loss (it treats the same mass as attacking every step), which when weighted
    # by ships catastrophically under-values a dominant garrison (200 vs 50 would
    # "erode" to a ~40% loss over the horizon). The ship-weighting already makes
    # the instantaneous leak load-bearing (0.05·200 vs 0.99·1 differ hugely), so
    # the cumulative compounding is neither needed nor correct here.
    norm = float(disc.sum())
    if inflight is not None:
        my_if = inflight[..., me]                                 # [C, H+1]
        inflight_margin = my_if - (inflight.sum(dim=-1) - my_if)  # [C, H+1]
    else:
        inflight_margin = None
    # Ship weight = current garrison + a POST-horizon production credit. A held
    # planet's production becomes future ships (the engine scores total ships at
    # game end); within H≈18 only a sliver has accrued, so owning a productive
    # planet must be credited its forward production stream or expansion is
    # under-valued (the producer's "terminal production value" for the same
    # reason). garrison already carries in-horizon production via the recurrence;
    # `prod·terminal` adds the beyond-horizon stream.
    w = garrison + prod.view(1, P, 1) * float(terminal)          # [C, P, H+1]
    s_mine = is_mine * (1.0 - leak)
    s_opp = is_opp + is_mine * leak
    if model_opp_expansion:
        s_opp = s_opp + is_neutral * leak    # ceded reachable neutrals -> opp

    # OFFENSIVE POTENTIAL (PI 2026-06-20, mirror of the defensive flip-hazard):
    # credit, on each enemy/neutral planet we can reach, the expected ship-mass we
    # could CONVERT by capturing it -- where `off_reach` is OUR routable mass onto
    # that planet (computed exactly like `atk_reach`, but with our planets as the
    # sources). Intent: HOLDING a massed frontier keeps this potential (patience),
    # dribbling ships into far neutrals destroys it (anti-scatter), a concentrated
    # capture realizes it (decisive action). REFUTED IN REPLAYS (kept default-OFF):
    # because the term sums over ALL reachable targets, "sit still and threaten
    # everything" out-scores "commit and take one", biasing toward passivity -- at
    # off_weight 0.5 it lost every test seed (2P 7/106499442 and 4P 1912745358 went
    # win->loss) while reducing scatter; lower weights were erratic. A correct
    # version would need a COMPETITIVE (commit-attributed) potential, not this
    # additive one. OFF (off_reach None or off_weight 0) => block skipped entirely.
    off_gain = None
    if off_reach is not None and off_weight != 0.0:
        off = off_reach.view(1, P, H1).to(fdtype)
        cap = flip_prob(off, garrison, steepness=off_steepness) \
            * (off > 0).to(fdtype)                                # [C, P, H+1] capture prob
        swing = is_opp * 2.0 + is_neutral                         # enemy=2w swing, neutral=+w
        off_gain = (float(off_weight) * cap * swing * w)          # [C, P, H+1]

    if not concentrate:
        margin_k = ((s_mine - s_opp) * w).sum(dim=1)             # [C, H+1]
        if off_gain is not None:
            margin_k = margin_k + off_gain.sum(dim=1)
        if inflight_margin is not None:
            margin_k = margin_k + inflight_margin
        return (margin_k * disc.view(1, H1)).sum(dim=1) / norm

    det_k = ((is_mine - is_opp) * w).sum(dim=1)                  # [C, H+1]
    if inflight_margin is not None:
        det_k = det_k + inflight_margin
    det_margin = (det_k * disc.view(1, H1)).sum(dim=1) / norm     # [C]
    loss_pk = is_mine * leak * w                                  # [C, P, H+1]
    loss_p = (loss_pk * disc.view(1, 1, H1)).sum(dim=2) / norm    # [C, P]
    worst = loss_p.max(dim=1).values                              # [C]
    neutral_cost = ((is_neutral * leak * w
                     * disc.view(1, 1, H1)).sum(dim=(1, 2))) / norm \
        if model_opp_expansion else torch.zeros_like(worst)
    return det_margin - worst - neutral_cost


def _inflight_by_owner(arrivals_c: Tensor) -> Tensor:
    """Per-owner ship mass still IN FLIGHT after each step -> ``[C, H+1, A]``.

    ``arrivals_c`` is ``[C, P, H, A]``; bucket ``j`` lands at step ``k=j+1``. The
    mass aloft after step ``k`` is everything scheduled for a later step."""
    C, P, H, A = arrivals_c.shape
    arr_owner = arrivals_c.sum(dim=1)                     # [C, H, A] over planets
    cum_landed = torch.cumsum(arr_owner, dim=1)           # [C, H, A]
    total = cum_landed[:, -1, :]                          # [C, A] all launched mass
    out = total.unsqueeze(1).expand(C, H + 1, A).clone()  # step 0: all aloft
    out[:, 1:, :] = total.unsqueeze(1) - cum_landed       # step k: minus landed<=k
    return out


def score_candidates_native(
    *,
    init_owner: Tensor, init_ships: Tensor, prod: Tensor,
    alive_by_step: Tensor, background_arrivals: Tensor,
    src: Tensor, tgt: Tensor, ships: Tensor, eta: Tensor,
    owner: Tensor, valid: Tensor,
    cross_dist: Tensor, cur_ships: Tensor, is_enemy: Tensor,
    me: int, steepness: float = 5.0, discount: float = 1.0,
    concentrate: bool = False, model_opp_expansion: bool = True,
    value_mode: str = "ships", terminal: float = 12.0,
    prod_growth_alpha: float = 0.0,
) -> Tensor:
    """End-to-end native value per candidate ``[C]`` (the Phase-A scorer)."""
    H = int(background_arrivals.shape[1])
    owner_traj, ships_traj, arrivals_c = build_candidate_trajectories(
        init_owner=init_owner, init_ships=init_ships, prod=prod,
        alive_by_step=alive_by_step, background_arrivals=background_arrivals,
        src=src, tgt=tgt, ships=ships, eta=eta, owner=owner, valid=valid,
    )
    atk_reach = reachable_enemy_mass(
        cross_dist=cross_dist, ships=cur_ships, is_enemy=is_enemy, H=H,
        prod=prod, growth_alpha=prod_growth_alpha,
    )
    ships_mode = value_mode != "ownership"
    inflight = _inflight_by_owner(arrivals_c) if ships_mode else None
    val = hazard_ownership_value(
        owner=owner_traj, ships=ships_traj, prod=prod, atk_reach=atk_reach,
        me=me, steepness=steepness, discount=discount, concentrate=concentrate,
        model_opp_expansion=model_opp_expansion,
        value_mode=value_mode, inflight=inflight, terminal=terminal,
    )
    # MARGINAL value: subtract the do-nothing baseline so the score is the
    # improvement over inaction, not an absolute board value. The producer's
    # chooser commits candidates whose score clears the (~1.5-ship) roi floor; an
    # absolute value is dominated by a constant (every candidate cedes the same
    # bulk of distant neutrals) and pushes everything below the floor -> idle.
    # The delta cancels that constant (incl. shared background in-flight fleets):
    # capturing a contested neutral / defending a threatened planet is a positive
    # marginal gain, doing nothing is exactly 0.
    base_owner, base_ships, base_arr = build_candidate_trajectories(
        init_owner=init_owner, init_ships=init_ships, prod=prod,
        alive_by_step=alive_by_step, background_arrivals=background_arrivals,
        src=torch.full((1, 1), -1, dtype=torch.long),
        tgt=torch.full((1, 1), -1, dtype=torch.long),
        ships=torch.zeros(1, 1), eta=torch.ones(1, 1),
        owner=torch.zeros(1, 1, dtype=torch.long),
        valid=torch.zeros(1, 1, dtype=torch.bool),
    )
    base_inflight = _inflight_by_owner(base_arr) if ships_mode else None
    base_val = hazard_ownership_value(
        owner=base_owner, ships=base_ships, prod=prod, atk_reach=atk_reach,
        me=me, steepness=steepness, discount=discount, concentrate=concentrate,
        model_opp_expansion=model_opp_expansion,
        value_mode=value_mode, inflight=base_inflight, terminal=terminal,
    )
    return val - base_val
