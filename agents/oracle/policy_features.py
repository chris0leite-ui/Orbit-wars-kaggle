"""Oracle agent — per-candidate policy features (shared train/serve).

For a candidate wave (source planet -> target planet, with a launchable
spare), describe the decision the way the replay data labels it: local pair
geometry + ledger-priced economics + a compact global context. The policy
net maps this to P(a top-rated player fires this wave now) and, for fired
waves, the size as a fraction of the source's launchable spare.

Everything here must be computable at runtime from one observation.
"""

import math

from .engine import fleet_speed, required_ships

# the trailing block of pair features is state-global (identical for every
# pair of a state); the state-level initiation head trains on exactly this
# slice, so KEEP IT LAST and update N_GLOBALS when it changes
N_GLOBALS = 11

POLICY_FEATURES = [
    # pair geometry / tempo
    "dist", "eta_full", "eta_required", "src_to_enemy_d", "tgt_to_enemy_d",
    "tgt_to_my_d",
    # source state
    "src_garrison", "src_spare", "src_spare_frac", "src_prod",
    "src_threat_in", "src_doomed", "src_is_comet", "src_comet_ttl",
    # target state
    "tgt_owner_me", "tgt_owner_enemy", "tgt_owner_neutral", "tgt_prod",
    "tgt_radius", "tgt_garrison_now", "tgt_garrison_at_eta", "tgt_required",
    "tgt_required_frac_spare", "tgt_is_comet", "tgt_comet_ttl",
    "tgt_falls_to_enemy", "tgt_flips_to_me", "tgt_incoming_mine",
    "tgt_incoming_enemy", "tgt_nearest_enemy_eta", "race_margin",
    # global context
    "step_frac", "n_players", "my_planets", "opp_planets", "my_prod_share",
    "my_score_share", "my_garrison", "opp_garrison", "my_inflight_frac",
    "neutral_left", "frontline_dist",
]
N_POLICY_FEATURES = len(POLICY_FEATURES)


class PolicyContext:
    """Per-turn precomputation for candidate featurization (one seat).

    `src_states` maps planet_idx -> (garrison, safe_spare, doomed) — see
    planner.source_states.
    """

    def __init__(self, world, src_states):
        w = self.world = world
        self.sp = src_states
        me = w.me
        n = w.n_planets
        self.enemies = [j for j in range(n)
                        if w.owner0[j] >= 0 and w.owner0[j] != me]
        self.mine = [j for j in range(n) if w.owner0[j] == me]
        my_prod = sum(w.prod[i] for i in self.mine)
        tot_prod = sum(w.prod[i] for i in range(n)
                       if w.owner0[i] >= 0) or 1
        my_g = sum(w.ships0[i] for i in self.mine)
        opp_g = sum(w.ships0[i] for i in range(n)
                    if w.owner0[i] >= 0 and w.owner0[i] != me)
        my_fl = sum(f[6] for f in w.fleets if f[1] == me)
        all_fl = sum(f[6] for f in w.fleets if f[1] >= 0)
        my_score = my_g + my_fl
        tot_score = my_score + opp_g + (all_fl - my_fl)
        players = {w.owner0[i] for i in range(n) if w.owner0[i] >= 0} | \
                  {int(f[1]) for f in w.fleets if f[1] >= 0}
        fd = 200.0
        for i in self.mine:
            for j in self.enemies:
                d = math.hypot(w.px[i] - w.px[j], w.py[i] - w.py[j])
                if d < fd:
                    fd = d
        self.globals = [
            w.step / 500.0,
            float(len(players)),
            float(len(self.mine)),
            float(len(self.enemies)),
            my_prod / tot_prod,
            my_score / tot_score if tot_score > 0 else 0.0,
            float(my_g),
            float(opp_g),
            my_fl / my_score if my_score > 0 else 0.0,
            float(sum(1 for i in range(n)
                      if w.owner0[i] == -1 and not w.is_comet[i])),
            fd,
        ]
        # incoming per planet (mine vs enemy mass within 12 ticks)
        self.inc_mine = [0.0] * n
        self.inc_enemy = [0.0] * n
        for i in range(n):
            for dt, slot in w.arrivals[i].items():
                if dt > 12:
                    continue
                for o, s in slot.items():
                    if o == me:
                        self.inc_mine[i] += s
                    elif o >= 0:
                        self.inc_enemy[i] += s
        # nearest-enemy distance per planet (garrison-weighted reach proxy)
        self.near_enemy_d = [200.0] * n
        for i in range(n):
            for j in self.enemies:
                d = math.hypot(w.px[i] - w.px[j], w.py[i] - w.py[j])
                if d < self.near_enemy_d[i]:
                    self.near_enemy_d[i] = d
        self.near_mine_d = [200.0] * n
        for i in range(n):
            for j in self.mine:
                if j == i:
                    continue
                d = math.hypot(w.px[i] - w.px[j], w.py[i] - w.py[j])
                if d < self.near_mine_d[i]:
                    self.near_mine_d[i] = d

    def pair(self, src, tgt, garrison, safe, doomed):
        """Feature vector for candidate (src -> tgt)."""
        w = self.world
        me = w.me
        H = w.horizon
        spare = max(1, int(garrison))
        d = math.hypot(w.px[tgt] - w.px[src], w.py[tgt] - w.py[src])
        v_full = fleet_speed(max(1, spare))
        eta_full = max(1.0, math.ceil(
            max(d - w.pr[src] - w.pr[tgt] - 0.1, 0.0) / v_full))
        eta_i = min(int(eta_full), H)
        req = required_ships(w, tgt, eta_i, me)
        req_n = float(req[0]) if req else 0.0
        v_req = fleet_speed(max(1, int(req_n) or 1))
        eta_req = max(1.0, math.ceil(
            max(d - w.pr[src] - w.pr[tgt] - 0.1, 0.0) / v_req))

        own = w.owner0[tgt]
        po = w.post_owner[tgt]
        falls = 0.0
        flips_me = 0.0
        if own == me:
            for dt in range(1, min(H, 20) + 1):
                if po[dt] is not None and po[dt] not in (me, -2):
                    falls = 1.0
                    break
        else:
            for dt in range(1, min(H, 20) + 1):
                if po[dt] == me:
                    flips_me = 1.0
                    break
        garr_eta = float(w.pre_ships[tgt][eta_i]
                         if w.pre_ships[tgt][eta_i] is not None else 0.0)

        # fastest enemy response to tgt with a comparable stack
        best_e = 99.0
        for j in self.enemies:
            sz = w.ships0[j]
            if sz < max(4, req_n * 0.5):
                continue
            dj = math.hypot(w.px[j] - w.px[tgt], w.py[j] - w.py[tgt]) \
                - w.pr[j] - w.pr[tgt]
            eta = max(1.0, math.ceil(max(dj, 0.0)
                                     / fleet_speed(max(1, sz))))
            if eta < best_e:
                best_e = eta
        race = best_e - eta_full

        ttl_src = float(w.alive_until[src]) if w.is_comet[src] else 99.0
        ttl_tgt = float(w.alive_until[tgt]) if w.is_comet[tgt] else 99.0

        return [
            d, eta_full, eta_req,
            self.near_enemy_d[src], self.near_enemy_d[tgt],
            self.near_mine_d[tgt],
            float(w.ships0[src]), float(safe),
            safe / max(1.0, float(w.ships0[src])), float(w.prod[src]),
            self.inc_enemy[src], 1.0 if doomed else 0.0,
            1.0 if w.is_comet[src] else 0.0, min(ttl_src, 99.0),
            1.0 if own == me else 0.0,
            1.0 if own >= 0 and own != me else 0.0,
            1.0 if own == -1 else 0.0,
            float(w.prod[tgt]), float(w.pr[tgt]),
            float(w.ships0[tgt]), garr_eta, req_n,
            req_n / max(1.0, float(spare)),
            1.0 if w.is_comet[tgt] else 0.0, min(ttl_tgt, 99.0),
            falls, flips_me,
            self.inc_mine[tgt], self.inc_enemy[tgt],
            best_e, race,
        ] + self.globals
