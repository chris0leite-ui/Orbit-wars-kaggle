"""Bundle producer_plus into a single Kaggle-submittable .py file.

Producer_plus depends on agents/producer/orbit_lite/* (the vendored
Producer engine). The bundler concatenates all orbit_lite modules in
topological order, strips intra-package imports, appends
agents/producer_plus/main.py with its `from orbit_lite.X` imports
stripped, and bakes the env vars (PRODUCER_PLUS_ADAPTIVE_K=1,
PRODUCER_PLUS_MULTI_SIZE=1) at the top.

Output: submissions/producer_plus_multi_size_on.py

CLI:
    python scripts/bundle_producer_plus.py [--out PATH] [--variant adaptive_k|multi_size]
"""
from __future__ import annotations

import argparse
import ast
import os
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# Topological order: each module must come after its orbit_lite-internal deps.
ORBIT_LITE_ORDER = [
    "constants",
    "aiming",
    "geometry",
    "obs",
    "movement_aiming",
    "movement",
    "distance_cache",
    "garrison_launch",
    "intercept_aim",
    "movement_step",
    "adapter",
    "planner_core",
    # opp_projection depends on garrison_launch, intercept_aim, movement,
    # obs, planner_core — all above. Must come AFTER planner_core.
    "opp_projection",
    # recapture imports DistanceCache, fleet_speed, PlanetGarrisonStatus —
    # all from modules above. No interdependence with opp_projection.
    "recapture",
    # strategic_value imports LaunchSet (garrison_launch), PlanetGarrisonStatus
    # (movement). No interdependence with recapture or opp_projection.
    "strategic_value",
]

ORBIT_LITE_DIR = REPO / "agents" / "producer" / "orbit_lite"
PRODUCER_PLUS_MAIN = REPO / "agents" / "producer_plus" / "main.py"

ENV_VARIANTS = {
    "adaptive_k": {
        "PRODUCER_PLUS_ADAPTIVE_K": "1",
    },
    "multi_size": {
        # Adaptive_K (Step 2) deliberately OFF: 16-game seat-alt A/B
        # 2026-06-04 showed Step 2+4 regressed to 5/16 vs producer while
        # Step 4 alone hit 10/16. Adaptive_K is preserved in main.py as
        # a gated path for future tuning.
        "PRODUCER_PLUS_MULTI_SIZE": "1",
    },
    "coalitions": {
        # Step 5: L=2 multi-source coalitions packed alongside single-
        # source candidates. Multi_size (Step 4) deliberately OFF to
        # avoid the 3-size × C(K,2)-pair candidate explosion — compose
        # as Step 5b later only if both lift independently.
        "PRODUCER_PLUS_COALITIONS": "1",
    },
    "composed": {
        # Step 4 + Step 5: BOTH multi_size and coalitions ON. The plan.py
        # `plan_lite_waves` composed branch packs S*T*N + T*C(K,2) candidates
        # at L=2. Tests the hypothesis that coalitions only lift when paired
        # with multi-size single-source variants.
        "PRODUCER_PLUS_MULTI_SIZE": "1",
        "PRODUCER_PLUS_COALITIONS": "1",
    },
    "opp_proj": {
        # Step 3 redux: per-turn opp multi-launch projection injected as
        # background LaunchSet slots in the scorer. Multi_size and coalitions
        # deliberately OFF — this is the standalone variant testing the
        # opp-projection mechanism in isolation.
        "PRODUCER_PLUS_OPP_PROJECTION": "1",
    },
    "multi_opp_def": {
        # Step 4 (multi_size) + opp_projection (Producer-mirror) + the
        # opp-aware defensive shortlist augmentation in friendly_flip_targets
        # (which activates unconditionally when background is non-empty).
        # Coalitions deliberately OFF — diagnostic at seed 7 showed they
        # barely fire and actively hurt the kitchen-sink variant (-1 win).
        # Local n=16: 12/16 wins vs producer (Wilson [50%, 90%]).
        "PRODUCER_PLUS_MULTI_SIZE": "1",
        "PRODUCER_PLUS_OPP_PROJECTION": "1",
    },
    "multi_tick_opp_K3": {
        # Multi_opp_def + K-round opp projection. Opp's planner runs K
        # successive rounds, each round seeing prior rounds' projected
        # launches as `background`; per-round launches are eta-shifted by
        # +k before merging. Addresses the cycle stalemate diagnosis
        # (knowledge-base/thoughts/2026-06-05-cycle-stalemate-and-horizon-
        # scaling.md): scorer was blind past tick ~8 because opp_proj only
        # projected one tick. 4P stalemate is the target pathology; 2P
        # gets K=2 so the 2P A/B harness can still detect breakage.
        # Horizon bump intentionally NOT baked here — separate A/B once
        # this variant lands.
        "PRODUCER_PLUS_MULTI_SIZE": "1",
        "PRODUCER_PLUS_OPP_PROJECTION": "1",
        "PRODUCER_PLUS_MULTI_TICK_OPP_K_4P": "3",
        "PRODUCER_PLUS_MULTI_TICK_OPP_K_2P": "2",
    },
    "recapture_penalty": {
        # Standalone recapture penalty: per-candidate leaf-scorer discount
        # for thin captures opp can recapture. Tests the mechanism in
        # isolation (no multi_size, no opp_proj, no multi-tick). See
        # agents/producer/orbit_lite/recapture.py for the math.
        "PRODUCER_PLUS_RECAPTURE_PENALTY": "1",
    },
    "multi_tick_recap": {
        # Composed: multi_size + opp_proj + multi-tick + recapture penalty.
        # The path that ships if the standalone A/B clears. Recapture
        # penalty's K_recap_eff = max(1, K_recap - K_opp) clips the
        # window to past what multi-tick already modeled, avoiding
        # double-counting.
        "PRODUCER_PLUS_MULTI_SIZE": "1",
        "PRODUCER_PLUS_OPP_PROJECTION": "1",
        "PRODUCER_PLUS_MULTI_TICK_OPP_K_4P": "3",
        "PRODUCER_PLUS_MULTI_TICK_OPP_K_2P": "2",
        "PRODUCER_PLUS_RECAPTURE_PENALTY": "1",
    },
    "denial": {
        # Standalone denial bonus. Requires opp_proj to be ON so opp_intent
        # in the background LaunchSet contributes; otherwise denial only
        # fires for already-opp-owned targets (no race-for-neutral
        # component).
        "PRODUCER_PLUS_OPP_PROJECTION": "1",
        "PRODUCER_PLUS_DENIAL_BONUS": "1",
    },
    "opening": {
        # Standalone opening bonus. Opp-agnostic, no opp_proj required.
        "PRODUCER_PLUS_OPENING_BONUS": "1",
    },
    "strategic": {
        # Both new bonuses on, with opp_proj on so denial works fully.
        "PRODUCER_PLUS_OPP_PROJECTION": "1",
        "PRODUCER_PLUS_DENIAL_BONUS": "1",
        "PRODUCER_PLUS_OPENING_BONUS": "1",
    },
    "multi_tick_strategic": {
        # Full composed: multi_size + opp_proj + multi-tick + recap +
        # denial + opening. The path that ships if the strategic A/B
        # confirms lift.
        "PRODUCER_PLUS_MULTI_SIZE": "1",
        "PRODUCER_PLUS_OPP_PROJECTION": "1",
        "PRODUCER_PLUS_MULTI_TICK_OPP_K_4P": "3",
        "PRODUCER_PLUS_MULTI_TICK_OPP_K_2P": "2",
        "PRODUCER_PLUS_RECAPTURE_PENALTY": "1",
        "PRODUCER_PLUS_DENIAL_BONUS": "1",
        "PRODUCER_PLUS_OPENING_BONUS": "1",
    },
    "force_concentration": {
        # Standalone force-concentration: relax the one-wave-per-target mutex
        # in _greedy_select to allow up to MAX_WAVES (default 2). Between
        # waves the candidates are re-scored against the committed waves so
        # wave 2 to a target sees wave 1's reinforcement (no double-count).
        # Tests the chooser-architecture lever in isolation against vanilla
        # producer; no opp_proj / multi-tick / scorer-term stack.
        "PRODUCER_PLUS_FORCE_CONCENTRATION": "1",
    },
    "multi_tick_force_concentration": {
        # Composed: multi_size + opp_proj + multi-tick + recap + force-
        # concentration. The path that ships if the standalone A/B clears.
        # Force-concentration relaxes the chooser's one-wave-per-target mutex
        # on top of the multi_tick_recap stack so high-value targets can be
        # reinforced rather than left under-funded while ships scatter.
        "PRODUCER_PLUS_MULTI_SIZE": "1",
        "PRODUCER_PLUS_OPP_PROJECTION": "1",
        "PRODUCER_PLUS_MULTI_TICK_OPP_K_4P": "3",
        "PRODUCER_PLUS_MULTI_TICK_OPP_K_2P": "2",
        "PRODUCER_PLUS_RECAPTURE_PENALTY": "1",
        "PRODUCER_PLUS_FORCE_CONCENTRATION": "1",
    },
    "denial_calibrated": {
        # Denial bonus on the proven multi_opp_def base, with the weight set
        # from the 2026-06-10 calibration probe (one full game, 141k candidate
        # scores): median acted-on competitive_score = 48 ship units; denial
        # bonus at weight 1.0 has median 354. Weight 0.01 puts the median
        # bonus at ~3.5 ship units = ~7% of the acted-on median (the 5-15%
        # nudge band from the 2026-06-05 lesson). The 0.1 default that
        # regressed 0/4 put it at ~35 = ~74%.
        "PRODUCER_PLUS_MULTI_SIZE": "1",
        "PRODUCER_PLUS_OPP_PROJECTION": "1",
        "PRODUCER_PLUS_DENIAL_BONUS": "1",
        "PRODUCER_PLUS_DENIAL_WEIGHT": "0.01",
    },
    "opening_calibrated": {
        # Opening bonus on the proven multi_opp_def base; same probe: opening
        # bonus at weight 1.0 has median 60 -> weight 0.04 ~= 2.4 ship units
        # = ~5% of the acted-on median.
        "PRODUCER_PLUS_MULTI_SIZE": "1",
        "PRODUCER_PLUS_OPP_PROJECTION": "1",
        "PRODUCER_PLUS_OPENING_BONUS": "1",
        "PRODUCER_PLUS_OPENING_WEIGHT": "0.04",
    },
    "strategic_calibrated": {
        # Both calibrated bonuses on the multi_opp_def base.
        "PRODUCER_PLUS_MULTI_SIZE": "1",
        "PRODUCER_PLUS_OPP_PROJECTION": "1",
        "PRODUCER_PLUS_DENIAL_BONUS": "1",
        "PRODUCER_PLUS_DENIAL_WEIGHT": "0.01",
        "PRODUCER_PLUS_OPENING_BONUS": "1",
        "PRODUCER_PLUS_OPENING_WEIGHT": "0.04",
    },
    "ffa_score": {
        # The proven multi_opp_def stack + the FFA objective fix: in 3+ player
        # games the opponent term of competitive_score becomes a strength-
        # weighted AVERAGE over rivals instead of an equal-weight SUM, so
        # mutual-damage trades stop scoring positive and damage is valued by
        # how much it shifts our standing vs the rivals that threaten us.
        # 2P path is byte-identical by construction (weights only built when
        # player_count >= 3). Motivated by the 2026-06-10 live-replay
        # diagnosis: 4P = 60% of ladder games, our 4P winrate 29%, 82/83 4P
        # losses end eliminated by 2+ opponents carving us mid-game.
        "PRODUCER_PLUS_MULTI_SIZE": "1",
        "PRODUCER_PLUS_OPP_PROJECTION": "1",
        "PRODUCER_PLUS_FFA_SCORE": "1",
    },
    "ffa_uniform": {
        # ffa_score with equal weight per living rival instead of strength-
        # proportional: isolates the trade-devaluation effect from the
        # hit-the-leader tilt.
        "PRODUCER_PLUS_MULTI_SIZE": "1",
        "PRODUCER_PLUS_OPP_PROJECTION": "1",
        "PRODUCER_PLUS_FFA_SCORE": "1",
        "PRODUCER_PLUS_FFA_WEIGHTS": "uniform",
    },
    "opp_def_h24": {
        # The proven multi_opp_def stack with the scorer horizon lifted from
        # the default 18 to 24 ticks (the 5feabd8 knob, flagged "separate A/B
        # once this variant lands" in the 2026-06-05 wrap and never run).
        # Motivation: the cycle-stalemate diagnosis — the scorer undervalues
        # holding/stockpiling because it can't see past H. Cost is linear in
        # H (multi_opp_def p50 62 ms -> ~85 ms, far under the 1000 ms cap).
        "PRODUCER_PLUS_MULTI_SIZE": "1",
        "PRODUCER_PLUS_OPP_PROJECTION": "1",
        "PRODUCER_PLUS_HORIZON_2P": "24",
        "PRODUCER_PLUS_HORIZON_4P": "24",
    },
    "opp_def_force_concentration": {
        # The proven multi_opp_def stack (multi_size + opp_proj, local 24/32
        # = 75% vs producer, live mu 1263-1285) with force-concentration added
        # and NOTHING else. Both live/local regressions to date
        # (multi_tick_recap mu=1099, lean_force_concentration 7/32) carry the
        # recapture penalty; this variant tests FC on the proven base without
        # that suspect term and without multi-tick.
        "PRODUCER_PLUS_MULTI_SIZE": "1",
        "PRODUCER_PLUS_OPP_PROJECTION": "1",
        "PRODUCER_PLUS_FORCE_CONCENTRATION": "1",
    },
    "lean_force_concentration": {
        # Same scorer stack as multi_tick_force_concentration MINUS the
        # opp-projection mechanism (and the multi-tick K-round expansion of
        # it). Tests "does the producer-mirror opp model still pull its
        # weight once force-concentration stops the source-scatter that the
        # opp_proj defensive shortlist was patching?" If parity-or-better
        # vs the with-opp variant, the opp model is dead weight and the
        # cheaper variant ships.
        "PRODUCER_PLUS_MULTI_SIZE": "1",
        "PRODUCER_PLUS_RECAPTURE_PENALTY": "1",
        "PRODUCER_PLUS_FORCE_CONCENTRATION": "1",
    },
    "tick4p": {
        # The proven multi_opp_def stack + multi-tick opp projection in
        # 4P ONLY (K_4P=3; no _2P key baked, so 2P falls back to the
        # common-key default 0 -> K_opp = max(1, 0) = 1 = single-tick =
        # byte-identical to the live champion in 2P). Motivation
        # (2026-06-10 economy mining): 4P losses are decided in the
        # step-20..80 brawl window — production peaks ~step 40 then
        # declines while the eventual winner's doubles; the planner is
        # blind to rival launches beyond the current tick, so it neither
        # anticipates the incoming brawl waves nor times its own. The
        # earlier multi-tick live regression (sub 53390700, mu 1099) was
        # measured composed with the recapture penalty AND with K_2P=2
        # active in the 2P games that dominated that evaluation — multi-
        # tick standalone in 4P was never measured (clean_ffa did not
        # exist yet).
        "PRODUCER_PLUS_MULTI_SIZE": "1",
        "PRODUCER_PLUS_OPP_PROJECTION": "1",
        "PRODUCER_PLUS_MULTI_TICK_OPP_K_4P": "3",
    },
    "reinforce_deficit": {
        # The proven multi_opp_def stack + the defense candidate-sizing fix:
        # owned targets the projection shows flipping at tick k_f get a
        # pre-flip reinforcement floor of (post-flip survivor + 1) — the
        # exact minimum send that holds the planet — instead of 1. The
        # multi-size enumeration then carries a right-sized hold candidate
        # (instead of junk 1/2-ship sends), and doomed under-sized trickles
        # are gated invalid. Motivated by the 2026-06-10 loss-anatomy
        # mining: losses are decided by production retention in the
        # step-20..80 brawl window.
        "PRODUCER_PLUS_MULTI_SIZE": "1",
        "PRODUCER_PLUS_OPP_PROJECTION": "1",
        "PRODUCER_PLUS_REINFORCE_DEFICIT": "1",
    },
    "overkill2": {
        # Mass-concentration attack sizing on the proven multi_opp_def
        # stack: the lo/mid multi-size variants are sized at 2x/4x the
        # projected defense (capped by safe_drain) instead of the bare
        # capture floor. Motivated by top-ladder behavioral mining
        # (audit/2026-06-10-top-ladder-behavior.md): 1600-1750 agents
        # launch ~half as often with 2-4x the fleet mass; in our own 2P
        # losses the opponent's median fleet is ~2x ours. Decisive
        # captures survive the counter-attack past the scorer horizon;
        # marginal ones churn.
        "PRODUCER_PLUS_MULTI_SIZE": "1",
        "PRODUCER_PLUS_OPP_PROJECTION": "1",
        "PRODUCER_PLUS_OVERKILL_FACTOR": "2.0",
    },
    "overkill3": {
        # Same with factor 3 — brackets the mass knob.
        "PRODUCER_PLUS_MULTI_SIZE": "1",
        "PRODUCER_PLUS_OPP_PROJECTION": "1",
        "PRODUCER_PLUS_OVERKILL_FACTOR": "3.0",
    },
    "mass": {
        # Mass-concentration composite on the proven multi_opp_def stack,
        # from the top-ladder behavioral mining (the 1600-1750 agents
        # launch ~half as often with 2-4x the fleet mass; our 2P losses
        # correlate with big-fleet opponents): score near-ties resolve
        # toward the larger send, the regroup lane convoys (>= 25 ships)
        # instead of dribbling 10-30-ship parcels, and lo/mid attack
        # variants are sized at 2x/4x the projected defense.
        "PRODUCER_PLUS_MULTI_SIZE": "1",
        "PRODUCER_PLUS_OPP_PROJECTION": "1",
        "PRODUCER_PLUS_MASS_TIEBREAK": "1",
        "PRODUCER_PLUS_REGROUP_MIN_SEND": "25",
        "PRODUCER_PLUS_OVERKILL_FACTOR": "2.0",
    },
    "mass2p_ffa": {
        # THE COMPOSED CANDIDATE: best-known 2P play + best-known 4P play
        # in one bundle, by player-count gating.
        # - 2P: mass mechanisms active (tiebreak + convoy 25 + overkill 2)
        #   -> beats the champion head-to-head 35/64, holds 22/32 vs
        #   producer. FFA score inactive in 2P by construction.
        # - 4P: mass gated OFF (cost first-place rate in the 4P pool),
        #   FFA uniform objective active -> champion 4P behavior + the
        #   trade-devaluation fix (live A/B sub 53527125).
        # Verification is by action-stream parity to the measured bundles
        # (2P == mass, 4P == ffa_uniform), so both pool results transfer.
        "PRODUCER_PLUS_MULTI_SIZE": "1",
        "PRODUCER_PLUS_OPP_PROJECTION": "1",
        "PRODUCER_PLUS_MASS_TIEBREAK": "1",
        "PRODUCER_PLUS_REGROUP_MIN_SEND": "25",
        "PRODUCER_PLUS_OVERKILL_FACTOR": "2.0",
        "PRODUCER_PLUS_MASS_2P_ONLY": "1",
        "PRODUCER_PLUS_FFA_SCORE": "1",
        "PRODUCER_PLUS_FFA_WEIGHTS": "uniform",
    },
    "termval12": {
        # Terminal production value, standalone attribution: the flow scorer
        # truncates a captured planet's payoff at the horizon, so neutral
        # captures whose production only repays the garrison cost in-horizon
        # score ~0 and never clear the roi threshold (seed-7 expansion
        # probe: dozens of valid neutral candidates per opening turn at
        # best-score 0..1 while the bank climbed to ~300). Credits the
        # production owned at the horizon's final step for 12 further steps.
        "PRODUCER_PLUS_MULTI_SIZE": "1",
        "PRODUCER_PLUS_OPP_PROJECTION": "1",
        "PRODUCER_PLUS_TERMINAL_PROD_VALUE": "12",
    },
    "mass_termval12": {
        # Mass mechanisms + terminal production value (the expansion fix).
        "PRODUCER_PLUS_MULTI_SIZE": "1",
        "PRODUCER_PLUS_OPP_PROJECTION": "1",
        "PRODUCER_PLUS_MASS_TIEBREAK": "1",
        "PRODUCER_PLUS_REGROUP_MIN_SEND": "25",
        "PRODUCER_PLUS_OVERKILL_FACTOR": "2.0",
        "PRODUCER_PLUS_TERMINAL_PROD_VALUE": "12",
    },
    "mass_splitovk": {
        # Class-split overkill from top-3 replay mining: ~1.3x garrison on
        # neutrals (cheap front-loaded expansion), 4x on enemy planets
        # (decisive strikes that survive the counter-punch; their median is
        # 2.6-4.6x with 60-89-ship fleets vs our 40).
        "PRODUCER_PLUS_MULTI_SIZE": "1",
        "PRODUCER_PLUS_OPP_PROJECTION": "1",
        "PRODUCER_PLUS_MASS_TIEBREAK": "1",
        "PRODUCER_PLUS_REGROUP_MIN_SEND": "25",
        "PRODUCER_PLUS_OVERKILL_FACTOR": "1.3",
        "PRODUCER_PLUS_OVERKILL_FACTOR_ENEMY": "4.0",
    },
    "mass_termval12_splitovk": {
        # The full master-agent candidate: mass + terminal production value
        # (expansion fix) + class-split overkill (strike sizing fix).
        "PRODUCER_PLUS_MULTI_SIZE": "1",
        "PRODUCER_PLUS_OPP_PROJECTION": "1",
        "PRODUCER_PLUS_MASS_TIEBREAK": "1",
        "PRODUCER_PLUS_REGROUP_MIN_SEND": "25",
        "PRODUCER_PLUS_OVERKILL_FACTOR": "1.3",
        "PRODUCER_PLUS_OVERKILL_FACTOR_ENEMY": "4.0",
        "PRODUCER_PLUS_TERMINAL_PROD_VALUE": "12",
    },
    "veto_only": {
        # Champion stack + response veto, NO mass knobs. The mass knobs are
        # a confirmed LIVE 2P regression (59% vs champion code's 71% on the
        # drift-free pair, 2026-06-10); the veto addresses a defect the
        # champion also has (attacks into anticipated parries: 30% of
        # capture-sized attacks fail, 65% to in-flight reinforcement).
        "PRODUCER_PLUS_MULTI_SIZE": "1",
        "PRODUCER_PLUS_OPP_PROJECTION": "1",
        "PRODUCER_PLUS_RESPONSE_VETO": "1",
    },
    "veto2p_ffa": {
        # SUBMISSION CANDIDATE (2026-06-10 evening): champion 2P + response
        # veto, 4P byte-identical to ffa_uniform (sub 53527125 — the live
        # 4P improvement: 37% vs champion's 29%). Replaces the mass2p_ffa
        # 2P half (confirmed live regression, 59% vs 71%). The veto stops
        # attacks into anticipated parries (30% of capture-sized attacks
        # failed, 65% to in-flight reinforcement, ~321 ships/game).
        "PRODUCER_PLUS_MULTI_SIZE": "1",
        "PRODUCER_PLUS_OPP_PROJECTION": "1",
        "PRODUCER_PLUS_RESPONSE_VETO": "1",
        "PRODUCER_PLUS_RESPONSE_VETO_2P_ONLY": "1",
        "PRODUCER_PLUS_FFA_SCORE": "1",
        "PRODUCER_PLUS_FFA_WEIGHTS": "uniform",
    },
    "veto_snipe": {
        # Veto + snipe-hold: drop parried attacks AND reserve idle ships
        # that have a dated toll-snipe appointment (opponent flip at k_f ->
        # arrive k_f+1 for survivor+1) instead of regrouping them away.
        "PRODUCER_PLUS_MULTI_SIZE": "1",
        "PRODUCER_PLUS_OPP_PROJECTION": "1",
        "PRODUCER_PLUS_RESPONSE_VETO": "1",
        "PRODUCER_PLUS_SNIPE_HOLD": "1",
    },
    "veto_ntv6": {
        # Veto + NEUTRAL-ONLY terminal production value (lambda=6): the
        # SiestaGuru loss anatomy — zero neutral captures steps 60-100 while
        # they bought 4 planets, plus 700 ships into failed enemy strikes.
        # Full termval failed in duels because its enemy-capture credit
        # counts DOUBLE and amplified doomed aggression; the neutral-only
        # version encourages expansion without touching strike valuation.
        "PRODUCER_PLUS_MULTI_SIZE": "1",
        "PRODUCER_PLUS_OPP_PROJECTION": "1",
        "PRODUCER_PLUS_RESPONSE_VETO": "1",
        "PRODUCER_PLUS_TERMINAL_PROD_VALUE": "6",
        "PRODUCER_PLUS_TERMINAL_PROD_NEUTRAL_ONLY": "1",
    },
    "veto_nq": {
        # Veto + neutral shortlist quota (6): fixes the VISIBILITY half of
        # the expansion stall — once a frontline forms, the proximity
        # shortlist fills with enemy planets and neutrals are never even
        # scored (SiestaGuru: zero neutral captures for 40 steps).
        "PRODUCER_PLUS_MULTI_SIZE": "1",
        "PRODUCER_PLUS_OPP_PROJECTION": "1",
        "PRODUCER_PLUS_RESPONSE_VETO": "1",
        "PRODUCER_PLUS_NEUTRAL_SHORTLIST": "6",
    },
    "vetorf2p_ffa": {
        # SUBMISSION CANDIDATE (2026-06-10 night): live veto2p_ffa + the
        # reactive floor (the only mechanism with positive attribution over
        # plain veto: ahead 7/7 games @80, +21.5% paired @120). 2P-gated;
        # 4P byte-identical to ffa_uniform / live sub 53542171.
        "PRODUCER_PLUS_MULTI_SIZE": "1",
        "PRODUCER_PLUS_OPP_PROJECTION": "1",
        "PRODUCER_PLUS_RESPONSE_VETO": "1",
        "PRODUCER_PLUS_RESPONSE_VETO_2P_ONLY": "1",
        "PRODUCER_PLUS_REACTIVE_FLOOR": "0.5",
        "PRODUCER_PLUS_REACTIVE_FLOOR_2P_ONLY": "1",
        "PRODUCER_PLUS_FFA_SCORE": "1",
        "PRODUCER_PLUS_FFA_WEIGHTS": "uniform",
    },
    "vetorf4p_ffa": {
        # 4P UNGATING measurement: veto + reactive floor active in ALL
        # player counts (no _2P_ONLY gates) on the FFA-objective stack.
        # 4P is 60% of ladder volume and every shipped mechanism is
        # 2P-gated there — the live agent plays 4P as plain ffa_uniform.
        "PRODUCER_PLUS_MULTI_SIZE": "1",
        "PRODUCER_PLUS_OPP_PROJECTION": "1",
        "PRODUCER_PLUS_RESPONSE_VETO": "1",
        "PRODUCER_PLUS_REACTIVE_FLOOR": "0.5",
        "PRODUCER_PLUS_FFA_SCORE": "1",
        "PRODUCER_PLUS_FFA_WEIGHTS": "uniform",
    },
    "vetorf4p_seq": {
        # 4P ungating attempt #2: vetorf4p_ffa + sequential reply
        # conditioning. The independent merge triple-counts defense (every
        # attack priced as if all 3 rivals parry simultaneously) -> chronic
        # passivity, eliminated ~step 200, panel 1/16. REPLY_SEQ conditions
        # each rival on earlier rivals' predicted launches.
        "PRODUCER_PLUS_MULTI_SIZE": "1",
        "PRODUCER_PLUS_OPP_PROJECTION": "1",
        "PRODUCER_PLUS_RESPONSE_VETO": "1",
        "PRODUCER_PLUS_REACTIVE_FLOOR": "0.5",
        "PRODUCER_PLUS_REPLY_SEQ": "1",
        "PRODUCER_PLUS_FFA_SCORE": "1",
        "PRODUCER_PLUS_FFA_WEIGHTS": "uniform",
    },
    "rf4p_ffa": {
        # 4P ungating, reactive floor ONLY (veto stays 2P-gated): the floor
        # prices reactive defense into attack sizing — anti-waste without
        # the veto's passivity risk. 2P behavior identical to the live
        # vetorf2p_ffa stack.
        "PRODUCER_PLUS_MULTI_SIZE": "1",
        "PRODUCER_PLUS_OPP_PROJECTION": "1",
        "PRODUCER_PLUS_RESPONSE_VETO": "1",
        "PRODUCER_PLUS_RESPONSE_VETO_2P_ONLY": "1",
        "PRODUCER_PLUS_REACTIVE_FLOOR": "0.5",
        "PRODUCER_PLUS_FFA_SCORE": "1",
        "PRODUCER_PLUS_FFA_WEIGHTS": "uniform",
    },
    "vetorf2p_open": {
        # Live 2P stack + in-agent opening search (beam over neutral-capture
        # schedules, first 25 steps, ported from opening_optimum.py). The
        # decision-step finding says opening production IS the game; the
        # champion measured ~3.6% below the beam benchmark.
        "PRODUCER_PLUS_MULTI_SIZE": "1",
        "PRODUCER_PLUS_OPP_PROJECTION": "1",
        "PRODUCER_PLUS_RESPONSE_VETO": "1",
        "PRODUCER_PLUS_REACTIVE_FLOOR": "0.5",
        "PRODUCER_PLUS_OPENING_SEARCH": "25",
    },
    "vetorf_rescue": {
        # Coalition rescue (zero new code): live stack + deficit-sized
        # defense floors + L=2 coalitions. The hold-rate gap (0.59 vs the
        # top teams' 0.74-0.85) is killed by avalanche waves one neighbour
        # can't match; coalitions let two neighbours fund the deficit. The
        # old anti-coalition verdict predates the floor era and used the
        # failed vs-producer referee.
        "PRODUCER_PLUS_MULTI_SIZE": "1",
        "PRODUCER_PLUS_OPP_PROJECTION": "1",
        "PRODUCER_PLUS_RESPONSE_VETO": "1",
        "PRODUCER_PLUS_REACTIVE_FLOOR": "0.5",
        "PRODUCER_PLUS_REINFORCE_DEFICIT": "1",
        "PRODUCER_PLUS_COALITIONS": "1",
    },
    "vetorf_trust": {
        # Live stack + online opponent-model verification: replies priced
        # at EMA-recall strength (matched by source planet + owner, ships
        # within 2x). Producer-likes: trust ~1, unchanged. Originals: the
        # veto stops parrying ghosts.
        "PRODUCER_PLUS_MULTI_SIZE": "1",
        "PRODUCER_PLUS_OPP_PROJECTION": "1",
        "PRODUCER_PLUS_RESPONSE_VETO": "1",
        "PRODUCER_PLUS_REACTIVE_FLOOR": "0.5",
        "PRODUCER_PLUS_REPLY_TRUST": "1",
    },
    "vetorf_cc": {
        # Commitment cost (ledger-branch port): each candidate pays
        # eps x ships x flight-turns. In-flight capital can't change course;
        # top teams strike at eta 4-5 vs our 7-8, and the replan family
        # measured that ships held home beat schemes for spending them.
        # eps=0.02 ~ a 50-ship 6-turn mission pays 6 ships of score.
        "PRODUCER_PLUS_MULTI_SIZE": "1",
        "PRODUCER_PLUS_OPP_PROJECTION": "1",
        "PRODUCER_PLUS_RESPONSE_VETO": "1",
        "PRODUCER_PLUS_REACTIVE_FLOOR": "0.5",
        "PRODUCER_PLUS_COMMIT_COST": "0.02",
    },
    "vetorf4p_seq_strength": {
        # vetorf4p_seq with STRENGTH-weighted FFA objective (leader focus).
        # Ledger-branch evidence that survives their dead-opponent
        # correction: 4P leader objective 9/16 vs 4/16 parity baseline,
        # and 64% live 4P. Our strength mode = damage valued by rival
        # strength = focus the leader; zero new code.
        "PRODUCER_PLUS_MULTI_SIZE": "1",
        "PRODUCER_PLUS_OPP_PROJECTION": "1",
        "PRODUCER_PLUS_RESPONSE_VETO": "1",
        "PRODUCER_PLUS_REACTIVE_FLOOR": "0.5",
        "PRODUCER_PLUS_REPLY_SEQ": "1",
        "PRODUCER_PLUS_FFA_SCORE": "1",
        "PRODUCER_PLUS_FFA_WEIGHTS": "strength",
    },
    "vetorf_fwd": {
        # Forward redistribution (Planet Wars canon + del Toro loss): rear
        # leftover garrisons stream toward the frontier every turn (pressure
        # delta gate 0.25 -> 0, flight cap 7 -> 12). Top-2010 lesson:
        # "reinforcement toward the front: simple-minded but works great";
        # the winner scored every friendly transfer with a be-near-the-enemy
        # positional term.
        "PRODUCER_PLUS_MULTI_SIZE": "1",
        "PRODUCER_PLUS_OPP_PROJECTION": "1",
        "PRODUCER_PLUS_RESPONSE_VETO": "1",
        "PRODUCER_PLUS_REACTIVE_FLOOR": "0.5",
        "PRODUCER_PLUS_REGROUP_FORWARD": "1",
    },
    "veto_rf": {
        # Veto + reactive floor (weight 0.5): enemy floors include half the
        # garrison the defender can route to the target within our flight
        # (minus a 2-turn reaction lag). Activates capture_floor's dormant
        # reinforcement hook — the 700-wasted-ships channel vs SiestaGuru.
        "PRODUCER_PLUS_MULTI_SIZE": "1",
        "PRODUCER_PLUS_OPP_PROJECTION": "1",
        "PRODUCER_PLUS_RESPONSE_VETO": "1",
        "PRODUCER_PLUS_REACTIVE_FLOOR": "0.5",
    },
    "vetorf4p_sync_garval": {
        # NIGHT BUILD: live 4P stack + the balance-of-force pair. Garrison
        # value prices PROACTIVE reinforcement of own planets whose local
        # balance vs uncommitted enemy reserves is negative (the live war
        # ledger: 4P winner = whoever reinforces more, 17x gap vs Blu3s);
        # source safety caps drains by the same model on the push side.
        "PRODUCER_PLUS_MULTI_SIZE": "1",
        "PRODUCER_PLUS_OPP_PROJECTION": "1",
        "PRODUCER_PLUS_RESPONSE_VETO": "1",
        "PRODUCER_PLUS_REACTIVE_FLOOR": "0.5",
        "PRODUCER_PLUS_REPLY_SEQ": "1",
        "PRODUCER_PLUS_FFA_SCORE": "1",
        "PRODUCER_PLUS_FFA_WEIGHTS": "strength",
        "PRODUCER_PLUS_SYNC": "1",
        "PRODUCER_PLUS_SOURCE_SAFETY": "0.5",
        "PRODUCER_PLUS_GARRISON_VALUE": "12",
    },
    "vetorf_srcsafe": {
        # Source-safety drain cap alone on the live 2P stack. Caps drain by
        # local balance of force (enemy uncommitted reserve vs production
        # growth + routable friendly help). Regression leg: must not induce
        # banker-grade passivity.
        "PRODUCER_PLUS_MULTI_SIZE": "1",
        "PRODUCER_PLUS_OPP_PROJECTION": "1",
        "PRODUCER_PLUS_RESPONSE_VETO": "1",
        "PRODUCER_PLUS_REACTIVE_FLOOR": "0.5",
        "PRODUCER_PLUS_SOURCE_SAFETY": "0.5",
    },
    "vetorf_holdval12_srcsafe": {
        # Rule 38 fix-verification pair: the holding-time capture credit
        # (routed 0/12 in the mirror via strikes on drained sources) PLUS
        # the source-safety cap that prices exactly that strike. If the
        # blindspot diagnosis is right, the rout disappears.
        "PRODUCER_PLUS_MULTI_SIZE": "1",
        "PRODUCER_PLUS_OPP_PROJECTION": "1",
        "PRODUCER_PLUS_RESPONSE_VETO": "1",
        "PRODUCER_PLUS_REACTIVE_FLOOR": "0.5",
        "PRODUCER_PLUS_HOLD_VALUE": "12",
        "PRODUCER_PLUS_SOURCE_SAFETY": "0.5",
    },
    "vetorf_holdval12": {
        # FOUNDATION fix v2: holding-time-priced capture credit. Post-
        # horizon production (lambda=12) credited ONLY for captures the
        # opponent cannot feasibly retake (projected garrison vs full
        # routable enemy mass at every later tick). Flat termval12 was
        # refuted (unsafe expansion punished before payback); this version
        # unlocks exactly the safe rear expansions the paralysis trace
        # showed scoring +0.0. Priced consistently in the veto re-score.
        "PRODUCER_PLUS_MULTI_SIZE": "1",
        "PRODUCER_PLUS_OPP_PROJECTION": "1",
        "PRODUCER_PLUS_RESPONSE_VETO": "1",
        "PRODUCER_PLUS_REACTIVE_FLOOR": "0.5",
        "PRODUCER_PLUS_HOLD_VALUE": "12",
    },
    "vetorf_termval12": {
        # FOUNDATION experiment: live stack + terminal production value
        # (λ=12 post-horizon steps credited for horizon-end ownership).
        # Built 2026-06-10 from ladder mining (top agents hold 8 planets by
        # step 40; in-horizon flow truncates capture payoffs) but never
        # measured on this stack — only composed with the dead mass lineage.
        # Independently re-derived 2026-06-11 from live losses of 53564198:
        # 2P games are decided by the production race at steps 40-70
        # (wins: prod ahead 16/17 @40; losses: behind 9/17, -8 median @70;
        # in-flight share NOT discriminative).
        "PRODUCER_PLUS_MULTI_SIZE": "1",
        "PRODUCER_PLUS_OPP_PROJECTION": "1",
        "PRODUCER_PLUS_RESPONSE_VETO": "1",
        "PRODUCER_PLUS_REACTIVE_FLOOR": "0.5",
        "PRODUCER_PLUS_TERMINAL_PROD_VALUE": "12",
    },
    "vetorf4p_sync": {
        # LIVE EXPERIMENT candidate: the current live submission's full env
        # (vetorf4p_seq_strength) + same-tick two-source coalitions
        # (PRODUCER_PLUS_SYNC, holds off at default DMAX=0). Local evidence:
        # exact mirror parity (7/12, -0.7%@250) — purpose is ladder
        # information on whether the multi-source capture capability
        # matters against the real field.
        "PRODUCER_PLUS_MULTI_SIZE": "1",
        "PRODUCER_PLUS_OPP_PROJECTION": "1",
        "PRODUCER_PLUS_RESPONSE_VETO": "1",
        "PRODUCER_PLUS_REACTIVE_FLOOR": "0.5",
        "PRODUCER_PLUS_REPLY_SEQ": "1",
        "PRODUCER_PLUS_FFA_SCORE": "1",
        "PRODUCER_PLUS_FFA_WEIGHTS": "strength",
        "PRODUCER_PLUS_SYNC": "1",
    },
    "vetorf_sync": {
        # Live stack + same-tick two-source coalitions in the multi-size
        # path: pair candidates on targets neither source cracks alone,
        # floor-proportional sizing. SYNC_DMAX defaults to 0 — the delayed
        # (hold) variant lost -46% to the mirror (telegraphed far leg);
        # override PRODUCER_PLUS_SYNC_DMAX>0 to re-enable holds.
        "PRODUCER_PLUS_MULTI_SIZE": "1",
        "PRODUCER_PLUS_OPP_PROJECTION": "1",
        "PRODUCER_PLUS_RESPONSE_VETO": "1",
        "PRODUCER_PLUS_REACTIVE_FLOOR": "0.5",
        "PRODUCER_PLUS_SYNC": "1",
    },
    "veto_rf_nq": {
        # Stacking test: the two tournament survivors together. Reactive
        # floor converted (6/8 wins, +34% @120) but the quota's +31% @80
        # early-expansion lead leaked away late — hypothesis: floor-aware
        # strike sizing is what the quota's thin frontier was missing.
        "PRODUCER_PLUS_MULTI_SIZE": "1",
        "PRODUCER_PLUS_OPP_PROJECTION": "1",
        "PRODUCER_PLUS_RESPONSE_VETO": "1",
        "PRODUCER_PLUS_REACTIVE_FLOOR": "0.5",
        "PRODUCER_PLUS_NEUTRAL_SHORTLIST": "6",
    },
    "vetorf_replan": {
        # One-ply replan ON TOP of the current best stack (veto + reactive
        # floor): pass 1 plans, the mirror predicts the reply, pass 2
        # re-plans against it (redirecting vetoed ships + planning defenses),
        # then the veto verifies pass 2 against a fresh reply.
        "PRODUCER_PLUS_MULTI_SIZE": "1",
        "PRODUCER_PLUS_OPP_PROJECTION": "1",
        "PRODUCER_PLUS_RESPONSE_VETO": "1",
        "PRODUCER_PLUS_REACTIVE_FLOOR": "0.5",
        "PRODUCER_PLUS_REPLAN": "1",
    },
    "vetorf_redirect": {
        # The replan's fix: keep plan->veto unchanged; when waves are
        # dropped, ONE extra pass spends only the freed budget (surviving
        # waves committed: sources debited, effects + reply in background).
        # No reopened commitments -> no phantom-parry under-aggression
        # (replan's measured failure: 16 capture-sized launches vs 24).
        "PRODUCER_PLUS_MULTI_SIZE": "1",
        "PRODUCER_PLUS_OPP_PROJECTION": "1",
        "PRODUCER_PLUS_RESPONSE_VETO": "1",
        "PRODUCER_PLUS_REACTIVE_FLOOR": "0.5",
        "PRODUCER_PLUS_REDIRECT": "1",
    },
    "replan_rf": {
        # Replan WITHOUT the veto: does the full re-plan subsume the
        # drop-only filter? (Reactive floor kept — orthogonal mechanism.)
        "PRODUCER_PLUS_MULTI_SIZE": "1",
        "PRODUCER_PLUS_OPP_PROJECTION": "1",
        "PRODUCER_PLUS_REACTIVE_FLOOR": "0.5",
        "PRODUCER_PLUS_REPLAN": "1",
    },
    "vetorf_deficit": {
        # Reinforce-deficit re-judge under the CORRECT referees (its only
        # prior verdict was the failed vs-producer 4P panel): pre-flip
        # reinforcement floors raised to the hold-the-planet deficit, on
        # top of the veto + reactive-floor stack. Hold-rate gap vs top
        # teams: 0.59 ours vs 0.74-0.85 theirs.
        "PRODUCER_PLUS_MULTI_SIZE": "1",
        "PRODUCER_PLUS_OPP_PROJECTION": "1",
        "PRODUCER_PLUS_RESPONSE_VETO": "1",
        "PRODUCER_PLUS_REACTIVE_FLOOR": "0.5",
        "PRODUCER_PLUS_REINFORCE_DEFICIT": "1",
    },
    "vetorf_bgf": {
        # Background-aware floors on the live stack: the sizing subsystem
        # (capture floors, defensive shortlist, safe_drain) reads garrison
        # trajectories with the opponent's PREDICTED launches applied
        # (exact recurrence), instead of the frozen do-nothing projection.
        # Unlocks: right-sized attacks through parries, toll-sniping their
        # captures, not draining planets a predicted strike is about to hit.
        "PRODUCER_PLUS_MULTI_SIZE": "1",
        "PRODUCER_PLUS_OPP_PROJECTION": "1",
        "PRODUCER_PLUS_RESPONSE_VETO": "1",
        "PRODUCER_PLUS_REACTIVE_FLOOR": "0.5",
        "PRODUCER_PLUS_BG_FLOORS": "1",
    },
    "bgf_replan_rf": {
        # The full consistent stack: one-ply replan (pass 2 plans against
        # the predicted reply) + background-aware floors (pass 2's SIZES
        # are also reply-aware) + reactive floor + veto verify.
        "PRODUCER_PLUS_MULTI_SIZE": "1",
        "PRODUCER_PLUS_OPP_PROJECTION": "1",
        "PRODUCER_PLUS_RESPONSE_VETO": "1",
        "PRODUCER_PLUS_REACTIVE_FLOOR": "0.5",
        "PRODUCER_PLUS_REPLAN": "1",
        "PRODUCER_PLUS_BG_FLOORS": "1",
    },
    "veto_upsize": {
        # "Beat the parry": veto + full-spare-budget retry of killed waves
        # (aim/eta recomputed for the bigger, faster fleet; the flow scorer
        # decides if stripping the source is safe).
        "PRODUCER_PLUS_MULTI_SIZE": "1",
        "PRODUCER_PLUS_OPP_PROJECTION": "1",
        "PRODUCER_PLUS_RESPONSE_VETO": "1",
        "PRODUCER_PLUS_RESPONSE_VETO_UPSIZE": "1",
    },
    "mass_veto": {
        # Response veto on the live mass stack: one extra mirror pass with
        # our chosen waves as background -> the opponent's predicted reply;
        # attack waves that no longer clear the roi threshold under that
        # reply are dropped. Motivated by live mining: 30% of capture-sized
        # attacks fail, 65% of failures die to in-flight reinforcement
        # (~321 ships/game thrown into anticipated parries).
        "PRODUCER_PLUS_MULTI_SIZE": "1",
        "PRODUCER_PLUS_OPP_PROJECTION": "1",
        "PRODUCER_PLUS_MASS_TIEBREAK": "1",
        "PRODUCER_PLUS_REGROUP_MIN_SEND": "25",
        "PRODUCER_PLUS_OVERKILL_FACTOR": "2.0",
        "PRODUCER_PLUS_RESPONSE_VETO": "1",
    },
    "mass_termval12_veto": {
        # The counterweight composite: terminal production value drives
        # expansion (it measured NEGATIVE alone vs the mass incumbent —
        # thin garrisons lose the step-20..50 brawl), and the response veto
        # kills exactly the over-extensions the opponent's predicted reply
        # punishes. Hypothesis: veto restores the brawl-window safety while
        # keeping the expansion the top-ladder profile demands.
        "PRODUCER_PLUS_MULTI_SIZE": "1",
        "PRODUCER_PLUS_OPP_PROJECTION": "1",
        "PRODUCER_PLUS_MASS_TIEBREAK": "1",
        "PRODUCER_PLUS_REGROUP_MIN_SEND": "25",
        "PRODUCER_PLUS_OVERKILL_FACTOR": "2.0",
        "PRODUCER_PLUS_TERMINAL_PROD_VALUE": "12",
        "PRODUCER_PLUS_RESPONSE_VETO": "1",
    },
    "mass_termval12_recap": {
        # Holdability-discounted expansion: terminal production value drives
        # the front-loaded expansion the top ladder shows, and the recapture
        # penalty discounts captures the opponent can take back — the
        # missing "only keepable planets" filter (termval alone lost the
        # n=8 margin triage at every lambda; the 1-ply response veto can't
        # see the 10-20-turn-later punishment). Recapture penalty was
        # rejected 2026-06-05 vs the DRIBBLE referee — re-judged here vs
        # the mass incumbent.
        "PRODUCER_PLUS_MULTI_SIZE": "1",
        "PRODUCER_PLUS_OPP_PROJECTION": "1",
        "PRODUCER_PLUS_MASS_TIEBREAK": "1",
        "PRODUCER_PLUS_REGROUP_MIN_SEND": "25",
        "PRODUCER_PLUS_OVERKILL_FACTOR": "2.0",
        "PRODUCER_PLUS_TERMINAL_PROD_VALUE": "12",
        "PRODUCER_PLUS_RECAPTURE_PENALTY": "1",
    },
    "mass_termval6": {
        # Half-strength terminal production value on the mass stack — the
        # n=8 margin triage of the full master candidate regressed hard
        # (behind by step 40 already); probes whether λ=12 over-expands
        # (thin garrisons in the decision window) vs the direction itself
        # being wrong against the mass incumbent.
        "PRODUCER_PLUS_MULTI_SIZE": "1",
        "PRODUCER_PLUS_OPP_PROJECTION": "1",
        "PRODUCER_PLUS_MASS_TIEBREAK": "1",
        "PRODUCER_PLUS_REGROUP_MIN_SEND": "25",
        "PRODUCER_PLUS_OVERKILL_FACTOR": "2.0",
        "PRODUCER_PLUS_TERMINAL_PROD_VALUE": "6",
    },
    "mass_convoy40": {
        # Mass sweep: convoy threshold 40 instead of 25. Top-3 teams'
        # fleet p50 is 83 vs our ~44 after the first mass pivot — probes
        # whether a stiffer regroup gate closes more of the gap or
        # starves the transfer lane.
        "PRODUCER_PLUS_MULTI_SIZE": "1",
        "PRODUCER_PLUS_OPP_PROJECTION": "1",
        "PRODUCER_PLUS_MASS_TIEBREAK": "1",
        "PRODUCER_PLUS_REGROUP_MIN_SEND": "40",
        "PRODUCER_PLUS_OVERKILL_FACTOR": "2.0",
    },
    "mass_overkill3": {
        # Mass sweep: attack sizing 3x instead of 2x, convoy unchanged.
        "PRODUCER_PLUS_MULTI_SIZE": "1",
        "PRODUCER_PLUS_OPP_PROJECTION": "1",
        "PRODUCER_PLUS_MASS_TIEBREAK": "1",
        "PRODUCER_PLUS_REGROUP_MIN_SEND": "25",
        "PRODUCER_PLUS_OVERKILL_FACTOR": "3.0",
    },
    "convoy_only": {
        # Regroup convoying alone (attribution variant).
        "PRODUCER_PLUS_MULTI_SIZE": "1",
        "PRODUCER_PLUS_OPP_PROJECTION": "1",
        "PRODUCER_PLUS_REGROUP_MIN_SEND": "25",
    },
    "ffa_uniform_tick4p": {
        # Composition of the two 4P mechanisms: FFA uniform objective +
        # 4P-only multi-tick. Build/measure only if tick4p standalone
        # shows a lift on the 4P pools.
        "PRODUCER_PLUS_MULTI_SIZE": "1",
        "PRODUCER_PLUS_OPP_PROJECTION": "1",
        "PRODUCER_PLUS_FFA_SCORE": "1",
        "PRODUCER_PLUS_FFA_WEIGHTS": "uniform",
        "PRODUCER_PLUS_MULTI_TICK_OPP_K_4P": "3",
    },
}


# Strip `from .X import ...` (orbit_lite intra-package).
RE_INTRA_IMPORT_SINGLE = re.compile(r"^from \.[a-z_][a-z_0-9]* import [^()\n]+$", re.MULTILINE)
# Strip `from orbit_lite.X import ...` (producer_plus -> orbit_lite).
RE_OL_IMPORT_SINGLE = re.compile(r"^from orbit_lite\.[a-z_][a-z_0-9]* import [^()\n]+$", re.MULTILINE)
# Multi-line versions: `from .X import (a, b, c)` may span lines.
RE_INTRA_IMPORT_MULTI = re.compile(r"^from \.[a-z_][a-z_0-9]* import \([^)]*\)", re.MULTILINE | re.DOTALL)
RE_OL_IMPORT_MULTI = re.compile(r"^from orbit_lite\.[a-z_][a-z_0-9]* import \([^)]*\)", re.MULTILINE | re.DOTALL)
# producer_plus/main.py has sys.path injection that's not needed in a bundle.
RE_SYS_PATH_BLOCK = re.compile(
    r"# Make the sibling[\s\S]+?if _HERE not in sys\.path:\n\s+sys\.path\.insert\(0, _HERE\)\n",
    re.MULTILINE,
)


def strip_imports(text: str, kind: str) -> str:
    if kind == "orbit_lite":
        text = RE_INTRA_IMPORT_MULTI.sub("", text)
        text = RE_INTRA_IMPORT_SINGLE.sub("", text)
    elif kind == "producer_plus":
        text = RE_OL_IMPORT_MULTI.sub("", text)
        text = RE_OL_IMPORT_SINGLE.sub("", text)
        text = RE_SYS_PATH_BLOCK.sub("", text)
    return text


def strip_future_imports(text: str) -> tuple[str, str]:
    """Aggregate ALL `from __future__` lines (anywhere in module) for the
    bundle header and remove them from the body — `from __future__` must
    appear before any other statement in the final file, but module
    docstrings can hide them in the middle of source.
    """
    futures: list[str] = []
    rest_lines: list[str] = []
    for ln in text.split("\n"):
        if ln.strip().startswith("from __future__"):
            futures.append(ln.strip())
        else:
            rest_lines.append(ln)
    return "\n".join(futures), "\n".join(rest_lines)


def build(env_vars: dict, out_path: Path) -> None:
    parts: list[str] = []
    all_futures: set[str] = set()

    for mod_name in ORBIT_LITE_ORDER:
        src = (ORBIT_LITE_DIR / f"{mod_name}.py").read_text()
        src = strip_imports(src, "orbit_lite")
        futures, body = strip_future_imports(src)
        for ln in futures.splitlines():
            if ln.strip():
                all_futures.add(ln.strip())
        parts.append(f"\n# === orbit_lite.{mod_name} ===\n{body}\n")

    main_src = PRODUCER_PLUS_MAIN.read_text()
    main_src = strip_imports(main_src, "producer_plus")
    futures, body = strip_future_imports(main_src)
    for ln in futures.splitlines():
        if ln.strip():
            all_futures.add(ln.strip())
    parts.append(f"\n# === producer_plus.main ===\n{body}\n")

    env_header = ""
    if env_vars:
        env_header = "import os as _pp_os\n"
        for k, v in env_vars.items():
            env_header += f'_pp_os.environ.setdefault({k!r}, {v!r})\n'

    futures_header = "\n".join(sorted(all_futures)) + "\n" if all_futures else ""

    bundle = futures_header + env_header + "".join(parts) + "\n"
    # Kaggle entry point: a top-level `agent(obs, configuration=None)`.
    # producer_plus.main defines `def agent(obs):` — wrap it to add the
    # configuration arg the harness expects.
    bundle += (
        "\n# === bundle entry point (Kaggle expects 2-arg agent) ===\n"
        "_pp_inner_agent = agent\n"
        "def agent(obs, configuration=None):  # noqa: F811  shadow for harness\n"
        "    return _pp_inner_agent(obs)\n"
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(bundle)
    # Verify the bundle parses.
    ast.parse(bundle)
    print(f"wrote {out_path}  ({len(bundle):_} bytes)")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=list(ENV_VARIANTS), default="multi_size")
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output path; default submissions/producer_plus_<variant>_on.py",
    )
    p.add_argument(
        "--set", action="append", default=[], metavar="KEY=VAL",
        help="override/add a baked env var on top of the variant "
             "(repeatable; used by scripts/knob_tune.py)",
    )
    args = p.parse_args()
    out = args.out or REPO / "submissions" / f"producer_plus_{args.variant}_on.py"
    env = dict(ENV_VARIANTS[args.variant])
    for kv in args.set:
        k, _, v = kv.partition("=")
        if not k or not _:
            raise SystemExit(f"--set expects KEY=VAL, got {kv!r}")
        env[k] = v
    build(env, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
