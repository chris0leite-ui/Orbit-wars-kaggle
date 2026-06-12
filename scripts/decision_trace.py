"""Decision trace: replay a live episode step through the planner and dump
every option it considered — shortlist membership, capture floors, candidate
scores vs the roi threshold, and veto survivors.

Usage:
  python scripts/decision_trace.py REPLAY_JSON SEAT STEP [STEP ...]

Env: set the stack's PRODUCER_PLUS_* vars before calling (defaults to the
live 2P stack: multi-size + opp projection + veto + reactive floor 0.5).
"""
from __future__ import annotations

import json
import math
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "agents", "producer"))
sys.path.insert(0, os.path.join(REPO, "agents", "producer_plus"))

os.environ.setdefault("PRODUCER_PLUS_MULTI_SIZE", "1")
os.environ.setdefault("PRODUCER_PLUS_OPP_PROJECTION", "1")
os.environ.setdefault("PRODUCER_PLUS_RESPONSE_VETO", "1")
os.environ.setdefault("PRODUCER_PLUS_REACTIVE_FLOOR", "0.5")

import torch
import main as agent_mod


def trace_step(rj: dict, seat: int, step: int) -> None:
    obs = dict(rj["steps"][step][seat]["observation"])
    obs.setdefault("step", rj["steps"][step][0]["observation"].get("step", step))
    obs["player"] = seat
    rj_planets_box = [obs["planets"]]

    cap: dict = {}
    orig_shortlist = agent_mod.build_target_shortlist
    orig_floor = agent_mod.capture_floor
    orig_drain = agent_mod.safe_drain
    orig_greedy = agent_mod._greedy_select
    orig_veto = agent_mod._apply_response_veto

    # The opponent mirror and the veto's reply pass run the SAME planner from
    # other seats — record every call and keep the one for OUR seat.
    def shortlist_spy(obs_arg, *a, **kw):
        out = orig_shortlist(obs_arg, *a, **kw)
        if int(obs_arg.player_id) == seat and "shortlist" not in cap:
            cap["shortlist"] = out
        return out

    def floor_spy(*a, **kw):
        out = orig_floor(*a, **kw)
        if (kw.get("player_id") == seat and "floor" not in cap
                and kw.get("target_idx") is not None):
            cap["floor"] = out
            cap["floor_tidx"] = kw["target_idx"]
        return out

    def drain_spy(*a, **kw):
        out = orig_drain(*a, **kw)
        if (kw.get("player_id") == seat and "drain" not in cap
                and kw.get("source_idx") is not None):
            cap["drain"] = out
            cap["drain_sidx"] = kw["source_idx"]
        return out

    my_slots = {i for i, p in enumerate(rj_planets_box[0]) if int(p[1]) == seat}
    opp_slots = {i for i, p in enumerate(rj_planets_box[0])
                 if int(p[1]) >= 0 and int(p[1]) != seat}

    def greedy_spy(**kw):
        # Padded candidate rows carry arbitrary clamped slots — vote by
        # overlap of ACTIVE sources with ours vs the opponent's.
        act = kw["cand_active"] & (kw["cand_send"] > 0)
        src_slots = set(kw["cand_src"][act].reshape(-1).tolist())
        mine_n = len(src_slots & my_slots)
        opp_n = len(src_slots & opp_slots)
        if "score" not in cap and (mine_n > opp_n or (not src_slots)):
            cap.update({k: kw[k] for k in (
                "score", "cand_src", "cand_send", "cand_eta", "cand_tgt_slot",
                "roi_threshold")})
            cap["cand_active"] = kw["cand_active"]
        return orig_greedy(**kw)

    def veto_spy(entries, **kw):
        cap["pre_veto"] = entries
        out = orig_veto(entries, **kw)
        cap["post_veto"] = out
        return out

    agent_mod.build_target_shortlist = shortlist_spy
    agent_mod.capture_floor = floor_spy
    agent_mod.safe_drain = drain_spy
    agent_mod._greedy_select = greedy_spy
    agent_mod._apply_response_veto = veto_spy
    try:
        agent_mod._RUNTIME.reset()
        moves = agent_mod.agent(obs)
    finally:
        agent_mod.build_target_shortlist = orig_shortlist
        agent_mod.capture_floor = orig_floor
        agent_mod.safe_drain = orig_drain
        agent_mod._greedy_select = orig_greedy
        agent_mod._apply_response_veto = orig_veto

    planets = obs["planets"]
    pid_of = {i: int(p[0]) for i, p in enumerate(planets)}
    own = {i: int(p[1]) for i, p in enumerate(planets)}
    garr = {i: float(p[5]) for i, p in enumerate(planets)}
    prod = {i: float(p[6]) for i, p in enumerate(planets)}

    print(f"\n================ step {step} (seat {seat}) ================")
    mine = [i for i in own if own[i] == seat]
    drain_by = {}
    if "drain" in cap:
        sidx = cap["drain_sidx"].tolist()
        dr = cap["drain"].tolist()
        drain_by = dict(zip(sidx, dr))
    print("MY PLANETS  (slot: id  garrison  prod  safe_drain)")
    for i in sorted(mine, key=lambda x: -garr[x]):
        d = drain_by.get(i)
        print(f"  slot {i:2d}: id={pid_of[i]:2d}  g={garr[i]:5.0f}  +{prod[i]:.0f}"
              + (f"  drain={d:5.1f}" if d is not None else "  (not a source)"))

    sl_idx, sl_ex = cap["shortlist"]
    short = set(sl_idx[sl_ex].tolist()) if sl_ex.any() else set()
    print("\nNEUTRAL / ENEMY TARGET OPTIONS")
    floor_t = cap.get("floor")
    ftidx = cap.get("floor_tidx")
    floor_by = {}
    if floor_t is not None:
        for r, slot in enumerate(ftidx.tolist()):
            row = floor_t[r]
            floor_by[slot] = (float(row.min()), float(row.max()))
    # best candidate score per target slot
    best = {}
    if "score" in cap:
        sc = cap["score"]
        tgt = cap["cand_tgt_slot"]
        send = cap["cand_send"]
        eta = cap["cand_eta"]
        for c in range(int(sc.shape[0])):
            s = float(sc[c])
            if s == float("-inf"):
                continue
            t = int(tgt[c])
            tot = float(send[c].sum())
            e = float(eta[c].max())
            if t not in best or s > best[t][0]:
                best[t] = (s, tot, e)
    thr = cap.get("roi_threshold")
    # reach: per target, min candidate eta among candidates with send>0
    reach = {}
    if "score" in cap:
        act_all = (cap["cand_send"] > 0).any(dim=-1)
        for c in range(int(cap["cand_send"].shape[0])):
            if not bool(act_all[c]):
                continue
            t = int(cap["cand_tgt_slot"][c])
            e = float(cap["cand_eta"][c].max())
            if t not in reach or e < reach[t]:
                reach[t] = e
    for i in sorted(own, key=lambda x: -prod[x]):
        if own[i] == seat or garr[i] < 0:
            pass
        if own[i] == seat:
            continue
        tag = "ENEMY " if own[i] >= 0 else "neutral"
        in_sl = "shortlisted" if i in short else "NOT-SHORTLISTED"
        fl = floor_by.get(i)
        fls = f"floor={fl[0]:.0f}..{fl[1]:.0f}" if fl else "floor=n/a"
        b = best.get(i)
        r = reach.get(i)
        rs = f"min-eta={r:4.1f}" if r is not None else "unreached"
        bs = (f"best: score={b[0]:+7.1f} send={b[1]:4.0f} eta={b[2]:4.1f}"
              if b else f"NO VALID CANDIDATE ({rs})")
        print(f"  slot {i:2d} id={pid_of[i]:2d} {tag} g={garr[i]:5.0f} +{prod[i]:.0f}  "
              f"{in_sl:16s} {fls:18s} {bs}")
    if thr is not None:
        print(f"\nroi threshold (fire only if score ≥): {float(thr):+.2f}")

    def show(entries, label):
        if entries is None:
            print(f"{label}: (none)")
            return
        v = entries.valid
        n = int(v.sum())
        print(f"{label}: {n} launches")
        for j in range(int(v.shape[0])):
            if not bool(v[j]):
                continue
            s = int(entries.source_slots[j]); t = int(entries.target_slots[j])
            print(f"   {pid_of[s]:2d} -> {pid_of[t]:2d}  ships={float(entries.ships[j]):5.0f} "
                  f"eta={float(entries.eta[j]):4.1f}")
    show(cap.get("pre_veto"), "\nWAVES pre-veto")
    show(cap.get("post_veto"), "WAVES post-veto")
    print(f"actual emitted moves: {len(moves) if moves else 0}")


def main() -> None:
    path, seat = sys.argv[1], int(sys.argv[2])
    rj = json.load(open(path))
    for s in sys.argv[3:]:
        trace_step(rj, seat, int(s))


if __name__ == "__main__":
    main()
