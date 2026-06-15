"""Mine producer_plus's real ladder loss replays for systematic bugs.

The method that worked in the 2026-06-15 session (see
knowledge-base/thoughts/2026-06-15-loss-mining-grounded-fixes.md). Aggregates
the loss landscape across all loss replays in a live-episodes dir:
  - early-expansion gap (planets/ships us vs winner @ ~step 60),
  - far high-value-planet neglect (the corner-neglect bug, generalised),
  - collapse detection (planets lost after our peak).

Replay JSON: info.TeamNames identifies us ('ChrisLeiteScha'); info.seed is the
reproduction seed (feed it to the producer_plus mirror to diagnose a specific
game); rewards marks the loser. Usage:

    python scripts/mine_losses.py [audit/live-episodes/<submission_id>]
"""
from __future__ import annotations
import json, glob, math, statistics, sys

US_NAME = "ChrisLeiteScha"


def _state(steps, step, ns):
    obs = steps[min(step, len(steps) - 1)][0]["observation"]
    pc = [0] * ns; sh = [0.0] * ns; far = []
    for p in obs["planets"]:
        o = int(p[1])
        if 0 <= o < ns:
            pc[o] += 1; sh[o] += float(p[5])
        if math.hypot(float(p[2]) - 50, float(p[3]) - 50) > 33 and float(p[6]) >= 3:
            far.append(o)
    for fl in obs.get("fleets", []):
        o = int(fl[1])
        if 0 <= o < ns:
            sh[o] += float(fl[6])
    return pc, sh, far


def analyze(dir_):
    files = sorted(glob.glob(f"{dir_}/episode-*-replay.json"))
    losses = []
    for f in files:
        try:
            r = json.load(open(f))
        except Exception:
            continue
        nm = r.get("info", {}).get("TeamNames", []); rew = r.get("rewards", [])
        if US_NAME not in nm or len(rew) != len(nm):
            continue
        us = nm.index(US_NAME)
        if rew[us] >= max(rew):          # losses only
            continue
        win = rew.index(max(rew)); ns = len(nm); steps = r["steps"]; N = len(steps)
        pc60, sh60, far60 = _state(steps, 60, ns)
        peak = max(_state(steps, s, ns)[0][us] for s in range(0, N, 20))
        final = _state(steps, N - 1, ns)[0][us]
        losses.append(dict(
            ep=r["info"]["EpisodeId"], seed=r["info"].get("seed"), ns=ns, N=N,
            pc_us=pc60[us], pc_win=pc60[win], sh_us=sh60[us], sh_win=sh60[win],
            far_us=sum(1 for o in far60 if o == us), far_win=sum(1 for o in far60 if o == win),
            far_neu=sum(1 for o in far60 if o == -1), peak=peak, final=final,
            lost_after_peak=peak - final))
    return losses


def report(losses):
    n = len(losses)
    med = lambda k: statistics.median([x[k] for x in losses])
    print(f"=== {n} loss replays ===")
    print(f"planet count @60  us vs winner: median {med('pc_us'):.0f} vs {med('pc_win'):.0f}")
    print(f"ship total  @60   us vs winner: median {med('sh_us'):.0f} vs {med('sh_win'):.0f}")
    print(f"far high-value @60 us/winner/neutral: {med('far_us'):.0f}/{med('far_win'):.0f}/{med('far_neu'):.0f}")
    print(f"trail planet-count @60: {sum(1 for x in losses if x['pc_win'] > x['pc_us'])}/{n}  (under-expansion)")
    print(f"lost >=3 planets after peak: {sum(1 for x in losses if x['lost_after_peak'] >= 3)}/{n}  (collapse)")
    print("worst corner-neglect (far-neutral @60) + seeds to reproduce:")
    for x in sorted(losses, key=lambda x: -x["far_neu"])[:5]:
        print(f"  ep{x['ep']} {x['ns']}P far_neu={x['far_neu']} planets {x['pc_us']}v{x['pc_win']} seed={x['seed']}")


if __name__ == "__main__":
    d = sys.argv[1] if len(sys.argv) > 1 else "audit/live-episodes/53564198"
    report(analyze(d))
