"""Mid-game neutral-launch OUTCOME trace (race-vs-churn vs target-selection).

For 4P WIN vs LOSS episodes, classifies each of our step-40..110
NEUTRAL-targeted launches by the target's owner at launch+eta+10:
ours (won the capture), enemy (lost the race / lost it back), still-neutral
(never captured). Ray-cast target attribution (~84% accurate — read as
directional). Usage: python scripts/diag_neutral_outcome.py 53595717
"""
import json, math, sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO))
from scripts.label_shot_outcomes import _infer_target_pid, _fleet_speed

def main():
    sub = sys.argv[1] if len(sys.argv) > 1 else "53595717"
    LIVE = REPO/"audit"/"live-episodes"/sub
    LO, HI, BUF = 40, 110, 10
    agg = {"WIN": {"ours":0,"enemy":0,"neutral":0}, "LOSS": {"ours":0,"enemy":0,"neutral":0}}
    for f in sorted(LIVE.glob("episode-*-replay.json")):
        try: r = json.loads(f.read_text())
        except Exception: continue
        teams = r.get("info",{}).get("TeamNames",[]); ours = [i for i,t in enumerate(teams) if t=="ChrisLeiteScha"]
        steps = r.get("steps",[])
        if len(ours)!=1 or not steps or len(steps[0])!=4: continue
        pid = ours[0]; rew = [steps[-1][i].get("reward") for i in range(4)]; rew=[x for x in rew if x is not None]
        my = steps[-1][pid].get("reward")
        if my is None or not rew: continue
        key = "WIN" if my==max(rew) else "LOSS"
        for t in range(LO, min(HI,len(steps))):
            obs = steps[t][pid].get("observation",{}) or {}; planets = obs.get("planets",[]) or []
            by = {int(p[0]):p for p in planets}
            for a in (steps[t][pid].get("action") or []):
                if not a or len(a)<3: continue
                try: src_id=int(a[0]); ang=float(a[1]); sh=float(a[2])
                except Exception: continue
                src = by.get(src_id)
                if src is None: continue
                tid = _infer_target_pid((float(src[2]),float(src[3])), ang, planets)
                tgt = by.get(tid) if tid is not None else None
                if tgt is None or int(tgt[1])!=-1: continue
                d = math.hypot(float(tgt[2])-float(src[2]), float(tgt[3])-float(src[3]))
                eta = int(math.ceil(d/max(_fleet_speed(sh),1e-6)))
                chk = min(t+eta+BUF, len(steps)-1)
                cby = {int(p[0]):p for p in (steps[chk][pid].get("observation",{}) or {}).get("planets",[]) or []}
                tc = cby.get(tid)
                if tc is None: continue
                o = int(tc[1])
                agg[key]["ours" if o==pid else ("neutral" if o==-1 else "enemy")] += 1
    for k in ("WIN","LOSS"):
        d = agg[k]; tot = sum(d.values()) or 1
        print(f"{k}: n={tot}  ours {d['ours']/tot:.0%}  enemy {d['enemy']/tot:.0%}  still-neutral {d['neutral']/tot:.0%}")

if __name__ == "__main__":
    main()
