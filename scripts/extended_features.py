"""Extended behavioural features beyond lib/fingerprint.py's 15.

For each replay × focal seat, compute:
  - first_launch_step          : step of first non-empty action (opening tempo)
  - comet_capture_rate         : fraction of own fleets aimed at a comet
                                 planet (target_pid is in obs.comet_planet_ids)
  - fleets_lost_to_sun         : count of own fleets that disappear after a
                                 step where their path crosses the sun
                                 (proxy via path_clears_sun on chord)
  - fleets_lost_to_oob         : count of own fleets that vanish with no
                                 capture (proxy: id in fleets at step S,
                                 not in fleets at step S+1, and no planet's
                                 owner flipped to focal)
  - gang_up_rate               : fraction of focal launch events where
                                 another focal-owned source launched at
                                 the same target within ±5 turns
  - recapture_rate             : fraction of planet-flips where the
                                 captured planet was OURS within the last
                                 50 steps (recapturing lost territory)
  - mean_planets_owned_late    : mean planet count over the LAST third of
                                 the game (steps 2N/3 → end)
  - mid_to_late_aggression     : ratio of mid-game (step 200-350) launches
                                 to total
  - total_steps_until_done     : when game ended (proxy for fast-finish)

Output: audit/2026-05-11-top-performer-extended.json
"""
from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lib.geometry import path_clears_sun  # noqa: E402

REPLAY_DIR = Path("audit/external/replays")
OUT_PATH = Path("audit/2026-05-11-top-performer-extended.json")

FNAME_RE = re.compile(r"^r(?P<rank>\d{2})-(?P<team>.+?)-(?P<size>[24]P)-(?P<wl>[WL])-(?P<eid>\d+)\.json$")
MIDPACK_RE = re.compile(r"^midpack-.*?-(?P<size>[24]P)-(?P<eid>\d+)\.json$")


def find_focal_idx(team_names: list, fname_team: str) -> int:
    norm = lambda s: s.replace("_", " ").replace("@", "@ ").strip()
    target_variants = {fname_team, norm(fname_team), fname_team.replace("_", " ")}
    for i, t in enumerate(team_names):
        if t in target_variants:
            return i
    # fuzzy
    for i, t in enumerate(team_names):
        if fname_team.lower().replace("_", "") in t.lower().replace(" ", ""):
            return i
    return 0


def infer_target_pid(src_xy, angle, planets):
    """Project a ray from src at `angle`; return planet id with smallest
    perpendicular distance among forward-projected candidates."""
    sx, sy = src_xy
    dx, dy = math.cos(angle), math.sin(angle)
    best_id = None
    best_score = float("inf")
    for p in planets:
        pid, _, px, py, prad, _, _ = p[0], p[1], p[2], p[3], p[4], p[5], p[6]
        if abs(px - sx) < 1e-6 and abs(py - sy) < 1e-6:
            continue
        rx, ry = px - sx, py - sy
        fwd = rx * dx + ry * dy
        if fwd <= 0:
            continue
        perp_x = rx - fwd * dx
        perp_y = ry - fwd * dy
        perp = math.hypot(perp_x, perp_y)
        # Within target radius → strong match; otherwise penalty
        score = perp + 0.001 * fwd
        if score < best_score:
            best_score = score
            best_id = int(pid)
    return best_id


def replay_extended(replay: dict, focal_idx: int) -> dict:
    steps = replay.get("steps", [])
    if not steps:
        return {}

    # Track each focal launch as (step, src_pid, target_pid, ships)
    focal_launches = []
    fleets_lost_to_sun = 0
    fleets_lost_unknown = 0
    comet_targeted = 0
    gang_up_count = 0
    recapture_count = 0
    flip_count = 0

    last_planet_owners = {}  # planet_id -> last 50 steps of owner history
    own_fleets_prev: dict[int, dict] = {}

    n_steps_done = sum(1 for s in steps if s and any(p.get("status") != "DONE" for p in s))
    n_steps_total = len(steps)

    early_launches = 0
    mid_launches = 0
    late_launches = 0
    first_launch_step = None

    third = n_steps_total // 3

    for step_idx, step in enumerate(steps):
        if focal_idx >= len(step):
            continue
        obs = step[focal_idx].get("observation", {}) or {}
        planets = obs.get("planets", []) or []
        fleets = obs.get("fleets", []) or []
        comet_pids = set(obs.get("comet_planet_ids", []) or [])
        action = step[focal_idx].get("action") or []
        by_pid = {int(p[0]): p for p in planets}

        # Track flips: planet ownership change
        for p in planets:
            pid = int(p[0])
            owner = int(p[1])
            hist = last_planet_owners.setdefault(pid, [])
            hist.append((step_idx, owner))
            # Trim
            while hist and hist[0][0] < step_idx - 50:
                hist.pop(0)
            # Detect a flip to focal-owned where it had been focal in last 50 steps
            if len(hist) >= 2:
                prev_step, prev_owner = hist[-2]
                if prev_owner != focal_idx and owner == focal_idx:
                    flip_count += 1
                    # Check if any earlier owner in window was focal
                    was_ours = any(o == focal_idx for s, o in hist[:-2])
                    if was_ours:
                        recapture_count += 1

        # Identify own fleets in flight
        own_fleets_now: dict[int, dict] = {}
        for f in fleets:
            fid = int(f[0])
            owner = int(f[1])
            if owner == focal_idx:
                own_fleets_now[fid] = {
                    "x": float(f[2]),
                    "y": float(f[3]),
                    "angle": float(f[4]),
                    "from_pid": int(f[5]),
                    "ships": float(f[6]),
                }
        # Fleets that disappeared between prev and now
        for fid, prev_state in own_fleets_prev.items():
            if fid not in own_fleets_now:
                # Did anything our colour gain ownership of a planet near
                # the fleet's path? If yes → captured; else → lost.
                cap_likely = False
                for p in planets:
                    if int(p[1]) == focal_idx:
                        # if planet near where fleet was
                        if math.hypot(float(p[2]) - prev_state["x"],
                                       float(p[3]) - prev_state["y"]) < 5.0:
                            cap_likely = True
                            break
                if not cap_likely:
                    # Was its trajectory crossing the sun?
                    px = prev_state["x"] + 100 * math.cos(prev_state["angle"])
                    py = prev_state["y"] + 100 * math.sin(prev_state["angle"])
                    if not path_clears_sun((prev_state["x"], prev_state["y"]),
                                              (px, py), safety=0.0):
                        fleets_lost_to_sun += 1
                    else:
                        fleets_lost_unknown += 1
        own_fleets_prev = own_fleets_now

        # Process focal action (launches)
        for a in action:
            if not a or len(a) < 3:
                continue
            try:
                src_pid = int(a[0])
                angle = float(a[1])
                ships = float(a[2])
            except (TypeError, ValueError):
                continue
            if first_launch_step is None:
                first_launch_step = step_idx
            if step_idx < third:
                early_launches += 1
            elif step_idx < 2 * third:
                mid_launches += 1
            else:
                late_launches += 1
            src = by_pid.get(src_pid)
            if src is None:
                continue
            tgt_pid = infer_target_pid((float(src[2]), float(src[3])), angle, planets)
            focal_launches.append({
                "step": step_idx, "src": src_pid, "tgt": tgt_pid, "ships": ships,
            })
            if tgt_pid is not None and tgt_pid in comet_pids:
                comet_targeted += 1

    # Gang-up detection: pairs of focal launches with same target within ±5 steps
    for i, L in enumerate(focal_launches):
        for j in range(max(0, i - 5), min(len(focal_launches), i + 6)):
            if i == j:
                continue
            M = focal_launches[j]
            if M["tgt"] == L["tgt"] and abs(M["step"] - L["step"]) <= 5 and L["src"] != M["src"]:
                gang_up_count += 1
                break  # count L once

    n_launches = len(focal_launches)
    out = {
        "first_launch_step": first_launch_step if first_launch_step is not None else -1,
        "comet_capture_rate": (comet_targeted / n_launches) if n_launches else 0.0,
        "fleets_lost_to_sun": fleets_lost_to_sun,
        "fleets_lost_unknown": fleets_lost_unknown,
        "gang_up_rate": (gang_up_count / n_launches) if n_launches else 0.0,
        "recapture_rate": (recapture_count / max(1, flip_count)),
        "flip_count": flip_count,
        "early_launches": early_launches,
        "mid_launches": mid_launches,
        "late_launches": late_launches,
        "n_launches_total": n_launches,
        "n_steps_total": n_steps_total,
    }
    return out


def main():
    rows = []
    files = sorted(REPLAY_DIR.glob("*.json"))
    print(f"Processing {len(files)} replay files.")
    for fpath in files:
        m = FNAME_RE.match(fpath.name) or MIDPACK_RE.match(fpath.name)
        if not m:
            continue
        try:
            replay = json.loads(fpath.read_text())
        except Exception as e:
            print(f"  ERR {fpath.name}: {e}")
            continue

        is_midpack = fpath.name.startswith("midpack-")
        team_names = replay.get("info", {}).get("TeamNames", [])
        if is_midpack:
            rank = None
            fname_team = "ChrisLeiteScha"
        else:
            rank = int(m.group("rank"))
            fname_team = m.group("team")
        focal_idx = find_focal_idx(team_names, fname_team)

        try:
            ext = replay_extended(replay, focal_idx)
        except Exception as e:
            print(f"  ERR ext {fpath.name}: {e}")
            continue

        rows.append({
            "rank": rank,
            "src": fpath.name,
            "team": team_names[focal_idx] if focal_idx < len(team_names) else fname_team,
            "size": int(m.group("size")[0]),
            "seat": focal_idx,
            "episode_id": m.group("eid"),
            **ext,
        })

    out_path = OUT_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"rows": rows}, indent=2))
    print(f"Wrote {out_path}: {len(rows)} rows.")


if __name__ == "__main__":
    main()
