"""First-cut divergence diagnostic for the three PI-flagged screenshot
games (52894340 / FND):
  - seed 1250638780 vs Mille Initiate (step 14: missed capture)
  - seed 1085160712 vs KoshinM      (step 12: long-arc routing)
  - seed 669336863  vs Aidan P5     (step 44: failed mid-game pivot)

For each game, dumps a side-by-side action log over the first 50 steps:
which planet did each side target, with what ships, from which source.
Also tracks planet ownership transitions so we can see WHO CAPTURED WHAT
WHEN. Writes a per-game markdown report.

This is read-only — no agent re-execution. Just walking the replay JSON.

CLI: python3 scripts/replay_compare_screenshots.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LIVE_DIR = REPO / "audit" / "live-episodes" / "52894340"
OUT_DIR = REPO / "audit" / "replays"

SCREENSHOTS = [
    # (label, episode_id, seed, focal_step, our_team)
    ("S1-mille-step14", "77321232", 1250638780, 14, "ChrisLeiteScha"),
    ("S2-koshin-step12", "77320686", 1085160712, 12, "ChrisLeiteScha"),
    ("S3-aidan-step44", "77323008", 669336863, 44, "ChrisLeiteScha"),
]


def planet_id_of_action(action: list, planets_at_step: list) -> str:
    """Given action [src_id, angle, ships], best-effort identify the
    target planet by ray-casting from src position. For now just return
    the src; target identification needs the trajectory primitive."""
    if not isinstance(action, list) or len(action) != 3:
        return "?"
    src_id = int(action[0])
    return f"src={src_id} angle={action[1]:+.2f} ships={int(action[2])}"


def planets_owned_by(obs, seat_player_id: int) -> set[int]:
    """Return set of planet IDs owned by `seat_player_id` in this obs.

    obs['planets'] is the per-step planet snapshot; format mirrors
    initial_planets: [id, owner, x, y, orbit_r, garrison, prod].
    """
    out = set()
    for p in obs.get("planets", []):
        if p[1] == seat_player_id:
            out.add(int(p[0]))
    return out


def render_one_game(label: str, ep_id: str, seed: int, focal_step: int,
                    our_team: str) -> str:
    path = LIVE_DIR / f"episode-{ep_id}-replay.json"
    if not path.is_file():
        return f"# {label}\n\nERROR: missing {path}\n"

    d = json.load(open(path))
    teams = d["info"]["TeamNames"]
    our_seat = teams.index(our_team)
    opp_seat = 1 - our_seat
    opp_team = teams[opp_seat]
    rewards = d.get("rewards")
    n_steps = len(d["steps"])

    lines = []
    lines.append(f"# {label}: seed={seed}, ep={ep_id}, {n_steps} steps")
    lines.append(f"- We: **{our_team}** (seat {our_seat})")
    lines.append(f"- Opp: **{opp_team}** (seat {opp_seat})")
    lines.append(f"- Rewards: {rewards} (ours={rewards[our_seat]})")
    lines.append(f"- PI-flagged focal step: **{focal_step}**")
    lines.append("")

    # Track ownership: who has which planets at each step.
    # Per-turn diff: "we captured P_X" or "we lost P_Y".
    # Get our player_id (could differ from seat in 4P; here 2P so equals seat).
    our_pid = d["steps"][0][our_seat]["observation"].get("player", our_seat)
    opp_pid = d["steps"][0][opp_seat]["observation"].get("player", opp_seat)

    # Initial ownership
    obs0 = d["steps"][0][our_seat]["observation"]
    our_owned = planets_owned_by(obs0, our_pid)
    opp_owned = planets_owned_by(obs0, opp_pid)
    lines.append(f"## Initial state")
    lines.append(f"- Our planets: {sorted(our_owned)} (n={len(our_owned)})")
    lines.append(f"- Opp planets: {sorted(opp_owned)} (n={len(opp_owned)})")
    n_planets_total = len(obs0.get("planets", []))
    lines.append(f"- Neutrals: {n_planets_total - len(our_owned) - len(opp_owned)}")
    lines.append("")

    # Step-by-step action log (limit to first 60 steps or focal+20)
    end_step = min(n_steps, max(60, focal_step + 20))
    lines.append(f"## Per-step actions (steps 0..{end_step - 1})")
    lines.append("")
    lines.append("| step | OUR actions | OPP actions | OUR Δ planets | OPP Δ planets |")
    lines.append("|---:|---|---|---|---|")

    prev_our = our_owned.copy()
    prev_opp = opp_owned.copy()
    for s in range(end_step):
        seat_steps = d["steps"][s]
        our_action = seat_steps[our_seat].get("action") or []
        opp_action = seat_steps[opp_seat].get("action") or []
        # Current ownership (after the step)
        obs_now = seat_steps[our_seat]["observation"]
        cur_our = planets_owned_by(obs_now, our_pid)
        cur_opp = planets_owned_by(obs_now, opp_pid)
        gained_our = cur_our - prev_our
        lost_our = prev_our - cur_our
        gained_opp = cur_opp - prev_opp
        lost_opp = prev_opp - cur_opp
        prev_our = cur_our
        prev_opp = cur_opp

        def fmt_actions(actions):
            if not actions:
                return ""
            return "; ".join(planet_id_of_action(a, None) for a in actions)

        def fmt_delta(gained, lost):
            parts = []
            if gained:
                parts.append(f"+{sorted(gained)}")
            if lost:
                parts.append(f"-{sorted(lost)}")
            return " ".join(parts)

        flag = " ⚠️" if s == focal_step else ""
        lines.append(
            f"| {s}{flag} | {fmt_actions(our_action)} | {fmt_actions(opp_action)} | "
            f"{fmt_delta(gained_our, lost_our)} | "
            f"{fmt_delta(gained_opp, lost_opp)} |")

    lines.append("")

    # Summary: by end of `end_step`, who owned what
    obs_end = d["steps"][end_step - 1][our_seat]["observation"]
    final_our = planets_owned_by(obs_end, our_pid)
    final_opp = planets_owned_by(obs_end, opp_pid)
    lines.append(f"## State at step {end_step - 1}")
    lines.append(f"- Our planets: {sorted(final_our)} (n={len(final_our)})")
    lines.append(f"- Opp planets: {sorted(final_opp)} (n={len(final_opp)})")
    lines.append("")

    # Capture log
    lines.append(f"## Capture timeline (first {end_step} steps)")
    lines.append("")
    cap_lines = []
    prev_our_t = our_owned.copy()
    prev_opp_t = opp_owned.copy()
    for s in range(end_step):
        obs = d["steps"][s][our_seat]["observation"]
        cur_our = planets_owned_by(obs, our_pid)
        cur_opp = planets_owned_by(obs, opp_pid)
        for p in cur_our - prev_our_t:
            cap_lines.append(f"  step {s}: OUR captured P{p}")
        for p in cur_opp - prev_opp_t:
            cap_lines.append(f"  step {s}: OPP captured P{p}")
        for p in prev_our_t - cur_our:
            if p not in cur_opp:
                cap_lines.append(f"  step {s}: OUR lost P{p} (to neutral?)")
        prev_our_t = cur_our
        prev_opp_t = cur_opp
    lines.extend(cap_lines if cap_lines else ["  (none in window)"])
    lines.append("")
    lines.append(f"OUR total captures in window: {sum('OUR captured' in c for c in cap_lines)}")
    lines.append(f"OPP total captures in window: {sum('OPP captured' in c for c in cap_lines)}")
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "screenshot-divergence.md"
    body = []
    for spec in SCREENSHOTS:
        body.append(render_one_game(*spec))
        body.append("---\n")
    out_path.write_text("".join(body))
    print(f"wrote {out_path}")

    # Also dump per-game JSON for downstream tooling
    for label, ep_id, seed, focal_step, our_team in SCREENSHOTS:
        path = LIVE_DIR / f"episode-{ep_id}-replay.json"
        d = json.load(open(path))
        teams = d["info"]["TeamNames"]
        our_seat = teams.index(our_team)
        opp_seat = 1 - our_seat
        actions = {
            "label": label,
            "episode_id": ep_id,
            "seed": seed,
            "focal_step": focal_step,
            "teams": teams,
            "our_seat": our_seat,
            "opp_seat": opp_seat,
            "rewards": d.get("rewards"),
            "actions_per_step": [
                {
                    "step": s,
                    "our": d["steps"][s][our_seat].get("action") or [],
                    "opp": d["steps"][s][opp_seat].get("action") or [],
                }
                for s in range(len(d["steps"]))
            ],
        }
        json_path = OUT_DIR / f"{label}-actions.json"
        json_path.write_text(json.dumps(actions, indent=2))
    print(f"wrote per-game action JSONs to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
