"""Confound-controlled re-audit of large->small launch hypothesis.

v2 of `scripts/large_to_small_audit.py`. Addresses four confounds the PI
flagged (2026-05-21 session):

1. **Selection bias** (high-production planets launch more). Per-ship NET
   (production gained - production lost) / ships deployed is the primary
   metric. Per-launch NET is kept for backward comparison.
2. **End-state attribution bias**. "src lost by end-of-game" was confounded
   in lost episodes (everything flips). v2 uses `src_lost_within_20_pre_relaunch`:
   src owner != our_seat in the window (landing_step, min(landing_step+20,
   next_launch_step_from_src, n_steps-1)]. Attributes per actual launch
   instead of cumulating to the last launch from each src.
3. **Landing-time vs end-of-game outcome**. `tgt_owned_at_landing` records
   ownership one tick after landing (combat resolves), separate from
   end-of-game holding.
4. **End-state vs early-game bias**. Stratify all metrics by
   episode_window: early (t<=150), mid (151..350), late (>350). Verdict
   hinges on early + mid only.

Tier convention (canonical going forward, fixes v1 prod=2 bug):
  small = prod in {1, 2};  mid = prod == 3;  large = prod >= 4

CLI:
    python -m scripts.large_to_small_audit_v2 <submission_id>
        [--limit N] [--team NAME]
        [--out-jsonl audit/2026-05-21-large-to-small-v2.jsonl]
"""
from __future__ import annotations
import argparse
import collections
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from lib.intent import World  # noqa: E402
from lib.trajectory import predict_fleet_fate  # noqa: E402
from scripts.live_episode_summary import detect_team_name  # noqa: E402


def tier(prod: int) -> str:
    if prod <= 2:
        return "small(1-2)"
    if prod == 3:
        return "mid(3)"
    return "large(4+)"


TIERS = ["small(1-2)", "mid(3)", "large(4+)"]


def planet_owner(planets_list: list, pid: int) -> int | None:
    for row in planets_list:
        if int(row[0]) == int(pid):
            return int(row[1])
    return None


def episode_window(t: int) -> str:
    if t <= 150:
        return "early"
    if t <= 350:
        return "mid"
    return "late"


def analyze_episode(replay_path: Path, our_team: str) -> list[dict]:
    """Per-launch rows with v2 confound-controlled fields."""
    d = json.loads(replay_path.read_text())
    teams = d["info"]["TeamNames"]
    if len(teams) != 4:
        return []
    rewards = d["rewards"]
    if any(r is None for r in rewards):
        return []
    our_seats = [i for i, t in enumerate(teams) if t == our_team]
    if not our_seats:
        return []

    steps = d["steps"]
    n_steps = len(steps)
    end_planets = steps[-1][0]["observation"]["planets"]
    rows: list[dict] = []

    for our_seat in our_seats:
        we_won = rewards[our_seat] == max(rewards)

        # First pass: collect this seat's raw launches with predicted landings.
        seat_rows: list[dict] = []
        for t, step in enumerate(steps):
            entry = step[our_seat]
            actions = entry.get("action") or []
            if not actions:
                continue
            obs = entry["observation"]
            world = World.from_obs(obs)

            for move in actions:
                if not (isinstance(move, list) and len(move) == 3):
                    continue
                src_id, angle, ships = move
                src_id = int(src_id)
                ships = int(ships)
                src = world.planets_by_id.get(src_id)
                if src is None or int(src.owner) != our_seat:
                    continue
                src_prod = int(src.production)
                src_ships_before = int(src.ships)
                src_ships_after = max(0, src_ships_before - ships)

                try:
                    fate = predict_fleet_fate(src, src, float(angle), ships, world, max_steps=200)
                except Exception:
                    continue
                if fate.outcome not in ("target", "planet", "timeout"):
                    seat_rows.append({
                        "t": t, "outcome": fate.outcome,
                        "src_id": src_id, "src_prod": src_prod,
                        "src_ships_before": src_ships_before,
                        "src_ships_after_launch": src_ships_after,
                        "ships": ships,
                        "tgt_id": None, "tgt_prod": 0, "tgt_owner_before": None,
                        "landing_step": None,
                    })
                    continue
                tgt_id = fate.hit_planet_id
                tgt = world.planets_by_id.get(int(tgt_id)) if tgt_id is not None else None
                if tgt is None:
                    continue
                seat_rows.append({
                    "t": t, "outcome": fate.outcome,
                    "src_id": src_id, "src_prod": src_prod,
                    "src_ships_before": src_ships_before,
                    "src_ships_after_launch": src_ships_after,
                    "ships": ships,
                    "tgt_id": int(tgt_id), "tgt_prod": int(tgt.production),
                    "tgt_owner_before": int(tgt.owner),
                    "landing_step": t + int(fate.step),
                })

        # Second pass: compute next_launch_step_from_src per (src_id).
        by_src: dict[int, list[int]] = collections.defaultdict(list)
        for i, r in enumerate(seat_rows):
            by_src[r["src_id"]].append(i)
        for src_id, idxs in by_src.items():
            idxs.sort(key=lambda i: seat_rows[i]["t"])
            for k, i in enumerate(idxs):
                next_t = seat_rows[idxs[k + 1]]["t"] if k + 1 < len(idxs) else n_steps
                seat_rows[i]["next_launch_step_from_src"] = next_t

        # Third pass: short-window src loss + landing-time tgt owner.
        for r in seat_rows:
            landing = r.get("landing_step")
            if landing is None:
                r["src_lost_within_20_pre_relaunch"] = None
                r["tgt_owned_at_landing"] = None
            else:
                next_t = r["next_launch_step_from_src"]
                window_end = min(landing + 20, next_t, n_steps - 1)
                src_lost = False
                for tt in range(landing + 1, window_end + 1):
                    if tt >= n_steps:
                        break
                    owner = planet_owner(
                        steps[tt][0]["observation"]["planets"], r["src_id"]
                    )
                    if owner is not None and owner != our_seat:
                        src_lost = True
                        break
                r["src_lost_within_20_pre_relaunch"] = src_lost

                # Landing-time tgt owner (one tick after landing for combat
                # resolution).
                check_t = min(n_steps - 1, landing + 1)
                tgt_owner_landing = planet_owner(
                    steps[check_t][0]["observation"]["planets"], r["tgt_id"]
                )
                r["tgt_owned_at_landing"] = (
                    tgt_owner_landing is not None and tgt_owner_landing == our_seat
                )

            # End-of-game owner.
            r["src_owner_end"] = planet_owner(end_planets, r["src_id"])
            r["tgt_owner_end"] = (
                planet_owner(end_planets, r["tgt_id"]) if r["tgt_id"] is not None else None
            )
            r["episode"] = replay_path.stem
            r["seat"] = our_seat
            r["we_won_episode"] = we_won
            r["episode_window"] = episode_window(r["t"])

        rows.extend(seat_rows)
    return rows


def fmt_signed(x: float) -> str:
    return f"{'+' if x >= 0 else '-'}{abs(x):.3f}"


def fmt_signed_int(x: int) -> str:
    return f"{'+' if x >= 0 else '-'}{abs(x)}"


def print_pivot(rows: list[dict], label: str, fn_filter=None) -> dict:
    """Print all pivot tables for a row-subset; return verdict dict."""
    if fn_filter is None:
        sub = [r for r in rows if r["tgt_id"] is not None]
    else:
        sub = [r for r in rows if r["tgt_id"] is not None and fn_filter(r)]
    print(f"\n========== {label} ==========")
    print(f"  rows in subset: {len(sub)}")
    if not sub:
        return {}

    stats = collections.defaultdict(lambda: {
        "n": 0, "ships": 0,
        "tgt_won_end": 0, "prod_gain_end": 0,
        "src_lost_end": 0, "prod_loss_end": 0,
        "tgt_won_landing": 0, "prod_gain_landing": 0,
        "src_lost_short": 0, "prod_loss_short": 0,
    })

    # End-of-game attribution dedup (legacy comparable to v1).
    last_src_end: dict = {}
    last_tgt_end: dict = {}
    for i, r in enumerate(sub):
        key_s = (r["episode"], r["seat"], r["src_id"])
        last_src_end[key_s] = i
        key_t = (r["episode"], r["seat"], r["tgt_id"])
        last_tgt_end[key_t] = i

    for i, r in enumerate(sub):
        k = (tier(r["src_prod"]), tier(r["tgt_prod"]))
        s = stats[k]
        s["n"] += 1
        s["ships"] += r["ships"]

        # End-of-game (legacy)
        key_s = (r["episode"], r["seat"], r["src_id"])
        key_t = (r["episode"], r["seat"], r["tgt_id"])
        if last_src_end[key_s] == i and r["src_owner_end"] not in (None, r["seat"]):
            s["src_lost_end"] += 1
            s["prod_loss_end"] += r["src_prod"]
        if last_tgt_end[key_t] == i and r["tgt_owner_before"] != r["seat"] \
           and r["tgt_owner_end"] == r["seat"]:
            s["tgt_won_end"] += 1
            s["prod_gain_end"] += r["tgt_prod"]

        # Landing-time tgt outcome (per-launch, no dedup needed)
        if r["tgt_owner_before"] != r["seat"] and r["tgt_owned_at_landing"]:
            s["tgt_won_landing"] += 1
            s["prod_gain_landing"] += r["tgt_prod"]

        # Short-window src loss (per-launch — no last-launch tautology)
        if r["src_lost_within_20_pre_relaunch"]:
            s["src_lost_short"] += 1
            s["prod_loss_short"] += r["src_prod"]

    # Print pivot.
    print(f"\n  {'src->tgt':<26} {'n':>4} {'ships':>6} "
          f"{'NETlnch':>8} {'NETship':>9} "
          f"{'landCap%':>9} {'srcLs%':>7} {'NETshrt':>9}")
    verdict_rows = {}
    for s_ in TIERS:
        for t_ in TIERS:
            s = stats[(s_, t_)]
            if s["n"] == 0:
                continue
            net_per_launch = (s["prod_gain_end"] - s["prod_loss_end"]) / s["n"]
            net_per_ship = (
                (s["prod_gain_end"] - s["prod_loss_end"]) / s["ships"]
                if s["ships"] else 0.0
            )
            land_cap_rate = s["tgt_won_landing"] / s["n"] if s["n"] else 0.0
            src_loss_short_rate = s["src_lost_short"] / s["n"] if s["n"] else 0.0
            net_short_per_ship = (
                (s["prod_gain_landing"] - s["prod_loss_short"]) / s["ships"]
                if s["ships"] else 0.0
            )
            print(
                f"  {s_+' -> '+t_:<26} {s['n']:>4} {s['ships']:>6} "
                f"{fmt_signed(net_per_launch):>8} {fmt_signed(net_per_ship):>9} "
                f"{land_cap_rate*100:>7.1f}%  {src_loss_short_rate*100:>5.1f}% "
                f"{fmt_signed(net_short_per_ship):>9}"
            )
            verdict_rows[(s_, t_)] = {
                "n": s["n"], "ships": s["ships"],
                "NET_per_launch": net_per_launch,
                "NET_per_ship": net_per_ship,
                "landing_capture_rate": land_cap_rate,
                "src_loss_short_rate": src_loss_short_rate,
                "NET_short_per_ship": net_short_per_ship,
                "prod_gain_landing": s["prod_gain_landing"],
                "prod_loss_short": s["prod_loss_short"],
            }
    return verdict_rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("submission_id")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--team", default=None)
    ap.add_argument("--out-jsonl", default=None)
    args = ap.parse_args(argv)

    ep_dir = REPO / "audit" / "live-episodes" / args.submission_id
    files = sorted(ep_dir.glob("episode-*-replay.json"))
    if not files:
        print(f"no replays in {ep_dir}", file=sys.stderr)
        return 2
    team = args.team or detect_team_name(files)
    print(f"team={team!r}  total_episodes={len(files)}")

    fourp_files = []
    for f in files:
        try:
            d = json.loads(f.read_text())
        except Exception:
            continue
        if len(d.get("info", {}).get("TeamNames", [])) == 4:
            fourp_files.append(f)
    if args.limit:
        fourp_files = fourp_files[: args.limit]
    print(f"4P episodes to scan: {len(fourp_files)}")

    all_rows: list[dict] = []
    for i, f in enumerate(fourp_files):
        all_rows.extend(analyze_episode(f, team))
        if (i + 1) % 5 == 0:
            print(f"  ...{i+1}/{len(fourp_files)} eps  cumulative launches={len(all_rows)}")
    print(f"total launches: {len(all_rows)}")

    if args.out_jsonl:
        out = Path(args.out_jsonl)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as h:
            for r in all_rows:
                h.write(json.dumps(r) + "\n")
        print(f"wrote {out}")

    # ============ Pivots ============
    overall = print_pivot(all_rows, "ALL 4P launches")
    early = print_pivot(all_rows, "EARLY window (t<=150)",
                        lambda r: r["episode_window"] == "early")
    mid = print_pivot(all_rows, "MID window (151..350)",
                      lambda r: r["episode_window"] == "mid")
    late = print_pivot(all_rows, "LATE window (>350)",
                       lambda r: r["episode_window"] == "late")

    # ============ Verdict ============
    print("\n========== VERDICT ==========")
    def cell(table, src_t, tgt_t, key, default=0.0):
        row = table.get((src_t, tgt_t))
        return row[key] if row else default

    ls_early = cell(early, "large(4+)", "small(1-2)", "NET_short_per_ship")
    sl_early = cell(early, "small(1-2)", "large(4+)", "NET_short_per_ship")
    ls_mid = cell(mid, "large(4+)", "small(1-2)", "NET_short_per_ship")
    sl_mid = cell(mid, "small(1-2)", "large(4+)", "NET_short_per_ship")

    print(f"  NET_short_per_ship[large->small] early={ls_early:+.4f} mid={ls_mid:+.4f}")
    print(f"  NET_short_per_ship[small->large] early={sl_early:+.4f} mid={sl_mid:+.4f}")
    delta_early = sl_early - ls_early
    delta_mid = sl_mid - ls_mid
    print(f"  delta (small->large) - (large->small): early={delta_early:+.4f} mid={delta_mid:+.4f}")

    leak_confirmed = (
        ls_early < 0 and ls_mid < 0
        and sl_early > 0 and sl_mid > 0
        and delta_early > 0.05 and delta_mid > 0.05
    )
    leak_rejected = (ls_early >= 0 and ls_mid >= 0)

    if leak_confirmed:
        print("  >>> LEAK CONFIRMED <<<")
        print("      Phase B (opp model cheap-capture bonus) recommended.")
    elif leak_rejected:
        print("  >>> LEAK REJECTED <<<")
        print("      Early+mid show no large->small loss; v1 signal was end-state bias.")
        print("      Null the A.8 leaf; do not implement Phase B.")
    else:
        print("  >>> AMBIGUOUS <<<")
        print("      Mixed signal across windows. Inspect per-window pivots; loop with PI.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
