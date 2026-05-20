"""Audit hypothesis: in 4P live games we drain LARGE-production sources to
capture SMALL-production targets, net null/negative on production.

Walks every 4P replay under audit/live-episodes/<sub_id>/, for each of OUR
launches (src_id, angle, ships):
  - Predicts landing planet via lib.trajectory.predict_fleet_fate.
  - Records (src_prod, tgt_prod, src_owner_then, tgt_owner_then, ships_left_on_src).
  - Attributes end-of-game ownership change of src and tgt back to launches.

Output (printed): pivot table of net production-per-launch by (src_prod, tgt_prod)
bucket plus loss-rate of large sources within 40 turns after a drain.

Run:
    python -m scripts.large_to_small_audit 52827111 [--limit N]
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


def planet_lookup(planets_list: list, pid: int) -> list | None:
    for row in planets_list:
        if int(row[0]) == int(pid):
            return row
    return None


def analyze_episode(replay_path: Path, our_team: str) -> list[dict]:
    """Return one row per OUR launch in this episode."""
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
        for t, step in enumerate(steps):
            entry = step[our_seat]
            actions = entry.get("action") or []
            if not actions:
                continue
            obs = entry["observation"]
            world = World.from_obs(obs)
            planets_now = obs["planets"]

            for move in actions:
                if not (isinstance(move, list) and len(move) == 3):
                    continue
                src_id, angle, ships = move
                src_id = int(src_id)
                ships = int(ships)
                src = world.planets_by_id.get(src_id)
                if src is None:
                    continue
                if int(src.owner) != our_seat:
                    # already captured this turn — skip
                    continue
                src_prod = int(src.production)
                src_ships_before = int(src.ships)
                src_ships_after = max(0, src_ships_before - ships)

                # Use src itself as the dummy "target" — we read hit_planet_id.
                try:
                    fate = predict_fleet_fate(src, src, float(angle), ships, world, max_steps=200)
                except Exception:
                    continue
                if fate.outcome not in ("target", "planet", "timeout"):
                    # sun / oob — count as a wasted launch but skip target attribution
                    rows.append({
                        "episode": replay_path.stem,
                        "seat": our_seat,
                        "t": t,
                        "outcome": fate.outcome,
                        "src_id": src_id,
                        "src_prod": src_prod,
                        "src_ships_before": src_ships_before,
                        "src_ships_after_launch": src_ships_after,
                        "ships": ships,
                        "tgt_id": None, "tgt_prod": 0, "tgt_owner_before": None,
                        "src_owner_end": None, "tgt_owner_end": None,
                        "src_lost_within_40": None,
                    })
                    continue

                tgt_id = fate.hit_planet_id
                tgt = world.planets_by_id.get(int(tgt_id)) if tgt_id is not None else None
                if tgt is None:
                    continue
                tgt_prod = int(tgt.production)
                tgt_owner_before = int(tgt.owner)

                landing_step = t + int(fate.step)
                # Source owner at landing_step + 40, capped at end
                check_idx = min(n_steps - 1, landing_step + 40)
                check_planets = steps[check_idx][0]["observation"]["planets"]
                src_row_then = planet_lookup(check_planets, src_id)
                src_owner_then = int(src_row_then[1]) if src_row_then else None

                src_row_end = planet_lookup(end_planets, src_id)
                tgt_row_end = planet_lookup(end_planets, int(tgt_id))
                src_owner_end = int(src_row_end[1]) if src_row_end else None
                tgt_owner_end = int(tgt_row_end[1]) if tgt_row_end else None

                rows.append({
                    "episode": replay_path.stem,
                    "seat": our_seat,
                    "t": t,
                    "outcome": fate.outcome,
                    "src_id": src_id,
                    "src_prod": src_prod,
                    "src_ships_before": src_ships_before,
                    "src_ships_after_launch": src_ships_after,
                    "ships": ships,
                    "tgt_id": int(tgt_id),
                    "tgt_prod": tgt_prod,
                    "tgt_owner_before": tgt_owner_before,
                    "src_owner_end": src_owner_end,
                    "tgt_owner_end": tgt_owner_end,
                    "src_owner_then": src_owner_then,
                    "src_lost_within_40": (
                        src_owner_then is not None and src_owner_then != our_seat
                    ),
                    "we_won_episode": rewards[our_seat] == max(rewards),
                })
    return rows


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
        rows = analyze_episode(f, team)
        all_rows.extend(rows)
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

    # ----- Pivots -----
    def tier(prod: int) -> str:
        return "small(1)" if prod <= 1 else ("mid(3)" if prod == 3 else "large(4+)")

    # 1. Launch frequency by (src_tier, tgt_tier)
    by_pair = collections.Counter()
    for r in all_rows:
        if r["tgt_id"] is None:
            continue
        by_pair[(tier(r["src_prod"]), tier(r["tgt_prod"]))] += 1
    print("\n== Launch counts (src tier -> tgt tier), 4P only ==")
    tiers = ["small(1)", "mid(3)", "large(4+)"]
    hdr = "src vs tgt"
    print(f"  {hdr:<12} " + " ".join(f"{t:>10}" for t in tiers) + "  row_total")
    for s in tiers:
        row_total = sum(by_pair[(s, tt)] for tt in tiers)
        print(f"  {s:<12} " + " ".join(f"{by_pair[(s, tt)]:>10}" for tt in tiers) + f"  {row_total:>9}")

    # 2. Source loss rate by src_tier (was source captured within 40 turns of OUR launch?)
    src_loss = collections.defaultdict(lambda: [0, 0])  # tier -> [losses, total]
    for r in all_rows:
        if r["tgt_id"] is None or r["src_lost_within_40"] is None:
            continue
        bucket = tier(r["src_prod"])
        src_loss[bucket][1] += 1
        if r["src_lost_within_40"]:
            src_loss[bucket][0] += 1
    print("\n== Source loss within 40 turns of OUR launch, by src tier ==")
    for s in tiers:
        l, n = src_loss[s]
        rate = (l / n) if n else 0.0
        print(f"  src={s:<10}  losses/launches = {l:>4}/{n:<4}  rate={rate:.2%}")

    # 3. Net production attribution
    # For each (episode, seat, src_id) attribute "src lost" to last launch only.
    last_launch_by_src: dict[tuple, int] = {}
    for i, r in enumerate(all_rows):
        if r["tgt_id"] is None:
            continue
        key = (r["episode"], r["seat"], r["src_id"])
        last_launch_by_src[key] = i
    # For each (episode, seat, tgt_id) attribute "tgt gained" to first launch
    # where we eventually own tgt at end (we're crediting this launch if it
    # initiated the chain). Simpler: credit to last launch into that tgt that
    # we ended up owning.
    last_launch_by_tgt: dict[tuple, int] = {}
    for i, r in enumerate(all_rows):
        if r["tgt_id"] is None:
            continue
        key = (r["episode"], r["seat"], r["tgt_id"])
        last_launch_by_tgt[key] = i

    # Net effect per launch (attribution-deduped)
    bucket_stats = collections.defaultdict(lambda: {
        "n": 0, "src_lost": 0, "tgt_gained": 0,
        "prod_lost": 0, "prod_gained": 0,
    })
    for i, r in enumerate(all_rows):
        if r["tgt_id"] is None:
            continue
        bkey = (tier(r["src_prod"]), tier(r["tgt_prod"]))
        st = bucket_stats[bkey]
        st["n"] += 1

        # Source-loss attribution
        key_src = (r["episode"], r["seat"], r["src_id"])
        is_last_for_src = last_launch_by_src.get(key_src) == i
        src_was_lost_by_end = (
            r["src_owner_end"] is not None and r["src_owner_end"] != r["seat"]
        )
        if is_last_for_src and src_was_lost_by_end:
            st["src_lost"] += 1
            st["prod_lost"] += r["src_prod"]

        # Target-gain attribution
        key_tgt = (r["episode"], r["seat"], r["tgt_id"])
        is_last_for_tgt = last_launch_by_tgt.get(key_tgt) == i
        tgt_was_other_before = (
            r["tgt_owner_before"] is not None and r["tgt_owner_before"] != r["seat"]
        )
        tgt_now_ours = (
            r["tgt_owner_end"] is not None and r["tgt_owner_end"] == r["seat"]
        )
        if is_last_for_tgt and tgt_was_other_before and tgt_now_ours:
            st["tgt_gained"] += 1
            st["prod_gained"] += r["tgt_prod"]

    print("\n== Per-bucket net production (attribution-deduped per src/tgt) ==")
    print(f"  {'src->tgt':<25} {'n':>5} {'tgt_won':>8} {'src_lost':>9} "
          f"{'prod_gain':>10} {'prod_loss':>10} {'NET':>7}")
    for s in tiers:
        for tt in tiers:
            st = bucket_stats[(s, tt)]
            if st["n"] == 0:
                continue
            net = st["prod_gained"] - st["prod_lost"]
            print(f"  {s+'->'+tt:<25} {st['n']:>5} {st['tgt_gained']:>8} "
                  f"{st['src_lost']:>9} {st['prod_gained']:>10} "
                  f"{st['prod_lost']:>10} {net:>+7}")

    # 4. Same pivot but only on LOST episodes vs WON episodes
    print("\n== Bucket counts split by episode outcome ==")
    won_pair = collections.Counter()
    lost_pair = collections.Counter()
    for r in all_rows:
        if r["tgt_id"] is None:
            continue
        key = (tier(r["src_prod"]), tier(r["tgt_prod"]))
        if r["we_won_episode"]:
            won_pair[key] += 1
        else:
            lost_pair[key] += 1
    won_total = sum(won_pair.values()) or 1
    lost_total = sum(lost_pair.values()) or 1
    print(f"  (won episodes total launches = {won_total}, lost = {lost_total})")
    print(f"  {'src->tgt':<25} {'%won':>7} {'%lost':>7}  delta_pp")
    for s in tiers:
        for tt in tiers:
            key = (s, tt)
            if won_pair[key] + lost_pair[key] == 0:
                continue
            pw = 100 * won_pair[key] / won_total
            pl = 100 * lost_pair[key] / lost_total
            print(f"  {s+'->'+tt:<25} {pw:>6.1f}% {pl:>6.1f}%  {pl-pw:+6.1f}")

    # 5. "Drain" specifically: large source -> small target with src_ships_after_launch low
    print("\n== Drain events: src_prod>=3, tgt_prod==1, src_ships_after_launch<=10 ==")
    drains = [r for r in all_rows if r["tgt_id"] is not None
              and r["src_prod"] >= 3 and r["tgt_prod"] == 1
              and r["src_ships_after_launch"] <= 10]
    n_drains = len(drains)
    n_drain_loss = sum(1 for r in drains if r["src_lost_within_40"])
    won_drains = sum(1 for r in drains if r["we_won_episode"])
    lost_drains = n_drains - won_drains
    print(f"  total drain launches: {n_drains}")
    if n_drains:
        print(f"  src lost within 40 turns: {n_drain_loss}/{n_drains} = {n_drain_loss/n_drains:.1%}")
        print(f"  in WON episodes: {won_drains}/{n_drains} = {won_drains/n_drains:.1%}")
        print(f"  in LOST episodes: {lost_drains}/{n_drains} = {lost_drains/n_drains:.1%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
