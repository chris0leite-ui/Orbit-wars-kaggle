"""H44 Phase 1 — Landing-capture-rate failure-mode diagnostic.

v2 audit (`scripts/large_to_small_audit_v2.py`) showed only 33-46% of
our attack launches own the planet at landing. This script attributes
each FAILED landing-capture to one of four mechanisms so a targeted
Phase 2 fix can be designed.

Failure modes (assigned by precedence A -> C -> B -> D):
  (A) src lost before landing: chooser over-bid; opp captured drained src.
  (C) race condition: third party flipped tgt before our arrival.
  (B) tgt production accrual: defender grew during flight beyond prediction.
  (D) cleanly under-delivered: math was right; chooser sized too small.

Inputs:
  - audit/2026-05-21-large-to-small-v2.jsonl (failure-set source)
  - audit/live-episodes/<sub_id>/episode-*-replay.json (observation snapshots)

CLI:
  python -m scripts.h44_landing_capture_diagnostic <submission_id>
      [--limit N]
      [--in-jsonl audit/2026-05-21-large-to-small-v2.jsonl]
      [--out-jsonl audit/2026-05-21-h44-phase1-landing-capture-diagnostic.jsonl]
"""
from __future__ import annotations
import argparse
import collections
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def tier(prod: int) -> str:
    if prod <= 2:
        return "small(1-2)"
    if prod == 3:
        return "mid(3)"
    return "large(4+)"


TIERS = ["small(1-2)", "mid(3)", "large(4+)"]


def planet_row(planets_list, pid):
    for row in planets_list:
        if int(row[0]) == int(pid):
            return row
    return None


def classify(row: dict) -> str:
    """Precedence: E (off-by-one) -> A -> C -> B -> D -> other.

    NOTE: We do NOT use fleet-list disappearance as a "destroyed" signal —
    fleets disappear from the list when combat resolves at the target,
    which happens at the landing step regardless of outcome. The earlier
    F flag over-counted because of this; removed.
    """
    if row["E_landed_within_5_after"]:
        return "E_prediction_off_by_one"
    if row["src_lost_pre_landing"]:
        return "A_src_lost_pre_landing"
    if row["third_party_flip"]:
        return "C_third_party_flip"
    if row["tgt_grew_more_than_predicted"]:
        return "B_tgt_production_accrual"
    if row["under_delivered_cleanly"]:
        return "D_under_delivered"
    if row["near_tie_combat"]:
        return "G_near_tie_combat"
    return "other"


def analyze_episode(replay_path: Path, failure_rows: list[dict]) -> list[dict]:
    """Add diagnostic fields for each failure-row in this episode."""
    if not failure_rows:
        return []
    d = json.loads(replay_path.read_text())
    steps = d["steps"]
    n_steps = len(steps)
    out: list[dict] = []

    for r in failure_rows:
        t = int(r["t"])
        landing = int(r["landing_step"])
        seat = int(r["seat"])
        src_id = int(r["src_id"])
        tgt_id = int(r["tgt_id"])
        ships = int(r["ships"])
        arrival = landing - t  # = wait_N + eta
        tgt_prod = int(r["tgt_prod"])

        # At launch
        obs_launch = steps[t][0]["observation"]
        planets_launch = obs_launch["planets"]
        fleets_launch = obs_launch.get("fleets", [])
        tgt_at_launch = planet_row(planets_launch, tgt_id)
        tgt_ships_launch = int(tgt_at_launch[5]) if tgt_at_launch else 0
        tgt_owner_launch = int(tgt_at_launch[1]) if tgt_at_launch else -1

        # At landing
        if landing >= n_steps:
            continue
        planets_landing = steps[landing][0]["observation"]["planets"]
        src_at_landing = planet_row(planets_landing, src_id)
        tgt_at_landing = planet_row(planets_landing, tgt_id)
        if src_at_landing is None or tgt_at_landing is None:
            continue
        src_owner_landing = int(src_at_landing[1])
        tgt_owner_landing = int(tgt_at_landing[1])
        tgt_ships_landing = int(tgt_at_landing[5])

        # E — we DO own the tgt within 5 turns after landing+1
        # (predict_fleet_fate off-by-one or fleet still in flight one step).
        E_landed_within_5_after = False
        for k in range(2, 7):  # check landing+2 .. landing+6
            idx = landing + k
            if idx >= n_steps:
                break
            late_planets = steps[idx][0]["observation"]["planets"]
            late = planet_row(late_planets, tgt_id)
            if late is not None and int(late[1]) == seat:
                E_landed_within_5_after = True
                break

        # F — our fleet died in flight. Check fleet existence at THREE
        # checkpoints: t+1 (was it launched), landing-1 (did it survive to
        # the last step before arrival), and landing (did it complete).
        # If launched but missing at landing-1 OR landing, it died.
        F_fleet_destroyed_in_flight = False
        if not E_landed_within_5_after and arrival >= 2:
            launch_fleets = (
                steps[t + 1][0]["observation"].get("fleets", [])
                if t + 1 < n_steps else []
            )
            launched_at_all = any(
                int(f[1]) == seat and int(f[5]) == src_id
                and abs(int(f[6]) - ships) <= 1
                for f in launch_fleets
            )
            if launched_at_all:
                # Check survival at landing-1 and landing.
                check_steps = [landing - 1, landing]
                alive_at_check = False
                for ck in check_steps:
                    if 0 <= ck < n_steps:
                        cs_fleets = steps[ck][0]["observation"].get("fleets", [])
                        if any(
                            int(f[1]) == seat and int(f[5]) == src_id
                            and abs(int(f[6]) - ships) <= 1
                            for f in cs_fleets
                        ):
                            alive_at_check = True
                            break
                if not alive_at_check:
                    F_fleet_destroyed_in_flight = True

        # Race detection: any opp fleet incoming on tgt at launch time?
        # Heuristic: opp fleet whose source planet was tgt's nearest opp
        # AND whose angle points toward tgt-ish. Cheap proxy: opp fleet
        # exists in fleets_launch whose ETA-to-tgt would land <= our
        # landing_step. We use the third_party_flip signal as the actual
        # ground truth; this is a side stat.
        opp_fleets_in_flight = sum(
            1 for f in fleets_launch if int(f[1]) != seat
        )

        # Derived signals.
        # Neutral planets DON'T accrue production (per proposer.py:514-517
        # and env mechanics) — match that here.
        if tgt_owner_launch == -1:
            predicted_defender = tgt_ships_launch
        else:
            predicted_defender = tgt_ships_launch + tgt_prod * arrival
        # actual_delivered estimate: if tgt_owner unchanged since launch,
        # we can compare tgt_ships_landing to predicted_defender.
        # If owner flipped (3rd party), we can't compute it cleanly.
        if tgt_owner_landing == tgt_owner_launch:
            actual_defender = tgt_ships_landing
            tgt_grew_more_than_predicted = actual_defender > predicted_defender + 2
            under_delivered_cleanly = (
                actual_defender <= predicted_defender + 2
                and ships <= actual_defender
            )
        else:
            actual_defender = None
            tgt_grew_more_than_predicted = False
            under_delivered_cleanly = False

        src_lost_pre_landing = (src_owner_landing != seat)
        third_party_flip = (
            tgt_owner_landing != tgt_owner_launch
            and tgt_owner_landing != seat
            and tgt_owner_landing != -1
        )
        # Special case: tgt became neutral during flight (e.g. comet expired).
        if tgt_owner_landing == -1 and tgt_owner_launch != -1:
            third_party_flip = True

        # G — near-tie combat: ships barely exceed (or barely match) the
        # predicted defender. Env's tie rule may keep planet with original
        # owner when attacker_ships ≈ defender_ships.
        near_tie_combat = (
            tgt_owner_landing == tgt_owner_launch
            and abs(ships - predicted_defender) <= 2
        )

        diag = {
            **r,
            "arrival_step": arrival,
            "tgt_ships_launch": tgt_ships_launch,
            "tgt_owner_launch": tgt_owner_launch,
            "tgt_ships_landing": tgt_ships_landing,
            "tgt_owner_landing": tgt_owner_landing,
            "src_owner_landing": src_owner_landing,
            "predicted_defender": predicted_defender,
            "actual_defender": actual_defender,
            "opp_fleets_in_flight_at_launch": opp_fleets_in_flight,
            "src_lost_pre_landing": src_lost_pre_landing,
            "third_party_flip": third_party_flip,
            "tgt_grew_more_than_predicted": tgt_grew_more_than_predicted,
            "under_delivered_cleanly": under_delivered_cleanly,
            "near_tie_combat": near_tie_combat,
            "E_landed_within_5_after": E_landed_within_5_after,
            "F_fleet_destroyed_in_flight": F_fleet_destroyed_in_flight,  # legacy, NOT used in classify()
        }
        diag["primary_diagnosis"] = classify(diag)
        out.append(diag)

    return out


def print_pivot(rows: list[dict], label: str, key_fn) -> None:
    """Per-bucket count + primary diagnosis distribution."""
    by_bucket = collections.defaultdict(lambda: collections.Counter())
    totals = collections.Counter()
    for r in rows:
        k = key_fn(r)
        by_bucket[k][r["primary_diagnosis"]] += 1
        totals[k] += 1

    diags = ["E_prediction_off_by_one", "A_src_lost_pre_landing",
             "C_third_party_flip", "B_tgt_production_accrual",
             "D_under_delivered", "G_near_tie_combat", "other"]
    short = ["E_late", "A_src", "C_race", "B_tgt", "D_under", "G_tie", "other"]
    print(f"\n== {label} ==")
    keys = sorted(by_bucket.keys(), key=lambda k: -totals[k])
    print(f"  {'bucket':<24} {'n':>5}  " + "  ".join(f"{s:>8}" for s in short))
    for k in keys:
        n = totals[k]
        cells = []
        for d in diags:
            v = by_bucket[k][d]
            pct = (100 * v / n) if n else 0
            cells.append(f"{v}({pct:>4.1f}%)")
        print(f"  {str(k):<24} {n:>5}  " + "  ".join(f"{c:>8}" for c in cells))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("submission_id")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--in-jsonl",
                    default="audit/2026-05-21-large-to-small-v2.jsonl")
    ap.add_argument("--out-jsonl",
                    default="audit/2026-05-21-h44-phase1-landing-capture-diagnostic.jsonl")
    args = ap.parse_args(argv)

    in_path = REPO / args.in_jsonl
    if not in_path.exists():
        print(f"missing input: {in_path}", file=sys.stderr)
        return 2

    all_rows = [json.loads(l) for l in open(in_path)]
    # Failure set: tried to capture (not own), didn't own at landing.
    failures = [
        r for r in all_rows
        if r.get("tgt_id") is not None
        and r.get("tgt_owner_before") is not None
        and r["tgt_owner_before"] != r["seat"]
        and r.get("tgt_owned_at_landing") is False
        and r.get("landing_step") is not None
    ]
    print(f"loaded {len(all_rows)} v2 rows; failure set = {len(failures)}")

    # Group by episode.
    by_ep: dict[str, list[dict]] = collections.defaultdict(list)
    for r in failures:
        by_ep[r["episode"]].append(r)
    episodes = sorted(by_ep.keys())
    if args.limit:
        episodes = episodes[: args.limit]
    print(f"episodes to process: {len(episodes)}")

    ep_dir = REPO / "audit" / "live-episodes" / args.submission_id
    diagnosed: list[dict] = []
    for i, ep_stem in enumerate(episodes):
        ep_path = ep_dir / f"{ep_stem}.json"
        if not ep_path.exists():
            print(f"  missing replay: {ep_path}", file=sys.stderr)
            continue
        rows = analyze_episode(ep_path, by_ep[ep_stem])
        diagnosed.extend(rows)
        if (i + 1) % 5 == 0:
            print(f"  ...{i+1}/{len(episodes)} eps  cumulative diagnosed={len(diagnosed)}")
    print(f"total diagnosed rows: {len(diagnosed)}")

    if args.out_jsonl:
        out = REPO / args.out_jsonl
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as h:
            for r in diagnosed:
                h.write(json.dumps(r) + "\n")
        print(f"wrote {out}")

    # ============ Pivot tables ============
    overall = collections.Counter(r["primary_diagnosis"] for r in diagnosed)
    n = len(diagnosed) or 1
    print(f"\n========== OVERALL failure-mode breakdown (n={n}) ==========")
    for diag in ["E_prediction_off_by_one", "A_src_lost_pre_landing",
                 "C_third_party_flip", "B_tgt_production_accrual",
                 "D_under_delivered", "G_near_tie_combat", "other"]:
        v = overall[diag]
        print(f"  {diag:<32} {v:>5}  ({100*v/n:>5.1f}%)")

    print_pivot(diagnosed, "By (src_tier -> tgt_tier)",
                lambda r: f"{tier(r['src_prod'])} -> {tier(r['tgt_prod'])}")
    print_pivot(diagnosed, "By episode_window",
                lambda r: r["episode_window"])
    print_pivot(diagnosed, "By episode outcome",
                lambda r: "won" if r["we_won_episode"] else "lost")

    # Sanity checks
    print("\n========== Sanity checks ==========")
    no_flag = sum(1 for r in diagnosed if r["primary_diagnosis"] == "other")
    print(f"  rows with no diagnosis flag (other): {no_flag} ({100*no_flag/n:.1f}%)")
    multi_flag = sum(
        1 for r in diagnosed
        if sum([r["src_lost_pre_landing"], r["third_party_flip"],
                r["tgt_grew_more_than_predicted"], r["under_delivered_cleanly"]]) >= 2
    )
    print(f"  rows with >=2 diagnosis flags (precedence resolves): {multi_flag} ({100*multi_flag/n:.1f}%)")

    # Verdict
    print("\n========== VERDICT ==========")
    sorted_diags = sorted(overall.items(), key=lambda x: -x[1])
    top_diag, top_count = sorted_diags[0] if sorted_diags else ("none", 0)
    top_pct = 100 * top_count / n if n else 0
    print(f"  dominant failure mode: {top_diag} ({top_pct:.1f}%)")
    if top_pct >= 40:
        print(f"  >>> DOMINANT MODE NAMED <<< Phase 2 fix axis: see plan decision tree.")
    else:
        print(f"  >>> MIXED <<< no mode >= 40%; multi-pronged fix or re-plan.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
