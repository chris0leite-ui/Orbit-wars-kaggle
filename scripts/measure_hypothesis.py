"""scripts/measure_hypothesis.py — post-submit metric measurement.

For a given submission ID, compute every metric in `lib.metrics` and
print a markdown table. The pre-registration doc declares thresholds in
human-readable prose; this script just produces the numbers.

Usage:
    python -m scripts.measure_hypothesis 52744856
    python -m scripts.measure_hypothesis 52744856 --pull
    python -m scripts.measure_hypothesis 52744856 --append-results

`--pull` first invokes `live_episode_summary --pull` to fetch replays
into `audit/live-episodes/<sub_id>/`. Required on first run.

`--append-results` appends a row to `audit/hypotheses/results.md` with
the computed metric values. The PI then back-fills the verdict column
after comparing against the pre-registration thresholds.

Outputs:
- stdout: markdown table of all metrics
- optional: appended row in audit/hypotheses/results.md

The script intentionally does NOT try to parse the pre-registration
doc's thresholds. Hypothesis verdicts are a human call; the script's
job is honest measurement.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from lib import metrics  # noqa: E402
from scripts.replay_mine import (  # noqa: E402
    detect_team_name,
    mine_one_submission,
)


def _live_episodes_dir(sub_id: str) -> Path:
    return REPO / "audit" / "live-episodes" / str(sub_id)


def _load_replays_with_team(sub_id: str,
                            team_name: str | None = None,
                            ) -> list[tuple[dict, str]]:
    """Load every `episode-*-replay.json` under
    audit/live-episodes/<sub_id>/ and pair it with the detected team
    name. Returns [] if the directory is missing/empty.
    """
    sub_dir = _live_episodes_dir(sub_id)
    if not sub_dir.is_dir():
        return []
    replays = sorted(sub_dir.glob("episode-*-replay.json"))
    if not replays:
        return []
    if team_name is None:
        team_name = detect_team_name(replays, None)
    out: list[tuple[dict, str]] = []
    for path in replays:
        try:
            replay = json.load(open(path))
        except Exception as e:
            print(f"  WARN: skip {path.name}: {type(e).__name__}: {e}",
                  file=sys.stderr)
            continue
        out.append((replay, team_name))
    return out


def _pull_replays(sub_id: str) -> int:
    """Invoke live_episode_summary --pull. Returns exit code."""
    cmd = [sys.executable, "-m", "scripts.live_episode_summary",
           str(sub_id), "--pull"]
    print(f"--- pulling replays for {sub_id} ---", file=sys.stderr)
    return subprocess.call(cmd, cwd=str(REPO))


def measure(sub_id: str,
            *,
            team_name: str | None = None,
            ) -> dict[str, float]:
    """Compute every registered metric for the named submission.

    Rollup metrics use `replay_mine.mine_one_submission(sub_id)`.
    Replay-walking metrics walk the same replay JSONs but at a finer
    granularity. Missing inputs (empty replay folder) yield 0.0 for
    every metric.

    Returns a dict `{metric_name: value, ...}`.
    """
    rollup = mine_one_submission(sub_id, team_name=team_name)
    if "error" in rollup:
        print(f"  ERROR mining sub {sub_id}: {rollup['error']}",
              file=sys.stderr)
        if "hint" in rollup:
            print(f"  HINT: {rollup['hint']}", file=sys.stderr)
        return {}

    results: dict[str, float] = {}
    # Rollup-based metrics.
    for name, fn in sorted(metrics._ROLLUP_METRICS.items()):
        try:
            results[name] = float(fn(rollup))
        except Exception as e:
            print(f"  ERROR computing {name}: {type(e).__name__}: {e}",
                  file=sys.stderr)
            results[name] = float("nan")

    # Replay-walking metrics — load the same replays once and pass to
    # every metric.
    replays_with_team = _load_replays_with_team(
        sub_id, team_name=rollup.get("team_name") or team_name,
    )
    for name, fn in sorted(metrics._REPLAY_METRICS.items()):
        try:
            results[name] = float(fn(replays_with_team))
        except Exception as e:
            print(f"  ERROR computing {name}: {type(e).__name__}: {e}",
                  file=sys.stderr)
            results[name] = float("nan")

    # Bonus context (NOT registered metrics, but useful for the row).
    results["__n_episodes"] = float(rollup.get("n_episodes", 0))
    results["__n_fleets"] = float(rollup.get("n_fleets", 0))
    return results


def render_markdown(sub_id: str, results: dict[str, float]) -> str:
    """Build the stdout-facing markdown report."""
    lines: list[str] = []
    lines.append(f"# Hypothesis measurement — sub {sub_id}")
    lines.append("")
    n_ep = int(results.get("__n_episodes", 0))
    n_fl = int(results.get("__n_fleets", 0))
    lines.append(f"Sample: **{n_ep}** episodes / **{n_fl}** fleets.")
    lines.append("")
    lines.append("## Metrics")
    lines.append("")
    lines.append("| metric | value | v15 | top10 | midpack |")
    lines.append("|---|---:|---:|---:|---:|")
    for name in metrics.list_metrics():
        value = results.get(name, float("nan"))
        v15 = metrics.baseline(name, "v15")
        t10 = metrics.baseline(name, "top10")
        mid = metrics.baseline(name, "midpack")

        def _cell(x):
            return "—" if x is None else f"{x:.3f}"

        lines.append(f"| `{name}` | {value:.3f} | {_cell(v15)} | "
                     f"{_cell(t10)} | {_cell(mid)} |")
    lines.append("")
    lines.append("PI: compare each `value` against the threshold the "
                 "pre-registration doc declared for that metric. "
                 "Verdict (confirmed / wrong_axis / refuted / mixed) goes "
                 "in `audit/hypotheses/results.md`.")
    return "\n".join(lines)


def append_results_row(sub_id: str,
                       results: dict[str, float],
                       *,
                       pre_register_doc: str | None = None,
                       results_path: Path | None = None,
                       ) -> None:
    """Append a measurement row to audit/hypotheses/results.md.

    The verdict / μ_delta / postmortem columns are left as 'TBD' for
    the PI to back-fill after looking at settled live μ.
    """
    if results_path is None:
        results_path = REPO / "audit" / "hypotheses" / "results.md"
    if not results_path.exists():
        print(f"  WARN: {results_path} missing — skip append",
              file=sys.stderr)
        return

    n_ep = int(results.get("__n_episodes", 0))
    waste = results.get("waste_attack_fraction", float("nan"))
    win = results.get("win_fraction", float("nan"))
    first = results.get("first_launch_step", float("nan"))

    row = (f"| {sub_id} | (measured-via-script) | TBD | TBD | "
           f"{pre_register_doc or 'TBD'} | TBD | TBD | TBD | "
           f"win={win:.3f} waste_atk={waste:.3f} first_launch={first:.1f} "
           f"(n_ep={n_ep}) | "
           f"win={win:.3f}, waste_atk={waste:.3f}, first={first:.1f} | "
           f"TBD | TBD |")

    # Insert AFTER the table header, BEFORE existing rows (newest-first
    # ordering per the README).
    text = results_path.read_text().splitlines()
    # Find the first data row (line starting with `| ` after the header
    # separator `|---|`). Insert above it.
    insert_idx = None
    for i, line in enumerate(text):
        if line.startswith("|---"):
            insert_idx = i + 1
            break
    if insert_idx is None:
        # Append to EOF as fallback.
        text.append(row)
    else:
        text.insert(insert_idx, row)

    results_path.write_text("\n".join(text) + ("\n" if not text[-1].endswith("\n") else ""))
    print(f"  appended row to {results_path}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="measure_hypothesis")
    ap.add_argument("sub_id",
                    help="Kaggle submission ID (e.g. 52744856).")
    ap.add_argument("--pull", action="store_true",
                    help="Pull replays via live_episode_summary --pull "
                         "before measuring (use on first run).")
    ap.add_argument("--team-name",
                    help="Override team detection (rarely needed).")
    ap.add_argument("--append-results", action="store_true",
                    help="Append a row to audit/hypotheses/results.md.")
    ap.add_argument("--pre-register-doc",
                    help="Path to the pre-registration doc (recorded in "
                         "the results row).")
    args = ap.parse_args(argv)

    if args.pull:
        rc = _pull_replays(args.sub_id)
        if rc != 0:
            print(f"WARN: --pull exited with {rc}; continuing",
                  file=sys.stderr)

    results = measure(args.sub_id, team_name=args.team_name)
    if not results:
        return 2

    print(render_markdown(args.sub_id, results))

    if args.append_results:
        append_results_row(
            args.sub_id, results,
            pre_register_doc=args.pre_register_doc,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
