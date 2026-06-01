# Postmortem — 2026-06-01 game-theory-winning-strategy-SEU7P

Covers the 2026-05-31 work-session (08:44–12:25 UTC, three commits) that
shipped `composite_universal` to live ladder and then attempted two
diagnostic / fix rounds against the resulting μ ≈ 1085 regression. No
prior wrap-up was run for that session; this artifact closes it.

## What went wrong

- **Round 1 Stage 1 refactor re-enabled a known-falsified flag.** Flipped
  `COMPOSITE_PRODUCTION_PV` from 0→1 in `agents/baseline/main.py`. The
  same file contained an in-source comment documenting the 2026-05-18
  falsification (39.6 % at n=96, "Default OFF"). Did not grep the file
  before editing. Rule 44 violation (state-of-truth read before
  subsystem edits) — Rule 44 didn't bind because the state-of-truth
  lived IN the file under edit, not in `MULTI_BRANCH.md`. Cost: n=27
  A/B at 24 %, reverted in commit `df3361c`. ~30 min compute.
- **Round 3 rotation infrastructure built before validating premise.**
  Self-play P0=30 / P1=1 was the evidence; hypothesis = "geometric seat
  asymmetry, rotation fixes." A ≤30-min instrumentation probe of
  `score_candidate_v4` with both seats fed the same canonical-frame
  input would have falsified the rotation plan (the divergence source
  is INSIDE the chooser, not in the obs). Instead built ~50 LOC + 4
  unit tests + integration test (~2 h). Outcome: rotation correct
  mathematically, self-play STILL P0=5 / P1=0 across 5 seeds. Infra
  preserved in commit `edd63b3` for next-session re-use.
- **`composite_universal` shipped on n=16 evidence pre-session.** Local
  A/B was 9/16 = 56 % vs champion, Wilson-lo = 0.32 — below Rule 45's
  n≥32 / Wlo≥0.50 submission gate. Live μ ≈ 1085, ~100 below champion's
  1188. Recurrence of `n16-falsely-shows-parity`. Rule 45 didn't bind
  because the submit happened against the rule.

## Frictions logged this session

See `audit/friction.md ## 2026-06-01` (three new tags appended in this
wrap-up):
- `same-file-falsification-comment-skipped`
- `infra-built-before-premise-validated`
- `n16-falsely-shows-parity` (recurrence; 3rd occurrence)

## Promotion candidates (PI ratified: NO)

Two candidates drafted; PI declined both. Recorded here for the audit
trail; not committed to `improvements.md`.

- **Candidate A — same-file falsification-comment scan before setdefault
  edits** (proposed Rule 49). Would require a same-file grep for the
  flag name and inspection of any "DISABLED" / "FAILED" / "Default OFF"
  / "n=" hits before a flag-default flip. PI declined.
- **Candidate B — premise-validation gate before ≥30-LOC infrastructure
  builds** (proposed Rule 50). Would require designing and running a
  ≤30-min falsification probe before any ≥30-LOC fix build. PI declined.

## PI additions (from step 4)

"Nothing to add or promote." Treated as a clean PI signal — the
frictions stay logged; no rule promotions land this cycle.

## Framework version at session-end

- Commit SHA: `edd63b3`
- Active rules: 1..48 (CLAUDE.md `## Operating rules — concise`)
- Loaded skills this session: postmortem

## Next-session priority

Instrument `score_candidate_v4` (or the proposer's candidate-emit path)
to find the earliest turn where P0 and P1 diverge when BOTH are fed
identical canonical-frame input. The rotation infra is already in
place; this isolates the seat-asymmetric subsystem (likely opp-model
iteration order, joint-chooser pair-id sort, or candidate dedup).
