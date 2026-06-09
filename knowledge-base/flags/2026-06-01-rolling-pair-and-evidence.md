# Flags — 2026-06-01

- **Rolling-pair eviction wrinkle.** Pair = [53259633 (expand-credit, ~1086),
  53248277 (size-balance, ~1138)]. 53259633 is the NEWEST, so the next submit
  evicts the GOOD 1138, keeping the weaker 1086. Any next push must clear
  ~1138 or it downgrades the pair. No rush — pair is stable; don't push a
  speculative variant without ≥1138 expectation.
- **Two submissions this session settled BELOW backstop** (size-balance ~1138
  vs champion ~1183; expand-credit ~1086), both pushed on weak/no winrate
  evidence under PI override. Reaffirm: n=16 triage and single-opponent A/Bs
  are not submit evidence (Rules 43/45). Calibration-probe framing held, but
  the floor drifted down ~1183 → ~1086/1138.
- **HANDOVER.md bloat:** 517 lines (budget 150). Needs an archive pass next
  session (WRAPUP step 5) — deferred today to avoid a large mechanical edit
  during wrap.
