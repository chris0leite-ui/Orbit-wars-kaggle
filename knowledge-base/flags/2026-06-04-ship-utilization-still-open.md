# 2026-06-04 — ship utilization observation still open

PI repeatedly observes (live games) that rear stockpiles sit idle while
the front fights. Three implementations of a circulation-style fix
falsified this session — see
`audit/2026-06-04-postmortem-champion-ml-graft-majestic-storm.md` —
but the OBSERVATION itself is NOT falsified.

**Watch for**: future opportunities to address ship utilization without
a thin post-pass over the existing chooser. Specifically:

- Chooser rewrite (pressure-aware scoring) — the way Biel's Producer
  works end-to-end.
- Goal-directed 2-hop pre-positioning when a concrete 2-hop attack is
  identifiable.
- Any chooser-internal mechanism that increases per-source firing rate
  for stockpile planets.

Do NOT propose another thin gradient-based post-pass without first
showing how the chooser would consume the pre-positioned ships.
