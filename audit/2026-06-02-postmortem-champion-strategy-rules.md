# Postmortem — 2026-06-02 champion-strategy-rules

## What went wrong
- No bad decisions given decision-time priors. The approved re-test plan executed
  cleanly: stack built + cost-smoked, A/B'd to a clean n=16 null; the garrison-axis
  gate was resolved with a free replay diagnostic *before* any build spend.
- **PI-overrides (2):** (a) PI rejected the aggressive "tech-and-kill" design after
  asking me to lay it out — a merits rejection, not a process miss. (b) PI: "no
  further brainstorming, wait for the A/B" — calibration signal that in tight-steering
  mode the PI wants less agent-initiated ideation and a straight run-it/report-it loop.
- **Rule-gap (caught, zero cost):** I nearly queued `STAGNANT_DRAIN` (rear-drain / H1
  axis) as the cheap probe for a gate finding that was about *early launch timing*
  (value-head axis). Same family as 2026-06-01 `wrong-ab-instrument-champion-mirror`.
  Caught before spend.

## Frictions logged this session
- `heavy-vs-heavy-play-smoke-timeout` (audit/friction.md 2026-06-02) — behavior-diff
  smoke timed out at 180s for two search agents; chained cost-smoke never ran.
- `cheap-probe-tests-wrong-axis` (audit/friction.md 2026-06-02) — planned cheap probe
  tested an adjacent-but-different axis than the diagnostic flagged; caught before spend.

## Promotion candidates (PI ratified: NO)
- `cheap-probe-tests-wrong-axis` → drafted as a Rule-44 axis/instrument-identity
  sub-clause. **PI: do not promote.**
- `heavy-vs-heavy-play-smoke-timeout` → not proposed for promotion (one-off tooling
  nuance). **PI: do not promote.**
- Net: nothing promoted to `improvements.md` this session. Both remain in friction.md.

## PI additions (from step 4)
- None. PI: "nothing to either to promote."

## Framework version at session-end
- Commit SHA: 5d41e25006718c71566342f93a3ea655e9fc35bb
- Active rules: CLAUDE.md Rules 0–48 (operating rules) + R-defaults R1–R8.
- Loaded skills this session: postmortem.
