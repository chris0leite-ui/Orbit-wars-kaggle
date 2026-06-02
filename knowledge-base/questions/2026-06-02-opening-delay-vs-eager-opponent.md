# Open question: does the ~7-10 turn opening delay cost games vs an EAGER opponent?

The early-capture gate (2026-06-02, `audit/2026-06-02-early-capture-gap-gate.md`)
found the champion delays its first affordable cheap-neutral launch by ~7-10 turns
(chooser picks a wait-band over fire-now). The delay rate is identical in the
champion's wins (2.32/step) and losses (2.13/step), so within these replays it does
not discriminate outcomes.

BUT all those replays are champion-vs-champion-derived A/Bs, so both sides delay
symmetrically — the metric structurally cannot detect a *universal* opening
inefficiency that a genuinely eager opponent would punish.

**Question to settle (only if/when we revisit opening tempo):** build a single
eager-opening variant (early fire-now bonus, scoped step<15) and A/B it vs the
champion. If it wins, the delay is a real universal weakness; if it ties/loses, the
wait-band is correctly valuing accumulation and the axis is dead.

Status: PARKED. PI rejected building the aggressive "tech-and-kill" re-framing this
session. Not blocking; revisit only if the conversion-axis work also stalls.
