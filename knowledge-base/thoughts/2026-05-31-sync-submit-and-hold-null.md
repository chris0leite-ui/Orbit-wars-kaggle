# 2026-05-31 — sync coalition submitted; size-to-hold is null; two durable lessons

Branch: `claude/champion-strategy-rules-00JzI`. Session: confirm the sync
coalition panel, build/test "size-to-hold" (Lever 1), submit.

## Outcome in one paragraph

The synchronized two-source team-up ("sync coalition") confirmed strong vs
the calibration panel (v7_0 90.6%, v4_planner 93.8%, v3.5.1 87.5%, all
Wilson-lo ≥ 0.72), so we submitted it to Kaggle as a calibration probe
(sub `53223160`). The size-to-hold refinement we built and tested this
session is a NULL (7/7 tie vs the champion) and is shelved default-OFF.

## Lesson 1 — a refinement that fixes a real leak can still be win-rate-neutral

Size-to-hold targets a real, measured failure: vs a strong counter-attacker
the sync coalitions capture a planet and then lose it 40% of the time
(garrison+1 sizing → ~1 ship planted → recaptured). The fix (size to survive
the predicted counter) demonstrably drops that to 0% in the trace. **But the
A/B was an exact tie** (7/16 both arms vs champion, 4 games flipped 2W/2L).

Why: holding the planets we take is worth ~the same as the captures we now
*decline* by being more cautious. The two cancel. The leak was also
**opponent-specific** — vs weak `v7_0` there was no leak at all (0% recapture
either way), and there hold-on merely declined a held-able capture (a mild
regression). Generalisable: "we fixed the leak in the trace" is necessary but
NOT sufficient — the conservatism a fix introduces must be priced in the
win-rate A/B, and the leak's prevalence depends on the opponent class. Always
run the isolation A/B; don't ship on the trace alone.

## Lesson 2 — the bundle-baking gotcha (Kaggle has no env vars)

This nearly shipped an inert agent. Our agents read config from `os.environ`
(`BASELINE_JOINT_SYNC`, `BASELINE_PV_ETA`, `BASELINE_ORBITAL_SAFETY`, …),
all defaulting OFF; local A/B drivers (`clean_ab`, `fast.py`) set those env
vars in the shell. **Kaggle sets none** → a vanilla bundle runs everything
OFF. The local "focal" bundles (`baseline_joint_sync_focal.py`,
`baseline_joint_sync_hold_focal.py`) do NOT bake config — they were only ever
valid because the harness supplied the env. A correct submission must prepend
an `os.environ.setdefault(...)` header with the full tested env block **above
the first inlined module** (modules read their constants at import time, so
the header must run first). The champion bundle does exactly this (lines
6–21, `_lr_os.environ.setdefault`).

Verification that actually catches it: a clean-env smoke — scrub every
`BASELINE_*`/`PV_*` from `os.environ`, import the bundle (register it in
`sys.modules` first or `@dataclass` resolution fails), assert the baked values
took, then run one full game. This is now in the HANDOVER resume notes.
Promote to a friction/rule if it recurs.

## Open question carried forward

Is sync a genuine ladder gain, or only a panel-beater? The submit answers it
empirically (read sub 53223160's μ). The missing control we never ran: the
*champion* vs the same 3-opponent panel — if it also scores ~90%, sync isn't
an upgrade over what's already live. Run that next session if the live μ is
ambiguous.
