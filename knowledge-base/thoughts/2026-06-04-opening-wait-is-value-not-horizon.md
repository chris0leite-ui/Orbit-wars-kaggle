# 2026-06-04 — "We wait too long in the opening" is a value-function fact, not a horizon fact

PI flagged a live loss (seed 722289020, perimeter-ring / central-sun map): we sit
idle early and the opponent (Merchant API) takes the whole ring before step 90. PI's
read: sparse map → targets out of reach within horizon K → no candidates → idle.

Built `scripts/opening_starvation.py` and tested it directly. The horizon story is
**false**: 0% of opening boards have their nearest neutral past K_OPEN=20, and on the
exact repro seed the agent had 2–12 launch candidates *every* opening turn and fired
on only 4 of 31. Zero turns were horizon-starved. The "sparseness" is cross-map arc
distance around the sun; adjacent-ring grabs are cheap (ETA 4–10). **We don't launch
because the value function chooses not to** — it hoards ships.

Why this matters as a second-brain note:
- It kills a tempting constant-bump fix (raise K_OPEN / add a far-launch fallback).
  Rule 40 instinct confirmed empirically: the symptom is not where the knob is.
- It relocates the lever to the **chooser/value function's early-expansion
  appetite**, which is a *known-hard, known-confounded* axis: "launch more early"
  already regressed — but in self-play, where waiting is symmetric and launching
  more just overextends.
- The real, untested question is **opponent-relative**: does our waiting get
  *punished* by an aggressive early-expander (the Merchant class), even though it's
  fine in self-play? If yes, the fix is an appetite that's *conditional* on facing an
  aggressive expander — a strategic-mode signal, not a global threshold change. That
  rhymes with the resource-ratio-as-mode idea from 2026-06-03.

Method note worth keeping: separating the **mechanism question** (is the proposer
starved? — cheap, map-only, no game) from the **behavioral question** (does the agent
launch? — needs the actual action stream) is what made this clean and fast. The
step-0 scan needs no game play at all; the launch overlay reads `env.steps[t][me]
["action"]` (a list of `[src, angle, ships]`; `len==0` ⇒ no launch). Reusable pattern
for any "why isn't the agent doing X" question.
