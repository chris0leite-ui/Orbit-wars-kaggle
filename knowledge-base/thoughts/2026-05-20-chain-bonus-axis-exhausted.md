# 2026-05-20 — chain-bonus axis exhausted; close-read is the missing reflex

The Phase 7→8→9 arc on btjeK base ended in Rule 37 exhaustion: three
variants of the chain-bonus mechanism all failed. The interesting
lesson isn't that the mechanism was broken (it was — see postmortem)
but **how late the broken-ness was discovered.**

## The shape of the failure

| Phase | Mechanism                              | n=16 winrate | Inspect                |
|-------|----------------------------------------|--------------|------------------------|
| 7     | Cheap-delta bump only                  | 7/16 (43.8%) | not run                |
| 8     | + chooser bypass (use cheap as Δ)      | not A/B'd    | 0/31 relay completions |
| 9     | + leg-2 ledger commit (force relay)    | 1/16 (6.2%)  | not re-run             |

Phase 7 was the first probe; the right next step was *not* "make
the bonus stronger" (Phase 8). The right next step was "watch a
game to see whether the bonus is even firing as designed." Both
"bonus not firing" and "bonus firing but ignored by chooser" predict
~50% winrate — A/B at n=16 can't distinguish them.

## What close-read revealed

Phase 8 inspect: 31 chain launches fired (in 146 turns of seed 0),
**0 relay completions**. The captured planet's surviving stack
never launched toward the predicted T2. So the bonus credited
leg-1 with leg-2 value that the agent never delivered.

Two diametrically opposed reads:
- (i) *force the relay* — agent's "don't commit" was the bug.
  → Phase 9 implements this.
- (ii) *the bonus is wrong* — agent's adaptive re-evaluation each
  turn was actually correct; forcing a stale plan destroys
  adaptivity.
  → Implies kill the mechanism.

PI chose (i). Phase 9 went to 1/16 = 6.2% (Wlo=0.011). The data
strongly supports (ii): adaptive re-evaluation was good, forcing
commitment harmed.

## The reflex that's missing

My default mental loop is: hypothesis → A/B → iterate. The
inserted reflex needs to be: hypothesis → close-read → A/B →
iterate.

Cost of the missing reflex this session: ~4h compute on Phases
8 + 9 that the close-read after Phase 7 would have killed. PI
prompted "looked at one game closely?" mid-iteration, which is a
calibration signal that my default workflow consistently understates
single-game inspection.

Promotion candidate logged in postmortem: a Rule 42 about close-
reading before scaling A/B beyond n=16.

## Generalisation

The chain-bonus axis is dead, but the failure pattern isn't unique
to chain-bonus. Any mechanism that **credits future actions the
agent isn't actually committed to taking** has this shape. A
generic check: when you add a +bonus for some predicted-future-
EV, ask "what happens if I run inspect and observe the future
never fires?" If "we eat the bonus's cost anyway" is the answer,
the mechanism is structurally broken. Phase 7 fits this; the
hold-feasibility filter (also on btjeK) does NOT — it gates
decisions, not credit.

## Artifacts to preserve

- `scripts/inspect_chain_game.py` — generalisable. Adapts to any
  mechanism that emits per-turn metadata; just swap the
  "mechanism-firing" check.
- `tests/fixtures/replays/claws_77164175_step223.json` — the
  reference state for any future relay-pattern work. Even if the
  Phase 7 mechanism is dead, the **observation** (Claws actually
  plays relays in 77164175) remains a load-bearing piece of
  evidence about what good opp play looks like.

## Next session

PI signed off "wrap up" — pivot mechanism family. Don't ship from
`claude/phase7-btjek-chain-bonus`. btjeK at HEAD `0b83734`
(chain-off) is the strict-best from this branch.
