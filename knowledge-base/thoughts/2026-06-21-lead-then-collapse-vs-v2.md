# 2026-06-21 — review of the vjpgup native-offense line + the lead-then-collapse pattern

> AI session note (diagnosis), written while reviewing branch
> `claude/submission-strategy-review-vjpgup` at PI request: "understand it, review
> it, fix it, think it through." Plain English (Rule 0). Not a PI voice-dump.

## What the branch does (the idea)

The live agent is `least_resistance` (producer `orbit_lite` garrison scorer + a
2-ply lookahead). The vjpgup line adds, all baked ON in the shipped stack:

- **native flip-hazard leaf** (`LR_NATIVE_LEAF`): the 2-ply leaf is no longer raw
  "my ships minus theirs at the horizon." It projects the board forward and lets
  the opponent's *reachable* ship mass leak ownership of planets they can reach, so
  an exposed capture scores lower than a holdable one.
- **reinforcement race** (`LR_NATIVE_REINFORCE`): a capture is only credited to the
  extent our *other* planets can route ships to hold it faster than the enemy can
  attack it. Finite, conserved reinforcement → favours fewer, closer, sustainable
  captures over a thin spread.
- **concentrate** (`LR_CONCENTRATE`): adds decisive single-fleet captures sized to
  beat the defender, plus a value-ordered commit so the 2-ply prefers
  concentration prefixes.
- **native offense** (`LR_NATIVE_OFFENSE`): a mirror of the defensive hazard — credit
  the capture *potential* of enemy/neutral planets our massed ships can reach.
  Intent: reward massing for a real capture, punish far dribble.

The idea is sound and it matches the PI's thesis (capture planets that **stick** so
our production grows and theirs shrinks). The live submission with the full stack
(sub **53906150**, "lr_offense") reads **μ ≈ 1154.5 after ~10 h** — the best of the
rolling history (concentrate 1107, robust 1090, native_reinforce 1033).

## Review finding #1 — stale "REFUTED" comment (fixed)

`native_forward.py` carried a comment declaring the offense term REFUTED /
passivity-biased / "kept default-OFF." That was written when offense was tested
*standalone* on the plain native leaf (commit 5b78471d). It was later baked ON on
the concentrate+reinforce stack (commit dca1504d) where it helps, and the comment
was never updated — it flatly contradicted the shipped, highest-μ config. Corrected
the comment this session (no behaviour change; default-OFF path still byte-identical).

## Review finding #2 — the shipped stack was never A/B'd at sample vs V2

The submission messages say "n=1 grounded" and "local n=8." The full
offense+concentrate+reinforce stack has **no n≥32 A/B vs Producer V2** behind it.
Its evidence is anecdotal seeds + ladder μ. Per PI directive this session we are
*not* running extensive A/B — we watch games — so this is noted, not actioned.

## Review finding #3 (the big one) — lead-then-collapse

Watching 2P games vs Producer V2 (production-share trace), the losses share one
signature: **we build a commanding production lead in the early-mid game and then
collapse and lose.** Examples (seat 0, production mine/opp):

| seed | early-mid peak | later | final |
|---|---|---|---|
| 6007 | s48 **354 / 145** (2.4:1) | s96 289/304 | loss |
| 6013 | s29 **201 / 76** (2.6:1) | s58 242/354 | loss |
| 6001 | s63 450/411 (ahead) | s94 157/401 | loss |
| 106499442 | s31 211/178 (ahead) | s62 95/167 | loss |

Wins look the opposite — **monotonic** share growth, captures compound:
1127764379 (32→45→54→92%), 6031 (9→42→57→95%), 6007 *seat 1* (14→26→27→74%).

So the win/loss discriminator is precisely the PI's thesis: do captures **stick and
compound**, or do we **peak and get overrun**. The earlier HANDOVER framing ("close
games, only −2.8 ship margin at step 50") under-stated it — in *production* terms we
are often 2:1+ ahead mid-game and still lose. Ship-margin hid it because V2 hoards a
concentrated ship reserve (few launches) while we convert ships into spread-out
production; once their reserve is big enough they punch through our thinly-defended
planets faster than we react, and the spread production cascades away.

## Review finding #4 — SEAT/MAP confound (do not over-read win-rate)

On seed 6007, **seat 0 loses and seat 1 wins — and in both runs the seat-1 player
won regardless of whether it was us or V2.** So that map carries a seat-1 edge. The
quick triage above was all seat 0, i.e. biased toward the bad seat; it is NOT
evidence of an overall low win-rate. (Contradicts the old "seat-invariant" claim for
at least some maps.) The lead-then-collapse *mechanism* is seat-independent —
blowing a 2.4:1 lead is a failure from either seat — but any win-rate number must
rotate seat across seeds (the standing methodology rule).

## Candidate fix direction (for PI sign-off, NOT yet implemented)

The collapse is a **defense / consolidation** failure against a **concentrated**
opponent strike, not a scatter or offense problem. The defensive hazard
(`atk_reach`) currently uses `aggregate="allocate"`, which *splits* the enemy's mass
across all of our reachable planets — so a single massed enemy stronghold reads as a
diffuse, survivable threat and we never garrison enough to stop the real punch.
`LR_NATIVE_THREAT_MAX=1` already exists (default-OFF): it makes each of our planets
see the **worst-case single enemy stronghold** that can reach it. That is the
smallest change that could let the agent feel the concentrated punch and hold a
reserve. Needs replay confirmation on the lead-then-collapse seeds before any submit.

## Artifacts this session
- Replays sent to PI: `g_6007_lead_lost.html` (seat0 loss from 2.4:1 lead),
  `g_6007_seat1_WIN.html` (same map, won, monotonic), `g_1127764379_on.html`
  (clean win), `g_6013_lead_lost.html` (early lead blown).
- Producer V2 re-pulled to `audit/external/agents/slawekbiel_the-producer-v2/`.
