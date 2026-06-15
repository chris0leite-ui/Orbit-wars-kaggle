# 2026-06-15 — Frontier/gateway value is a scoring term; corner-neglect is a generation bug (different layers)

Session arc: PI offered a positional-game framing (win key positions that open
better options). We made it rigorous (asset worth ≈ own-income + options-it-opens
× tenure × denial), found the champion scores only the near-term own-income slice,
and built the "options" term first: **frontier/gateway value** — credit a capture
for the new neutral frontier it unlocks as a launch base. Then the PI (rightly)
said: mine a real CPMP loss and verify before spending a ladder slot.

## The finding (a clean, useful negative)

Downloaded the real CPMP 2P losses (`kaggle competitions replay`), pulled seeds
from `info.seed`, ran the symmetric producer_plus mirror on them.

On the canonical corner-neglect seed **641308308** the mirror reproduces the
failure exactly: base leaves two garrison-41, prod-4 corners (ids 5,6, dist 63)
neutral for the **entire 500-step game**. Four-way mirror comparison:

| agent | corners 5,6 captured? |
|---|---|
| base | never |
| **frontier alone** | **never — identical to base** |
| wideshortlist (generation) | yes, ~step 499 |
| **frontier + wideshortlist** | yes, **step 95** |

**Frontier alone does not fix corner-neglect.** Why: corner-neglect is a
candidate-**generation** truncation (the far corner is outside the nearest-K
shortlist → never a candidate). `frontier_bonus` is a candidate-**scoring**
term — it can only boost candidates that already exist. A scoring term cannot
surface a target generation never proposes. **Layer mismatch.**

The constructive half: once generation surfaces the corner (wideshortlist),
frontier **accelerates its capture from ~step 499 to step 95**. Frontier is a
**multiplier on a generation fix**, not a standalone fix.

## The deeper lesson (worth keeping)

A scoring/value term and a generation/shortlist term live at different layers.
When a target is *neglected* (never acted on), ask first: is it **not a
candidate** (generation) or **a low-scoring candidate** (scoring)? Our two
documented loss drivers are mostly the former or neither:
- **Under-expansion / far-economic-prize neglect** (#1, ~76%): the prize is a
  far high-garrison neutral *outside the shortlist*. Fix = **generation**
  (wideshortlist) + value its long-run production (**horizon/opening**). That's
  the `expand` variant already on the ladder. Frontier is orthogonal here —
  these corners are extreme (unlock little), so even their gateway value is low;
  what they need is to be *seen* and their own production valued.
- **Collapse** (#2): peak then lose everything (seed 1506374610: both reach 8
  then die). That's **tenure/durability** — the proposed *second* term — not
  frontier and not generation.

So frontier targets a third phenomenon — **mid-board gateways that are
candidates but under-valued** — which we have **not yet confirmed exists** in
our real losses. Frontier may be a solution looking for a problem; or its real
home is *composed* with generation (`expand + frontier`), accelerating capture
of surfaced far targets. That composition is the next thing to test.

## Status of the code
- `frontier_bonus` is committed (correct, 9/9 unit, default-OFF byte-identical,
  144 ms max turn, gated). Harmless when off. NOT submitted.
- Spec + full reproduction verdict: `audit/2026-06-15-frontier-gateway-value-spec.md`.

## Open questions
- Does `expand + frontier` beat `expand` alone on the ladder (n≥32)? (Does
  frontier add lift *on top of* the generation fix?)
- Is there a real loss exhibiting mid-board gateway neglect (candidate but
  under-valued), the phenomenon frontier is actually built for?
- The collapse/tenure term (#2 driver) is unbuilt and is what 1506374610 (and
  the back half of 1692894782) actually need.

## Method note (reusable)
The mirror + replay-seed loop worked cleanly: `kaggle competitions replay <ep>`
→ `info.seed` → symmetric mirror on the seed → measure the specific neglected
planets at steps 60/95/final. `/tmp/repro_mirror.py` lists the exact neglected
high-value far neutrals (id/pos/garrison) so "is the failure reproduced / fixed"
is a concrete check, not a vibe. Flags are global env → only symmetric mirror
in-process; cross-config comparison is separate-process.
