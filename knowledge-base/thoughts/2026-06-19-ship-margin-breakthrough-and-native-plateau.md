# Ship-margin was the breakthrough; native still plateaus below the bolt-on

2026-06-19, branch claude/dropout-plan-review-rb5817.

## The one insight worth keeping

The dropout-NATIVE forward model optimized **production-weighted ownership** —
`Σ prod·(P_mine − P_opp)`. That is a *proxy*. The engine decides the winner by
**total ships** (planets + fleets). In the churn regime those diverge: the agent
out-produced V2 on planets while bleeding ~12,760 ships/game into thin captures
that reflip — it led on the proxy and lost the real game. Reformulating the value
to **expected ship-margin** (weight ownership by ship count + a post-horizon
production credit, add in-flight mass, instantaneous leak, discounted mean) took
native from 0/40 → 13/40 and eliminated the churn. *Optimize the scored quantity,
not a correlate of it.* This is the transferable lesson.

## Why dropout alone couldn't fix the churn

The flip hazard refines *which planets you keep* — but it values ownership, and
charges nothing for the ships a churning capture consumes/transfers to the
opponent on a reflip. The producer's own `competitive_score` already counts net
ships (`produced − lost_to_combat`); replacing it with pure ownership-probability
dropped that ship-economy term. Dropout was sharpening the wrong objective.

## The plateau (honest)

After ship-margin (13/40), FOUR closing-levers all failed to reach base's 21/40:
wide neutral shortlist (no change), λ production-credit sweep (12 optimal, higher
hurts), force-concentration (worse), anticipatory threat growth (α=0.25 EXACT
parity Δ=0.000, higher over-suppresses). Consistent message: the remaining ~8-win
gap is **structural** — multi-ply sequencing and the calibrated coalition/sync
machinery the mature bolt-on has — not a single A/B-gated knob. A one-ply
mean-field hazard value, even correctly aligned to ships, sits below a tuned
multi-mechanism scorer. Banked native at 13/40; base stays agent of record.

## Process scar (see postmortem)

The native scorer threw on 100% of turns for the first half of the session
(silent `except: pass`), so the original "hazard inert / refuted" verdict measured
the static fallback. Caught only by a code review. Lesson promoted to the
postmortem: verify a gated scorer actually executes (strict-raise / executed
count) before trusting its A/B; green unit tests are necessary but not sufficient.
