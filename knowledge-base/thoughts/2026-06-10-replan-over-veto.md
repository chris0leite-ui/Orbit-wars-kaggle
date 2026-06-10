# Why the replan should eventually retire the veto (2026-06-10, night)

The response veto was the first mechanism that made the opponent model
load-bearing, and it worked live (touched 1309 within two hours). But its
shape is wrong in a way the PI's "model it correctly, don't restrict it"
rule predicts: it can only say NO. When the predicted reply kills a wave,
the right answer is almost never "do nothing with those ships" — it's
"capture the neutral one hop over", "reinforce the planet their counter
targets", or "send a bigger fleet that beats the parry" (the upsize tried
the last one in isolation and failed attribution — because in isolation it
could only re-ask the SAME question louder, not change the subject).

The one-ply replan asks the planner the whole question again under the new
information. Pass 2 sees the reply in every candidate's flow diff, in the
defensive shortlist, and in the roi normalization. If it works, veto-on-top
becomes a cheap verification pass (replan, then check pass 2 against a
fresh reply), and the natural next rung is iterating to a small fixed point
— plan/reply/plan/reply until stable or budget out — which is the honest
approximation of simultaneous-move equilibrium this game actually wants.

Risk to watch: oscillation. Pass 2 best-responds to the reply-to-pass-1,
but the live opponent reacts to pass 2, not pass 1. If pass 2 routinely
deviates far from pass 1 (e.g. drops the attack that provoked the parry,
then "defends" against a parry that will never come), margins will show it
— that's what the attribution leg vs the live stack is for.
