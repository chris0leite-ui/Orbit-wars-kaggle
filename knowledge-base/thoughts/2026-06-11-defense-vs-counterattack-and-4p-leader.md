# PI observation — defense vs asymmetric counter-attack; 4P leader avoidance
(2026-06-11, transcribed from chat)

Two observations, PI verbatim in substance:

1. "I wonder if defense might even be the right answer at all times or if
   it rather be an asymmetric attack. I think I have observed that."
   — i.e. when we get attacked, routing ships home to defend may be worse
   than counter-attacking somewhere else (trading planets instead of
   parrying). PI believes they have seen the trade play out favorably.

2. "In 4-player games we seem not to attack the strongest opponent."
   — despite the strength-weighted free-for-all objective (leader focus)
   shipped in sub 53564198, observed behavior is that our waves go at the
   weaker rivals, not the leader.

Context links: same session as the holdability autopsy (rolling
recaptures, expansion-was-winning finding) — all three observations
circle the attack/defense exchange-rate question.

## Measurement (same day, 33 live 4-player games + 2-player set, both live subs)

**Leader targeting, refined.** We DO attack the strongest rival — 62% of
all enemy-directed waves, 684 of 1106. The real pattern hiding under the
impression: ship-share aimed at the leader is 67% in games we WIN but
52% in games we LOSE (when a rival outranks us). So it is not absence of
leader focus; it is leader focus DILUTING exactly when the leader
snowballs — consistent with feasibility, not valuation: the leader's
garrisons + reactive floors grow past what our waves can clear, the
leader candidates die in the floor/veto, and the ships go to whoever is
still attackable. Unproven; needs decision_trace on behind-ticks of a
live loss.

**Defense vs counter-attack.** Against inbound enemy waves >= 20 ships
(2,356 events): we already respond with counter-attack 65% of the time
(1,532), defend only 21% (484). Exchange outcome (our strength minus
attacker's, launch to arrival+5): defend +120 mean / 63% positive;
counter-attack +13 / 51%; expand-elsewhere -23; freeze -156. Heavy
selection bias (we defend when defense is winnable), so this does NOT
prove defense > counter-attack causally — but it shows no large untapped
counter-attack alpha in OUR behavior. The PI may have observed opponents
using asymmetric attack against US — need the specific replay.
