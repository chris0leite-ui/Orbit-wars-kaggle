# 2026-06-10 — PI directive: forget everything, rebuild from first principles

PI (verbatim intent): "Forget everything we have done so far. What would
be the simplest, yet feasible most powerful solution to this game? Think
it through and implement it. You have all night, iterate as fast as you
can." Mid-night tempo correction: "Iterate quickly, a win 4 seeds clearly
then move on."

## What the night taught us (plain English)

1. **The game is an economy, not a battle.** Every fight destroys equal
   ships on both sides, so the only currencies are production-time and
   the garrisons you pay for neutral planets. We verified this in real
   games: the final ship gap equals the production-integral gap almost
   exactly.
2. **Exactness is buyable and cheap.** A few hundred lines predict the
   engine perfectly (proved by test). Every prior agent here ran on
   approximations of things that are simply knowable.
3. **Passive defense cannot work.** Every attempt to hold bigger
   garrisons made things worse. What works: pricing the enemy's feasible
   response into the value of each attack, and just-in-time rescue
   against fleets that are actually in the air.
4. **Liquidity is a war resource.** Ships in flight cannot turn around.
   The cost of committing them scales with how close the war is — but
   only when you are ahead on production; when behind, the bank must be
   spent (it is the only thing not producing).
5. **Analytic pricing proposes, simulation disposes.** The hand-built
   value model kept being wrong in one direction or another; a tiny
   event-driven simulation with a crude reactive opponent, used only as
   a veto between variants of the same plan, differenced out its own
   model error and was the breakthrough.

## Result

One night, one file: beats the production baseline 28/32 and sweeps the
public "Producer" (our live champion's nemesis) 32/32 locally. Ready to
submit pending PI sign-off.
