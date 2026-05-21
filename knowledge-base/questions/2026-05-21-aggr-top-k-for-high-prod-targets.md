# 2026-05-21 — question: should AGGR TOP_K=5 lift for high-prod enemy targets?

PI image (step 122, 2P, sub 52882014) shows our 319-ship planet
sitting idle adjacent to combat, while we attack a +5 enemy planet
from elsewhere. Hypothesis: chooser's JOINT enumeration considers
only the TOP_K=5 cheapest-Δ sources per target; the 319-ship planet
isn't in that top-5 (or it's behind some filter). Drain mechanisms
shipped today don't address this — they're post-pass, after the
chooser's source-selection has already happened.

**Question:** does lifting `JOINT_TOP_K` from 5 to (say) 8 when the
target has production ≥ +4 net-positive in 4P? It'd let the
multi-source enumeration include more sources, potentially catching
the "obvious-big-source-but-unranked" case.

**Risk:** more JOINT pairs to score → wallclock pressure. AGGR
already enumerates 60 max pairs; doubling top-K doubles candidate
volume.
