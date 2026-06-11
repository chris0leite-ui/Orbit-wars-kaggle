# 2026-06-11 — Oracle track open questions

1. **Does cloned top-ladder behavior transfer off-manifold?** The policy
   net is sharp on expert-distribution states; our agent will visit
   states the experts never created (especially when winning differently
   or losing unusually). How gracefully does it degrade, and does the
   exact-engine sizing + value veto contain the damage?
2. **Is the state-initiation head's calibration stable across board
   geometries?** Far-home boards may genuinely warrant later first
   strikes; check initiation timing by frontline-distance bucket against
   expert games.
3. **4P seat dynamics**: the policy is trained on a 2P+4P mixture with no
   seat-objective distinction; the engine's 4P reward is binary. Does the
   data carry enough 4P-specific caution (don't bleed into bystanders)?
4. **Self-play fine-tuning upside**: with the JAX batch interpreter and
   Kaggle GPU available, does PPO fine-tuning from the BC initialization
   buy a real lift within the remaining days, or is data-scale +
   architecture iteration on the BC stack the better spend?
5. **Replay freshness**: the meta shifts (the ladder population changes
   weekly). Should the dataset be re-scraped + retrained near deadline?
