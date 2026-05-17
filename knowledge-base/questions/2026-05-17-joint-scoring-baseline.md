# Open question — Direction B joint-scoring baseline (2026-05-17)

When scoring joint candidates (k-tuples of launches), what's the
right baseline?

Three options, each with a different bias:

1. **Same idle baseline as singles.** Each joint candidate's Δ =
   `favor(leaf_with_all_launches) − baseline_favors[h]`. Bias:
   joints systematically score higher than singles in expectation
   (more action = more board flux). Direct comparison singles-vs-
   joints is uninterpretable.

2. **Best-single baseline per joint.** Δ_joint = `favor(leaf_joint) −
   max(favor(leaf_single_i))`. Measures marginal lift of joint
   over any single constituent. Cleaner for ablation but more
   expensive (k+1 fast_sim runs per joint).

3. **Sum-of-singles baseline.** Δ_joint = `favor(leaf_joint) −
   sum(Δ_single_i)`. Measures the INTERACTION term: how much extra
   does the joint produce beyond the sum of its parts. Most
   theoretically clean — captures only synergy/antagonism.

Pre-implementation choice: probably (2) or (3). Pick (2) for first
cut (matches how v15's chooser already scores against idle baseline;
just changes the comparison floor per candidate). Switch to (3) if
results suggest synergy detection is the load-bearing signal.

PI to weigh in before implementation.
