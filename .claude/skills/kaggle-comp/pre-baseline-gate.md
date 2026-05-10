> **Orbit Wars note (code/agent comp).** The seven tabular items below
> are reference material. For Orbit Wars the gate reduces to:
> 1. **Read all comp rules verbatim** (Kaggle's eval logic, kernel
>    runtime cap, allowed external data, agent IO contract).
> 2. **Inventory the agent IO spec** — observation tensor shape, action
>    space, episode length, who-moves-when ordering.
> 3. **Watch ≥3 sample replays** end-to-end and write 1 sentence each
>    on what each player attempted.
> 4. **Inventory the baseline-opponent panel** Kaggle ships (random,
>    rules-based, last-year winners, etc.).
> 5. **Replicate the simplest reference notebook end-to-end**:
>    kernel-pull → run locally if `kaggle_environments` permits →
>    submit-format-check (dry-run with `kaggle_environments.evaluate()`
>    against the random baseline for ≥10 episodes). Do NOT push to
>    Kaggle yet — this is the gate, not the first submission.

# Pre-baseline understanding gate

Mandatory before the first `baseline_lgbm` smoke run. PI must
sign off on all seven items below.

**Use dedicated research agents in parallel** for the web /
research items (1, 5, 6, 7) — single message, foreground, three
`general-purpose` subagents. Doing this in the main thread bloats
context and is the documented failure mode (kickoff-#2 friction,
2026-05-04, `playground-series-s6e5`). Local Bash for the schema
items (2, 3, 4).

## Why this gate exists

Without it the agent autopilots into a baseline on default columns
with zero domain understanding, no group-CV check, no published
prior art, and no metric-specific tactics. A "smoke run" framed as
a time probe quietly becomes the first calibration anchor and
locks in bad assumptions.

## The seven items

### Item 1 — host brief verbatim *(web-research agent)*

WebFetch `https://www.kaggle.com/competitions/<slug>`. Extract
**verbatim** (exact host wording — DO NOT summarise) the
Description / Overview, Evaluation, Data, Rules tabs and any
pinned host posts from Discussion. Write to `brief.md` using the
section headers from `templates/brief-template.md`. Cap at 150
lines.

Fallback if WebFetch is gated: `kaggle competitions view <slug>`
and per-tab scraping with the kaggle CLI.

### Item 2 — full schema dump *(local Bash)*

```python
df = pd.read_csv("data/train.csv")
schema = pd.DataFrame({
  "dtype": df.dtypes,
  "n_unique": df.nunique(),
  "n_null": df.isna().sum(),
  "top3": [df[c].value_counts().head(3).to_dict() for c in df.columns],
})
schema.to_markdown("audit/schema-<date>.md")
```

Same for `data/test.csv`. Diff the two — flag any column missing
from test (potential leakage source).

### Item 3 — per-feature target-rate *(local Bash)*

For each top-10 numeric: bin into 20 quantiles, compute target
rate per bin, plot. For each categorical: target rate per level.
Embed in `plots/eda/report.html`. Extends `scripts/eda.py`.

### Item 4 — GroupKFold candidate keys *(local Bash)*

Identify columns that look like a group key (race-id, driver-id,
sequence-id, anything with cardinality ≪ N_rows but ≫ 10).
Document at least one candidate; row-count distribution per
group; whether the candidate appears in test (if not, can't fold
on it without aggregating).

### Item 5 — top-3 public notebooks *(web-research agent)*

Same agent as item 1. WebFetch top-3 most-upvoted notebooks for
the slug. For each capture: title, author, votes, CV scheme
(stratified? group? group key?), feature engineering, model
class, reported OOF / LB, any leakage / group-CV warning. Output a
`prior_art:` YAML block for `comp-context.md`.

Fallback: `kaggle kernels list -c <slug> --sort-by voteCount`
+ `kaggle kernels pull <ref>`.

### Item 6 — domain-knowledge paragraph *(domain-research agent)*

WebSearch real-world decision drivers behind the target. Cite
≥3 sources (Wikipedia, domain press, academic papers, prior-comp
writeups). Output an 8-line `domain_notes:` paragraph plus a
column-to-driver mapping table for `comp-context.md`.

### Item 7 — metric-specific notes *(metric-research agent)*

WebSearch best practices for the comp's metric at this imbalance
/ row count / group structure. Six bullets:

1. Imbalance handling — does class-weighting help this metric?
2. Calibration plan — does the metric require it? When?
3. CV scheme — stratified vs group, with citation.
4. Blend topology — what historically wins this metric on
   Playground.
5. Metric-specific gotchas (e.g. AUC: probability monotonicity,
   public-private split sensitivity).
6. Best public-LB tactics at this scale (early stopping, depth,
   learning rate).

Cite ≥4 sources. Output as a `metric_notes:` YAML block.

## Agent dispatch pattern

One message, three parallel `Agent` calls (subagent_type:
`general-purpose`). Each agent writes its detailed artifact to
disk and returns a ≤200-word summary so the main thread doesn't
inhale 10k tokens of research output.

```
Agent(web-research)    → writes brief.md, returns prior_art YAML
Agent(domain-research) → returns domain_notes YAML
Agent(metric-research) → returns metric_notes YAML
```

The main thread merges the three YAML blocks into `comp-context.md`
under new top-level keys. Schema / target-rate / group-keys are
local Bash, written to `audit/schema-<date>.md` and
`audit/<date>-pre-baseline-gate.md`.

## Gate-exit chat (Q5b)

> "Pre-baseline gate done: brief / prior_art / domain_notes /
> metric_notes / schema / per-feature target-rate / group-keys.
> Posting agent summaries. Cleared? [yes / fix what / show me X]"

PI must explicitly say "cleared". Then proceed to Bash batch D.

## Friction handling

Any agent that returns thin / unverifiable findings: log to
`audit/friction.md` with `tag: research-thin`. Re-spawn with a
sharper prompt or surface the gap to PI. Do not let a stub
finding count as a cleared item.
