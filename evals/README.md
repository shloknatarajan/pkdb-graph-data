# `evals` — testing models against the PK-DB graph benchmark

Runs a model over `benchmark/benchmark.jsonl` (figure image + paper text) and scores
the timecourses it recovers against PK-DB's curator digitizations.

## Reusable single-file extractor

The complete dataset-independent implementation lives in
[`timecourse_extractor.py`](timecourse_extractor.py): model defaults, provider
selection, prompt, JSON schema, result format, image handling, and API calls. It
returns visible series with panel identity, the verbatim plotted label, normalized
metadata, points, API usage, and consistency warnings:

```python
from evals import TimecourseExtractor

extract = TimecourseExtractor(
    provider="openai",       # "openai", "gemini", or "anthropic"
    model="gpt-5.6-sol",     # optional; each provider has a default in the file
)
result = extract(
    "path/to/figure.png",
    paper_text="optional caption, methods, or full paper",
    paper_title="optional title",
    substance_vocabulary=["optional", "canonical", "names"],
)

print(result.curve_count, result.timecourses, result.warnings)
```

Equivalent model selections are:

```python
TimecourseExtractor(provider="gemini", model="gemini-3.6-flash")
TimecourseExtractor(provider="anthropic", model="claude-opus-5")
```

The core extractor has no PK-DB codes or annotation rules. Its prompt inventories
panels/legends/styles before digitizing, treats visible evidence as authoritative,
preserves the plotted label separately from normalized substance identity, refuses
to infer regular sampling schedules, and returns human-readable interventions.

`extractor.py` and `prompt.py` are retained only as compatibility/benchmark adapter
modules; the reusable extraction behavior is defined in `timecourse_extractor.py`.

```bash
# latest flagship GPT model (requires OPENAI_API_KEY)
uv run python -m evals.run run --backend openai --workers 2

# end-to-end: predict then score
uv run python -m evals.run run --backend claude-cli --model sonnet --limit 12

# separately
uv run python -m evals.run predict --backend claude-cli --model opus --out predictions/opus.jsonl
uv run python -m evals.run score   --predictions predictions/opus.jsonl --out reports/opus.json
```

## Backends

| `--backend` | What it does |
|---|---|
| `openai` | Calls the Responses API with the figure at `original` detail and strict structured output. Needs `OPENAI_API_KEY`. Defaults to `gpt-5.6-sol` with high reasoning effort; use `--model` to pin another model or snapshot. |
| `gemini` | Calls Google's Interactions API with an inline figure and structured JSON output. Needs `GEMINI_API_KEY`. Defaults to `gemini-3.6-flash` with high thinking effort. |
| `claude-cli` | Shells out to the authenticated `claude` CLI headlessly. Works wherever Claude Code does — no API key needed. `--model sonnet\|opus\|haiku`. |
| `anthropic` | Calls the Messages API directly (base64 image block + schema-constrained JSON). Needs `ANTHROPIC_API_KEY` or an `ant auth login` profile. Defaults to `claude-opus-4-8`. |
| `oracle` | Returns the gold answer. Must score exactly 1.0 — this is the scorer's regression test. |
| `noisy-oracle` | Gold answer with multiplicative noise, for checking the metrics degrade smoothly. |

Useful flags: `--limit N`, `--ids ID [ID ...]`, `--workers N`, `--paper-chars N` (`-1`
for the whole paper), `--substance-hint` (supply the study's substance vocabulary),
`--rel-tol` (the recovery threshold, default 0.20).

For a cheap smoke test before running all 38 scoreable figures:

```bash
export OPENAI_API_KEY=...
uv run python -m evals.run run --backend openai --limit 2 --workers 1
```

Predictions are saved under `predictions/`. Pass `--out reports/gpt-5.6-sol.json`
to `score` when you want to retain the full per-figure and per-curve report.

## How scoring works

A model returns an *unordered set* of curves, so curves must be matched to gold
before anything numeric can be compared.

1. **Match.** Linear-sum assignment (`scipy`) over a cost blending metadata agreement
   with a *scale-free* shape distance. Scale-free matters: a curve reported in `ng/ml`
   still matches its `microg/ml` gold counterpart, and the unit error is then charged
   during scoring rather than corrupting the assignment.

2. **Align.** Gold points are grouped by time. A group of size one is a sample of a
   function, so a model that traced the same curve with fewer markers is scored by
   interpolating its curve. A group of size > 1 is several subjects measured at one
   time — there is no function to interpolate, so the value multisets are paired in
   sorted order. Extrapolation is refused; a model that only digitized the first two
   hours loses *coverage* rather than gaining free credit on the tail.

3. **Compare.** A predicted curve earns credit only if its unit converts into the gold
   unit, it covers ≥ 50% of the gold time range, and its median relative error is
   within `--rel-tol`. Numeric accuracy is reported both as `median_rel_err` and as
   `median_log10_err`, because these curves span decades on a log axis.

**Headline metric:** `micro_f1` — recovered curves against predicted (precision) and
against gold (recall), pooled over all curves. `macro_f1` averages per figure, so it
weights a 2-curve figure the same as a 20-curve one.

### Reading the metadata numbers

`substance`, `tissue` and `measurement_type` come from a controlled vocabulary, so
they are scored by exact match. `intervention` and `group` are free-text curator
codes and are scored as fuzzy token sets.

**`intervention` is close to ungradeable and should not be read as a model failure.**
Gold values are PK-DB-internal dose codes — `COD`, `M6G_200, SUL, APAP`, `Dcaf`,
`apap_20mgkg_po` — that are not derivable from the figure or the paper. A model that
writes `propranolol` where gold says `paracetamol,propanolol` (sic) scores low for
reasons that have nothing to do with reading the plot.

## Known data caveats

- `PKDB00063__Fig1` has 10 gold timecourses carrying times but no values, units, or
  substances. It has no answer and is excluded (`benchmark.SKIPPED_EMPTY_GOLD`),
  leaving **38 scoreable figures / 204 curves** of the 39 / 216 advertised.
- Two further curves elsewhere carry times but no values and are dropped individually.
- Nine gold curves repeat time points (several subjects sampled at one time). This is
  why alignment cannot simply interpolate — `np.interp` silently returns garbage on a
  non-monotonic grid.

## Results so far

Both runs used `claude-cli` with truncated paper text (40k chars) and no substance hint.

**Sonnet, 12 figures** ($7.30, 34 model-minutes). The aggregate number hides a hard split:

| figure class | figs | gold curves | micro F1 | median rel err | points within 20% |
|---|---|---|---|---|---|
| group-mean curves | 10 | 43 | **0.66** | **16%** | 58% |
| per-subject curves (`S1_…`) | 2 | 36 | 0.06 | 77% | 16% |

**Sonnet vs Opus**, on the same 6 group-mean figures / 27 curves:

| | sonnet | opus |
|---|---|---|
| micro F1 | 0.56 | **0.78** |
| median rel err | 18.4% | **9.7%** |
| points within 20% | 52% | **65%** |
| unit error rate | 0% | 0% |
| substance accuracy | 91% | 92% |
| cost | $3.65 | **$2.41** |

Findings:

- **Curve counting and unit reading are close to solved.** 83% of figures get the curve
  count exactly right (mean |count error| 0.33), and neither model made a single unit
  error once `µg ml-1` and `mg/h` were parsed correctly.
- **Numeric precision is the bottleneck**, and it responds strongly to model capability:
  Opus roughly halves the median relative error and lands within 10% on nearly half of
  all points. At `--rel-tol 0.20` that is F1 0.78.
- **Dense per-subject figures fail outright.** On the 16- and 20-curve plots the model
  emits the right *number* of curves but idealizes the sampling schedule onto a regular
  grid (`0, 0.25, 0.5, 1, 2, 3`) instead of reading actual marker positions, and the
  per-subject identities are not recoverable from the plot anyway.
- Log-scale axes are the main source of residual error on group-mean figures
  (`PKDB00024__Fig1`: sonnet 38% error, opus 5%).
