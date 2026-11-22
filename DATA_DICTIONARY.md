# Numerical data dictionary

## Shared identifiers and units

- `query_id`: opaque generated experiment key, not a graph node ID. `config`,
  `dataset`, and `corpus` identify a workload or fixed corpus.
- `seed`/`cluster`: query or graph seed block, depending on the stage. Rows are
  not independent timing replications for inferential purposes.
- `policy`/`chosen_policy`/`arm`: fixed execution policy or serving path.
- `wall_ms`: one recorded invocation in milliseconds; `cost_ms`: the median of
  the declared repetitions for the corresponding request/path.
- `repeat`, `position`, `order`, and `query_sequence`: recorded measurement
  order; they are preserved for balance and chronology checks.
- `*_us` denotes microseconds; 1 ms = 1,000 microseconds. Ratios are unitless.
- `recall`, exact-match and legality fields: recorded post-execution audits.
  Empty-reference conventions are retained; they do not imply perfect nonempty
  retrieval. Node IDs and vectors are not distributed.

## Retained stages (`analysis/retained/data`)

`costs.csv` and `costs_bcr.csv` contain long-form policy costs and static
`f_*` syntax/catalog covariates. `predictions.csv` contains frozen out-of-fold
policy choices and their measured costs. `folds.csv` records held-out grouping.
`router_timings.csv` and `native_overhead.csv` are measured inference/training
summaries; inference hot loops alone are not complete serving-path timings.
`summary.csv` is the original numerical reference used to verify exact tests.
The adjacent input manifest hashes every numeric snapshot.

## Leakage diagnostics (`analysis/leakage/data`)

The `posthoc_features_*.csv` caches contain measured result cardinality,
selectivity, endpoint fractions, or quality statistics by stage. They are
deliberately excluded from the admissible online feature set. The third mode
directly exposes target costs as an explicitly invalid control. The public
script preserves fold keys, depth 4, minimum leaf size 20, seed 1837, and the
original cost/penalty/uncertainty targets. Reference predictions enable exact
choice comparisons; source graph reconstruction is not claimed.

## Prospective records (`data/prospective`)

The plan is the original pre-outcome statistical plan. Each corpus has 640
queries, ten seed blocks, and eight serving paths with eight timed repetitions.
The gzip raw files retain timings, choices, order, answer size, and equality
hashes. `engine_session_ms` is an inner boundary; outer-minus-inner time is not
isolated causal inference overhead. Scalar `quality.json` excludes actual and
reference node lists. Minimal completion and plan metadata replace deployment
identifiers; the corresponding projection is recorded in provenance.

## Factorial records (`data/factorial`)

Sessions A and B contain the same five corpora and four order/prewarm cells.
Every cell has the two Eager/RACE policies with eight repetitions per query.
UTC measurement timestamps support chronology checks and do not identify a
machine or account. Scalar raw projections omit nested engine answer lists and
distances. `quality.csv` and schedule records retain query-level audit flags,
counts, and randomized ordering. Environment files explicitly mark omitted
machine/deployment identifiers rather than inventing a replacement environment.

## Censored scan (`data/scan`)

`observations.csv` retains all four policy attempts per query. A censored
`cost_ms` is missing, not zero. Watchdog/publication elapsed values are not
substituted for native costs. The reduced raw records retain only the status
and publication-race fields needed by the scan audit. The 725 all-complete
requests are a selected subset, not the full uncensored workload.

## Public-baseline records (`data/external`)

All three declared search budgets and Eager/RACE controls are retained on three
corpora. `bitmap_ms` and `search_ms` are per-call components; their separately
computed medians need not sum to the total-time median. `quality.csv` records
scalar recalls, underfill, and legality counts, excluding returned/reference
ID lists. Completion hashes refer explicitly to these public projections.
The raw/median relationship is verified; exact graph eligibility is not rerun.

## References and provenance

`reference/*/*.csv` are retained numerical outputs. The reproduction command
checks their dimensions, labels, and numerical values. Original historical
Figure 3 intervals are audit-only and are not the revised plotted intervals.
`MANIFEST_SHA256.json` covers the distributed input/code/reference files.
`provenance/transformations.json` distinguishes copies from field projections.
No absent record, timed-out cost, or unfavorable result is reconstructed or
replaced by a synthetic measurement.
