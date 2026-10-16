# Day 12 — Baseline vs Level 1 Comparison

## Objective
Compare baseline model performance against performance on the Level 1 anonymized dataset.

## Method
- Models: Logistic Regression and Random Forest.
- Evaluation metrics: accuracy, precision, recall, and F1-score.
- Level 1 transformations: age generalization and ZIP-code masking.
- Comparison source: saved baseline and Level 1 experiment metrics.

## Results
The comparison table is saved in `results/baseline_vs_level1_comparison.csv`.

The charts are saved in the `figures/` directory.

## Findings
Use the calculated metric differences to describe which metrics increased or decreased for each model. Report changes in percentage points and avoid claiming that a difference is statistically significant without an appropriate test.

## Privacy–Utility Interpretation
This comparison measures model utility changes. It does not, by itself, demonstrate the magnitude of privacy improvement. Privacy properties require a separate evaluation of the anonymization method.

## Next Steps
Review the results with the team and prepare the next experiment according to the project schedule.
