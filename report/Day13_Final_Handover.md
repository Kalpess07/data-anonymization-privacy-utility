# Day 13 — Level 1 Results and Team Handover

## 1. Work Completed

- Prepared the data-cleaning, EDA, preprocessing, and baseline modeling notebooks.
- Trained Logistic Regression and Random Forest baseline models.
- Tested the ML preprocessing pipeline on Dhiraj's Level 1 anonymized dataset.
- Trained and evaluated both models on Level 1 data.
- Created a baseline-versus-Level 1 model comparison.
- Reviewed the available k-anonymity evaluation results.

## 2. Model Utility Findings

The saved results show broadly similar baseline and Level 1 model performance.

- Logistic Regression accuracy: 85.51% baseline and 85.55% Level 1.
- Random Forest accuracy: 86.07% baseline and 86.15% Level 1.
- F1-scores also increased slightly in the recorded Level 1 results.

These are descriptive comparisons. Confirm consistent splits, preprocessing, and evaluation settings before drawing stronger conclusions.

## 3. Privacy Evaluation Findings

The available privacy metrics evaluate `age` and `zip_code` as quasi-identifiers.

- Total records: 32,561.
- Equivalence classes reported: 5,647.
- Minimum achieved k: 1.
- For target k=3, 2,170 records (6.66%) are reported in violating classes.
- For target k=5, 5,548 records (17.04%) are reported in violating classes.
- The k-anonymity criterion is reported as not achieved for either threshold.

The current Level 1 transformation therefore does not satisfy the tested k-anonymity criteria. Further generalization or suppression may be needed, followed by a fresh privacy and utility evaluation.

## 4. Main Deliverables

- `notebooks/01_data_cleaning.ipynb`
- `notebooks/02_eda.ipynb`
- `notebooks/04_baseline_models.ipynb`
- `notebooks/08_level1_encoding.ipynb`
- `notebooks/09_test_level1_pipeline.ipynb`
- `notebooks/10_level1_logistic_regression.ipynb`
- `notebooks/11_level1_random_forest.ipynb`
- `notebooks/12_baseline_vs_level1.ipynb`
- `results/final_experiment_summary.csv`
- `results/baseline_vs_level1_comparison.csv`
- `results/privacy_metrics_level1_cleaned.csv`

## 5. Handover and Next Steps

1. Share the results and findings with Kalpesh and the project team.
2. Coordinate with Dhiraj on improving generalization or suppression to satisfy the intended k-anonymity criterion.
3. Re-evaluate privacy metrics after any transformation changes.
4. Retrain both ML models with consistent experimental settings.
5. Compare privacy improvement against any change in model utility.

## Conclusion

The Level 1 dataset supported successful ML training with broadly similar recorded utility metrics. However, the available privacy evaluation indicates that the tested k-anonymity criteria were not achieved. Further anonymization and re-evaluation are required before claiming that Level 1 meets the intended privacy requirements.
