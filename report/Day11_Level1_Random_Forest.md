# Day 11 — Random Forest on Level 1 Anonymized Data

## Objective
Train and evaluate Random Forest on Dhiraj's Level 1 anonymized dataset using the agreed baseline model settings.

## Experimental Setup
- Dataset: Level 1 anonymized Adult dataset
- Train/test split: 80:20, stratified
- Random state: 42
- Random Forest: 100 estimators, random_state=42
- Training rows: 26,048
- Test rows: 6,513
- Processed features: 1,013

## Results

| Metric | Random Forest |
|---|---:|
| Accuracy | 0.861508 |
| Precision | 0.760972 |
| Recall | 0.619260 |
| F1-score | 0.682841 |

## Conclusion
Random Forest trained successfully on the Level 1 anonymized dataset. It achieved an accuracy of approximately 86.15% and an F1-score of approximately 68.28%.

These results provide the Level 1 Random Forest metrics for comparison with the baseline model. They do not independently establish the privacy benefit of anonymization.

## Next Step
Compare baseline and Level 1 metrics using consistent data splits, preprocessing, and evaluation settings.
