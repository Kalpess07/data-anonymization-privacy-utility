# Day 9 — Level 1 Pipeline Testing

## Objective
Test whether the machine learning preprocessing pipeline can handle Dhiraj's Level 1 anonymized dataset.

## Dataset Validation
- Dataset size: 32,561 rows and 16 columns.
- Duplicate rows detected: 0.
- Age generalized into decade ranges.
- ZIP codes masked using the first three digits followed by `**`.
- Income target retained with no missing values.
- Missing values remain in workclass, occupation, and native_country; the preprocessing pipeline handles them using most-frequent imputation.

## Preprocessing Results
- Training rows: 26,048.
- Test rows: 6,513.
- Numeric input features: 5.
- Categorical input features: 9.
- Encoded features: 1,013.
- Preprocessing test: Passed.

## Model Results

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.8555 | 0.7303 | 0.6339 | 0.6787 |
| Random Forest | 0.8615 | 0.7610 | 0.6193 | 0.6828 |

## Conclusion
Both models trained successfully on the Level 1 anonymized dataset. Random Forest achieved the highest accuracy and F1-score, while Logistic Regression achieved the highest recall.

## Next Steps
Compare these results with baseline metrics using the same data split and evaluation settings. Investigate any preprocessing or schema mismatches before drawing conclusions about the privacy–utility trade-off.