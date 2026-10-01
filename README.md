# Hide Personal Data and Still Use It

**An Empirical Research Investigation on Data Anonymization Techniques: Evaluating the Privacy-Utility Trade-off in ML Models**

**Internship Program | Problem 18 | 9 September 2026 – 9 December 2026**

## 1. Project Overview

Data-driven work in sensitive domains such as healthcare, finance and government has to balance two needs: individual-level data is analytically valuable, but personal privacy must be protected, both ethically and legally.

This project takes a real, publicly available dataset, applies anonymization techniques (generalization, suppression, masking and k-anonymity) at three increasing levels, and measures empirically what each level costs in model performance. The aim is to understand and defend the **privacy-utility trade-off**.

## 2. Objectives

1. Apply anonymization techniques (generalization and suppression) to sensitive fields of a real dataset.
2. Understand and apply k-anonymity at a basic level.
3. Train the same ML models on the original and the anonymized data, and compare the results.
4. Compare multiple anonymization levels (Low, Medium, High) side by side.
5. Articulate the privacy-utility trade-off and the re-identification risks that anonymization does not fully remove.
6. Deliver a research report, a position paper on responsible data practices, and a Streamlit tool for anonymizing CSV files.

## 3. Data Policy

Only publicly available datasets are used. No synthetic, fabricated or scraped data is used. Large raw data files are not committed to this repository; see `data/README.md` or the source link above to download them.

## 4. Methodology

| Stage | Description |
|---|---|
| **Baseline** | EDA, then Logistic Regression and Random Forest trained on the original data. Accuracy is logged as the reference. |
| **Level 1 (Low)** | Name removal, basic masking, age bucketing, zip-code masking. |
| **Level 2 (Medium)** | Broader age bins, partial zip suppression, k-anonymity (k = 3 and 5) through generalization plus suppression. |
| **Level 3 (High)** | Aggressive generalization and suppression, testing where the model breaks (over-anonymization check). |
| **Comparison** | The same models are retrained at every level. Accuracy and other metrics are compared against the baseline. |

## 5. References

Add as the team uses them: k-anonymity literature, the Netflix Prize re-identification paper, dataset source, library documentation (pandas, scikit-learn, Streamlit).

## 6. License

To be decided by the team (for example MIT).
