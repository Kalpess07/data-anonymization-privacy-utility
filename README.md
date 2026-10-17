Hide Personal Data and Still Use It
An Empirical Research Investigation on Data Anonymization Techniques: Evaluating the Privacy-Utility Trade-off in ML Models
Internship Program | Problem 18 | 9 September 2026 – 9 December 2026
---
1. Project Overview
Data-driven work in sensitive domains such as healthcare, finance, and government must balance analytical usefulness with individual privacy.
This project investigates anonymization techniques such as generalization, suppression, masking, and k-anonymity at increasing levels. Model performance on original and anonymized data is compared to study the privacy-utility trade-off.
The goal is to understand how privacy-oriented transformations affect machine-learning utility and what re-identification risks may remain.
2. Objectives
Apply generalization, suppression, and masking to selected dataset fields.
Understand and implement basic k-anonymity.
Train the same machine-learning models on original and anonymized data and compare results.
Compare low, medium, and high anonymization levels when those experiments are implemented.
Examine privacy-utility trade-offs and limitations of anonymization.
Prepare a research report and position paper on responsible data practices.
Develop a Streamlit application for anonymizing CSV files.
3. Data Policy
Use publicly available datasets and document their source.
Do not use fabricated or scraped data for the experiments.
Do not commit large raw data files to the repository.
Document dataset download instructions and place local raw datasets under `data/raw/` when appropriate.
Check that any data shared publicly is permitted for the intended use.
4. Repository Structure
```text
.
├── data/
├── figures/
├── notebooks/
├── report/
├── results/
├── src/
├── tests/
├── CONTRIBUTING.md
├── README.md
└── requirements.txt
```
The folders are intended for datasets, generated figures, analysis notebooks, reports, experiment outputs, source code, and tests respectively.
5. Workflow
Inspect and clean the dataset.
Perform exploratory data analysis (EDA).
Prepare preprocessing pipelines.
Train and evaluate baseline models.
Apply an anonymization level and document its transformations.
Retrain/evaluate models on the transformed data using comparable evaluation conditions.
Compare privacy-related measures and model utility.
Document limitations, results, and reproducibility details.
6. Current Repository Contents
The repository currently contains notebooks for data cleaning, EDA, preprocessing, baseline modelling, model tuning, evaluation, feature importance, Level 1 encoding, Level 1 model evaluation, and baseline-versus-Level 1 comparison. CSV result files are stored under `results/`.
This section describes files present in the repository; it does not by itself guarantee that every experiment has been independently rerun or validated. See the notebooks and result CSVs for the actual recorded outputs.
7. Environment Setup
Use a supported Python environment. From the repository root, install the listed dependencies with:
```bash
python -m pip install -r requirements.txt
```
To use the notebooks, start Jupyter from the environment in which the dependencies are installed:
```bash
jupyter notebook
```
Run notebooks in the intended order and check that input data paths match your local setup. Dataset files may not be included in the repository.
8. Reproducibility and Reporting
Record the dataset source, preprocessing, train/test split, random seeds, model settings, and evaluation metrics.
Use equivalent evaluation conditions when comparing baseline and anonymized data.
Preserve actual experiment outputs and avoid filling missing results with estimates.
Clearly label limitations and distinguish observed results from interpretations.
Document any required dataset files that are not committed to GitHub.
9. Privacy and Limitations
Anonymization techniques can reduce some disclosure risks but do not guarantee that re-identification is impossible. The privacy implications depend on the data, transformation rules, available auxiliary information, and threat model. Report the chosen quasi-identifiers and assumptions explicitly.
10. Contributions
Please see CONTRIBUTING.md for repository contribution and collaboration guidelines.

## 11. Reproducing the Level 1 Encoding Notebook

The notebook `notebooks/08_level1_encoding.ipynb` applies the project's Level 1 transformations to the cleaned Adult dataset. Before running it, confirm that the repository has this local input file:

```text
data/cleaned/adult_cleaned.csv
```

The dataset may be excluded from GitHub, so obtain it from the project's documented dataset source and run the project's data-cleaning workflow first if the cleaned file is not available. Do not substitute a different dataset without recording that change.

From the repository root, create and activate your Python environment, then install the dependencies:

```bash
python -m pip install -r requirements.txt
jupyter notebook
```

In Jupyter, open `notebooks/08_level1_encoding.ipynb` and run its cells from top to bottom:

1. **Load the cleaned dataset:** reads `data/cleaned/adult_cleaned.csv`, treating `zip_code` as text so leading zeros are preserved.
2. **Test the transformation functions:** imports `generalize_age`, `mask_zip_code`, and `prepare_level1_features` from `src/level1_encoding.py` and applies the feature-preparation function to a small example.
3. **Apply Level 1 encoding:** transforms the full dataframe and displays the first ten rows of the `age` and `zip_code` fields.

The notebook is a transformation demonstration. It does not, by itself, retrain models or produce a complete privacy-utility comparison. Use the separate model-evaluation and baseline-versus-Level 1 comparison notebooks for those analyses, and confirm their required input CSV files and settings before running them.

### Troubleshooting

- **`FileNotFoundError`:** check that `data/cleaned/adult_cleaned.csv` exists and that Jupyter was launched from the repository root or the `notebooks/` directory.
- **`ModuleNotFoundError: src...`:** check that `src/level1_encoding.py` exists and that you opened the notebook from the repository's `notebooks/` folder or root.
- **Missing package:** activate the same Python environment used by Jupyter and run `python -m pip install -r requirements.txt` again.
- **Different results:** record the dataset version, preprocessing steps, split settings, random seeds, and model configuration before comparing metrics.

Do not describe results as reproduced until the notebook has been run successfully and its outputs have been checked.
