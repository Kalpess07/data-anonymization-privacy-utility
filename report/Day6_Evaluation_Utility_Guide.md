# Day 6: Evaluation Utility Development

## Objective
Develop a reusable evaluation utility for measuring machine-learning classification performance across baseline and anonymized datasets.

## Metrics
- Accuracy
- Precision for the `>50K` income class
- Recall for the `>50K` income class
- F1-score for the `>50K` income class

## Files
- `src/evaluation.py`: reusable evaluation and logging functions
- `results/results_log.csv`: cumulative experiment results
- `notebooks/06_evaluation_utility.ipynb`: testing notebook

## Usage
The `evaluate_model()` function accepts the true labels, predicted labels, model name, experiment identifier, anonymization level, and positive class label.

The `append_result()` function stores each experiment's metrics in the cumulative results CSV.

The `print_evaluation_report()` function displays class-wise metrics and a confusion matrix.

## Reproducibility
All model experiments should use a consistent train/test split and documented model parameters. The test set should not be used to fit preprocessing or train models.

## Limitations
Evaluation metrics measure predictive utility, not privacy directly. Privacy risk must be evaluated separately using the project's anonymization and re-identification analysis.
