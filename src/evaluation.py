from pathlib import Path
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


def evaluate_model(
    y_true,
    y_pred,
    model_name,
    experiment="baseline",
    anonymization_level="none",
    positive_label=">50K",
):
    """Calculate classification metrics for a model experiment."""

    return {
        "experiment": experiment,
        "anonymization_level": anonymization_level,
        "model": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true, y_pred,
            pos_label=positive_label,
            zero_division=0,
        ),
        "recall": recall_score(
            y_true, y_pred,
            pos_label=positive_label,
            zero_division=0,
        ),
        "f1_score": f1_score(
            y_true, y_pred,
            pos_label=positive_label,
            zero_division=0,
        ),
    }


def append_result(result, csv_path):
    """Append one experiment result to a cumulative CSV log."""

    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    new_row = pd.DataFrame([result])

    if path.exists() and path.stat().st_size > 0:
        existing = pd.read_csv(path)

        # Keep the same schema across all experiment rows.
        columns = list(dict.fromkeys(
            list(existing.columns) + list(new_row.columns)
        ))

        existing = existing.reindex(columns=columns)
        new_row = new_row.reindex(columns=columns)

        combined = pd.concat(
            [existing, new_row],
            ignore_index=True,
        )
    else:
        combined = new_row

    combined.to_csv(path, index=False)


def print_evaluation_report(y_true, y_pred):
    """Print class-wise metrics and the confusion matrix."""

    print("Classification Report:")
    print(classification_report(
        y_true, y_pred, zero_division=0
    ))

    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))