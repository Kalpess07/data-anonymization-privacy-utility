
"""Utilities for measuring and validating k-anonymity."""

from pathlib import Path

import pandas as pd

from src.level1_encoding import prepare_level1_features


def equivalence_class_sizes(
    df: pd.DataFrame,
    quasi_identifiers: list[str],
) -> pd.DataFrame:
    """Return each quasi-identifier combination and its record count."""

    if not quasi_identifiers:
        raise ValueError("At least one quasi-identifier is required.")

    if len(set(quasi_identifiers)) != len(quasi_identifiers):
        raise ValueError("Quasi-identifiers must be unique.")

    missing = [
        col for col in quasi_identifiers
        if col not in df.columns
    ]
    if missing:
        raise KeyError(f"Missing quasi-identifier columns: {missing}")

    if df.empty:
        return pd.DataFrame(
            columns=[*quasi_identifiers, "class_size"]
        )

    return (
        df.groupby(
            quasi_identifiers,
            dropna=False,
            observed=True,
        )
        .size()
        .rename("class_size")
        .reset_index()
    )


def measure_k_anonymity(
    df: pd.DataFrame,
    quasi_identifiers: list[str],
) -> dict:
    """Measure the minimum equivalence-class size."""

    sizes = equivalence_class_sizes(df, quasi_identifiers)

    if sizes.empty:
        return {
            "rows": len(df),
            "minimum_k": None,
            "equivalence_classes": 0,
            "class_sizes": [],
            "target_met": None,
        }

    class_counts = sizes["class_size"].astype(int).tolist()

    return {
        "rows": len(df),
        "minimum_k": min(class_counts),
        "equivalence_classes": len(class_counts),
        "class_sizes": class_counts,
        "target_met": None,
    }


def suppress_records_below_k(
    df: pd.DataFrame,
    quasi_identifiers: list[str],
    k: int,
) -> tuple[pd.DataFrame, dict]:
    """Remove records belonging to equivalence classes smaller than k."""

    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise ValueError("k must be a positive integer.")

    equivalence_class_sizes(df, quasi_identifiers)

    if df.empty:
        return df.copy(), {
            "input_rows": 0,
            "output_rows": 0,
            "suppressed_rows": 0,
            "suppressed_fraction": 0.0,
            "minimum_k_after": None,
            "target_met": None,
        }

    group_sizes = (
        df.groupby(
            quasi_identifiers,
            dropna=False,
            observed=True,
        )[quasi_identifiers[0]]
        .transform("size")
    )

    kept = df.loc[group_sizes >= k].copy()
    suppressed = len(df) - len(kept)
    measured = measure_k_anonymity(kept, quasi_identifiers)

    return kept, {
        "input_rows": len(df),
        "output_rows": len(kept),
        "suppressed_rows": suppressed,
        "suppressed_fraction": suppressed / len(df),
        "minimum_k_after": measured["minimum_k"],
        "target_met": (
            measured["minimum_k"] >= k
            if measured["minimum_k"] is not None
            else True
        ),
    }


def validate_k_anonymity(
    df: pd.DataFrame,
    quasi_identifiers: list[str],
    k: int,
) -> dict:
    """Report k-anonymity violations without modifying the dataset."""

    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise ValueError("k must be a positive integer.")

    if df.empty:
        raise ValueError("Dataset cannot be empty.")

    if "income" in quasi_identifiers:
        raise ValueError(
            "'income' is the prediction target, not a default QI."
        )

    sizes = equivalence_class_sizes(df, quasi_identifiers)
    class_sizes = sizes["class_size"]

    if class_sizes.empty:
        raise ValueError("No equivalence classes were found.")

    violating_classes = class_sizes[class_sizes < k]
    total_rows = len(df)
    violating_records = int(violating_classes.sum())
    achieved_k = int(class_sizes.min())

    return {
        "target_k": k,
        "achieved_min_k": achieved_k,
        "number_of_equivalence_classes": len(class_sizes),
        "total_rows": total_rows,
        "records_in_violating_classes": violating_records,
        "violating_records_percentage": round(
            100 * violating_records / total_rows, 2
        ),
        "criterion_achieved": achieved_k >= k,
        "quasi_identifiers": ",".join(quasi_identifiers),
    }


def validate_dataset(
    input_path: str = "data/cleaned/adult_cleaned.csv",
    output_path: str = "results/privacy_metrics_level1_cleaned.csv",
) -> pd.DataFrame:
    """Evaluate k=3 and k=5 after Level 1 transformations."""

    source = Path(input_path)

    if not source.is_file():
        raise FileNotFoundError(f"Dataset not found: {source}")

    df = pd.read_csv(source)
    anonymized_df = prepare_level1_features(df)

    quasi_identifiers = ["age", "zip_code"]

    if anonymized_df["zip_code"].isna().any():
        raise ValueError(
            "Missing ZIP values found after Level 1 masking."
        )

    results = []

    for k in (3, 5):
        metrics = validate_k_anonymity(
            anonymized_df,
            quasi_identifiers,
            k,
        )
        metrics["dataset_version"] = "cleaned_level1"
        metrics["input_path"] = str(source)
        results.append(metrics)

    results_df = pd.DataFrame(results)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output, index=False)

    return results_df


if __name__ == "__main__":
    metrics = validate_dataset()
    print(metrics.to_string(index=False))
