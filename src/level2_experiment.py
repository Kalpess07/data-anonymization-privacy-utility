
"""Run Level 2 anonymization experiments and evaluate privacy-utility trade-offs."""

from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


from src.k_anonymity import measure_k_anonymity, suppress_records_below_k
from src.pipeline import LEVEL_CONFIG, apply_anonymization

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "cleaned" / "adult_cleaned.csv"
RESULTS_DIR = PROJECT_ROOT / "results"



QUASI_IDENTIFIERS = ["age", "zip_code"]

def load_dataset():
    """Load the cleaned dataset and validate required columns."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH, dtype={"zip_code": "string"})

    required_columns = {"age", "zip_code", "income"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: {sorted(missing_columns)}"
        )

    df = df.dropna(subset=["income"]).copy()
    df["income"] = df["income"].astype(str).str.strip()

    return df

def run_anonymization_level(df, level):
    """Apply one anonymization level and return its privacy summary."""
    anonymized_df, summary = apply_anonymization(
        df,
        level=level,
        quasi_identifiers=QUASI_IDENTIFIERS,
    )

    return anonymized_df, summary

def run_all_levels(df):
    """Run each configured anonymization level and collect the results."""
    results = {}

    for level in LEVEL_CONFIG:
        anonymized_df, summary = run_anonymization_level(df, level)
        results[level] = {
            "data": anonymized_df,
            "summary": summary,
        }

    return results

def build_preprocessor(X):
    """Build preprocessing steps for numeric and categorical features."""
    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(
        include=["object", "string", "category", "bool"]
    ).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )

    return preprocessor

def evaluate_models(df, level):
    """Train and evaluate both models for one anonymization level."""
    if df.empty:
        raise ValueError(f"The dataset for level {level!r} is empty.")

    X = df.drop(columns=["income"])
    y = df["income"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
        ),
    }

    results = []

    for model_name, model in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(X_train)),
                ("classifier", model),
            ]
        )

        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)

        results.append(
            {
                "experiment": "level2",
                "anonymization_level": level,
                "model": model_name,
                "accuracy": accuracy_score(y_test, predictions),
                "precision": precision_score(
                    y_test, predictions, pos_label=">50K", zero_division=0
                ),
                "recall": recall_score(
                    y_test, predictions, pos_label=">50K", zero_division=0
                ),
                "f1_score": f1_score(
                    y_test, predictions, pos_label=">50K", zero_division=0
                ),
            }
        )

    return results

def save_results(privacy_results, utility_results):
    """Save Level 2 privacy and utility results to separate CSV files."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    privacy_path = RESULTS_DIR / "level2_privacy_metrics.csv"
    utility_path = RESULTS_DIR / "level2_utility_metrics.csv"

    pd.DataFrame(privacy_results).to_csv(privacy_path, index=False)
    pd.DataFrame(utility_results).to_csv(utility_path, index=False)

    print(f"Privacy results saved to: {privacy_path}")
    print(f"Utility results saved to: {utility_path}")

def collect_privacy_summary(level, summary):
    """Extract key privacy measurements into one CSV-friendly record."""
    suppression = summary.get("suppression") or {}

    before = summary.get("k_anonymity_before_suppression") or {}
    after = summary.get("k_anonymity_after_suppression") or {}

    return {
        "level": level,
        "input_rows": summary["input_rows"],
        "output_rows_before_suppression": summary[
            "output_rows_before_suppression"
        ],
        "minimum_k_before_suppression": before.get("minimum_k"),
        "target_k": LEVEL_CONFIG[level]["target_k"],
        "suppressed_records": suppression.get("suppressed_rows", 0),
        "suppression_fraction": suppression.get("suppressed_fraction", 0),
        "output_rows_after_suppression": suppression.get(
            "output_rows", summary["output_rows_before_suppression"]
        ),
        "minimum_k_after_suppression": (
            after.get("minimum_k") if after else before.get("minimum_k")
        ),
    }

def main():
    """Run all Level 2 experiments and save privacy and utility results."""
    df = load_dataset()

    level_results = run_all_levels(df)

    privacy_results = []
    utility_results = []

    for level, result in level_results.items():
        privacy_results.append(
            collect_privacy_summary(level, result["summary"])
        )

        if level == "medium":
            output_dir = (
                PROJECT_ROOT / "data" / "anonymized" / "level2"
            )
            output_dir.mkdir(parents=True, exist_ok=True)

            output_path = output_dir / "adult_level2_medium.csv"
            result["data"].to_csv(output_path, index=False)

            print(f"Medium Level 2 dataset saved to: {output_path}")

        utility_results.extend(
            evaluate_models(result["data"], level)
        )

    save_results(privacy_results, utility_results)


if __name__ == "__main__":
    main()


