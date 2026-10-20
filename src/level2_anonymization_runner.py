"""Level 2 anonymization and privacy-utility experiments.

Implements configurable age generalization, ZIP masking, categorical
generalization, k-anonymity suppression, risk comparison, and ML evaluation.
"""

from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from k_anonymity import measure_k_anonymity, suppress_records_below_k


ROOT = Path(__file__).resolve().parent.parent

SOURCE_PATH = ROOT / "data" / "cleaned" / "adult_cleaned.csv"
LEVEL1_PATH = ROOT / "data" / "anonymized" / "level1" / "adult_level1.csv"
OUTPUT_DIR = ROOT / "data" / "anonymized" / "level2"
RESULTS_DIR = ROOT / "results"

QI_COLUMNS = ["age", "zip_code"]
TARGET = "income"
RANDOM_STATE = 42
TEST_SIZE = 0.20

# Proposed experimental settings. Confirm with the team before reporting
# these as the finalized Low / Medium / High definitions.
LEVEL_CONFIG = {
    "low": {
        "age_bin_width": 10,
        "zip_keep_digits": 2,
        "target_k": None,
        "generalize_education": False,
        "generalize_marital_status": False,
        "generalize_native_country": False,
    },
    "medium": {
        "age_bin_width": 20,
        "zip_keep_digits": 1,
        "target_k": 3,
        "generalize_education": True,
        "generalize_marital_status": True,
        "generalize_native_country": True,
    },
    "high": {
        "age_bin_width": 20,
        "zip_keep_digits": 1,
        "target_k": 5,
        "generalize_education": True,
        "generalize_marital_status": True,
        "generalize_native_country": True,
    },
}

# Proposed broad education categories.
EDUCATION_GROUPS = {
    "Preschool": "Basic or school",
    "1st-4th": "Basic or school",
    "5th-6th": "Basic or school",
    "7th-8th": "Basic or school",
    "9th": "Basic or school",
    "10th": "Basic or school",
    "11th": "Basic or school",
    "12th": "Basic or school",
    "HS-grad": "High school",
    "Some-college": "Post-secondary",
    "Assoc-voc": "Post-secondary",
    "Assoc-acdm": "Post-secondary",
    "Bachelors": "University",
    "Masters": "Postgraduate",
    "Prof-school": "Postgraduate",
    "Doctorate": "Postgraduate",
}

# Proposed broad marital-status categories.
MARITAL_GROUPS = {
    "Married-civ-spouse": "Married",
    "Married-AF-spouse": "Married",
    "Married-spouse-absent": "Previously/temporarily married",
    "Divorced": "Previously/temporarily married",
    "Separated": "Previously/temporarily married",
    "Widowed": "Previously/temporarily married",
    "Never-married": "Never married",
}

# Proposed regional grouping for common Adult Census country values.
COUNTRY_GROUPS = {
    "United-States": "North America",
    "Canada": "North America",
    "Mexico": "North America",
    "Puerto-Rico": "Caribbean",
    "Cuba": "Caribbean",
    "Jamaica": "Caribbean",
    "Haiti": "Caribbean",
    "Dominican-Republic": "Caribbean",
    "Trinadad&Tobago": "Caribbean",
    "El-Salvador": "Central America",
    "Guatemala": "Central America",
    "Honduras": "Central America",
    "Nicaragua": "Central America",
    "Costa-Rica": "Central America",
    "Outlying-US(Guam-USVI-etc)": "Other territories",
    "Columbia": "South America",
    "Ecuador": "South America",
    "Peru": "South America",
    "Honduras": "Central America",
    "Brazil": "South America",
    "England": "Europe",
    "Germany": "Europe",
    "Greece": "Europe",
    "Italy": "Europe",
    "Poland": "Europe",
    "Portugal": "Europe",
    "Ireland": "Europe",
    "France": "Europe",
    "Scotland": "Europe",
    "Hungary": "Europe",
    "Yugoslavia": "Europe",
    "Holand-Netherlands": "Europe",
    "India": "South Asia",
    "Iran": "Middle East",
    "Philippines": "Southeast Asia",
    "Cambodia": "Southeast Asia",
    "Thailand": "Southeast Asia",
    "Laos": "Southeast Asia",
    "Vietnam": "Southeast Asia",
    "China": "East Asia",
    "Japan": "East Asia",
    "Taiwan": "East Asia",
    "Hong": "East Asia",
}


def load_dataset(path=SOURCE_PATH):
    """Load a dataset and validate required columns."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path, dtype={"zip_code": "string"})

    required = set(QI_COLUMNS + [TARGET])
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df.dropna(subset=[TARGET]).copy()
    df[TARGET] = df[TARGET].astype("string").str.strip()

    if df.empty:
        raise ValueError(f"No usable records in {path}")

    return df


def generalize_age(series, bin_width):
    """Convert numeric ages to configurable categorical ranges."""
    age = pd.to_numeric(series, errors="coerce")

    if bin_width <= 0:
        raise ValueError("bin_width must be positive")

    upper = max(100, ((int(age.max()) // bin_width) + 1) * bin_width)
    bins = list(range(0, upper + bin_width, bin_width))
    labels = [
        f"{start}-{start + bin_width - 1}"
        for start in bins[:-1]
    ]

    result = pd.cut(age, bins=bins, labels=labels, right=False)
    return result.astype("string")


def mask_zip_code(series, keep_digits):
    """Keep leading ZIP digits and mask the remaining digits."""
    if keep_digits < 0:
        raise ValueError("keep_digits cannot be negative")

    values = series.astype("string").str.strip()
    result = pd.Series(pd.NA, index=series.index, dtype="string")

    valid = values.str.fullmatch(r"\d+", na=False)
    for idx in values.index[valid]:
        value = values.loc[idx]
        if len(value) <= keep_digits:
            result.loc[idx] = value
        else:
            result.loc[idx] = (
                value[:keep_digits] + "*" * (len(value) - keep_digits)
            )

    return result


def generalize_categories(df, config):
    """Generalize selected categorical columns using approved mappings."""
    out = df.copy()

    if config["generalize_education"] and "education" in out:
        out["education"] = (
            out["education"].astype("string")
            .map(EDUCATION_GROUPS)
            .fillna(out["education"].astype("string"))
        )

    if config["generalize_marital_status"] and "marital_status" in out:
        out["marital_status"] = (
            out["marital_status"].astype("string")
            .map(MARITAL_GROUPS)
            .fillna(out["marital_status"].astype("string"))
        )

    if config["generalize_native_country"] and "native_country" in out:
        out["native_country"] = (
            out["native_country"].astype("string")
            .map(COUNTRY_GROUPS)
            .fillna(out["native_country"].astype("string"))
        )

    return out


def anonymize_dataset(df, level):
    """Apply one Level 2 configuration and enforce its k target if set."""
    if level not in LEVEL_CONFIG:
        raise ValueError(f"Unknown level: {level}")

    config = LEVEL_CONFIG[level]
    out = df.copy()

    out["age"] = generalize_age(out["age"], config["age_bin_width"])
    out["zip_code"] = mask_zip_code(
        out["zip_code"], config["zip_keep_digits"]
    )
    out = generalize_categories(out, config)

    before_k = measure_k_anonymity(out, QI_COLUMNS)
    target_k = config["target_k"]

    suppression_summary = {
        "suppressed_rows": 0,
        "suppressed_fraction": 0.0,
        "output_rows": len(out),
    }

    if target_k is not None:
        out, suppression_summary = suppress_records_below_k(
            out, QI_COLUMNS, target_k
        )

    after_k = measure_k_anonymity(out, QI_COLUMNS)

    summary = {
        "level": level,
        "input_rows": len(df),
        "rows_before_suppression": len(df) if False else len(df),
        "rows_after_generalization": len(df),
        "target_k": target_k,
        "minimum_k_before_suppression": before_k.get("minimum_k"),
        "suppressed_rows": suppression_summary.get("suppressed_rows", 0),
        "suppression_fraction": suppression_summary.get(
            "suppressed_fraction", 0.0
        ),
        "output_rows": len(out),
        "minimum_k_after_suppression": after_k.get("minimum_k"),
        "age_bin_width": config["age_bin_width"],
        "zip_keep_digits": config["zip_keep_digits"],
    }

    return out, summary


def count_unique_qi_combinations(df):
    """Count unique age + ZIP combinations and identify singleton rows."""
    counts = df.groupby(QI_COLUMNS, dropna=False).size()

    return {
        "rows": len(df),
        "unique_qi_combinations": int(len(counts)),
        "singleton_combinations": int((counts == 1).sum()),
        "rows_in_singleton_combinations": int(counts[counts == 1].sum()),
        "minimum_class_size": int(counts.min()) if len(counts) else 0,
    }


def build_preprocessor(X):
    """Impute and encode features without leaking test-set information."""
    numeric_columns = X.select_dtypes(include="number").columns.tolist()
    categorical_columns = X.select_dtypes(
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

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ]
    )


def evaluate_models(df, level):
    """Evaluate Logistic Regression and Random Forest for one level."""
    if df.empty:
        raise ValueError(f"{level} dataset is empty after suppression.")

    X = df.drop(columns=[TARGET])
    y = df[TARGET].astype(str)

    if y.nunique() < 2:
        raise ValueError(f"{level} dataset has fewer than two target classes.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    rows = []

    for model_name, model in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(X_train)),
                ("classifier", model),
            ]
        )
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)

        rows.append({
            "experiment": "level2",
            "anonymization_level": level,
            "model": model_name,
            "train_rows": len(X_train),
            "test_rows": len(X_test),
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
        })

    return rows


def main():
    """Run all levels and save datasets, privacy, risk, and utility results."""
    source = load_dataset()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    privacy_rows = []
    risk_rows = []
    utility_rows = []
    level_datasets = {}

    source_risk = count_unique_qi_combinations(source)
    risk_rows.append({"dataset": "Original", **source_risk})

    for level in LEVEL_CONFIG:
        print(f"Processing {level.title()}...")

        anonymized, summary = anonymize_dataset(source, level)
        level_datasets[level] = anonymized

        privacy_rows.append(summary)
        risk_rows.append({
            "dataset": level.title(),
            **count_unique_qi_combinations(anonymized),
        })

        utility_rows.extend(evaluate_models(anonymized, level))

    # Compare against Level 1 when that dataset is available.
    if LEVEL1_PATH.exists():
        level1 = load_dataset(LEVEL1_PATH)
        risk_rows.append({
            "dataset": "Level 1",
            **count_unique_qi_combinations(level1),
        })

    # Write Medium dataset and result tables.
    level_datasets["medium"].to_csv(
        OUTPUT_DIR / "adult_level2_medium.csv", index=False
    )
    pd.DataFrame(privacy_rows).to_csv(
        RESULTS_DIR / "level2_privacy_metrics.csv", index=False
    )
    pd.DataFrame(utility_rows).to_csv(
        RESULTS_DIR / "level2_utility_metrics.csv", index=False
    )
    pd.DataFrame(risk_rows).to_csv(
        RESULTS_DIR / "level2_risk_comparison.csv", index=False
    )

    print("\nExperiment completed.")
    print("Privacy rows:", len(privacy_rows))
    print("Utility rows:", len(utility_rows))
    print("Risk comparison rows:", len(risk_rows))


if __name__ == "__main__":
    main()

