
import pandas as pd


def generalize_age(age_series):
    """Convert numeric ages into 10-year categorical ranges."""
    age = pd.to_numeric(age_series, errors="coerce")

    bins = list(range(0, 101, 10))
    labels = [f"{start}-{start + 9}" for start in range(0, 100, 10)]

    result = pd.cut(
        age,
        bins=bins,
        labels=labels,
        right=False
    )

    # Preserve missing and out-of-range values as missing.
    return result.astype("string")


def mask_zip_code(zip_series):
    """Mask the last two characters of five-character ZIP-like values."""
    values = zip_series.astype("string").str.strip()

    valid = values.str.fullmatch(r"\d{5}", na=False)

    result = pd.Series(
        pd.NA,
        index=values.index,
        dtype="string"
    )

    result.loc[valid] = values.loc[valid].str[:3] + "**"

    return result


def prepare_level1_features(data):
    """Create Level 1 age and ZIP features without modifying the input."""
    output = data.copy()

    if "age" in output.columns:
        output["age"] = generalize_age(output["age"])

    if "zip_code" in output.columns:
        output["zip_code"] = mask_zip_code(output["zip_code"])

    return output
