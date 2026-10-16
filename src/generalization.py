

import pandas as pd


def generalize_quasi_identifiers(
    df: pd.DataFrame,
    age_band_width: int = 10,
    zip_prefix_length: int = 2,
) -> pd.DataFrame:
    """
    Generalize age and a ZIP-like field for experimentation.

    The input DataFrame is not modified.
    Original columns are preserved.
    """

    if age_band_width < 1:
        raise ValueError("age_band_width must be positive.")

    if zip_prefix_length < 1:
        raise ValueError("zip_prefix_length must be positive.")

    required = {"age", "zip_code"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    result = df.copy()

    # Generalize age into fixed-width bands.
    age = pd.to_numeric(result["age"], errors="coerce")

    age_group = (
        (age // age_band_width) * age_band_width
    )

    lower = age_group.astype("Int64").astype("string")
    upper = (
        age_group + age_band_width - 1
    ).astype("Int64").astype("string")

    result["age_group"] = lower + "-" + upper
    result.loc[age.isna(), "age_group"] = "Unknown"

    # Preserve leading zeros by converting the ZIP-like
    # field to text before taking its prefix.
    zip_code = result["zip_code"].astype("string").str.strip()

    invalid_zip = zip_code.isna() | zip_code.eq("")
    result["zip_prefix"] = zip_code.str[:zip_prefix_length]
    result.loc[invalid_zip, "zip_prefix"] = "Unknown"

    return result
