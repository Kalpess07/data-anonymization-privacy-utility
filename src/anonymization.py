"""Basic, reusable transformations for the Adult-with-ZIP project dataset."""
from __future__ import annotations
import pandas as pd


def generalize_age(df: pd.DataFrame, column: str = "age", bin_width: int = 10) -> pd.DataFrame:
    """Replace numeric ages with labels such as '30-39'; missing values stay missing."""
    if column not in df.columns:
        raise KeyError(f"Column not found: {column}")
    if not isinstance(bin_width, int) or isinstance(bin_width, bool) or bin_width < 1:
        raise ValueError("bin_width must be a positive integer")
    out = df.copy()
    numeric_age = pd.to_numeric(out[column], errors="coerce")
    valid = numeric_age.notna() & (numeric_age >= 0)
    labels = pd.Series(pd.NA, index=out.index, dtype="string")
    lower = (numeric_age[valid] // bin_width * bin_width).astype(int)
    upper = lower + bin_width - 1
    labels.loc[valid] = lower.astype(str) + "-" + upper.astype(str)
    out[column] = labels
    return out


def mask_zip(df: pd.DataFrame, column: str = "zip_code", keep_digits: int = 2) -> pd.DataFrame:
    """Keep leading postal-code digits and replace remaining characters with '*'."""
    if column not in df.columns:
        raise KeyError(f"Column not found: {column}")
    if not isinstance(keep_digits, int) or isinstance(keep_digits, bool) or keep_digits < 0:
        raise ValueError("keep_digits must be a non-negative integer")
    out = df.copy()
    values = out[column].astype("string").str.strip()
    # Treat missing/blank values as missing, not the literal string 'nan'.
    values = values.mask(values.eq(""))
    def mask_one(value):
        if pd.isna(value):
            return pd.NA
        digits = "".join(ch for ch in str(value) if ch.isdigit())
        if not digits:
            return "*" * max(1, keep_digits)
        prefix = digits[:keep_digits]
        return prefix + "*" * max(0, len(digits) - len(prefix))
    out[column] = values.map(mask_one).astype("string")
    return out


def remove_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Drop configured direct-identifier columns; fail if a requested column is absent."""
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise KeyError(f"Columns not found: {missing}")
    return df.drop(columns=columns).copy()
