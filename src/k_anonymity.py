"""Utilities for measuring k-anonymity over selected quasi-identifiers."""
from __future__ import annotations
import pandas as pd


def equivalence_class_sizes(df: pd.DataFrame, quasi_identifiers: list[str]) -> pd.DataFrame:
    """Return one row per quasi-identifier combination and its record count."""
    if not quasi_identifiers:
        raise ValueError("quasi_identifiers must contain at least one column")
    missing = [col for col in quasi_identifiers if col not in df.columns]
    if missing:
        raise KeyError(f"Missing quasi-identifier columns: {missing}")
    if df.empty:
        return pd.DataFrame(columns=[*quasi_identifiers, "class_size"])
    return (
        df.groupby(quasi_identifiers, dropna=False, observed=True)
        .size().rename("class_size").reset_index()
    )


def measure_k_anonymity(df: pd.DataFrame, quasi_identifiers: list[str]) -> dict:
    """Measure the minimum equivalence-class size; this does not anonymize data."""
    sizes = equivalence_class_sizes(df, quasi_identifiers)
    if sizes.empty:
        return {"rows": len(df), "minimum_k": None, "equivalence_classes": 0,
                "class_sizes": [], "target_met": None}
    class_counts = sizes["class_size"].astype(int).tolist()
    return {"rows": len(df), "minimum_k": min(class_counts),
            "equivalence_classes": len(class_counts), "class_sizes": class_counts,
            "target_met": None}


def suppress_records_below_k(
    df: pd.DataFrame, quasi_identifiers: list[str], k: int
) -> tuple[pd.DataFrame, dict]:
    """Remove entire records belonging to classes smaller than k, then report impact."""
    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise ValueError("k must be a positive integer")
    # Validate columns even when df is empty.
    equivalence_class_sizes(df, quasi_identifiers)
    if df.empty:
        return df.copy(), {"input_rows": 0, "output_rows": 0, "suppressed_rows": 0,
                           "suppressed_fraction": 0.0, "minimum_k_after": None, "target_met": None}
    group_sizes = df.groupby(quasi_identifiers, dropna=False, observed=True)[quasi_identifiers[0]].transform("size")
    kept = df.loc[group_sizes >= k].copy()
    suppressed = len(df) - len(kept)
    measured = measure_k_anonymity(kept, quasi_identifiers)
    return kept, {"input_rows": len(df), "output_rows": len(kept), "suppressed_rows": suppressed,
                  "suppressed_fraction": suppressed / len(df),
                  "minimum_k_after": measured["minimum_k"],
                  "target_met": (measured["minimum_k"] >= k) if measured["minimum_k"] is not None else True}
