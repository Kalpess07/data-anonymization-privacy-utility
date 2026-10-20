"""Configuration-driven anonymization pipeline."""
from __future__ import annotations
import pandas as pd
from anonymization import generalize_age, mask_zip, remove_columns
from k_anonymity import measure_k_anonymity, suppress_records_below_k

# Proposed defaults only. Confirm these rules and the quasi-identifiers with the team.
LEVEL_CONFIG = {
    "low": {"age_bin_width": 10, "zip_keep_digits": 2, "target_k": None, "remove_columns": []},
    "medium": {"age_bin_width": 20, "zip_keep_digits": 1, "target_k": 3, "remove_columns": []},
    "high": {"age_bin_width": 30, "zip_keep_digits": 1, "target_k": 5, "remove_columns": []},
}


def apply_anonymization(
    df: pd.DataFrame,
    level: str,
    quasi_identifiers: list[str] | None = None,
    config: dict | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Return transformed data and a run summary. No original data is mutated."""
    configurations = LEVEL_CONFIG if config is None else config
    if level not in configurations:
        raise ValueError(f"Unknown level {level!r}; expected one of {list(configurations)}")
    settings = configurations[level]
    out = df.copy()
    out = remove_columns(out, settings.get("remove_columns", []))
    if "age" in out.columns:
        out = generalize_age(out, "age", settings["age_bin_width"])
    if "zip_code" in out.columns:
        out = mask_zip(out, "zip_code", settings["zip_keep_digits"])
    summary = {"level": level, "input_rows": len(df), "output_rows_before_suppression": len(out),
               "transformations": {"age_bin_width": settings.get("age_bin_width"),
                                   "zip_keep_digits": settings.get("zip_keep_digits")},
               "suppression": None, "k_anonymity_before_suppression": None,
               "k_anonymity_after_suppression": None}
    if quasi_identifiers:
        before = measure_k_anonymity(out, quasi_identifiers)
        summary["k_anonymity_before_suppression"] = before
        target_k = settings.get("target_k")
        if target_k is not None:
            out, suppression_summary = suppress_records_below_k(out, quasi_identifiers, target_k)
            summary["suppression"] = suppression_summary
            summary["k_anonymity_after_suppression"] = measure_k_anonymity(out, quasi_identifiers)
    return out, summary
