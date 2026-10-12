import pandas as pd
import pytest
from src.anonymization import generalize_age, mask_zip, remove_columns
from src.k_anonymity import measure_k_anonymity, suppress_records_below_k
from src.pipeline import apply_anonymization


def test_generalize_age_bins_and_missing():
    df = pd.DataFrame({"age": [0, 29, 30, 39, 40, None]})
    result = generalize_age(df, bin_width=10)
    assert result["age"].iloc[:5].tolist() == ["0-9", "20-29", "30-39", "30-39", "40-49"]
    assert pd.isna(result["age"].iloc[5])
    assert df["age"].iloc[1] == 29  # input remains unchanged


def test_mask_zip():
    df = pd.DataFrame({"zip_code": [12345, "9876", None, "ABC"]})
    result = mask_zip(df, keep_digits=2)
    assert result["zip_code"].iloc[0] == "12***"
    assert result["zip_code"].iloc[1] == "98**"
    assert pd.isna(result["zip_code"].iloc[2])
    assert result["zip_code"].iloc[3] == "**"


def test_measure_k_anonymity():
    df = pd.DataFrame({"age_band": ["20-29", "20-29", "30-39"], "zip": ["12*", "12*", "98*"]})
    result = measure_k_anonymity(df, ["age_band", "zip"])
    assert result["minimum_k"] == 1
    assert result["equivalence_classes"] == 2


def test_suppress_records_below_k():
    df = pd.DataFrame({"age_band": ["A", "A", "A", "B", "B"], "zip": ["X", "X", "X", "Y", "Y"]})
    result, summary = suppress_records_below_k(df, ["age_band", "zip"], 3)
    assert len(result) == 3
    assert summary["suppressed_rows"] == 2
    assert summary["minimum_k_after"] == 3
    assert summary["target_met"] is True


def test_missing_quasi_identifier_is_grouped_consistently():
    df = pd.DataFrame({"age_band": ["A", "A", None], "zip": ["X", "X", "Z"]})
    result = measure_k_anonymity(df, ["age_band", "zip"])
    assert result["minimum_k"] == 1


def test_invalid_k_and_missing_column():
    df = pd.DataFrame({"age": [20]})
    with pytest.raises(ValueError):
        suppress_records_below_k(df, ["age"], 0)
    with pytest.raises(KeyError):
        measure_k_anonymity(df, ["not_a_column"])


def test_pipeline_low_level():
    df = pd.DataFrame({"age": [21, 24], "zip_code": [12345, 12346], "income": ["<=50K", ">50K"]})
    result, summary = apply_anonymization(df, "low", quasi_identifiers=["age", "zip_code"])
    assert result["age"].tolist() == ["20-29", "20-29"]
    assert result["zip_code"].tolist() == ["12***", "12***"]
    assert summary["level"] == "low"
    assert len(result) == len(df)


def test_unknown_level():
    with pytest.raises(ValueError):
        apply_anonymization(pd.DataFrame({"age": [20]}), "extreme")
