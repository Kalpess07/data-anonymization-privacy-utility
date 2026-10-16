
import pandas as pd
import pytest

from src.generalization import generalize_quasi_identifiers


def test_generalization_creates_expected_bands():
    df = pd.DataFrame({
        "age": [17, 25, 39, 40],
        "zip_code": [12345, 12346, 67890, 67891],
        "income": ["<=50K", ">50K", "<=50K", ">50K"],
    })

    result = generalize_quasi_identifiers(df)

    assert result["age_group"].tolist() == [
        "10-19", "20-29", "30-39", "40-49"
    ]
    assert result["zip_prefix"].tolist() == [
        "12", "12", "67", "67"
    ]
    assert result["income"].equals(df["income"])


def test_input_is_not_modified():
    df = pd.DataFrame({
        "age": [25, 35],
        "zip_code": [411001, 411002],
    })
    original = df.copy(deep=True)

    generalize_quasi_identifiers(df)

    pd.testing.assert_frame_equal(df, original)


def test_missing_values_are_handled():
    df = pd.DataFrame({
        "age": [25, None],
        "zip_code": [12345, None],
    })

    result = generalize_quasi_identifiers(df)

    assert result.loc[1, "age_group"] == "Unknown"
    assert result.loc[1, "zip_prefix"] == "Unknown"


def test_invalid_width_is_rejected():
    df = pd.DataFrame({
        "age": [25],
        "zip_code": [12345],
    })

    with pytest.raises(ValueError):
        generalize_quasi_identifiers(df, age_band_width=0)
