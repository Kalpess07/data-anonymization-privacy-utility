
import pandas as pd
import pytest

from src.k_anonymity import validate_k_anonymity


def test_k3_satisfied():
    df = pd.DataFrame({
        "age": [25, 25, 25, 35, 35, 35],
        "zip_code": [411, 411, 411, 412, 412, 412],
    })

    result = validate_k_anonymity(
        df, ["age", "zip_code"], 3
    )

    assert result["achieved_min_k"] == 3
    assert result["criterion_achieved"] is True


def test_k5_not_satisfied():
    df = pd.DataFrame({
        "age": [25, 25, 25, 35, 35, 35, 35, 35],
        "zip_code": [411, 411, 411, 412, 412, 412, 412, 412],
    })

    result = validate_k_anonymity(
        df, ["age", "zip_code"], 5
    )

    assert result["achieved_min_k"] == 3
    assert result["criterion_achieved"] is False


def test_target_is_rejected_as_quasi_identifier():
    df = pd.DataFrame({
        "age": [25, 25, 25],
        "income": ["<=50K", "<=50K", ">50K"],
    })

    with pytest.raises(ValueError):
        validate_k_anonymity(df, ["income"], 3)


def test_invalid_k_is_rejected():
    df = pd.DataFrame({"age": [25, 25]})

    with pytest.raises(ValueError):
        validate_k_anonymity(df, ["age"], 0)
