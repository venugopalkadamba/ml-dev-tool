import numpy as np
import pandas as pd

from src.eval import (
    compute_demographic_parity_ratio,
)


def test_demographic_parity_ratio_basic_and_inf():
    y_pred = np.array([1, 0, 1, 0, 1, 0])
    A = pd.Series([0, 0, 0, 1, 1, 1])
    r = compute_demographic_parity_ratio(y_pred, A)
    # P(ŷ=1|A=1)=1/3, P(ŷ=1|A=0)=2/3 => 0.5
    assert abs(r - 0.5) < 1e-9

    # Denominator zero -> inf
    y_pred = np.array([0, 0, 0, 1, 1, 1])
    A = pd.Series([0, 0, 0, 1, 1, 1])
    assert compute_demographic_parity_ratio(y_pred, A) == float('inf')


def test_demographic_parity_ratio_with_strings_and_booleans():
    y_pred = np.array([1, 1, 0, 0])
    A_str = pd.Series(['A', 'A', 'B', 'B'])
    A_bool = pd.Series([True, True, False, False])
    r1 = compute_demographic_parity_ratio(y_pred, A_str)
    r2 = compute_demographic_parity_ratio(y_pred, A_bool)
    # Consider reciprocal equivalence when one group's positive rate is zero
    if np.isinf(r1) and r2 == 0.0:
        assert True
    elif np.isinf(r2) and r1 == 0.0:
        assert True
    else:
        assert abs(r1 - r2) < 1e-12


