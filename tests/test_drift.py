import pandas as pd

from churn.drift import calculate_psi, interpret_psi


def test_identical_distributions_have_low_psi():
    reference = pd.Series([10, 20, 30, 40, 50])
    current = pd.Series([10, 20, 30, 40, 50])

    psi = calculate_psi(reference, current)

    assert psi < 0.10


def test_different_distributions_have_higher_psi():
    reference = pd.Series([10, 20, 30, 40, 50])
    current = pd.Series([100, 110, 120, 130, 140])

    psi = calculate_psi(reference, current)

    assert psi > 0.10


def test_psi_interpretation():
    assert interpret_psi(0.05) == "low"
    assert interpret_psi(0.15) == "moderate"
    assert interpret_psi(0.30) == "high"