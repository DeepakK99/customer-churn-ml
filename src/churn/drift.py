import numpy as np
import pandas as pd


def calculate_psi(
    reference: pd.Series,
    current: pd.Series,
    bins: int = 10,
) -> float:
    breakpoints = np.linspace(0, 1, bins + 1)

    edges = reference.quantile(breakpoints).values
    edges = np.unique(edges)

    if len(edges) < 3:
        return 0.0

    reference_counts, _ = np.histogram(reference, bins=edges)
    current_counts, _ = np.histogram(current, bins=edges)

    reference_pct = reference_counts / len(reference)
    current_pct = current_counts / len(current)

    epsilon = 1e-6

    reference_pct = np.clip(reference_pct, epsilon, None)
    current_pct = np.clip(current_pct, epsilon, None)

    psi = np.sum((current_pct - reference_pct) * np.log(current_pct / reference_pct))

    return float(psi)


def interpret_psi(psi: float) -> str:
    if psi < 0.10:
        return "low"
    elif psi < 0.25:
        return "moderate"
    else:
        return "high"
