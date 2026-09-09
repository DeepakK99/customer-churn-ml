import pandas as pd
import joblib

from churn.drift import calculate_psi, interpret_psi

REFERENCE_PATH = "models/reference_data.csv" # training data
DATA_PATH = "data/raw/production_sample.csv" # dummy production simulated data
MODEL_PATH = "models/churn_pipeline.joblib" 

NUMERICAL_FEATURES = [
    "credit_score",
    "age",
    "tenure",
    "balance",
    "products_number",
    "credit_card",
    "active_member",
    "estimated_salary",
]

def get_predictions(model, data):
    X = data.drop(columns=["customer_id", "churn"], errors="ignore")

    return model.predict_proba(X)[:, 1]

def check_data_drift():
    reference = pd.read_csv(REFERENCE_PATH)
    current = pd.read_csv(DATA_PATH)

    drifted_features = []

    for feature in NUMERICAL_FEATURES:
        psi = calculate_psi(
            reference[feature],
            current[feature],
        )
        if psi >= 0.25:
            drifted_features.append(feature)

        print(f"{feature:20s} " f"PSI={psi:.4f} " f"({interpret_psi(psi)})")

    if drifted_features:
        print(f"Drift detected: {', '.join(drifted_features)}")
    else:
        print("No significant drift detected.")

def check_prediction_drift():
    reference = pd.read_csv(REFERENCE_PATH)
    current = pd.read_csv(DATA_PATH)
    model = joblib.load(MODEL_PATH)
    
    reference_predictions = get_predictions(
        model,
        reference,
    )
    current_predictions = get_predictions(
        model,
        current,
    )
    psi = calculate_psi(
        pd.Series(reference_predictions),
        pd.Series(current_predictions),
    )
    print(
        f"\nprediction_probability "
        f"PSI={psi:.4f} "
        f"({interpret_psi(psi)})"
    )

if __name__ == "__main__":
    check_data_drift()
    check_prediction_drift()
