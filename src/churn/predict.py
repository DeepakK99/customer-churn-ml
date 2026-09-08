import joblib
import pandas as pd


MODEL_PATH = "models/churn_pipeline.joblib"
THRESHOLD = 0.31


def load_model():
    return joblib.load(MODEL_PATH)


def predict_churn(model, customer: dict):
    X = pd.DataFrame([customer])

    probability = model.predict_proba(X)[0, 1]

    prediction = int(probability >= THRESHOLD)

    return {
        "churn_probability": float(probability),
        "churn_prediction": prediction,
    }

if __name__ == "__main__":
    model = load_model()

    customer = {
        "credit_score": 650,
        "country": "Germany",
        "gender": "Female",
        "age": 45,
        "tenure": 3,
        "balance": 120000,
        "products_number": 2,
        "credit_card": 1,
        "active_member": 0,
        "estimated_salary": 90000,
    }

    result = predict_churn(model, customer)

    print(result)