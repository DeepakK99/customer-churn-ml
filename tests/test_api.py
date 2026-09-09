import numpy as np
from fastapi.testclient import TestClient

from app.main import app, get_model

client = TestClient(app)


class FakeModel:
    def predict_proba(self, X):
        return np.array([[0.2, 0.8]])


app.dependency_overrides[get_model] = lambda: FakeModel()

client = TestClient(app)


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict():
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

    response = client.post("/predict", json=customer)

    assert response.status_code == 200

    result = response.json()

    assert "churn_probability" in result
    assert "churn_prediction" in result

    assert 0 <= result["churn_probability"] <= 1
    assert result["churn_prediction"] in [0, 1]


def test_predict_invalid_age():
    customer = {
        "credit_score": 650,
        "country": "Germany",
        "gender": "Female",
        "age": -5,
        "tenure": 3,
        "balance": 120000,
        "products_number": 2,
        "credit_card": 1,
        "active_member": 0,
        "estimated_salary": 90000,
    }

    response = client.post("/predict", json=customer)

    assert response.status_code == 422
