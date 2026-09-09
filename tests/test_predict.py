import numpy as np

from churn.predict import predict_churn


class FakeModel:
    def predict_proba(self, X):
        return np.array([[0.2, 0.8]])


def test_predict_churn():
    model = FakeModel()

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

    assert "churn_probability" in result
    assert "churn_prediction" in result

    assert result["churn_probability"] == 0.8
    assert result["churn_prediction"] == 1
