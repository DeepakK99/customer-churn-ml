from churn.predict import load_model, predict_churn


def test_predict_churn():
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

    assert "churn_probability" in result
    assert "churn_prediction" in result

    assert 0 <= result["churn_probability"] <= 1
    assert result["churn_prediction"] in [0, 1]