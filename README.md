# Customer Churn Prediction API

A machine learning project that predicts whether a bank customer is likely to churn.

The project covers the full workflow:

- Data preprocessing
- Model training
- Validation and model selection
- Business-based threshold selection
- Model serialization
- Prediction
- REST API serving with FastAPI
- Input validation with Pydantic
- Automated testing with pytest

## Model

The final model is an XGBoost classifier.

Categorical features are encoded using `OneHotEncoder`, while numerical features are passed through unchanged.

The prediction threshold is set to `0.31` based on a business-cost analysis performed on the validation set.

The final production pipeline is trained on the combined training and validation data.

The final test set remains untouched during model selection and threshold tuning.

## Features

The model uses:

### Numerical features

- `credit_score`
- `age`
- `tenure`
- `balance`
- `products_number`
- `credit_card`
- `active_member`
- `estimated_salary`

### Categorical features

- `country`
- `gender`

`customer_id` is excluded because it is an identifier rather than a predictive feature.

## Project Structure

```text
customer-churn-ml/
├── app/
│   └── main.py
│
├── src/
│   └── churn/
│       ├── __init__.py
│       ├── preprocessing.py
│       ├── train.py
│       ├── predict.py
│       └── evaluate.py
│
├── data/
│   └── raw/
│       └── churn.csv
│
├── models/
│   └── churn_pipeline.joblib
│
├── tests/
│   ├── test_predict.py
│   ├── test_schema.py
│   └── test_api.py
│
├── pyproject.toml
└── README.md