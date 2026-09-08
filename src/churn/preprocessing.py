from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


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

CATEGORICAL_FEATURES = [
    "country",
    "gender",
]


def create_preprocessor():
    return ColumnTransformer(
        transformers=[
            (
                "num",
                "passthrough",
                NUMERICAL_FEATURES,
            ),
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )