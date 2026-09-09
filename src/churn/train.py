import joblib
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from churn.preprocessing import create_preprocessor

DATA_PATH = "data/raw/Bank CUstomer Churn Prediction.csv"
MODEL_PATH = "models/churn_pipeline.joblib"

THRESHOLD = 0.31


def create_model():
    return XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        min_child_weight=1,
        scale_pos_weight=1,
        random_state=42,
        eval_metric="logloss",
    )


def create_pipeline():
    return Pipeline(
        [
            ("preprocessor", create_preprocessor()),
            ("model", create_model()),
        ]
    )


def train():
    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=["churn", "customer_id"])
    y = df["churn"]

    # -------------------------
    # 1. Development / test split
    # -------------------------

    X_dev, X_test, y_dev, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=42,
    )

    # -------------------------
    # 2. Train / validation split
    # -------------------------

    X_train, X_val, y_train, y_val = train_test_split(
        X_dev,
        y_dev,
        test_size=0.20,
        stratify=y_dev,
        random_state=42,
    )

    # -------------------------
    # 3. Model selection
    # -------------------------

    pipeline = create_pipeline()

    pipeline.fit(X_train, y_train)

    val_proba = pipeline.predict_proba(X_val)[:, 1]

    val_pr_auc = average_precision_score(
        y_val,
        val_proba,
    )

    print(f"Validation PR-AUC: {val_pr_auc:.4f}")

    # -------------------------
    # 4. Validate business threshold
    # -------------------------

    val_pred = (val_proba >= THRESHOLD).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_val,
        val_pred,
    ).ravel()

    cost = fp * 20 + fn * 500

    print(f"Threshold: {THRESHOLD}")
    print(f"FP: {fp}")
    print(f"FN: {fn}")
    print(f"TP: {tp}")
    print(f"TN: {tn}")
    print(f"Business cost: ${cost:,}")

    # -------------------------
    # 5. Final training
    # -------------------------

    X_final = pd.concat([X_train, X_val])

    y_final = pd.concat([y_train, y_val])

    final_pipeline = create_pipeline()

    final_pipeline.fit(
        X_final,
        y_final,
    )

    # -------------------------
    # 6. Save artifact
    # -------------------------

    joblib.dump(
        final_pipeline,
        MODEL_PATH,
    )

    print(f"Final model saved to {MODEL_PATH}")


if __name__ == "__main__":
    train()
