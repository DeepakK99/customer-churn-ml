import os
from pathlib import Path

import boto3

MODEL_PATH = Path(
    os.getenv(
        "MODEL_PATH",
        "models/churn_pipeline.joblib",
    )
)

S3_BUCKET = os.getenv("S3_BUCKET")
S3_MODEL_KEY = os.getenv(
    "S3_MODEL_KEY",
    "models/churn_pipeline.joblib",
)


def model_exists() -> bool:
    return MODEL_PATH.exists()


def download_model() -> None:
    if not S3_BUCKET:
        raise RuntimeError("S3_BUCKET environment variable is not set")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    s3 = boto3.client("s3")

    s3.download_file(
        S3_BUCKET,
        S3_MODEL_KEY,
        str(MODEL_PATH),
    )


def ensure_model() -> None:
    if model_exists():
        return

    download_model()
