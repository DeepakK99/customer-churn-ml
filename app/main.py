import logging
import time
from fastapi import FastAPI
from churn.predict import load_model, predict_churn

from pydantic import BaseModel, Field
from app.logging_config import configure_logging

configure_logging()

logger = logging.getLogger(__name__)


class CustomerRequest(BaseModel):
    credit_score: int = Field(ge=300, le=850)
    country: str
    gender: str
    age: int = Field(ge=18, le=100)
    tenure: int = Field(ge=0, le=10)
    balance: float = Field(ge=0)
    products_number: int = Field(ge=1, le=4)
    credit_card: int = Field(ge=0, le=1)
    active_member: int = Field(ge=0, le=1)
    estimated_salary: float = Field(ge=0)


app = FastAPI(
    title="Customer Churn Prediction API",
    version="1.0.0",
)


model = load_model()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(customer: CustomerRequest):
    start = time.perf_counter()

    result = predict_churn(model, customer.model_dump())

    duration = time.perf_counter() - start

    logger.info(
        "prediction_completed | probability=%.4f | prediction=%d | duration_ms=%.2f",
        result["churn_probability"],
        result["churn_prediction"],
        duration * 1000,
    )

    return result
