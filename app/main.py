import logging
import time
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI
from pydantic import BaseModel, Field

from app.logging_config import configure_logging
from churn.model_manager import ensure_model
from churn.predict import load_model, predict_churn

configure_logging()

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("application_startup | checking model")

    ensure_model()

    app.state.model = load_model()

    logger.info("application_startup | model loaded")

    yield

    logger.info("application_shutdown")


def get_model():
    return app.state.model
    # return load_model()


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


app = FastAPI(title="Customer Churn Prediction API", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(customer: CustomerRequest, model=Depends(get_model)):
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
