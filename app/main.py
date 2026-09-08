from fastapi import FastAPI
from churn.predict import load_model, predict_churn

from pydantic import BaseModel, Field


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
    return predict_churn(model, customer.model_dump())
