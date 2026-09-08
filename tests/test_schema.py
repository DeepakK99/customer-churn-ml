import pytest
from pydantic import ValidationError

from app.main import CustomerRequest


def test_valid_customer():
    customer = CustomerRequest(
        credit_score=650,
        country="Germany",
        gender="Female",
        age=45,
        tenure=3,
        balance=120000,
        products_number=2,
        credit_card=1,
        active_member=0,
        estimated_salary=90000,
    )

    assert customer.age == 45
    assert customer.country == "Germany"


def test_invalid_age():
    with pytest.raises(ValidationError):
        CustomerRequest(
            credit_score=650,
            country="Germany",
            gender="Female",
            age=-5,
            tenure=3,
            balance=120000,
            products_number=2,
            credit_card=1,
            active_member=0,
            estimated_salary=90000,
        )