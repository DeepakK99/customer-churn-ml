# Customer Churn Prediction — End-to-End ML Deployment

An end-to-end machine learning project for predicting customer churn, covering the complete lifecycle from model development and evaluation to API serving, containerization, CI/CD, AWS deployment, and monitoring.

The project was intentionally designed around practical ML engineering concepts rather than unnecessary infrastructure complexity.

---

## Project Overview

The goal is to predict whether a bank customer is likely to churn.

The project uses the **Bank Customer Churn** dataset containing 10,000 customers and combines:

- Exploratory data analysis
- Classification modeling
- Model evaluation
- Threshold optimization
- Business-cost evaluation
- Model interpretation
- Data drift monitoring
- Automated testing
- FastAPI model serving
- Docker
- GitHub Actions
- AWS ECR
- AWS S3
- AWS EC2
- AWS Systems Manager
- GitHub OIDC authentication

The final system can take customer information through a REST API and return a churn prediction and probability.

---

## Architecture

```text
                         ┌─────────────────────┐
                         │     GitHub Repo      │
                         │ customer-churn-ml   │
                         └──────────┬──────────┘
                                    │
                                  Push
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │   GitHub Actions    │
                         │                     │
                         │ • Ruff              │
                         │ • Pytest            │
                         │ • Docker build      │
                         └──────────┬──────────┘
                                    │
                              GitHub OIDC
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │      AWS IAM        │
                         │  Temporary creds    │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │        ECR          │
                         │ Docker image        │
                         └──────────┬──────────┘
                                    │
                              SSM deployment
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │        EC2          │
                         │                     │
                         │ Docker              │
                         │   ↓                 │
                         │ FastAPI             │
                         └──────────┬──────────┘
                                    │
                                    │ model download
                                    ▼
                         ┌─────────────────────┐
                         │         S3          │
                         │ churn_pipeline      │
                         │      .joblib        │
                         └─────────────────────┘
```

---

# 1. Dataset

The project uses a bank customer churn dataset containing:

* 10,000 rows
* 12 original columns
* Binary churn target

Important features include:

* Credit score
* Country
* Gender
* Age
* Tenure
* Balance
* Number of products
* Credit card ownership
* Active membership
* Estimated salary

`customer_id` was excluded because it is an identifier rather than a predictive feature.

The churn distribution is approximately:

```text
No churn: 79.63%
Churn:    20.37%
```

This class imbalance made **precision-recall metrics particularly useful**.

---

# 2. Data Splitting

The dataset was split into three sets:

```text
10,000 samples
      │
      ├── 8,000 development
      │       │
      │       ├── 6,400 train
      │       └── 1,600 validation
      │
      └── 2,000 final test
```

All splits were stratified.

The final test set was kept untouched during model selection and threshold tuning.

---

# 3. Preprocessing

The preprocessing pipeline uses `ColumnTransformer`.

### Numerical features

Numerical features are passed through without scaling because the final model is XGBoost.

### Categorical features

Categorical features are encoded using:

```python
OneHotEncoder(handle_unknown="ignore")
```

`handle_unknown="ignore"` makes the inference pipeline more resilient if a previously unseen category appears in production.

The transformed feature matrix contains 13 features:

```text
8 numerical
+
3 country categories
+
2 gender categories
=
13 features
```

The preprocessing pipeline is fitted only on training data to avoid data leakage.

---

# 4. Model Development

Several classification approaches were explored during the learning process:

* Logistic Regression
* Decision Trees
* Random Forest
* XGBoost

XGBoost provided the strongest validation performance for this problem.

The final model is based on:

```python
XGBClassifier(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    random_state=42,
    eval_metric="logloss"
)
```

---

# 5. Model Evaluation

Because churn is an imbalanced classification problem, accuracy alone was not considered sufficient.

The project evaluates:

* Accuracy
* Precision
* Recall
* F1
* ROC-AUC
* PR-AUC
* Confusion matrix
* Business cost

PR-AUC was particularly useful because the positive churn class represents only about 20% of customers.

---

# 6. Probability Threshold

The default classification threshold of `0.5` was not treated as automatically optimal.

The threshold was evaluated based on the business objective.

A validation threshold of:

```text
0.31
```

was selected for the project.

The threshold determines the tradeoff between:

```text
Lower threshold
    ↓
More customers flagged
    ↓
Higher recall
    ↓
More false positives
```

and:

```text
Higher threshold
    ↓
Fewer customers flagged
    ↓
Lower recall
    ↓
Fewer false positives
```

The threshold was selected using the validation set and then evaluated on the locked final test set.

---

# 7. Model Interpretation

Two approaches were used to understand the model.

## Feature Importance

The most influential features included:

1. `products_number`
2. `active_member`
3. `age`
4. `country_Germany`
5. `gender_Female`
6. `balance`

Feature importance indicates how much features contribute to model splits. It does **not** establish causality or explain the direction of the relationship.

## SHAP

SHAP was used to explain individual predictions and understand how feature values push the model prediction toward or away from churn.

For example:

```text
Higher age
      ↓
pushes prediction toward churn

Inactive membership
      ↓
pushes prediction toward churn

Higher number of products
      ↓
can strongly influence churn prediction
```

SHAP values were interpreted as explanations of the model's behavior rather than causal explanations of customer behavior.

---

# 8. Model Artifact

The trained preprocessing + model pipeline is stored as:

```text
models/churn_pipeline.joblib
```

The production architecture intentionally keeps the model artifact outside the Docker image.

The model is stored in Amazon S3:

```text
models/churn_pipeline.joblib
```

The Docker container downloads the model when it starts if it is not already available locally.

This separates:

```text
Application code
        +
Model artifact
```

and allows the model to be updated independently from the application image.

---

# 9. Production Project Structure

```text
customer-churn-ml/
│
├── data/
│   └── raw/
│       └── churn.csv
│
├── models/
│   └── churn_pipeline.joblib
│
├── src/
│   └── churn/
│       ├── __init__.py
│       ├── preprocessing.py
│       ├── train.py
│       ├── predict.py
│       ├── evaluate.py
│       ├── drift.py
│       └── monitor.py
│
├── app/
│   ├── main.py
│   └── logging_config.py
│
├── tests/
│   ├── test_schema.py
│   ├── test_predict.py
│   ├── test_api.py
│   └── test_drift.py
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── Dockerfile
├── pyproject.toml
└── README.md
```

---

# 10. FastAPI

The trained model is exposed through FastAPI.

### Health endpoint

```text
GET /health
```

### Prediction endpoint

```text
POST /predict
```

Example request:

```json
{
  "credit_score": 650,
  "country": "Germany",
  "gender": "Female",
  "age": 45,
  "tenure": 3,
  "balance": 80000,
  "products_number": 2,
  "credit_card": 1,
  "active_member": 0,
  "estimated_salary": 70000
}
```

The API returns the predicted class and churn probability.

Pydantic validation is used to reject invalid input before it reaches the model.

---

# 11. Testing

The project includes automated tests for:

* Request schema validation
* Invalid input handling
* Prediction logic
* FastAPI endpoints
* Application startup/model loading
* Data drift calculations

Production model artifacts are not required to run the API unit tests because dependency injection and fake models are used where appropriate.

This keeps CI independent from local model files.

---

# 12. Data Drift Monitoring

A lightweight Population Stability Index (PSI) implementation was added to monitor feature distributions.

The project uses the following interpretation:

```text
PSI < 0.10       Low / little change
0.10 – 0.25      Moderate change
> 0.25            High change
```

Example simulated drift in the `age` feature produced a high PSI and triggered the monitoring alert.

Prediction drift was also considered by comparing distributions of model probabilities.

### Important distinction

Drift does not automatically mean the model needs retraining.

The monitoring system provides a signal that should lead to further investigation.

---

# 13. CI/CD

GitHub Actions is used for the CI/CD workflow.

The pipeline performs:

```text
Code push
   ↓
Run tests
   ↓
Ruff formatting check
   ↓
Ruff lint
   ↓
Build Docker image
   ↓
Authenticate with AWS using OIDC
   ↓
Push image to ECR
   ↓
Deploy to EC2 through SSM
```

The workflow is currently configured for **manual execution** so that the AWS infrastructure used for the learning exercise is not accidentally recreated or deployed to.

The complete workflow is retained in:

```text
.github/workflows/ci.yml
```

and can be manually enabled/run when needed.

---

# 14. AWS Architecture

The deployment used the following AWS services:

### Amazon S3

Stores the trained model artifact:

```text
customer-churn-ml-models-deepak/
└── models/
    └── churn_pipeline.joblib
```

### Amazon ECR

Stores the Docker image:

```text
customer-churn-api
```

### Amazon EC2

Runs the FastAPI Docker container.

### AWS Systems Manager

Used by GitHub Actions to execute deployment commands on EC2 without requiring SSH-based deployment.

### AWS IAM

IAM roles were used to implement least-privilege access.

---

# 15. GitHub OIDC Authentication

GitHub Actions does not use long-lived AWS access keys.

Instead:

```text
GitHub Actions
      ↓
GitHub OIDC token
      ↓
AWS STS
      ↓
GitHubActionsECRRole
      ↓
Temporary AWS credentials
```

The IAM trust policy restricts access to the specific GitHub repository and `main` branch.

This avoids storing permanent AWS credentials in GitHub.

---

# 16. Deployment Flow

The production-style deployment process demonstrated in this project is:

```text
Developer
    │
    │ git push
    ▼
GitHub
    │
    ▼
GitHub Actions
    │
    ├── tests
    ├── lint
    └── Docker build
            │
            ▼
          ECR
            │
            ▼
          SSM
            │
            ▼
          EC2
            │
            ├── docker pull
            ├── stop old container
            └── start new container
                    │
                    ▼
                  S3
                    │
                    ▼
             Load ML model
                    │
                    ▼
                 FastAPI
```

---

# 17. Key Engineering Decisions

### Model artifact outside Docker

The model is stored in S3 rather than baked into the Docker image.

**Reason:** separates model artifacts from application code and keeps the container image independent of the trained model.

### `handle_unknown="ignore"`

Categorical preprocessing can handle previously unseen categories.

**Reason:** prevents a new category from immediately breaking inference.

### Locked test set

The final test set was not used during model selection or threshold tuning.

**Reason:** preserves an unbiased final evaluation.

### PR-AUC

PR-AUC was emphasized alongside ROC-AUC.

**Reason:** churn is an imbalanced classification problem.

### Business threshold

The classification threshold was selected based on business cost rather than blindly using `0.5`.

**Reason:** ML predictions ultimately support business decisions.

### OIDC instead of AWS access keys

GitHub Actions uses temporary credentials.

**Reason:** avoids long-lived AWS credentials in CI/CD.

### SSM instead of SSH deployment

GitHub Actions communicates with EC2 through Systems Manager.

**Reason:** avoids managing SSH deployment keys and keeps the deployment mechanism simple.

---

# 18. What This Project Demonstrates

This project demonstrates the complete ML engineering lifecycle:

```text
Data
 ↓
Exploration
 ↓
Model development
 ↓
Evaluation
 ↓
Business threshold
 ↓
Model interpretation
 ↓
Testing
 ↓
API
 ↓
Docker
 ↓
CI/CD
 ↓
Cloud deployment
 ↓
Monitoring
```

The emphasis was on understanding how the pieces connect rather than building unnecessarily complex infrastructure.

---

# 19. Local Development

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it:

### Linux/macOS

```bash
source .venv/bin/activate
```

### Windows

```bash
.venv\Scripts\activate
```

Install the project:

```bash
pip install -e ".[dev]"
```

Run tests:

```bash
pytest
```

Run formatting check:

```bash
ruff format --check .
```

Run linting:

```bash
ruff check .
```

---

# 20. Run the API Locally

Start FastAPI:

```bash
uvicorn app.main:app --reload
```

Then open:

```text
http://localhost:8000/docs
```

The interactive Swagger UI can be used to test `/predict`.

---

# 21. Docker

Build:

```bash
docker build -t customer-churn-api .
```

Run:

```bash
docker run -p 8000:8000 \
  -e S3_BUCKET=<bucket-name> \
  -e S3_MODEL_KEY=models/churn_pipeline.joblib \
  customer-churn-api
```

---

# 22. Cleanup

The AWS infrastructure used for the deployment exercise was intentionally torn down after verification to avoid unnecessary ongoing costs.

The deployment workflow remains in the repository as a reference but is disabled from automatic execution.

---

# 23. Future Improvements

Possible future improvements include:

* Model versioning
* Automated model retraining
* Better experiment tracking
* Prediction logging
* Delayed-label performance monitoring
* Model calibration
* Automated drift alerts
* Infrastructure as Code
* More robust deployment strategies

These were intentionally kept outside the scope of this project to maintain a practical ROI-focused learning path.

---

## Conclusion

This project goes beyond training a classifier in a notebook.

It demonstrates how a machine learning model can move from:

```text
Training
   ↓
Evaluation
   ↓
Interpretation
   ↓
Testing
   ↓
API
   ↓
Container
   ↓
CI/CD
   ↓
Cloud deployment
   ↓
Monitoring
```

The result is a complete, production-oriented ML workflow while keeping the infrastructure and implementation complexity proportional to the project's goals.