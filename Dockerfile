FROM python:3.11-slim

WORKDIR /app

# 1. Copy ONLY the config file first
COPY pyproject.toml .

# 1.5. Create a fake src folder
RUN mkdir src/

# 2. Heavy Lifting: Download and cache all ML dependencies from your toml
RUN pip install --no-cache-dir .

# 3. Copy your live, changing code and files AFTER dependencies are installed
COPY src/ ./src/
COPY app/ ./app/
COPY models/ ./models/

# 4. Instant Link: Register your 'src' package locally without re-downloading anything
RUN pip install --no-cache-dir --no-deps .

EXPOSE 8000

# 5. Run Uvicorn pointing to main.py inside the app directory
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
