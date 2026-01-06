"""FastAPI application for serving heart disease predictions.

This module defines a simple REST API using FastAPI. It exposes a `/predict`
endpoint that accepts patient health data as JSON, applies the same
preprocessing pipeline used during training, and returns the predicted
probability of heart disease along with the binary prediction.

The app logs incoming requests and predictions using the standard
logging module. To run the API locally:

```bash
uvicorn app.main:app --reload
```

When containerized, the Dockerfile will start the server using gunicorn.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import numpy as np
import logging

import time
from starlette.responses import Response
from starlette.middleware.base import BaseHTTPMiddleware

from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load model and preprocessor at startup
MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
MODEL_FILE = None
PREPROCESSOR_FILE = None

# ---------------------------
# Prometheus Monitoring
# ---------------------------

# General HTTP metrics
HTTP_REQUESTS_TOTAL = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "path", "status_code"],
)

HTTP_REQUEST_DURATION_SECONDS = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency in seconds",
    ["method", "path"],
    # buckets tuned for APIs; adjust if needed
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10),
)

# App-specific metrics for predictions
PREDICTIONS_TOTAL = Counter(
    "heart_disease_predictions_total",
    "Total predictions made",
    ["result"],  # "success" or "error"
)

PREDICT_DURATION_SECONDS = Histogram(
    "heart_disease_predict_duration_seconds",
    "Latency for /predict handler in seconds",
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5),
)



def _find_latest_model(directory: Path) -> Path:
    """Return the path to the first model file found in the directory."""
    candidates = sorted(directory.glob("*_model.pkl"))
    if not candidates:
        raise FileNotFoundError("No model file found in the models directory.")
    return candidates[0]


def load_assets():
    """Load the model and preprocessor from disk into global variables."""
    global MODEL_FILE, PREPROCESSOR_FILE, model, preprocessor
    model_path = _find_latest_model(MODEL_DIR)
    preproc_path = MODEL_DIR / "preprocessor.pkl"
    MODEL_FILE = model_path
    PREPROCESSOR_FILE = preproc_path
    model = joblib.load(model_path)
    preprocessor = joblib.load(preproc_path)
    logger.info("Model loaded from %s", model_path)
    logger.info("Preprocessor loaded from %s", preproc_path)


# Define request schema using Pydantic
class HeartData(BaseModel):
    """Schema for incoming patient data. Fields mirror the dataset columns."""

    Age: int = Field(..., example=63)
    Sex: str = Field(..., example="M")
    ChestPainType: str = Field(..., example="ATA")
    RestingBP: int = Field(..., example=145)
    Cholesterol: int = Field(..., example=233)
    FastingBS: int = Field(..., example=1)
    RestingECG: str = Field(..., example="Normal")
    MaxHR: int = Field(..., example=150)
    ExerciseAngina: str = Field(..., example="N")
    Oldpeak: float = Field(..., example=2.3)
    ST_Slope: str = Field(..., example="Up")


# Initialize FastAPI
app = FastAPI(title="Heart Disease Prediction API", version="1.0.0")


class PrometheusMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        # NOTE: avoid high-cardinality labels. Keep 'path' stable.
        # Using request.url.path is fine here because your routes are fixed (/predict, /).
        method = request.method
        path = request.url.path

        start = time.perf_counter()
        status_code = 500

        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        except Exception:
            # will be handled by FastAPI exception handlers; still record as 500
            raise
        finally:
            elapsed = time.perf_counter() - start
            HTTP_REQUESTS_TOTAL.labels(method=method, path=path, status_code=str(status_code)).inc()
            HTTP_REQUEST_DURATION_SECONDS.labels(method=method, path=path).observe(elapsed)


# Register middleware
app.add_middleware(PrometheusMiddleware)


@app.get("/metrics")
async def metrics():
    """Prometheus scrape endpoint."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.on_event("startup")
async def startup_event() -> None:
    """Load the model and preprocessing pipeline on startup."""
    try:
        load_assets()
    except Exception as exc:
        logger.exception("Failed to load model and preprocessor: %s", exc)
        raise


@app.get("/")
async def read_root() -> dict[str, str]:
    """Health check endpoint."""
    return {"message": "Heart Disease Prediction API is running"}


@app.post("/predict")
async def predict(data: HeartData) -> dict[str, float | int]:
    """Predict the probability of heart disease for a single patient.

    Args:
        data: Patient information as a HeartData object.

    Returns:
        A dictionary containing the predicted class (0/1) and probability of heart disease.
    """
    start = time.perf_counter()
    try:
        # Convert input data to DataFrame with one row
        input_df = data.model_dump()
        input_df = {k: [v] for k, v in input_df.items()}
        import pandas as pd  # Local import to avoid circular dependency
        df = pd.DataFrame.from_dict(input_df)
        # Preprocess features
        X_processed = preprocessor.transform(df)
        # Predict probability
        prob = model.predict_proba(X_processed)[0][1]
        pred = int(model.predict(X_processed)[0])
        logger.info("Prediction made successfully")
        return {"prediction": pred, "probability": float(prob)}
    except Exception as exc:
        PREDICTIONS_TOTAL.labels(result="error").inc()
        logger.exception("Error during prediction: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        PREDICT_DURATION_SECONDS.observe(time.perf_counter() - start)