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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load model and preprocessor at startup
MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
MODEL_FILE = None
PREPROCESSOR_FILE = None

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
        logger.exception("Error during prediction: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))