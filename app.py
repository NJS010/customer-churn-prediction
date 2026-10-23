
from pathlib import Path
from typing import Any
import logging

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "optimized_churn_model.pkl"
FEATURE_PATHS = [
    BASE_DIR / "model_featues.pkl",
    BASE_DIR / "model_featues.pkl"
]

app = FastAPI(
    title="Customer Churn Prediction API",
    version="1.0.0",
    description="API for predicting telecom customer churn."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"]
)

try:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    model = joblib.load(MODEL_PATH)

    feature_path = next(
        (p for p in FEATURE_PATHS if p.is_file()),
        None
    )

    if feature_path is None:
        raise FileNotFoundError("Model feature file was not found.")

    saved_features = joblib.load(feature_path)

    if hasattr(model, "feature_names_in_"):
        model_features = list(model.feature_names_in_)
    elif isinstance(saved_features, pd.DataFrame):
        model_features = saved_features.columns.tolist()
    elif isinstance(saved_features, pd.Index):
        model_features = saved_features.tolist()
    elif isinstance(saved_features, dict):
        model_features = list(saved_features.keys())
    else:
        model_features = list(saved_features)

    if not model_features:
        raise ValueError("The model feature list is empty.")

    if len(model_features) != len(set(model_features)):
        raise ValueError("Duplicate model features were found.")

    if not callable(getattr(model, "predict", None)):
        raise TypeError("Loaded model does not support predict().")

    logger.info("Model artifacts loaded successfully.")
    logger.info("Number of model features: %d", len(model_features))

except Exception:
    logger.exception("Model loading failed.")
    raise


@app.get("/")
def health_check():
    return {
        "status": "active",
        "message": "Customer churn prediction API is running.",
        "model_loaded": True,
        "feature_count": len(model_features)
    }


@app.get("/features")
def get_features():
    return {
        "feature_count": len(model_features),
        "features": model_features
    }


@app.post("/predict")
def predict_churn(payload: dict[str, Any] = Body(...)):
    try:
        if not isinstance(payload, dict) or not payload:
            raise HTTPException(
                status_code=422,
                detail="Provide a non-empty JSON object."
            )

        if isinstance(payload.get("data"), dict):
            customer_data = payload["data"]
        else:
            customer_data = payload

        if not customer_data:
            raise HTTPException(
                status_code=422,
                detail="Customer feature data cannot be empty."
            )

        unknown_features = [
            name for name in customer_data
            if name not in model_features
        ]

        if unknown_features:
            logger.warning(
                "Ignoring unknown input features: %s",
                unknown_features
            )

        valid_data = {
            name: value
            for name, value in customer_data.items()
            if name in model_features
        }

        if not valid_data:
            raise HTTPException(
                status_code=422,
                detail={
                    "message": "No supplied features match the model.",
                    "expected_features": model_features,
                    "received_features": list(customer_data.keys())
                }
            )

        df = pd.DataFrame([valid_data])
        df = df.reindex(columns=model_features, fill_value=0)

        try:
            df = df.apply(pd.to_numeric, errors="raise")
        except (ValueError, TypeError) as exc:
            raise HTTPException(
                status_code=422,
                detail=(
                    "Feature values must be numeric and use the "
                    "same encoding as the training data."
                )
            ) from exc

        if not np.isfinite(df.to_numpy(dtype=float)).all():
            raise HTTPException(
                status_code=422,
                detail="Feature values cannot be NaN or infinite."
            )

        prediction = int(model.predict(df)[0])

        if not callable(getattr(model, "predict_proba", None)):
            raise HTTPException(
                status_code=500,
                detail="The model does not support predict_proba()."
            )

        probabilities = model.predict_proba(df)[0]
        classes = list(model.classes_)

        if 1 not in classes:
            raise HTTPException(
                status_code=500,
                detail={
                    "message": "Churn class 1 was not found in the model.",
                    "model_classes": [str(c) for c in classes]
                }
            )

        probability = float(probabilities[classes.index(1)])

        return {
            "churn_prediction": prediction,
            "churn_probability": round(probability, 4),
            "risk_status": (
                "High Risk" if probability >= 0.5 else "Low Risk"
            ),
            "status_code": 200
        }

    except HTTPException:
        raise

    except Exception:
        logger.exception("Prediction failed.")
        raise HTTPException(
            status_code=500,
            detail="Prediction failed. Check the terminal logs."
        )
