from pathlib import Path
import logging

import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "optimized_churn_model.pkl"
FEATURE_PATHS = [
    BASE_DIR / "model_features.pkl",
    BASE_DIR / "model_featues.pkl",
]

# Load model and feature names
model = joblib.load(MODEL_PATH)

feature_path = next((p for p in FEATURE_PATHS if p.is_file()), None)
if feature_path is None:
    raise FileNotFoundError("Model feature file not found.")

saved_features = joblib.load(feature_path)

if hasattr(model, "feature_names_in_"):
    model_features = list(model.feature_names_in_)
elif isinstance(saved_features, pd.DataFrame):
    model_features = saved_features.columns.tolist()
else:
    model_features = list(saved_features)

app = FastAPI(
    title="Customer Churn Prediction API",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

def explain_churn(df):
    """Return the features pushing the prediction toward or away from churn."""
    try:
        explainer = shap.TreeExplainer(model)
        values = explainer.shap_values(df)

        if isinstance(values, list):
            values = values[1] if len(values) > 1 else values[0]
        elif hasattr(values, "values"):
            values = values.values

        values = np.asarray(values)

        if values.ndim == 3:
            class_index = (
                list(model.classes_).index(1)
                if 1 in list(model.classes_) else 0
            )
            if values.shape[1] == len(model_features):
                values = values[0, :, class_index]
            else:
                values = values[0, class_index, :]
        elif values.ndim == 2:
            values = values[0]

        if values.ndim != 1 or len(values) != len(model_features):
            raise ValueError("Unexpected SHAP output shape.")

        contributions = [
            {
                "feature": feature,
                "value": float(df.iloc[0][feature]),
                "contribution": round(float(score), 4),
                "effect": (
                    "increases churn risk" if score > 0
                    else "decreases churn risk" if score < 0
                    else "no effect"
                ),
            }
            for feature, score in zip(model_features, values)
            if np.isfinite(score)
        ]

        contributions.sort(
            key=lambda item: abs(item["contribution"]),
            reverse=True,
        )

        reasons = [
            item for item in contributions
            if item["contribution"] > 0
        ][:5]

        protective_factors = [
            item for item in contributions
            if item["contribution"] < 0
        ][:5]

        return reasons, protective_factors

    except Exception:
        logger.exception("SHAP explanation failed.")
        return [], []


def retention_suggestions(reasons):
    suggestions = {
        "Contract_Month-to-month":
            "Offer a suitable longer-term contract option.",
        "MonthlyCharges":
            "Review the bill and discuss suitable lower-cost plans.",
        "tenure":
            "Consider an onboarding or loyalty offer.",
        "InternetService_Fiber optic":
            "Review service quality, pricing, and customer satisfaction.",
        "PaymentMethod_Electronic check":
            "Offer convenient alternative payment methods.",
        "TechSupport_Yes":
            "Check whether the customer's support issues are resolved.",
        "OnlineSecurity_Yes":
            "Review satisfaction with the security service.",
        "OnlineBackup_Yes":
            "Review satisfaction with the backup service.",
    }

    results = [
        suggestions[item["feature"]]
        for item in reasons
        if item["feature"] in suggestions
    ]

    return list(dict.fromkeys(results)) or [
        "Contact the customer to understand their concerns."
    ]


@app.get("/")
def health_check():
    return {"status": "active", "model_loaded": True}


@app.get("/features")
def get_features():
    return {"features": model_features}


@app.post("/predict")
def predict_churn(payload: dict = Body(...)):
    customer = payload.get("data", payload)

    if not isinstance(customer, dict) or not customer:
        raise HTTPException(
            status_code=422,
            detail="Provide customer feature data.",
        )

    valid_data = {
        key: value for key, value in customer.items()
        if key in model_features
    }

    if not valid_data:
        raise HTTPException(
            status_code=422,
            detail="No supplied features match the model.",
        )

    df = pd.DataFrame([valid_data]).reindex(
        columns=model_features,
        fill_value=0,
    )

    try:
        df = df.apply(pd.to_numeric, errors="raise")
    except (ValueError, TypeError) as exc:
        raise HTTPException(
            status_code=422,
            detail="All features must be numeric and correctly encoded.",
        ) from exc

    if not np.isfinite(df.to_numpy(dtype=float)).all():
        raise HTTPException(
            status_code=422,
            detail="Feature values cannot be NaN or infinite.",
        )

    try:
        prediction = int(model.predict(df)[0])
        classes = list(model.classes_)
        probabilities = model.predict_proba(df)[0]

        if 1 not in classes:
            raise HTTPException(
                status_code=500,
                detail="Churn class 1 not found in model.",
            )

        probability = float(probabilities[classes.index(1)])
        reasons, protective_factors = explain_churn(df)

        return {
            "churn_prediction": prediction,
            "churn_probability": round(probability, 4),
            "risk_status": (
                "High Risk" if probability >= 0.5 else "Low Risk"
            ),
            "top_churn_reasons": reasons if prediction == 1 else [],
            "factors_reducing_churn_risk": protective_factors,
            "retention_suggestions": (
                retention_suggestions(reasons) if prediction == 1 else []
            ),
            "explanation_note": (
                "These factors explain the model's prediction; "
                "they do not prove the customer's actual reasons for leaving."
            ),
        }

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Prediction failed.")
        raise HTTPException(
            status_code=500,
            detail="Prediction failed. Check the API terminal.",
        ) from exc

