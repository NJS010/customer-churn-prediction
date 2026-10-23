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

model = joblib.load(MODEL_PATH)
feature_path = next((p for p in FEATURE_PATHS if p.is_file()), None)

if feature_path is None:
    raise FileNotFoundError("Model feature file not found.")

saved_features = joblib.load(feature_path)
model_features = (
    list(model.feature_names_in_)
    if hasattr(model, "feature_names_in_")
    else saved_features.columns.tolist()
    if isinstance(saved_features, pd.DataFrame)
    else list(saved_features)
)

app = FastAPI(title="Customer Churn Prediction API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def explain_churn(df):
    try:
        values = shap.TreeExplainer(model).shap_values(df)

        if isinstance(values, list):
            values = values[1] if len(values) > 1 else values[0]
        elif hasattr(values, "values"):
            values = values.values

        values = np.asarray(values)

        if values.ndim == 3:
            class_index = list(model.classes_).index(1)
            values = (
                values[0, :, class_index]
                if values.shape[1] == len(model_features)
                else values[0, class_index, :]
            )
        elif values.ndim == 2:
            values = values[0]

        if values.ndim != 1 or len(values) != len(model_features):
            raise ValueError("Unexpected SHAP output shape.")

        contributions = [
            {
                "feature": feature,
                "value": float(df.iloc[0][feature]),
                "contribution": round(float(score), 4),
                "effect": "increases churn risk" if score > 0
                else "decreases churn risk" if score < 0 else "no effect",
            }
            for feature, score in zip(model_features, values)
            if np.isfinite(score)
        ]

        contributions.sort(
            key=lambda item: abs(item["contribution"]), reverse=True
        )
        return (
            [x for x in contributions if x["contribution"] > 0][:5],
            [x for x in contributions if x["contribution"] < 0][:5],
        )

    except Exception:
        logger.exception("SHAP explanation failed.")
        return [], []


def retention_suggestions(reasons):
    suggestions = {
        "Contract_Month-to-month": "Offer a suitable longer-term contract.",
        "MonthlyCharges": "Review the bill and discuss lower-cost plans.",
        "tenure": "Consider an onboarding or loyalty offer.",
        "InternetService_Fiber optic": "Review service quality and pricing.",
        "PaymentMethod_Electronic check": "Offer convenient alternative payment methods.",
        "TechSupport_Yes": "Check whether the customer's support issues are resolved.",
        "OnlineSecurity_Yes": "Review satisfaction with the security service.",
        "OnlineBackup_Yes": "Review satisfaction with the backup service.",
    }
    results = list(dict.fromkeys(
        suggestions[x["feature"]]
        for x in reasons if x["feature"] in suggestions
    ))
    return results or ["Contact the customer to understand their concerns."]


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
        raise HTTPException(422, "Provide customer feature data.")

    unknown = set(customer) - set(model_features)
    if unknown:
        raise HTTPException(
            422, detail=f"Unknown features: {sorted(unknown)}"
        )

    missing = set(model_features) - set(customer)
    if missing:
        raise HTTPException(
            422, detail=f"Missing features: {sorted(missing)}"
        )

    try:
        df = pd.DataFrame(
            [{feature: customer[feature] for feature in model_features}]
        ).apply(pd.to_numeric, errors="raise")
    except (ValueError, TypeError) as exc:
        raise HTTPException(422, "All features must be numeric.") from exc

    if not np.isfinite(df.to_numpy(dtype=float)).all():
        raise HTTPException(422, "Feature values cannot be NaN or infinite.")

    if (df["tenure"] < 0).any() or (df["tenure"] > 120).any():
        raise HTTPException(422, "Tenure must be between 0 and 120 months.")

    if (df["MonthlyCharges"] < 0).any() or (df["TotalCharges"] < 0).any():
        raise HTTPException(422, "Charges cannot be negative.")

    try:
        prediction = int(model.predict(df)[0])
        classes = list(model.classes_)
        probabilities = model.predict_proba(df)[0]

        if 1 not in classes:
            raise HTTPException(500, "Churn class 1 not found in model.")

        probability = float(probabilities[classes.index(1)])
        reasons, protective_factors = explain_churn(df)

        return {
            "churn_prediction": prediction,
            "churn_probability": round(probability, 4),
            "risk_status": "High Risk" if probability >= 0.5 else "Low Risk",
            "top_churn_reasons": reasons if prediction == 1 else [],
            "factors_reducing_churn_risk": protective_factors,
            "retention_suggestions": (
                retention_suggestions(reasons) if prediction == 1 else []
            ),
            "explanation_note": (
                "These factors explain the model's prediction; they do not "
                "prove the customer's actual reasons for leaving."
            ),
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Prediction failed.")
        raise HTTPException(500, "Prediction failed. Check the API terminal.") from exc