
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests

BASE_DIR = Path(__file__).resolve().parent
API_URL = "http://127.0.0.1:8000/predict"

MODEL_PATH = BASE_DIR / "optimized_churn_model.pkl"
FEATURE_PATHS = [
    BASE_DIR / "model_features.pkl",
    BASE_DIR / "model_featues.pkl"
]

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

customer = {
    "SeniorCitizen": 0,
    "tenure": 2,
    "MonthlyCharges": 85.5,
    "TotalCharges": 171.0,
    "Partner_Yes": 0,
    "Dependents_Yes": 0,
    "InternetService_Fiber optic": 1,
    "InternetService_No": 0,
    "OnlineSecurity_No internet service": 0,
    "OnlineSecurity_Yes": 0,
    "OnlineBackup_No internet service": 0,
    "OnlineBackup_Yes": 0,
    "DeviceProtection_No internet service": 0,
    "DeviceProtection_Yes": 0,
    "TechSupport_No internet service": 0,
    "TechSupport_Yes": 0,
    "StreamingTV_No internet service": 0,
    "StreamingTV_Yes": 1,
    "StreamingMovies_No internet service": 0,
    "StreamingMovies_Yes": 1,
    "Contract_One year": 0,
    "Contract_Two year": 0,
    "PaperlessBilling_Yes": 1,
    "PaymentMethod_Credit card (automatic)": 0,
    "PaymentMethod_Electronic check": 1,
    "PaymentMethod_Mailed check": 0
}

missing = [f for f in model_features if f not in customer]

if missing:
    raise ValueError(f"Test customer is missing features: {missing}")

customer = {f: customer[f] for f in model_features}

df = pd.DataFrame([customer], columns=model_features)
df = df.apply(pd.to_numeric, errors="raise")

if not np.isfinite(df.to_numpy(dtype=float)).all():
    raise ValueError("Test input contains invalid numeric values.")

local_prediction = int(model.predict(df)[0])

if not hasattr(model, "predict_proba"):
    raise TypeError("Model does not support predict_proba().")

probabilities = model.predict_proba(df)[0]
classes = list(model.classes_)

if 1 not in classes:
    raise ValueError(f"Churn class 1 not found. Model classes: {classes}")

local_probability = float(probabilities[classes.index(1)])

print("\nLOCAL MODEL RESULT")
print("------------------")
print("Prediction:", local_prediction)
print("Churn probability:", round(local_probability, 4))

try:
    response = requests.post(
        API_URL,
        json={"data": customer},
        timeout=15
    )
except requests.RequestException as exc:
    raise SystemExit(
        f"Could not connect to the API. Is Uvicorn running? Details: {exc}"
    )

print("\nAPI HTTP STATUS:", response.status_code)

if response.status_code != 200:
    print("API ERROR:", response.text)
    raise SystemExit("API test failed.")

api_result = response.json()

print("\nAPI RESULT")
print("----------")
print(api_result)

api_prediction = api_result.get("churn_prediction")
api_probability = api_result.get("churn_probability")

if api_prediction is None or api_probability is None:
    raise SystemExit("API response is missing prediction or probability.")

prediction_matches = int(api_prediction) == local_prediction
probability_matches = np.isclose(
    float(api_probability),
    local_probability,
    atol=0.00011,
    rtol=0
)

print("\nVERIFICATION")
print("------------")
print("Prediction matches:", prediction_matches)
print("Probability matches:", probability_matches)

if prediction_matches and probability_matches:
    print("\nPASS: API and local model predictions match.")
else:
    print("\nFAIL: API and local model predictions differ.")
    raise SystemExit(1)
