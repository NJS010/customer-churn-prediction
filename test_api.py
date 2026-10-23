import requests

API_URL = "http://127.0.0.1:8000/predict"

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
    "PaymentMethod_Mailed check": 0,
}

tests = [
    (
        "Valid customer",
        lambda: requests.post(
            API_URL, json={"data": customer}, timeout=10
        ),
        lambda r: r.status_code == 200
        and r.json().get("churn_prediction") in (0, 1)
        and 0 <= float(r.json().get("churn_probability", -1)) <= 1,
    ),
    (
        "Empty request",
        lambda: requests.post(API_URL, json={}, timeout=10),
        lambda r: r.status_code in (400, 422),
    ),
    (
        "Missing customer features",
        lambda: requests.post(
            API_URL, json={"data": {"tenure": 12}}, timeout=10
        ),
        lambda r: r.status_code in (400, 422),
    ),
    (
        "Invalid numeric value",
        lambda: requests.post(
            API_URL, json={"data": {"tenure": "invalid"}}, timeout=10
        ),
        lambda r: r.status_code in (400, 422),
    ),
    (
        "Negative tenure",
        lambda: requests.post(
            API_URL,
            json={"data": {**customer, "tenure": -5}},
            timeout=10,
        ),
        lambda r: r.status_code in (400, 422),
    ),
    (
        "Negative monthly charges",
        lambda: requests.post(
            API_URL,
            json={"data": {**customer, "MonthlyCharges": -50}},
            timeout=10,
        ),
        lambda r: r.status_code in (400, 422),
    ),
    (
        "Null request data",
        lambda: requests.post(
            API_URL, json={"data": None}, timeout=10
        ),
        lambda r: r.status_code in (400, 422),
    ),
    (
        "Unknown feature",
        lambda: requests.post(
            API_URL,
            json={"data": {**customer, "UnknownFeature": 123}},
            timeout=10,
        ),
        lambda r: r.status_code in (400, 422),
    ),
    (
        "Malformed JSON",
        lambda: requests.post(
            API_URL,
            data='{"data" {"tenure": 12}}',
            headers={"Content-Type": "application/json"},
            timeout=10,
        ),
        lambda r: r.status_code == 422,
    ),
    (
        "API documentation available",
        lambda: requests.get(
            "http://127.0.0.1:8000/docs", timeout=10
        ),
        lambda r: r.status_code == 200,
    ),
]

passed = 0
failed = 0

print("\nCUSTOMER CHURN API TESTS")
print("=" * 45)

try:
    requests.get("http://127.0.0.1:8000/docs", timeout=3)
except requests.RequestException:
    print("API is not running. Start it with:")
    print("uvicorn app:app --reload")
    raise SystemExit(1)

for name, send_request, check in tests:
    try:
        response = send_request()
        success = check(response)

        if success:
            print(f"PASS: {name}")
            passed += 1
        else:
            print(f"FAIL: {name} (HTTP {response.status_code})")
            print("      Response:", response.text[:300])
            failed += 1

    except (requests.RequestException, ValueError, TypeError, AttributeError) as exc:
        print(f"FAIL: {name} ({exc})")
        failed += 1

print("=" * 45)
print(f"Passed: {passed}/{len(tests)}")
print(f"Failed: {failed}/{len(tests)}")

if failed:
    print("Some tests failed. Review the results above.")
    raise SystemExit(1)

print("All tests passed!")
