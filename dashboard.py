
import streamlit as st
import requests
import plotly.graph_objects as go
import pandas as pd

st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide",
)

API_URL = "http://127.0.0.1:8000/predict"

st.title("📊 Customer Churn Prediction Dashboard")
st.caption("Predict churn risk and understand the factors influencing it.")

with st.form("customer_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Customer Details")
        tenure = st.number_input("Tenure (months)", 0, 120, 2)
        monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=85.50)
        total_charges = st.number_input("Total Charges", min_value=0.0, value=171.00)
        senior = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Partner", ["No", "Yes"])
        dependents = st.selectbox("Dependents", ["No", "Yes"])

    with col2:
        st.subheader("Account Details")
        internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        contract = st.selectbox(
            "Contract", ["Month-to-month", "One year", "Two year"]
        )
        paperless = st.selectbox("Paperless Billing", ["No", "Yes"])
        payment = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Credit card (automatic)",
                "Bank transfer (automatic)",
            ],
        )

    with col3:
        st.subheader("Additional Services")
        if internet == "No":
            security = backup = protection = support = tv = movies = "No internet service"
            st.info("Internet-dependent services set to No internet service.")
        else:
            security = st.selectbox("Online Security", ["No", "Yes"])
            backup = st.selectbox("Online Backup", ["No", "Yes"])
            protection = st.selectbox("Device Protection", ["No", "Yes"])
            support = st.selectbox("Tech Support", ["No", "Yes"])
            tv = st.selectbox("Streaming TV", ["No", "Yes"])
            movies = st.selectbox("Streaming Movies", ["No", "Yes"])

    submitted = st.form_submit_button("Predict Churn", use_container_width=True)


def is_yes(value):
    return int(value == "Yes")


def is_no_internet(value):
    return int(value == "No internet service")


if submitted:
    payload = {
        "SeniorCitizen": is_yes(senior),
        "tenure": tenure,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "Partner_Yes": is_yes(partner),
        "Dependents_Yes": is_yes(dependents),
        "InternetService_Fiber optic": int(internet == "Fiber optic"),
        "InternetService_No": int(internet == "No"),
        "OnlineSecurity_No internet service": is_no_internet(security),
        "OnlineSecurity_Yes": is_yes(security),
        "OnlineBackup_No internet service": is_no_internet(backup),
        "OnlineBackup_Yes": is_yes(backup),
        "DeviceProtection_No internet service": is_no_internet(protection),
        "DeviceProtection_Yes": is_yes(protection),
        "TechSupport_No internet service": is_no_internet(support),
        "TechSupport_Yes": is_yes(support),
        "StreamingTV_No internet service": is_no_internet(tv),
        "StreamingTV_Yes": is_yes(tv),
        "StreamingMovies_No internet service": is_no_internet(movies),
        "StreamingMovies_Yes": is_yes(movies),
        "Contract_One year": int(contract == "One year"),
        "Contract_Two year": int(contract == "Two year"),
        "PaperlessBilling_Yes": is_yes(paperless),
        "PaymentMethod_Credit card (automatic)": int(
            payment == "Credit card (automatic)"
        ),
        "PaymentMethod_Electronic check": int(payment == "Electronic check"),
        "PaymentMethod_Mailed check": int(payment == "Mailed check"),
    }

    try:
        with st.spinner("Predicting churn..."):
            response = requests.post(
                API_URL, json={"data": payload}, timeout=30
            )
            response.raise_for_status()
            result = response.json()

        probability = float(result["churn_probability"])
        prediction = int(result["churn_prediction"])

        if not 0 <= probability <= 1 or prediction not in (0, 1):
            raise ValueError("Invalid prediction values returned by the API.")

        st.divider()
        st.subheader("Prediction Results")

        c1, c2, c3 = st.columns(3)
        c1.metric("Churn Probability", f"{probability:.2%}")
        c2.metric("Prediction", "Likely to Churn" if prediction else "Likely to Stay")
        c3.metric("Risk Status", result.get("risk_status", "Unknown"))

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={"suffix": "%"},
            title={"text": "Churn Risk"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#dc2626" if prediction else "#16a34a"},
                "steps": [
                    {"range": [0, 50], "color": "#dcfce7"},
                    {"range": [50, 100], "color": "#fee2e2"},
                ],
            },
        ))
        fig.update_layout(height=280)
        st.plotly_chart(fig, use_container_width=True)

        reasons = result.get("top_churn_reasons", [])
        if prediction and reasons:
            st.subheader("Why the Model Predicts Churn")

            reason_df = pd.DataFrame(reasons)
            fig = go.Figure(go.Bar(
                x=reason_df["contribution"],
                y=reason_df["feature"],
                orientation="h",
                marker_color="#dc2626",
                text=reason_df["contribution"].round(4),
                textposition="auto",
            ))
            fig.update_layout(
                xaxis_title="SHAP contribution",
                yaxis_title="Feature",
                yaxis={"autorange": "reversed"},
                height=max(280, 55 * len(reason_df)),
            )
            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(
                reason_df[["feature", "value", "contribution", "effect"]],
                hide_index=True,
                use_container_width=True,
            )
        elif prediction:
            st.info("No explanation available. Check SHAP support in the API.")

        protective = result.get("factors_reducing_churn_risk", [])
        if protective:
            st.subheader("Factors Reducing Churn Risk")
            st.dataframe(pd.DataFrame(protective), hide_index=True, use_container_width=True)

        if prediction:
            st.subheader("Retention Suggestions")
            suggestions = result.get("retention_suggestions", [])
            if suggestions:
                for suggestion in suggestions:
                    st.markdown(f"- {suggestion}")
            else:
                st.write("Contact the customer to understand their concerns.")
        else:
            st.success("The model predicts low churn risk.")

        if result.get("explanation_note"):
            st.caption(result["explanation_note"])

    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to the API. Run `uvicorn app:app --reload` first.")
    except requests.exceptions.RequestException as exc:
        st.error(f"API request failed: {exc}")
    except (KeyError, TypeError, ValueError) as exc:
        st.error(f"Invalid API response: {exc}")
