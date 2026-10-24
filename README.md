# Telecom Customer Churn Prediction

## About the Project

This project predicts whether a telecom customer is likely to leave the company using machine learning.

I built a web application where users can enter customer details and get a churn prediction, churn probability, risk level, and SHAP-based explanations. The project combines a machine learning model, a FastAPI backend, and a Streamlit dashboard.

## Features

- Predicts whether a customer is likely to leave the company.
- Displays the predicted churn probability.
- Shows the customer's risk level.
- Uses SHAP to explain which features influence the model's prediction.
- Provides an interactive dashboard for entering customer details and viewing results.
- Uses FastAPI to serve predictions through an API.
- Uses Pydantic to validate API input data.
- Connects the dashboard to the backend API.
- Supports local execution using Docker Compose.
- Deployed online using Render.

## Technologies Used

- **Python** – Main programming language.
- **Pandas and NumPy** – Data handling and numerical operations.
- **Scikit-learn** – Machine learning utilities.
- **XGBoost** – Machine learning algorithm.
- **SHAP** – Explaining model predictions.
- **FastAPI** – Building the prediction API.
- **Pydantic** – Validating API input data.
- **Streamlit** – Building the interactive dashboard.
- **Plotly** – Creating interactive charts.
- **Matplotlib** – Data visualization.
- **Joblib** – Saving and loading model artifacts.
- **Requests** – Sending requests from the dashboard to the API.
- **Uvicorn** – Running the FastAPI application.
- **Docker and Docker Compose** – Containerizing and running the application.
- **Render** – Deploying the application online.
- **Git and GitHub** – Version control and project hosting.

## How SHAP Is Used

SHAP stands for SHapley Additive exPlanations. It helps explain how different customer features influence the machine learning model's predictions.

SHAP explanations can show which features contribute toward or away from a churn prediction. This makes the model easier to interpret instead of showing only a prediction and probability.

These explanations describe the model's reasoning; they do not prove the actual reason a particular customer will leave.

## Live Demo

- **Dashboard:** https://customer-churn-dashboard-92o1.onrender.com
- **API:** https://customer-churn-prediction-205m.onrender.com
- **API Documentation:** https://customer-churn-prediction-205m.onrender.com/docs
- **GitHub Repository:** https://github.com/NJS010/customer-churn-prediction

## How to Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/NJS010/customer-churn-prediction.git
cd customer-churn-prediction
```

### 2. Install the dependencies

```bash
pip install -r requirements.txt
```

Make sure the required model and feature files are available in the project directory.

### 3. Start the FastAPI backend

```bash
uvicorn app:app --reload
```

Open the API documentation in your browser:

http://127.0.0.1:8000/docs

### 4. Start the Streamlit dashboard

Open another terminal in the project directory and run:

```bash
export API_URL=http://127.0.0.1:8000/predict
streamlit run dashboard.py
```

Open the local dashboard URL displayed in the terminal.

### 5. Run using Docker Compose

If Docker and Docker Compose are installed and the required configuration files are included in the repository, run:

```bash
docker compose up --build
```

To stop the application, run:

```bash
docker compose down
```

## Testing

I tested the application locally and deployed it using Render. I tried different customer profiles and confirmed that the API returned predictions and probabilities and that the dashboard displayed the results successfully.

## What I Learned

- Building a machine learning solution for customer churn prediction.
- Working with XGBoost and machine learning tools.
- Using SHAP to interpret model predictions.
- Building an API with FastAPI and validating inputs with Pydantic.
- Creating an interactive dashboard using Streamlit.
- Creating visualizations with Plotly and Matplotlib.
- Connecting a frontend dashboard to a backend API.
- Managing model files and Python dependencies.
- Running an application using Docker and Docker Compose.
- Deploying a machine learning application using Render.

## Conclusion

This project helped me understand how machine learning can be used to predict customer churn and how SHAP can make predictions easier to interpret. It also gave me practical experience in building, testing, connecting, and deploying a complete machine learning web application.
