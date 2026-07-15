"""
Example: loading the exported pipeline and scoring new, raw customer records.
No manual preprocessing needed -- the pipeline handles imputation, scaling,
and one-hot encoding internally.
"""

import joblib
import pandas as pd

# Load the full pipeline (preprocessing + trained model) in one shot
pipeline = joblib.load("churn_pipeline.joblib")

# A couple of raw, unprocessed customer records (same schema as the training CSV,
# minus customerID and Churn)
new_customers = pd.DataFrame([
    {
        "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
        "tenure": 1, "PhoneService": "No", "MultipleLines": "No phone service",
        "InternetService": "DSL", "OnlineSecurity": "No", "OnlineBackup": "Yes",
        "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
        "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check", "MonthlyCharges": 29.85, "TotalCharges": 29.85,
    },
    {
        "gender": "Male", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
        "tenure": 60, "PhoneService": "Yes", "MultipleLines": "Yes",
        "InternetService": "Fiber optic", "OnlineSecurity": "Yes", "OnlineBackup": "Yes",
        "DeviceProtection": "Yes", "TechSupport": "Yes", "StreamingTV": "Yes",
        "StreamingMovies": "Yes", "Contract": "Two year", "PaperlessBilling": "No",
        "PaymentMethod": "Bank transfer (automatic)", "MonthlyCharges": 95.5, "TotalCharges": 5730.2,
    },
])

predictions = pipeline.predict(new_customers)
probabilities = pipeline.predict_proba(new_customers)[:, 1]

for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
    label = "Churn" if pred == 1 else "No Churn"
    print(f"Customer {i}: {label}  (churn probability: {prob:.2%})")
