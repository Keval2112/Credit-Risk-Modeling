# app.py

import streamlit as st
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder

# Load the trained LightGBM model
model = joblib.load("credit_risk_model.pkl") # Make sure you've saved model as this

st.title("Credit Risk Prediction App")
st.write("Enter loan application details to predict if the loan will be approved.")

# User input form
gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])
applicant_income = st.number_input("Applicant Income", min_value=0)
coapplicant_income = st.number_input("Coapplicant Income", min_value=0)
loan_amount = st.number_input("Loan Amount (in thousands)", min_value=0)
loan_term = st.selectbox("Loan Amount Term", [12, 36, 60, 84, 120, 180, 240, 300, 360, 480])
credit_history = st.selectbox("Credit History", [1.0, 0.0])
property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

# Preprocess input
total_income = applicant_income + coapplicant_income
loan_to_income = loan_amount / total_income if total_income != 0 else 0

# Encode categorical inputs (match the label encoding used in training)
gender = 1 if gender == "Male" else 0
married = 1 if married == "Yes" else 0
education = 1 if education == "Graduate" else 0
self_employed = 1 if self_employed == "Yes" else 0
property_area_map = {"Urban": 2, "Semiurban": 1, "Rural": 0}
property_area = property_area_map[property_area]
dependents_map = {"0": 0, "1": 1, "2": 2, "3+": 3}
dependents = dependents_map[dependents]

# Create input DataFrame
input_data = pd.DataFrame({
    'Gender': [gender],
    'Married': [married],
    'Dependents': [dependents],
    'Education': [education],
    'Self_Employed': [self_employed],
    'LoanAmount': [loan_amount],
    'Loan_Amount_Term': [loan_term],
    'Credit_History': [credit_history],
    'Property_Area': [property_area],
    'TotalIncome': [total_income],
    'Loan_to_Income': [loan_to_income]
})

# Prediction
if st.button("Predict Loan Approval"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]
    
    if prediction == 1:
        st.success(f"✅ Loan Approved! (Confidence: {probability:.2f})")
    else:
        st.error(f"❌ Loan Not Approved (Confidence: {1 - probability:.2f})")
