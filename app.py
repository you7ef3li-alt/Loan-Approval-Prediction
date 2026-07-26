import streamlit as st
import joblib
import pandas as pd

# Load the trained model
model = joblib.load("loan_model.pkl")

st.set_page_config(page_title="Loan Approval Prediction", page_icon="🏦")

st.title("🏦 Loan Approval Prediction System")
st.write("Enter the applicant information below:")

# User Inputs
no_of_dependents = st.number_input("Number of Dependents", min_value=0, step=1)

education = st.selectbox(
    "Education",
    ["Graduate", "Not Graduate"]
)

self_employed = st.selectbox(
    "Self Employed",
    ["Yes", "No"]
)

income_annum = st.number_input("Annual Income", min_value=0)

loan_amount = st.number_input("Loan Amount", min_value=0)

loan_term = st.number_input("Loan Term (Years)", min_value=1)

cibil_score = st.number_input(
    "CIBIL Score",
    min_value=300,
    max_value=900
)

residential_assets_value = st.number_input("Residential Assets Value", min_value=0)

commercial_assets_value = st.number_input("Commercial Assets Value", min_value=0)

luxury_assets_value = st.number_input("Luxury Assets Value", min_value=0)

bank_asset_value = st.number_input("Bank Asset Value", min_value=0)

# Convert text to numbers
education = 0 if education == "Graduate" else 1
self_employed = 1 if self_employed == "Yes" else 0

# Prediction
if st.button("Predict Loan Status"):

    data = pd.DataFrame([[
        no_of_dependents,
        education,
        self_employed,
        income_annum,
        loan_amount,
        loan_term,
        cibil_score,
        residential_assets_value,
        commercial_assets_value,
        luxury_assets_value,
        bank_asset_value
    ]], columns=[
        "no_of_dependents",
        "education",
        "self_employed",
        "income_annum",
        "loan_amount",
        "loan_term",
        "cibil_score",
        "residential_assets_value",
        "commercial_assets_value",
        "luxury_assets_value",
        "bank_asset_value"
    ])

    prediction = model.predict(data)

    if prediction[0] == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")