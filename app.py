
import streamlit as st
import pandas as pd
import joblib
import os

# Page configuration
st.set_page_config(
    page_title="Banking Loan Prediction",
    page_icon="🏦",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
.stApp {
    background-color: #f4f7fc;
}
h1, h2, h3 {
    color: #142d4e;
}
[data-testid="stSidebar"] {
    background-color: #142d4e;
}
[data-testid="stSidebar"] * {
    color: white !important;
}
div.stButton > button {
    background-color: #142d4e;
    color: white;
    border-radius: 8px;
    height: 45px;
    width: 100%;
    font-weight: bold;
}
div.stButton > button:hover {
    background-color: #244b78;
    color: white;
}
.metric-card {
    background-color: white;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #dce3ed;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# Load trained models
@st.cache_resource
def load_models():
    classifier = joblib.load("models/loan_classifier.pkl")
    regressor = joblib.load("models/loan_regressor.pkl")
    features = joblib.load("models/feature_names.pkl")
    return classifier, regressor, features

classifier, regressor, features = load_models()

# Header
st.title("🏦 Banking Loan Prediction System")
st.caption("Smart Loan Analysis | Machine Learning Powered")
st.divider()

# Sidebar
st.sidebar.title("🏦 Banking System")
st.sidebar.markdown("---")
st.sidebar.subheader("Applicant Details")
st.sidebar.info("Enter applicant information to analyze loan eligibility.")

# Main layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("👤 Personal Information")

    age = st.number_input(
        "Applicant Age",
        min_value=21,
        max_value=60,
        value=30
    )

    employment = st.selectbox(
        "Employment Type",
        ["Salaried", "Self Employed", "Business"]
    )

    annual_income = st.number_input(
        "Annual Income (₹)",
        min_value=150000,
        max_value=1500000,
        value=500000,
        step=10000
    )

with col2:
    st.subheader("💳 Financial Information")

    credit_score = st.slider(
        "Credit Score",
        min_value=300,
        max_value=850,
        value=700
    )

    existing_loan = st.number_input(
        "Existing Loan Amount (₹)",
        min_value=0,
        max_value=500000,
        value=50000,
        step=5000
    )

    loan_amount = st.number_input(
        "Requested Loan Amount (₹)",
        min_value=50000,
        max_value=1000000,
        value=300000,
        step=10000
    )

st.divider()

# Prediction
if st.button("🔍 Analyze Loan Application"):

    input_data = pd.DataFrame([{
        "age": age,
        "annual_income": annual_income,
        "credit_score": credit_score,
        "existing_loan": existing_loan,
        "loan_amount_requested": loan_amount,
        "employment_type_Business": int(employment == "Business"),
        "employment_type_Salaried": int(employment == "Salaried"),
        "employment_type_Self Employed": int(employment == "Self Employed")
    }])

    input_data = input_data.reindex(
        columns=features,
        fill_value=0
    )

    # Classification prediction
    prediction = classifier.predict(input_data)[0]

    # Regression prediction
    recommended_amount = regressor.predict(input_data)[0]

    st.subheader("📊 Loan Analysis Results")

    result1, result2 = st.columns(2)

    with result1:
        if prediction == 1:
            st.success("✅ LOAN APPROVED")
        else:
            st.error("❌ LOAN REJECTED")

    with result2:
        st.metric(
            "Recommended Loan Amount",
            f"₹{recommended_amount:,.2f}"
        )

    st.divider()

    st.subheader("📋 Applicant Summary")

    summary = pd.DataFrame({
        "Parameter": [
            "Age",
            "Employment Type",
            "Annual Income",
            "Credit Score",
            "Existing Loan",
            "Requested Loan"
        ],
        "Value": [
            f"{age} Years",
            employment,
            f"₹{annual_income:,.0f}",
            credit_score,
            f"₹{existing_loan:,.0f}",
            f"₹{loan_amount:,.0f}"
        ]
    })

    st.dataframe(summary, use_container_width=True, hide_index=True)

    st.info(
        "Note: This project uses a synthetic sample dataset "
        "for educational demonstration. Predictions are not "
        "actual bank lending decisions."
    )

st.divider()

st.markdown(
    "<center>Banking Loan Prediction System | "
    "Powered by Machine Learning</center>",
    unsafe_allow_html=True
)
