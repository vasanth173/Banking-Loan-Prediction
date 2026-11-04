
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Banking Loan Prediction",
    page_icon="🏦",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
    .stApp {
        background-color: #f4f6fb;
    }
    h1, h2, h3 {
        color: #172b4d;
    }
    div.stButton > button {
        background-color: #17365d;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 20px;
    }
    div.stButton > button:hover {
        background-color: #24558a;
        color: white;
    }
</style>
""", unsafe_allow_html=True)


# ---------------- LOAD MODELS ----------------
@st.cache_resource
def load_models():
    classifier = joblib.load(MODEL_DIR / "loan_classifier.pkl")
    regressor = joblib.load(MODEL_DIR / "loan_regressor.pkl")
    features = joblib.load(MODEL_DIR / "feature_names.pkl")
    return classifier, regressor, features


try:
    classifier, regressor, features = load_models()
except Exception as e:
    st.error(f"Model loading error: {e}")
    st.stop()


# ---------------- PREPARE INPUT ----------------
def prepare_input(df):

    required_columns = [
        "age",
        "annual_income",
        "credit_score",
        "employment_type",
        "existing_loan",
        "loan_amount_requested"
    ]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing columns: " + ", ".join(missing)
        )

    data = df[required_columns].copy()

    numeric_columns = [
        "age",
        "annual_income",
        "credit_score",
        "existing_loan",
        "loan_amount_requested"
    ]

    for col in numeric_columns:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    if data[numeric_columns].isnull().any().any():
        raise ValueError("Please check missing or invalid numeric values.")

    data["employment_type"] = (
        data["employment_type"].astype(str).str.strip()
    )

    valid_types = ["Business", "Salaried", "Self Employed"]

    invalid_types = ~data["employment_type"].isin(valid_types)

    if invalid_types.any():
        raise ValueError(
            "Employment type must be Business, Salaried, or Self Employed."
        )

    model_input = pd.DataFrame(index=data.index)

    model_input["age"] = data["age"]
    model_input["annual_income"] = data["annual_income"]
    model_input["credit_score"] = data["credit_score"]
    model_input["existing_loan"] = data["existing_loan"]
    model_input["loan_amount_requested"] = data["loan_amount_requested"]

    model_input["employment_type_Business"] = (
        data["employment_type"] == "Business"
    ).astype(int)

    model_input["employment_type_Salaried"] = (
        data["employment_type"] == "Salaried"
    ).astype(int)

    model_input["employment_type_Self Employed"] = (
        data["employment_type"] == "Self Employed"
    ).astype(int)

    model_input = model_input.reindex(
        columns=features,
        fill_value=0
    )

    return model_input


# ---------------- HEADER ----------------
st.title("🏦 Banking Loan Prediction System")
st.caption("Smart Loan Analysis | Machine Learning Powered")
st.divider()

st.sidebar.title("🏦 Banking System")
st.sidebar.info(
    "Analyze individual customers or predict loan "
    "eligibility for multiple customers."
)

st.sidebar.markdown("### Project Features")
st.sidebar.write("👤 Individual Loan Prediction")
st.sidebar.write("📂 Bulk Customer Prediction")
st.sidebar.write("📊 Dashboard Analytics")
st.sidebar.write("📥 Download Prediction Report")


# ---------------- TABS ----------------
individual_tab, bulk_tab = st.tabs([
    "👤 Individual Prediction",
    "📂 Bulk Customer Prediction"
])


# ==================================================
# INDIVIDUAL CUSTOMER PREDICTION
# ==================================================
with individual_tab:

    st.header("👤 Individual Customer Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Personal Information")

        age = st.number_input(
            "Applicant Age",
            min_value=18,
            max_value=100,
            value=30
        )

        employment = st.selectbox(
            "Employment Type",
            ["Salaried", "Business", "Self Employed"]
        )

        annual_income = st.number_input(
            "Annual Income (₹)",
            min_value=0,
            value=500000,
            step=10000
        )

    with col2:
        st.subheader("Financial Information")

        credit_score = st.slider(
            "Credit Score",
            min_value=300,
            max_value=900,
            value=700
        )

        existing_loan = st.number_input(
            "Existing Loan Amount (₹)",
            min_value=0,
            value=50000,
            step=5000
        )

        loan_amount = st.number_input(
            "Requested Loan Amount (₹)",
            min_value=0,
            value=300000,
            step=10000
        )

    st.divider()

    if st.button("🔍 Analyze Loan Application", key="individual_predict"):

        input_df = pd.DataFrame([{
            "age": age,
            "annual_income": annual_income,
            "credit_score": credit_score,
            "employment_type": employment,
            "existing_loan": existing_loan,
            "loan_amount_requested": loan_amount
        }])

        try:
            model_input = prepare_input(input_df)

            prediction = classifier.predict(model_input)[0]
            recommended_amount = regressor.predict(model_input)[0]

            st.subheader("📊 Loan Prediction Result")

            if prediction == 1:
                st.success("✅ Loan Approved")
            else:
                st.error("❌ Loan Rejected")

            recommended_amount = max(0, recommended_amount)

            metric1, metric2, metric3 = st.columns(3)

            metric1.metric("Annual Income", f"₹{annual_income:,.0f}")
            metric2.metric("Credit Score", credit_score)
            metric3.metric(
                "Recommended Loan",
                f"₹{recommended_amount:,.0f}"
            )

            st.subheader("📋 Applicant Summary")

            summary = pd.DataFrame({
                "Details": [
                    "Age",
                    "Employment Type",
                    "Annual Income",
                    "Credit Score",
                    "Existing Loan",
                    "Requested Loan",
                    "Loan Status",
                    "Recommended Loan Amount"
                ],
                "Values": [
                    age,
                    employment,
                    f"₹{annual_income:,.0f}",
                    str(credit_score),
                    f"₹{existing_loan:,.0f}",
                    f"₹{loan_amount:,.0f}",
                    "Approved" if prediction == 1 else "Rejected",
                    f"₹{recommended_amount:,.0f}"
                ]
            })

            st.dataframe(summary, use_container_width=True, hide_index=True)

            st.subheader("💰 Financial Analysis")

            chart_data = pd.DataFrame({
                "Category": [
                    "Annual Income",
                    "Existing Loan",
                    "Requested Loan",
                    "Recommended Loan"
                ],
                "Amount": [
                    annual_income,
                    existing_loan,
                    loan_amount,
                    recommended_amount
                ]
            })

            chart_data = chart_data[chart_data["Amount"] > 0]

            if not chart_data.empty:
                st.vega_lite_chart(
                    chart_data,
                    {
                        "mark": {
                            "type": "arc",
                            "innerRadius": 50
                        },
                        "encoding": {
                            "theta": {
                                "field": "Amount",
                                "type": "quantitative"
                            },
                            "color": {
                                "field": "Category",
                                "type": "nominal"
                            },
                            "tooltip": [
                                {"field": "Category"},
                                {"field": "Amount", "type": "quantitative"}
                            ]
                        }
                    },
                    use_container_width=True
                )

        except Exception as e:
            st.error(f"Prediction error: {e}")


# ==================================================
# BULK CUSTOMER PREDICTION
# ==================================================
with bulk_tab:

    st.header("📂 Bulk Customer Loan Prediction")

    st.write(
        "Upload a CSV file containing multiple customer details "
        "to predict their loan eligibility."
    )

    st.info(
        "Required columns: age, annual_income, credit_score, "
        "employment_type, existing_loan, loan_amount_requested"
    )

    sample_path = DATA_DIR / "banking_customers.csv"

    if sample_path.exists():

        with open(sample_path, "rb") as file:
            sample_data = file.read()

        st.download_button(
            label="📥 Download Sample Customer Dataset",
            data=sample_data,
            file_name="banking_customers.csv",
            mime="text/csv"
        )

    uploaded_file = st.file_uploader(
        "Upload Customer CSV File",
        type=["csv"],
        key="bulk_csv"
    )

    if uploaded_file is not None:

        try:
            customers = pd.read_csv(uploaded_file)

            st.subheader("📋 Uploaded Customer Data")
            st.write(f"Total customers uploaded: {len(customers)}")
            st.dataframe(customers.head(10), use_container_width=True)

            if st.button("🚀 Predict All Customers", key="bulk_predict"):

                if customers.empty:
                    st.warning("The uploaded CSV file is empty.")

                else:
                    model_input = prepare_input(customers)

                    predictions = classifier.predict(model_input)
                    recommended = regressor.predict(model_input)

                    results = customers.copy()

                    results["Loan Prediction"] = predictions
                    results["Loan Status"] = [
                        "Approved" if p == 1 else "Rejected"
                        for p in predictions
                    ]

                    results["Recommended Loan Amount"] = (
                        recommended.clip(min=0)
                    ).round(2)

                    total = len(results)
                    approved = int((predictions == 1).sum())
                    rejected = total - approved
                    approval_rate = (approved / total) * 100

                    st.session_state["bulk_results"] = results

                    st.success("Bulk prediction completed!")

                    st.subheader("📊 Bulk Prediction Dashboard")

                    m1, m2, m3, m4 = st.columns(4)

                    m1.metric("Total Customers", total)
                    m2.metric("Approved", approved)
                    m3.metric("Rejected", rejected)
                    m4.metric("Approval Rate", f"{approval_rate:.1f}%")

                    st.subheader("📈 Loan Approval Analysis")

                    chart_df = pd.DataFrame({
                        "Loan Status": ["Approved", "Rejected"],
                        "Customers": [approved, rejected]
                    })

                    st.vega_lite_chart(
                        chart_df,
                        {
                            "mark": {
                                "type": "arc",
                                "innerRadius": 45
                            },
                            "encoding": {
                                "theta": {
                                    "field": "Customers",
                                    "type": "quantitative"
                                },
                                "color": {
                                    "field": "Loan Status",
                                    "type": "nominal"
                                },
                                "tooltip": [
                                    {"field": "Loan Status"},
                                    {"field": "Customers"}
                                ]
                            }
                        },
                        use_container_width=True
                    )

                    st.subheader("📋 Customer Prediction Results")

                    st.dataframe(
                        results,
                        use_container_width=True,
                        hide_index=True
                    )

                    csv_report = results.to_csv(index=False).encode("utf-8")

                    st.download_button(
                        label="📥 Download Prediction Report",
                        data=csv_report,
                        file_name="loan_prediction_report.csv",
                        mime="text/csv"
                    )

        except Exception as e:
            st.error(f"Bulk prediction error: {e}")

    if "bulk_results" in st.session_state:
        st.caption(
            "Your latest bulk prediction is available above. "
            "Upload another CSV to run a new prediction."
        )


# ---------------- FOOTER ----------------
st.divider()

st.caption(
    "Banking Loan Prediction System | Machine Learning Project"
)

st.warning(
    "Educational demonstration only. Predictions use a synthetic "
    "sample dataset and should not be used for actual lending decisions."
)
