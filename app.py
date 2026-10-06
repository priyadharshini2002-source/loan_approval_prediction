import os
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from imblearn.over_sampling import SMOTE


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon="🏦",
    layout="wide"
)


# ============================================================
# CSS
# ============================================================

st.markdown("""
<style>

.main {
    background-color: #f8fafc;
}

.block-container {
    padding-top: 2rem;
}

.title {
    font-size: 42px;
    font-weight: 700;
    text-align: center;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: #64748b;
    margin-bottom: 30px;
}

.section-title {
    font-size: 25px;
    font-weight: 650;
    margin-top: 20px;
    margin-bottom: 15px;
}

.success-box {
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    font-size: 30px;
    font-weight: 700;
    margin-top: 20px;
}

.approved {
    background-color: #dcfce7;
    color: #166534;
    border: 2px solid #22c55e;
}

.rejected {
    background-color: #fee2e2;
    color: #991b1b;
    border: 2px solid #ef4444;
}

.info-box {
    background-color: #eff6ff;
    padding: 18px;
    border-radius: 12px;
    border-left: 5px solid #2563eb;
    margin-bottom: 20px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# FILES
# ============================================================

MODEL_FILE = "loan_approval_model.pkl"
FEATURE_FILE = "loan_features.pkl"
DATA_FILE = "loan_approval_dataset.csv"


# ============================================================
# CHECK FILES
# ============================================================

if not os.path.exists(MODEL_FILE):
    st.error("❌ loan_approval_model.pkl not found.")
    st.stop()

if not os.path.exists(FEATURE_FILE):
    st.error("❌ loan_features.pkl not found.")
    st.stop()


# ============================================================
# LOAD MODEL USING JOBLIB ONLY
# ============================================================

try:

    model = joblib.load(MODEL_FILE)
    feature_columns = joblib.load(FEATURE_FILE)

except Exception as e:

    st.error("❌ Model files could not be loaded.")
    st.error(f"{type(e).__name__}: {e}")
    st.stop()


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="title">🏦 Loan Approval Prediction System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Machine Learning Based Loan Eligibility Prediction'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("📌 About the Project")

    st.write(
        """
        This application predicts whether a loan application
        is likely to be **Approved** or **Rejected**
        using Machine Learning.
        """
    )

    st.markdown("---")

    st.subheader("Algorithms Used")

    st.write("• Logistic Regression")
    st.write("• Decision Tree")
    st.write("• SMOTE")

    st.markdown("---")

    st.subheader("Dataset")

    st.write("Loan Approval Dataset")

    st.markdown("---")

    st.subheader("Best Model")

    st.write("Decision Tree Classifier")


# ============================================================
# APPLICANT INFORMATION
# ============================================================

st.markdown(
    '<div class="section-title">📝 Applicant Information</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="info-box">'
    'Enter the applicant details below to predict the loan approval status.'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# INPUTS
# ============================================================

col1, col2, col3 = st.columns(3)

with col1:

    no_of_dependents = st.number_input(
        "👨‍👩‍👧 Number of Dependents",
        min_value=0,
        max_value=10,
        value=2
    )

with col2:

    education = st.selectbox(
        "🎓 Education",
        ["Graduate", "Not Graduate"]
    )

with col3:

    self_employed = st.selectbox(
        "💼 Self Employed",
        ["Yes", "No"]
    )


col4, col5, col6 = st.columns(3)

with col4:

    income_annum = st.number_input(
        "💰 Annual Income",
        min_value=0,
        max_value=1000000000,
        value=5000000,
        step=100000
    )

with col5:

    loan_amount = st.number_input(
        "🏦 Loan Amount",
        min_value=0,
        max_value=1000000000,
        value=15000000,
        step=100000
    )

with col6:

    loan_term = st.number_input(
        "📅 Loan Term (Years)",
        min_value=1,
        max_value=50,
        value=10
    )


col7, col8, col9 = st.columns(3)

with col7:

    cibil_score = st.number_input(
        "📈 CIBIL Score",
        min_value=300,
        max_value=900,
        value=700
    )

with col8:

    residential_assets_value = st.number_input(
        "🏠 Residential Assets Value",
        min_value=0,
        max_value=1000000000,
        value=5000000,
        step=100000
    )

with col9:

    commercial_assets_value = st.number_input(
        "🏢 Commercial Assets Value",
        min_value=0,
        max_value=1000000000,
        value=2000000,
        step=100000
    )


col10, col11, col12 = st.columns(3)

with col10:

    luxury_assets_value = st.number_input(
        "🚗 Luxury Assets Value",
        min_value=0,
        max_value=1000000000,
        value=3000000,
        step=100000
    )

with col11:

    bank_asset_value = st.number_input(
        "🏦 Bank Asset Value",
        min_value=0,
        max_value=1000000000,
        value=4000000,
        step=100000
    )


# ============================================================
# PREDICTION BUTTON
# ============================================================

st.markdown("---")

predict_button = st.button(
    "🔮 Predict Loan Approval",
    type="primary",
    use_container_width=True
)


# ============================================================
# PREDICTION
# ============================================================

if predict_button:

    try:

        education_encoded = (
            1 if education == "Graduate" else 0
        )

        self_employed_encoded = (
            1 if self_employed == "Yes" else 0
        )


        input_data = pd.DataFrame({

            "no_of_dependents": [no_of_dependents],

            "education": [education_encoded],

            "self_employed": [self_employed_encoded],

            "income_annum": [income_annum],

            "loan_amount": [loan_amount],

            "loan_term": [loan_term],

            "cibil_score": [cibil_score],

            "residential_assets_value": [
                residential_assets_value
            ],

            "commercial_assets_value": [
                commercial_assets_value
            ],

            "luxury_assets_value": [
                luxury_assets_value
            ],

            "bank_asset_value": [
                bank_asset_value
            ]
        })


        # EXACT FEATURE ORDER
        input_data = input_data[feature_columns]


        prediction = model.predict(input_data)[0]


        # ====================================================
        # RESULT
        # ====================================================

        st.markdown(
            '<div class="section-title">📊 Prediction Result</div>',
            unsafe_allow_html=True
        )


        if prediction == 1:

            st.markdown(
                '<div class="success-box approved">'
                '✅ LOAN APPROVED'
                '</div>',
                unsafe_allow_html=True
            )

            st.success(
                "The model predicts that this loan application "
                "is likely to be approved."
            )

        else:

            st.markdown(
                '<div class="success-box rejected">'
                '❌ LOAN REJECTED'
                '</div>',
                unsafe_allow_html=True
            )

            st.warning(
                "The model predicts that this loan application "
                "is likely to be rejected."
            )


        # ====================================================
        # SUMMARY
        # ====================================================

        st.markdown(
            '<div class="section-title">👤 Applicant Summary</div>',
            unsafe_allow_html=True
        )

        c1, c2, c3 = st.columns(3)

        with c1:

            st.metric(
                "CIBIL Score",
                cibil_score
            )

            st.metric(
                "Annual Income",
                f"₹{income_annum:,.0f}"
            )

        with c2:

            st.metric(
                "Loan Amount",
                f"₹{loan_amount:,.0f}"
            )

            st.metric(
                "Loan Term",
                f"{loan_term} Years"
            )

        with c3:

            st.metric(
                "Dependents",
                no_of_dependents
            )

            st.metric(
                "Education",
                education
            )


    except Exception as e:

        st.error("❌ Prediction failed.")
        st.error(f"{type(e).__name__}: {e}")


# ============================================================
# MODEL PERFORMANCE
# ============================================================

if os.path.exists(DATA_FILE):

    st.markdown("---")

    st.markdown(
        '<div class="section-title">📊 Model Performance</div>',
        unsafe_allow_html=True
    )

    try:

        data = pd.read_csv(DATA_FILE)

        # ----------------------------------------------------
        # CLEAN COLUMN NAMES
        # ----------------------------------------------------

        data.columns = data.columns.str.strip()


        # ----------------------------------------------------
        # CLEAN OBJECT COLUMNS
        # ----------------------------------------------------

        for column in data.select_dtypes(
            include="object"
        ).columns:

            data[column] = (
                data[column]
                .astype(str)
                .str.strip()
            )


        # ----------------------------------------------------
        # ENCODING
        # ----------------------------------------------------

        data["education"] = (
            data["education"]
            .str.lower()
            .map({
                "graduate": 1,
                "not graduate": 0
            })
        )

        data["self_employed"] = (
            data["self_employed"]
            .str.lower()
            .map({
                "yes": 1,
                "no": 0
            })
        )

        data["loan_status"] = (
            data["loan_status"]
            .str.lower()
            .map({
                "approved": 1,
                "rejected": 0
            })
        )


        # ----------------------------------------------------
        # NUMERIC CONVERSION
        # ----------------------------------------------------

        numeric_columns = [
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
            "bank_asset_value",
            "loan_status"
        ]

        for column in numeric_columns:

            data[column] = pd.to_numeric(
                data[column],
                errors="coerce"
            )


        data = data.dropna(
            subset=numeric_columns
        ).reset_index(drop=True)


        # ----------------------------------------------------
        # X AND Y
        # ----------------------------------------------------

        X = data.drop(
            ["loan_status", "loan_id"],
            axis=1
        )

        y = data["loan_status"]


        # ----------------------------------------------------
        # TRAIN TEST SPLIT
        # ----------------------------------------------------

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42
        )


        # ====================================================
        # LOGISTIC REGRESSION
        # ====================================================

        scaler = StandardScaler()

        X_train_scaled = scaler.fit_transform(X_train)

        X_test_scaled = scaler.transform(X_test)


        log_model = LogisticRegression(
            max_iter=1000
        )

        log_model.fit(
            X_train_scaled,
            y_train
        )

        log_pred = log_model.predict(
            X_test_scaled
        )


        # ====================================================
        # DECISION TREE
        # ====================================================

        dt_model = DecisionTreeClassifier(
            random_state=42
        )

        dt_model.fit(
            X_train,
            y_train
        )

        dt_pred = dt_model.predict(
            X_test
        )


        # ====================================================
        # SMOTE
        # ====================================================

        smote = SMOTE(
            random_state=42
        )

        X_train_smote, y_train_smote = smote.fit_resample(
            X_train,
            y_train
        )


        # ====================================================
        # LOGISTIC REGRESSION + SMOTE
        # ====================================================

        scaler_smote = StandardScaler()

        X_train_smote_scaled = scaler_smote.fit_transform(
            X_train_smote
        )

        X_test_smote_scaled = scaler_smote.transform(
            X_test
        )


        log_smote_model = LogisticRegression(
            max_iter=1000
        )

        log_smote_model.fit(
            X_train_smote_scaled,
            y_train_smote
        )

        log_smote_pred = log_smote_model.predict(
            X_test_smote_scaled
        )


        # ====================================================
        # DECISION TREE + SMOTE
        # ====================================================

        dt_smote_model = DecisionTreeClassifier(
            random_state=42
        )

        dt_smote_model.fit(
            X_train_smote,
            y_train_smote
        )

        dt_smote_pred = dt_smote_model.predict(
            X_test
        )


        # ====================================================
        # METRICS FUNCTION
        # ====================================================

        def get_metrics(y_true, y_pred):

            return {

                "Accuracy": accuracy_score(
                    y_true,
                    y_pred
                ),

                "Precision": precision_score(
                    y_true,
                    y_pred,
                    average="macro",
                    zero_division=0
                ),

                "Recall": recall_score(
                    y_true,
                    y_pred,
                    average="macro",
                    zero_division=0
                ),

                "F1-Score": f1_score(
                    y_true,
                    y_pred,
                    average="macro",
                    zero_division=0
                )
            }


        # ====================================================
        # RESULTS
        # ====================================================

        results = {

            "Logistic Regression":
                get_metrics(
                    y_test,
                    log_pred
                ),

            "Decision Tree":
                get_metrics(
                    y_test,
                    dt_pred
                ),

            "Logistic Regression + SMOTE":
                get_metrics(
                    y_test,
                    log_smote_pred
                ),

            "Decision Tree + SMOTE":
                get_metrics(
                    y_test,
                    dt_smote_pred
                )
        }


        results_df = pd.DataFrame(
            results
        ).T


        # ====================================================
        # BEST MODEL
        # ====================================================

        best_model_name = results_df[
            "F1-Score"
        ].idxmax()

        best_f1 = results_df.loc[
            best_model_name,
            "F1-Score"
        ]


        st.success(
            f"🏆 Best Model: {best_model_name} "
            f"with F1-Score of {best_f1 * 100:.2f}%"
        )


        # ====================================================
        # METRIC CARDS
        # ====================================================

        m1, m2, m3, m4 = st.columns(4)


        with m1:

            st.metric(
                "Accuracy",
                f"{results_df.loc[best_model_name, 'Accuracy'] * 100:.2f}%"
            )


        with m2:

            st.metric(
                "Precision",
                f"{results_df.loc[best_model_name, 'Precision'] * 100:.2f}%"
            )


        with m3:

            st.metric(
                "Recall",
                f"{results_df.loc[best_model_name, 'Recall'] * 100:.2f}%"
            )


        with m4:

            st.metric(
                "F1-Score",
                f"{results_df.loc[best_model_name, 'F1-Score'] * 100:.2f}%"
            )


        # ====================================================
        # CHART
        # ====================================================

        st.markdown(
            "#### 📈 Model Metrics Comparison"
        )

        chart_data = (
            results_df[
                [
                    "Accuracy",
                    "Precision",
                    "Recall",
                    "F1-Score"
                ]
            ] * 100
        )

        st.bar_chart(
            chart_data
        )


        # ====================================================
        # TABLE
        # ====================================================

        st.markdown(
            "#### 📋 Detailed Performance Table"
        )

        display_df = (
            results_df * 100
        ).round(2)

        display_df.columns = [
            "Accuracy (%)",
            "Precision (%)",
            "Recall (%)",
            "F1-Score (%)"
        ]

        st.dataframe(
            display_df,
            use_container_width=True
        )


        # ====================================================
        # CONFUSION MATRIX
        # ====================================================

        st.markdown(
            "#### 🔍 Decision Tree Confusion Matrix"
        )

        cm = confusion_matrix(
            y_test,
            dt_pred
        )


        fig, ax = plt.subplots(
            figsize=(6, 4)
        )

        ax.imshow(cm)

        ax.set_title(
            "Decision Tree Confusion Matrix"
        )

        ax.set_xlabel(
            "Predicted Label"
        )

        ax.set_ylabel(
            "Actual Label"
        )

        ax.set_xticks([0, 1])

        ax.set_yticks([0, 1])

        ax.set_xticklabels(
            ["Rejected", "Approved"]
        )

        ax.set_yticklabels(
            ["Rejected", "Approved"]
        )


        for i in range(2):

            for j in range(2):

                ax.text(
                    j,
                    i,
                    cm[i, j],
                    ha="center",
                    va="center"
                )


        st.pyplot(fig)

        plt.close(fig)


        # ====================================================
        # CLASSIFICATION REPORT
        # ====================================================

        st.markdown(
            "#### 📄 Classification Report"
        )


        report = classification_report(
            y_test,
            dt_pred,
            target_names=[
                "Rejected",
                "Approved"
            ],
            output_dict=True,
            zero_division=0
        )


        report_df = pd.DataFrame(
            report
        ).transpose()


        st.dataframe(
            report_df.round(4),
            use_container_width=True
        )


    except Exception as e:

        st.error(
            "❌ Could not generate model performance."
        )

        st.error(
            f"{type(e).__name__}: {e}"
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.markdown(
    """
    <div style="text-align:center; color:#64748b;">
        <b>Loan Approval Prediction System</b><br>
        Machine Learning Project | MSc Data Science
    </div>
    """,
    unsafe_allow_html=True
)