from pathlib import Path

import numpy as np
import pandas as pd
import shap
import streamlit as st
from scipy.stats import boxcox
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from streamlit_shap import st_shap
from xgboost import XGBClassifier
import streamlit as st
import pickle
import xgboost

# Load the model you just saved
model = pickle.load(open('maternal_model.pkl', 'rb'))


LOCAL_DATA_CANDIDATES = ["iris.csv"]
REMOTE_DATA_URL = "/kaggle/input/maternal-health-risk-data/Maternal Health Risk Data Set.csv"
LABEL_MAP = {"high risk": 2, "mid risk": 1, "low risk": 0}
REVERSE_LABEL_MAP = {0: "Low Risk", 1: "Mid Risk", 2: "High Risk"}

# Full feature set from your notebook
FEATURES = ["Age", "SystolicBP", "DiastolicBP", "BS", "BodyTemp", "HeartRate"]
TARGET_COL = "RiskLevel"
feature_names = ["Age", "SystolicBP", "DiastolicBP", "BS", "BodyTemp", "HeartRate"]


def _looks_like_valid_dataset(df: pd.DataFrame) -> bool:
    required = set(FEATURES + [TARGET_COL])
    return required.issubset(set(df.columns))


def _load_local_dataset() -> pd.DataFrame:
    # First try known filenames.
    for file_name in LOCAL_DATA_CANDIDATES:
        path = Path(file_name)
        if path.exists():
            df = pd.read_csv(path)
            if _looks_like_valid_dataset(df):
                return df

    # Then scan nearby CSV/SC files and pick one with required columns.
    for path in Path(".").glob("*"):
        if path.is_file() and path.suffix.lower() in {".csv", ".sc"}:
            try:
                df = pd.read_csv(path)
                if _looks_like_valid_dataset(df):
                    return df
            except Exception:
                continue

    raise FileNotFoundError("No local dataset file with required columns was found.")


@st.cache_resource
def train_full_model():
    """
    Train a full model using the same ideas as your notebook:
    - Encode RiskLevel (low/mid/high -> 0/1/2)
    - Log-transform Age
    - Box-Cox transform BS
    - Standard-scale all numeric features
    - XGBoost classifier on the transformed features
    """
    try:
        df = _load_local_dataset()
    except FileNotFoundError:
        df = pd.read_csv(REMOTE_DATA_URL)

    df = df.copy()
    df[TARGET_COL] = df[TARGET_COL].replace(LABEL_MAP)

    # Transformations to mirror the notebook
    df_transformed = df.copy()
    df_transformed["Age"] = df_transformed["Age"].apply(np.log)

    # Box-Cox for BS (requires positive values; dataset already satisfies this)
    bc_values, bc_lambda = boxcox(df_transformed["BS"])
    df_transformed["BS"] = bc_values

    X = df_transformed[FEATURES]
    y = df_transformed[TARGET_COL]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, _, y_train, _ = train_test_split(
        X_scaled, y, test_size=0.3, random_state=101, stratify=y
    )

    # Use an XGBoost classifier similar to the one in your notebook.
    model = XGBClassifier(
        n_estimators=50,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="mlogloss",
        random_state=101,
    )
    model.fit(X_train, y_train)

    # SHAP explainer for XGBoost (tree-based, like in your notebook)
    explainer = shap.TreeExplainer(model)

    return model, scaler, bc_lambda, explainer


def main():
    st.set_page_config(page_title="Pregnancy Risk Predictor", page_icon=":hospital:", layout="wide")
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700&display=swap');
        :root {
            color-scheme: light;
        }
        .stApp {
            background: linear-gradient(180deg, #FFF5F8 0%, #FFFFFF 55%, #FFE7EE 100%);
            color: #333333;
            font-family: "Nunito", "Segoe UI", system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
        }
        .block-container {
            max-width: 95vw !important;
            padding-top: 1.2rem !important;
            padding-left: 2rem !important;
            padding-right: 2rem !important;
        }
        section[data-testid="stSidebar"] {
            background-color: #FFE0EA;
            color: #333333;
        }
        section[data-testid="stSidebar"] * {
            color: #333333 !important;
            font-size: 0.95rem;
        }
        [data-testid="stForm"] label,
        [data-testid="stForm"] .stMarkdown,
        [data-testid="stForm"] p,
        [data-testid="stForm"] span,
        [data-testid="stForm"] div {
            color: #333333 !important;
        }
        .custom-header {
            background: #FFFFFF;
            border-radius: 18px;
            padding: 20px 24px;
            box-shadow: 0 6px 20px rgba(149, 86, 107, 0.20);
            border: 1px solid #F4C7D6;
            margin-bottom: 16px;
        }
        .hero-image {
            border-radius: 18px;
            overflow: hidden;
            border: 1px solid #F4C7D6;
            box-shadow: 0 8px 24px rgba(149, 86, 107, 0.22);
            margin-bottom: 12px;
        }
        .hero-banner {
            width: 100%;
            height: 48vh;
            min-height: 280px;
            max-height: 520px;
            border-radius: 18px;
            border: 1px solid #F4C7D6;
            box-shadow: 0 8px 24px rgba(149, 86, 107, 0.22);
            background-image: url('https://images.pexels.com/photos/1257110/pexels-photo-1257110.jpeg');
            background-size: cover;
            background-position: center 35%;
            margin-bottom: 12px;
        }
        .hero-title {
            text-align: center;
            font-size: 1.75rem;
            font-weight: 700;
            color: #3E3035;
            margin: 4px 0 14px 0;
        }
        .support-card {
            background: #FFFFFF;
            border-radius: 18px;
            border: 1px solid #F4C7D6;
            box-shadow: 0 6px 20px rgba(149, 86, 107, 0.16);
            padding: 18px 20px;
            margin: 8px 0 18px 0;
            color: #4C3C42;
            line-height: 1.6;
            font-size: 1.03rem;
        }
        .custom-header h2 {
            font-size: 1.6rem;
            margin: 0;
            color: #3E3035;
        }
        .custom-subtitle {
            color: #7A5A66;
            margin-top: 6px;
            margin-bottom: 0;
            font-size: 1.05rem;
        }
        .analysis-card-high {
            background: #FFE0E8;
            border-radius: 18px;
            padding: 16px 18px;
            border: 1px solid #F18AA7;
            color: #5A1F31;
            margin: 8px 0 12px 0;
            font-size: 1.05rem;
        }
        .analysis-card-low {
            background: #FFF9EC;
            border-radius: 18px;
            padding: 16px 18px;
            border: 1px solid #F0D7A6;
            color: #4C4A43;
            margin: 8px 0 12px 0;
            font-size: 1.05rem;
        }
        .analysis-card-mid {
            background: #FDEFFF;
            border-radius: 18px;
            padding: 16px 18px;
            border: 1px solid #E1C1EF;
            color: #503254;
            margin: 8px 0 12px 0;
            font-size: 1.05rem;
        }
        .result-card {
            border-radius: 16px;
            padding: 16px 18px;
            margin: 10px 0 12px 0;
            font-size: 1.04rem;
            line-height: 1.6;
            font-weight: 600;
        }
        .result-high {
            background: #FFE3E6;
            color: #5C1220;
            border: 1px solid #F2A0AC;
        }
        .result-low {
            background: #E8F8EE;
            color: #114A2B;
            border: 1px solid #99D7B3;
        }
        .result-mid {
            background: #FFF4D8;
            color: #5A4400;
            border: 1px solid #E6CB85;
        }
        .project-footer {
            text-align: center;
            color: #6D5A61;
            margin-top: 28px;
            font-size: 0.95rem;
        }
        .section-heading {
            font-size: 1.2rem;
            font-weight: 700;
            color: #3E3035;
            margin: 6px 0 12px 0;
        }
        .metric-label {
            font-size: 1.02rem;
            font-weight: 700;
            color: #333333;
            margin-bottom: 6px;
        }
        .input-grid-spacer {
            margin-bottom: 8px;
        }
        .shap-bottom-space {
            margin-top: 16px;
        }
        div.stButton > button {
            background-color: #FF8DA1 !important;
            color: #FFFFFF !important;
            border-radius: 999px !important;
            border: 1px solid #F0708A !important;
            font-weight: 700 !important;
            font-size: 1.02rem !important;
            padding: 0.6rem 1.8rem !important;
            min-height: 52px !important;
            width: 100% !important;
        }
        div.stButton > button:hover {
            background-color: #F0708A !important;
            border-color: #E55B78 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="hero-banner"></div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-title">Maternal Health Predictor: Your Wellness Partner.</div>',
        unsafe_allow_html=True,
    )

    with st.container():
        st.markdown(
            """
            <div class="support-card">
            <strong>To the mothers of the world:</strong> thank you for the love, strength, and courage you bring into every home.
            Pregnancy is beautiful, but we also know it can feel overwhelming, emotional, and full of uncertainty.
            You deserve support that is kind, clear, and always available. <br><br>
            That is why we proudly built this Maternal Health Predictor as part of our Data Science project:
            to turn health data into helpful guidance you can understand in seconds. Our mission is to make
            this platform a trusted companion that encourages early awareness, better conversations with doctors,
            and healthier decisions for both mother and baby. <br><br>
            We are excited to grow and promote this website so it can reach more families, more communities,
            and more mothers who deserve accessible digital support. Your wellness matters, your journey matters,
            and we are honored to walk this path with you.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="section-heading">Enter Details Below:</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="input-grid-spacer"></div>', unsafe_allow_html=True)

    model, scaler, bc_lambda, explainer = train_full_model()

    with st.form("risk_form"):
        c1, c2, c3 = st.columns(3, gap="medium")

        with c1:
            st.markdown('<div class="metric-label">Age (years)</div>', unsafe_allow_html=True)
            age = st.number_input(
                "Age (years)",
                min_value=10,
                max_value=70,
                value=30,
                step=1,
                key="age_input",
                label_visibility="collapsed",
            )

        with c2:
            st.markdown('<div class="metric-label">Systolic BP (mmHg)</div>', unsafe_allow_html=True)
            systolic_bp = st.number_input(
                "Systolic BP (mmHg)",
                min_value=70,
                max_value=200,
                value=120,
                step=1,
                key="sbp_input",
                label_visibility="collapsed",
            )

        with c3:
            st.markdown('<div class="metric-label">Diastolic BP (mmHg)</div>', unsafe_allow_html=True)
            diastolic_bp = st.number_input(
                "Diastolic BP (mmHg)",
                min_value=40,
                max_value=120,
                value=80,
                step=1,
                key="dbp_input",
                label_visibility="collapsed",
            )

        st.write("")
        c4, c5, c6 = st.columns(3, gap="medium")

        with c4:
            st.markdown('<div class="metric-label">Blood Sugar (BS)</div>', unsafe_allow_html=True)
            bs = st.number_input(
                "Blood Sugar (BS)",
                min_value=1.0,
                max_value=25.0,
                value=7.0,
                step=0.1,
                key="bs_input",
                label_visibility="collapsed",
            )

        with c5:
            st.markdown('<div class="metric-label">Body Temperature</div>', unsafe_allow_html=True)
            body_temp = st.number_input(
                "Body Temperature",
                min_value=90.0,
                max_value=110.0,
                value=98.0,
                step=0.1,
                key="temp_input",
                label_visibility="collapsed",
            )

        with c6:
            st.markdown('<div class="metric-label">Heart Rate (bpm)</div>', unsafe_allow_html=True)
            heart_rate = st.number_input(
                "Heart Rate (bpm)",
                min_value=40,
                max_value=180,
                value=80,
                step=1,
                key="hr_input",
                label_visibility="collapsed",
            )

        st.write("")
        center_col_left, center_col_button, center_col_right = st.columns([1, 2, 1], gap="small")
        with center_col_button:
            submitted = st.form_submit_button("Predict Risk Level")

    if submitted:
        raw_row = {
            "Age": float(age),
            "SystolicBP": float(systolic_bp),
            "DiastolicBP": float(diastolic_bp),
            "BS": float(bs),
            "BodyTemp": float(body_temp),
            "HeartRate": float(heart_rate),
        }

        # Apply the same transformations as training
        row = raw_row.copy()
        row["Age"] = np.log(row["Age"])
        # Box-Cox inverse of training step; here we apply the same lambda to the new BS value
        row["BS"] = (row["BS"] ** bc_lambda - 1) / bc_lambda if bc_lambda != 0 else np.log(
            row["BS"]
        )

        input_df = pd.DataFrame([row], columns=FEATURES)
        input_scaled = scaler.transform(input_df)

        pred_num = int(model.predict(input_scaled)[0])
        pred_label = REVERSE_LABEL_MAP.get(pred_num, "Unknown")

        st.subheader("Analysis Result")
        if pred_label == "High Risk":
            st.markdown(
                '<div class="result-card result-high">You are doing great, Mom! This is just a prediction based on data, not a medical diagnosis. '
                "Because your results show High Risk, please contact your doctor for a detailed check-up just to be safe. "
                "We are with you on this journey.</div>",
                unsafe_allow_html=True,
            )
        elif pred_label == "Low Risk":
            st.markdown(
                '<div class="result-card result-low">Great news! Your results suggest a Low Risk level. Continue focusing on good nutrition, rest, and prenatal care. '
                "Keep taking good care of yourself and the baby. Wishing you a wonderful pregnancy!</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="result-card result-mid">Your results suggest a Mid Risk level. Please continue routine prenatal care and discuss these markers '
                "with your doctor for personalized guidance.</div>",
                unsafe_allow_html=True,
            )
        st.caption(
            "Note:This app is for educational purposes only and does not provide a medical diagnosis. Always consult a healthcare professional for medical advice. "
        )

        st.markdown('<div class="shap-bottom-space"></div>', unsafe_allow_html=True)
        # -------- Explanation (SHAP) --------
        st.subheader("Explanation")

        # Compute SHAP values for this specific person.
        # For multi-class XGBoost, slice the predicted class for a 1D explanation.
        shap_values_all = explainer(
            input_scaled,
            check_additivity=False,  # matches common notebook usage for tree models
        )
        shap_values_for_class = shap_values_all[:, :, pred_num]

        # Build a named SHAP explanation so waterfall labels are real feature names.
        shap_row = shap.Explanation(
            values=shap_values_for_class.values[0],
            base_values=float(np.array(shap_values_all.base_values[0])[pred_num]),
            data=input_df.iloc[0].values,
            feature_names=feature_names,
        )

        st.write("Waterfall plot showing how each feature pushed your risk up or down:")
        st_shap(shap.plots.waterfall(shap_row), height=300)
        st.info("Color guide: Red = Increasing Risk, Blue = Decreasing Risk.")

        # Key Factors summary text
        contrib = shap_values_for_class.values[0]
        idx = int(np.argmax(np.abs(contrib)))
        top_feature = feature_names[idx]
        direction = "increased" if contrib[idx] > 0 else "decreased"

        st.markdown(
            f"**Key Factors**: Your **{top_feature}** was the biggest contributor and **{direction}** this predicted risk level."
        )

    st.markdown(
        '<div class="project-footer">College Data Science Project | Built with Streamlit &amp; XGBoost.</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
