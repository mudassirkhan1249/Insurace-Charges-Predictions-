import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import shap
import streamlit as st


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Medical Insurance AI Hub",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# =========================================================
# PATH CONFIGURATION
# =========================================================
# This file is:
#
# Project/
#     frontend/
#         app.py
#
# Therefore project root = frontend.parent

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent

MODEL_DIR = PROJECT_ROOT / "Trained_Model"

MODEL_PATH = MODEL_DIR / "insurance_charges_model.pkl"
SHAP_BACKGROUND_PATH = MODEL_DIR / "shap_background.pkl"
SHAP_FEATURE_NAMES_PATH = MODEL_DIR / "shap_feature_names.pkl"


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    /* Main page */
    .main {
        padding-top: 1rem;
    }

    /* Header */
    .main-header {
        font-size: clamp(24px, 5vw, 38px);
        font-weight: 800;
        text-align: center;
        margin-bottom: 10px;
    }

    .subtitle {
        text-align: center;
        color: #94A3B8;
        font-size: 15px;
        margin-bottom: 25px;
    }

    /* Cards */
    .info-card {
        padding: 18px;
        border-radius: 14px;
        border: 1px solid rgba(148, 163, 184, 0.25);
        background: rgba(30, 41, 59, 0.65);
        margin-bottom: 15px;
    }

    /* Mobile optimization */
    @media (max-width: 768px) {

        .main-header {
            font-size: 25px;
            line-height: 1.25;
        }

        .subtitle {
            font-size: 13px;
        }

        .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }

        div[data-testid="stMetric"] {
            padding: 8px;
        }

        div[data-testid="stMetricValue"] {
            font-size: 22px;
        }

        .stButton button {
            min-height: 48px;
            font-size: 15px;
        }

    }

    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# HEADER
# =========================================================

st.markdown(
    "<div class='main-header'>🏥 Medical Insurance Intelligence & Prediction System</div>",
    unsafe_allow_html=True,
)

st.markdown(
    "<div class='subtitle'>AI-powered insurance premium prediction with Explainable AI</div>",
    unsafe_allow_html=True,
)


# =========================================================
# MODEL LOADING
# =========================================================

@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found:\n{MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)


@st.cache_resource
def load_shap_artifacts():

    background = None
    feature_names = None

    background_error = None

    if SHAP_BACKGROUND_PATH.exists():

        try:
            background = joblib.load(SHAP_BACKGROUND_PATH)
        except Exception as e:
            background_error = str(e)

    else:

        background_error = (
            "shap_background.pkl was not found."
        )

    if SHAP_FEATURE_NAMES_PATH.exists():

        try:
            feature_names = joblib.load(
                SHAP_FEATURE_NAMES_PATH
            )
        except Exception as e:
            background_error = str(e)

    else:

        background_error = (
            "shap_feature_names.pkl was not found."
        )

    return background, feature_names, background_error


# =========================================================
# LOAD EVERYTHING
# =========================================================

try:

    model_pipeline = load_model()

    (
        shap_background,
        shap_feature_names,
        shap_error,
    ) = load_shap_artifacts()

    MODEL_READY = True

except Exception as e:

    MODEL_READY = False

    st.error(
        "❌ Model could not be loaded."
    )

    st.code(str(e))

    st.stop()


# =========================================================
# FEATURE ENGINEERING
# =========================================================

def preprocess_features(df: pd.DataFrame) -> pd.DataFrame:

    df_transformed = df.copy()

    # Obesity flag
    df_transformed["is_obese"] = np.where(
        df_transformed["bmi"] >= 30.0,
        "yes",
        "no",
    )

    # Children flag
    df_transformed["has_children"] = np.where(
        df_transformed["children"] > 0,
        "yes",
        "no",
    )

    # Obese smoker interaction
    df_transformed["obese_smoker"] = np.where(
        (df_transformed["bmi"] >= 30.0)
        & (
            df_transformed["smoker"]
            .astype(str)
            .str.lower()
            == "yes"
        ),
        "yes",
        "no",
    )

    return df_transformed


# =========================================================
# PIPELINE COMPONENTS
# =========================================================

def get_pipeline_components():

    if not hasattr(model_pipeline, "named_steps"):

        return None, model_pipeline

    named_steps = model_pipeline.named_steps

    preprocessor = named_steps.get(
        "preprocessor"
    )

    if (
        preprocessor is None
        and len(model_pipeline.steps) > 1
    ):

        preprocessor = model_pipeline.steps[0][1]

    regressor = named_steps.get("model")

    if regressor is None:

        regressor = named_steps.get(
            "regressor"
        )

    if regressor is None:

        regressor = model_pipeline.steps[-1][1]

    return preprocessor, regressor


# =========================================================
# PREDICTION FUNCTION
# =========================================================

def predict_insurance(
    age,
    sex,
    bmi,
    children,
    smoker,
):

    input_df = pd.DataFrame(
        [
            {
                "age": age,
                "sex": sex,
                "bmi": bmi,
                "children": children,
                "smoker": smoker,
            }
        ]
    )

    processed_df = preprocess_features(
        input_df
    )

    prediction = model_pipeline.predict(
        processed_df
    )[0]

    return (
        float(prediction),
        processed_df,
    )


# =========================================================
# SHAP FUNCTION
# =========================================================

def explain_prediction(
    processed_df,
):

    if (
        shap_background is None
        or shap_feature_names is None
    ):

        raise RuntimeError(
            shap_error
            or "SHAP artifacts are unavailable."
        )

    preprocessor, regressor = (
        get_pipeline_components()
    )

    # Transform user input
    if preprocessor is not None:

        transformed_input = (
            preprocessor.transform(
                processed_df
            )
        )

    else:

        transformed_input = processed_df

    # Convert sparse matrix
    if hasattr(
        transformed_input,
        "toarray",
    ):

        transformed_input = (
            transformed_input.toarray()
        )

    transformed_input = np.asarray(
        transformed_input,
        dtype=float,
    )

    # Prepare background
    background = shap_background

    if hasattr(
        background,
        "toarray",
    ):

        background = background.toarray()

    background = np.asarray(
        background,
        dtype=float,
    )

    # Validate dimensions
    if background.ndim != 2:

        raise ValueError(
            "SHAP background must be 2-D. "
            f"Received shape: {background.shape}"
        )

    if (
        transformed_input.shape[1]
        != background.shape[1]
    ):

        raise ValueError(
            "SHAP feature mismatch.\n\n"
            f"Background features: {background.shape[1]}\n"
            f"Input features: {transformed_input.shape[1]}"
        )

    if (
        len(shap_feature_names)
        != transformed_input.shape[1]
    ):

        raise ValueError(
            "SHAP feature-name mismatch.\n\n"
            f"Feature names: {len(shap_feature_names)}\n"
            f"Input features: {transformed_input.shape[1]}"
        )

    # Create SHAP explainer
    explainer = shap.Explainer(
        regressor,
        background,
    )

    shap_values = explainer(
        transformed_input
    )

    values = np.asarray(
        shap_values.values,
        dtype=float,
    )

    if values.ndim == 1:

        values = values.reshape(
            1, -1
        )

    elif values.ndim > 2:

        values = np.squeeze(values)

        if values.ndim == 1:

            values = values.reshape(
                1, -1
            )

    values = values[0]

    # Base value
    base_values = np.asarray(
        shap_values.base_values,
        dtype=float,
    ).reshape(-1)

    base_value = (
        float(base_values[0])
        if len(base_values)
        else None
    )

    # Model prediction
    prediction = float(
        model_pipeline.predict(
            processed_df
        )[0]
    )

    # SHAP reconstruction
    reconstructed = None

    if base_value is not None:

        reconstructed = (
            base_value
            + float(np.sum(values))
        )

    return (
        values,
        shap_feature_names,
        base_value,
        prediction,
        reconstructed,
    )


# =========================================================
# SESSION STATE
# =========================================================

if "last_payload" not in st.session_state:

    st.session_state.last_payload = None

if "last_prediction" not in st.session_state:

    st.session_state.last_prediction = None

if "last_processed_df" not in st.session_state:

    st.session_state.last_processed_df = None


# =========================================================
# NAVIGATION
# =========================================================

tab_predict, tab_analytics, tab_about = st.tabs(
    [
        "🎯 Prediction Hub",
        "📊 Explainability & Analytics",
        "📖 About",
    ]
)


# =========================================================
# TAB 1 — PREDICTION
# =========================================================

with tab_predict:

    st.subheader(
        "Client Policy Profile"
    )

    st.write(
        "Enter beneficiary information to estimate the annual insurance premium."
    )

    st.divider()

    col_a, col_b = st.columns(
        2,
        gap="large",
    )

    # -----------------------------------------------------
    # Demographics
    # -----------------------------------------------------

    with col_a:

        st.markdown(
            "### 👤 Demographic Details"
        )

        age = st.slider(
            "Beneficiary Age",
            min_value=18,
            max_value=100,
            value=32,
        )

        sex = st.radio(
            "Biological Sex",
            options=[
                "female",
                "male",
            ],
            horizontal=True,
        )

        bmi = st.number_input(
            "Body Mass Index (BMI)",
            min_value=10.0,
            max_value=60.0,
            value=28.4,
            step=0.1,
        )

    # -----------------------------------------------------
    # Risk metrics
    # -----------------------------------------------------

    with col_b:

        st.markdown(
            "### 🩺 Health & Risk Metrics"
        )

        children = st.number_input(
            "Number of Children / Dependents",
            min_value=0,
            max_value=20,
            value=1,
            step=1,
        )

        smoker = st.radio(
            "Smoking History",
            options=[
                "no",
                "yes",
            ],
            horizontal=True,
        )

    st.divider()

    # -----------------------------------------------------
    # Derived information
    # -----------------------------------------------------

    info_col1, info_col2, info_col3 = st.columns(
        3
    )

    bmi_category = (
        "Obese"
        if bmi >= 30
        else "Normal / Overweight"
    )

    obese_smoker = (
        bmi >= 30
        and smoker == "yes"
    )

    info_col1.metric(
        "BMI Category",
        bmi_category,
    )

    info_col2.metric(
        "Dependents",
        children,
    )

    info_col3.metric(
        "Obese Smoker",
        "YES" if obese_smoker else "NO",
    )

    st.divider()

    # -----------------------------------------------------
    # Prediction button
    # -----------------------------------------------------

    predict_btn = st.button(
        "🚀 Calculate Estimated Premium",
        use_container_width=True,
        type="primary",
    )

    if predict_btn:

        with st.spinner(
            "Running Machine Learning inference..."
        ):

            try:

                prediction, processed_df = (
                    predict_insurance(
                        age=age,
                        sex=sex,
                        bmi=bmi,
                        children=children,
                        smoker=smoker,
                    )
                )

                # Save state
                st.session_state.last_payload = {
                    "age": age,
                    "sex": sex,
                    "bmi": bmi,
                    "children": children,
                    "smoker": smoker,
                }

                st.session_state.last_prediction = (
                    prediction
                )

                st.session_state.last_processed_df = (
                    processed_df
                )

                st.success(
                    "✅ Prediction computed successfully!"
                )

                # Results
                result1, result2, result3 = (
                    st.columns(3)
                )

                result1.metric(
                    "Estimated Annual Premium",
                    f"${prediction:,.2f}",
                )

                result2.metric(
                    "BMI Classification",
                    bmi_category,
                )

                result3.metric(
                    "High-Risk Profile",
                    "YES" if obese_smoker else "NO",
                )

            except Exception as e:

                st.error(
                    "Prediction failed."
                )

                st.exception(e)


# =========================================================
# TAB 2 — SHAP + ANALYTICS
# =========================================================

with tab_analytics:

    st.subheader(
        "Model Decision Interpretation & Analytics"
    )

    # -----------------------------------------------------
    # SHAP
    # -----------------------------------------------------

    if (
        st.session_state.last_payload
        is None
    ):

        st.info(
            "👈 First generate a prediction from the Prediction Hub."
        )

    else:

        st.markdown(
            "### 🔍 Feature Attribution"
        )

        with st.spinner(
            "Calculating SHAP explanation..."
        ):

            try:

                (
                    shap_values,
                    feature_names,
                    base_value,
                    prediction,
                    reconstructed,
                ) = explain_prediction(
                    st.session_state.last_processed_df
                )

                flat_values = np.asarray(
                    shap_values,
                    dtype=float,
                ).reshape(-1)

                if (
                    len(feature_names)
                    != len(flat_values)
                ):

                    st.error(
                        "SHAP feature/value mismatch."
                    )

                else:

                    cleaned_names = [
                        str(feature)
                        .split("__")[-1]
                        for feature in feature_names
                    ]

                    df_shap = pd.DataFrame(
                        {
                            "Feature": cleaned_names,
                            "SHAP Contribution ($)": flat_values,
                        }
                    )

                    df_shap["abs_val"] = (
                        df_shap[
                            "SHAP Contribution ($)"
                        ].abs()
                    )

                    df_shap = (
                        df_shap
                        .sort_values(
                            "abs_val",
                            ascending=True,
                        )
                        .reset_index(
                            drop=True
                        )
                    )

                    # Colors
                    bar_colors = [
                        "#EF4444"
                        if value >= 0
                        else "#10B981"
                        for value in df_shap[
                            "SHAP Contribution ($)"
                        ]
                    ]

                    # Plot
                    fig_shap = go.Figure()

                    fig_shap.add_trace(
                        go.Bar(
                            x=df_shap[
                                "SHAP Contribution ($)"
                            ],
                            y=df_shap[
                                "Feature"
                            ],
                            orientation="h",
                            marker=dict(
                                color=bar_colors,
                                line=dict(
                                    width=1
                                ),
                            ),
                            text=[
                                f"{value:+,.2f}"
                                for value in df_shap[
                                    "SHAP Contribution ($)"
                                ]
                            ],
                            textposition="outside",
                            cliponaxis=False,
                            hovertemplate=(
                                "<b>%{y}</b><br>"
                                "SHAP contribution: %{x:+,.2f}"
                                "<extra></extra>"
                            ),
                        )
                    )

                    max_abs = float(
                        df_shap[
                            "SHAP Contribution ($)"
                        ].abs().max()
                    )

                    axis_limit = max(
                        max_abs * 1.25,
                        1.0,
                    )

                    fig_shap.update_layout(
                        template="plotly_dark",
                        title=(
                            "Local Feature Attribution "
                            "to Insurance Cost"
                        ),
                        xaxis_title=(
                            "SHAP Contribution ($)"
                        ),
                        yaxis_title="Feature",
                        height=500,
                        margin=dict(
                            l=70,
                            r=100,
                            t=70,
                            b=60,
                        ),
                        xaxis=dict(
                            range=[
                                -axis_limit,
                                axis_limit,
                            ],
                            showgrid=True,
                            zeroline=True,
                            zerolinewidth=2,
                        ),
                        showlegend=False,
                    )

                    st.plotly_chart(
                        fig_shap,
                        use_container_width=True,
                    )

                    # -------------------------------------------------
                    # SHAP metrics
                    # -------------------------------------------------

                    metric1, metric2, metric3 = (
                        st.columns(3)
                    )

                    metric1.metric(
                        "Model Prediction",
                        f"${prediction:,.2f}",
                    )

                    if base_value is not None:

                        metric2.metric(
                            "SHAP Base Value",
                            f"${base_value:,.2f}",
                        )

                    if (
                        reconstructed is not None
                    ):

                        metric3.metric(
                            "SHAP Reconstructed",
                            f"${reconstructed:,.2f}",
                        )

                    st.markdown(
                        """
                        **How to read the chart:**

                        🔴 **Positive SHAP value** → pushes the prediction higher.

                        🟢 **Negative SHAP value** → pushes the prediction lower.

                        📈 **Larger absolute value** → stronger influence on this prediction.

                        The SHAP base value represents the model's reference prediction.
                        """
                    )

                    # -------------------------------------------------
                    # Automatic interpretation
                    # -------------------------------------------------

                    top_features = (
                        df_shap
                        .nlargest(
                            3,
                            "abs_val",
                        )
                    )

                    st.markdown(
                        "### 🧠 Local Interpretation"
                    )

                    for _, row in (
                        top_features.iterrows()
                    ):

                        value = float(
                            row[
                                "SHAP Contribution ($)"
                            ]
                        )

                        direction = (
                            "increases"
                            if value >= 0
                            else "decreases"
                        )

                        st.write(
                            f"• **{row['Feature']}** "
                            f"{direction} the predicted premium "
                            f"by approximately "
                            f"**${abs(value):,.2f}**."
                        )

            except Exception as e:

                st.error(
                    "SHAP explanation failed."
                )

                st.exception(e)

    # =====================================================
    # SIMULATED ANALYTICS
    # =====================================================

    st.divider()

    st.markdown(
        "### 📊 Multi-Variable Risk Analysis"
    )

    # Fixed seed so charts remain consistent
    np.random.seed(42)

    sample_size = 500

    sim_age = np.random.randint(
        18,
        65,
        size=sample_size,
    )

    sim_bmi = np.random.normal(
        30,
        6,
        size=sample_size,
    ).clip(
        15,
        50,
    )

    sim_smoker = np.random.choice(
        [
            "yes",
            "no",
        ],
        size=sample_size,
        p=[
            0.2,
            0.8,
        ],
    )

    sim_children = np.random.choice(
        [
            0,
            1,
            2,
            3,
            4,
        ],
        size=sample_size,
        p=[
            0.4,
            0.3,
            0.15,
            0.1,
            0.05,
        ],
    )

    sim_charges = (
        1000
        + (sim_age * 250)
        + (sim_bmi * 320)
        + np.where(
            sim_smoker == "yes",
            23000,
            0,
        )
        + np.where(
            (sim_bmi >= 30)
            & (sim_smoker == "yes"),
            12000,
            0,
        )
        + (sim_children * 500)
        + np.random.normal(
            0,
            1500,
            size=sample_size,
        )
    )

    df_sim = pd.DataFrame(
        {
            "Age": sim_age,
            "BMI": sim_bmi,
            "Smoker": sim_smoker,
            "Children": sim_children,
            "Charges": sim_charges,
            "Obese_Smoker": np.where(
                (sim_bmi >= 30)
                & (sim_smoker == "yes"),
                "Obese Smoker",
                "Other",
            ),
        }
    )

    # -----------------------------------------------------
    # BMI vs Charges
    # -----------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        fig_bmi = px.scatter(
            df_sim,
            x="BMI",
            y="Charges",
            color="Smoker",
            hover_data=[
                "Age",
                "Children",
            ],
            title=(
                "BMI vs Charges Distribution"
            ),
            color_discrete_map={
                "yes": "#EF4444",
                "no": "#10B981",
            },
            template="plotly_dark",
        )

        st.plotly_chart(
            fig_bmi,
            use_container_width=True,
        )

    # -----------------------------------------------------
    # Age vs Charges
    # -----------------------------------------------------

    with col2:

        fig_age = px.scatter(
            df_sim,
            x="Age",
            y="Charges",
            color="Smoker",
            hover_data=[
                "BMI"
            ],
            title=(
                "Age vs Insurance Premium Trends"
            ),
            color_discrete_map={
                "yes": "#EF4444",
                "no": "#10B981",
            },
            template="plotly_dark",
        )

        st.plotly_chart(
            fig_age,
            use_container_width=True,
        )

    # -----------------------------------------------------
    # Smoker box plot
    # -----------------------------------------------------

    col3, col4 = st.columns(2)

    with col3:

        fig_box = px.box(
            df_sim,
            x="Smoker",
            y="Charges",
            color="Smoker",
            title=(
                "Insurance Cost by Smoking Status"
            ),
            color_discrete_map={
                "yes": "#EF4444",
                "no": "#10B981",
            },
            template="plotly_dark",
        )

        st.plotly_chart(
            fig_box,
            use_container_width=True,
        )

    # -----------------------------------------------------
    # Obese smoker
    # -----------------------------------------------------

    with col4:

        fig_risk = px.box(
            df_sim,
            x="Obese_Smoker",
            y="Charges",
            color="Obese_Smoker",
            title=(
                "High-Risk Profile: Obese Smoker"
            ),
            color_discrete_map={
                "Obese Smoker": "#DC2626",
                "Other": "#3B82F6",
            },
            template="plotly_dark",
        )

        st.plotly_chart(
            fig_risk,
            use_container_width=True,
        )


# =========================================================
# TAB 3 — ABOUT
# =========================================================

with tab_about:

    st.subheader(
        "📖 Project Architecture & Documentation"
    )

    st.markdown(
        """
        ## 📌 Project Overview

        The **Medical Insurance Intelligence System** is an
        end-to-end Machine Learning application designed to
        estimate individual annual insurance costs using
        demographic and health-related features.

        ---

        ## 🛠️ Technology Stack

        **Frontend**

        - Streamlit
        - Plotly
        - Responsive UI

        **Machine Learning**

        - Scikit-Learn
        - Joblib
        - Pandas
        - NumPy

        **Explainable AI**

        - SHAP

        ---

        ## 🧠 Model Features

        The application uses:

        - Age
        - Sex
        - BMI
        - Number of children/dependents
        - Smoking status
        - Obesity indicator
        - Children indicator
        - Obese-smoker interaction

        ---

        ## 🚀 Deployment

        This application is designed to run directly on
        **Streamlit Community Cloud**.

        No FastAPI server or external backend is required.

        ---

        ## 👨‍💻 Developer

        **Mudassir Khan**

        **Specialization:** AI & Data Science
        """
    )

    st.divider()

    st.markdown(
        "### 🔧 Model Status"
    )

    status1, status2, status3 = (
        st.columns(3)
    )

    status1.metric(
        "ML Model",
        "Loaded" if MODEL_READY else "Error",
    )

    status2.metric(
        "SHAP Background",
        (
            "Loaded"
            if shap_background is not None
            else "Missing"
        ),
    )

    status3.metric(
        "SHAP Features",
        (
            "Loaded"
            if shap_feature_names is not None
            else "Missing"
        ),
    )
