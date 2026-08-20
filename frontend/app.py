import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import seaborn as sns
import streamlit as st

# Application Configuration
st.set_page_config(
    page_title="Medical Insurance AI Hub",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

BACKEND_URL = "http://127.0.0.1:8000"

# Custom Styling
st.markdown(
    """
    <style>
    .main-header { font-size: 32px; font-weight: 700; color: #3B82F6; text-align: center; margin-bottom: 20px; }
    .card-box { background-color: #1E293B; border-radius: 10px; padding: 20px; border: 1px solid #334155; }
    </style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    "<div class='main-header'>🏥 Medical Insurance Intelligence & Prediction System</div>",
    unsafe_allow_html=True,
)

# Navigation Tabs
tab_predict, tab_analytics, tab_about = st.tabs([
    "🎯 Prediction Hub",
    "📊 Explainability & Analytics (SHAP)",
    "📖 About & Documentation",
])

# ---------------------------------------------------------
# TAB 1: PREDICTION HUB
# ---------------------------------------------------------
with tab_predict:
    st.subheader("Client Policy Profile")
    st.write(
        "Fill in the beneficiary details below to calculate the estimated"
        " insurance premium."
    )

    col_a, col_b = st.columns([1, 1], gap="medium")

    with col_a:
        with st.container():
            st.markdown("#### Demographic Details")
            age = st.slider(
                "Beneficiary Age", min_value=18, max_value=100, value=32
            )
            sex = st.radio(
                "Biological Sex", options=["female", "male"], horizontal=True
            )
            bmi = st.number_input(
                "Body Mass Index (BMI)",
                min_value=10.0,
                max_value=55.0,
                value=28.4,
                step=0.1,
            )

    with col_b:
        with st.container():
            st.markdown("#### Health & Risk Metrics")
            children = st.number_input(
                "Number of Children / Dependents",
                min_value=0,
                max_value=10,
                value=1,
            )
            smoker = st.radio(
                "Smoking History", options=["no", "yes"], horizontal=True
            )

    st.divider()

    predict_btn = st.button(
        "🚀 Calculate Estimated Premium",
        use_container_width=True,
        type="primary",
    )

    if predict_btn:
        payload = {
            "age": age,
            "sex": sex,
            "bmi": bmi,
            "children": children,
            "smoker": smoker,
        }

        with st.spinner("Executing Machine Learning Inference Pipeline..."):
            try:
                res = requests.post(
                    f"{BACKEND_URL}/predict", json=payload, timeout=10
                )
                if res.status_code == 200:
                    data = res.json()
                    val = data["prediction_amount"]

                    st.session_state["last_payload"] = payload
                    st.session_state["last_prediction"] = val

                    st.success("Prediction Computed Successfully!")

                    res_col1, res_col2, res_col3 = st.columns(3)
                    res_col1.metric(
                        label="Estimated Annual Premium", value=f"${val:,.2f}"
                    )
                    res_col2.metric(
                        label="BMI Category Classification",
                        value="Obese" if bmi >= 30 else "Normal / Overweight",
                    )
                    res_col3.metric(
                        label="High-Risk Profile (Obese Smoker)",
                        value=(
                            "YES" if (bmi >= 30 and smoker == "yes") else "NO"
                        ),
                    )

                else:
                    st.error(
                        f"Backend Server Error: Response Code {res.status_code}"
                    )
            except Exception as ex:
                st.error(
                    f"Connection Failure: Please verify that the FastAPI"
                    f" backend is running on {BACKEND_URL}. Details: {str(ex)}"
                )

# ---------------------------------------------------------
# TAB 2: EXPLAINABILITY & ANALYTICS (SHAP)
# ---------------------------------------------------------
with tab_analytics:
    st.subheader("Model Decision Interpretation & Analytical Insights")

    if "last_payload" not in st.session_state:
        st.info(
            "⚠️ Please navigate to the 'Prediction Hub' tab and compute a "
            "prediction first to generate custom SHAP explainability insights."
        )
    else:
        st.markdown("### Feature Attribution & SHAP Value Breakdown")

        with st.spinner("Calculating SHAP values via backend microservice..."):
            try:
                payload = st.session_state["last_payload"]
                shap_res = requests.post(
                    f"{BACKEND_URL}/explain", json=payload, timeout=30
                )

                if shap_res.status_code == 200:
                    shap_data = shap_res.json()
                    feature_names = shap_data.get("feature_names", [])
                    shap_values = shap_data.get("shap_values", [])
                    base_value = shap_data.get("base_value")
                    prediction = shap_data.get("prediction")
                    reconstructed = shap_data.get("reconstructed_prediction")

                    # SHAP values and feature names must correspond one-to-one.
                    flat_vals = np.asarray(shap_values, dtype=float).reshape(-1)

                    if len(feature_names) != len(flat_vals):
                        st.error(
                            "SHAP feature/value mismatch: "
                            f"{len(feature_names)} feature names but "
                            f"{len(flat_vals)} SHAP values."
                        )
                    else:
                        cleaned_names = [
                            str(f).split("__")[-1] for f in feature_names
                        ]

                        df_shap = pd.DataFrame({
                            "Feature": cleaned_names,
                            "SHAP Contribution ($)": flat_vals,
                        })

                        df_shap["abs_val"] = df_shap[
                            "SHAP Contribution ($)"
                        ].abs()
                        df_shap = df_shap.sort_values(
                            by="abs_val", ascending=True
                        ).reset_index(drop=True)

                        bar_colors = [
                            "#EF4444" if val >= 0 else "#10B981"
                            for val in df_shap["SHAP Contribution ($)"]
                        ]

                        fig_shap = go.Figure()
                        fig_shap.add_trace(
                            go.Bar(
                                x=df_shap["SHAP Contribution ($)"],
                                y=df_shap["Feature"],
                                orientation="h",
                                marker=dict(
                                    color=bar_colors,
                                    line=dict(width=1),
                                ),
                                text=[
                                    f"{v:+,.2f}"
                                    for v in df_shap["SHAP Contribution ($)"]
                                ],
                                textposition="outside",
                                cliponaxis=False,
                                hovertemplate=(
                                    "<b>%{y}</b><br>"
                                    "SHAP contribution: %{x:+,.2f}<extra></extra>"
                                ),
                            )
                        )

                        # Give the x-axis enough room for text labels while
                        # keeping zero clearly visible.
                        max_abs = float(
                            df_shap["SHAP Contribution ($)"].abs().max()
                        )
                        axis_limit = max(max_abs * 1.25, 1.0)

                        fig_shap.update_layout(
                            template="plotly_dark",
                            title="Local Feature Attribution to Insurance Cost Prediction",
                            xaxis_title="SHAP Contribution ($)",
                            yaxis_title="Feature",
                            height=500,
                            margin=dict(l=70, r=100, t=60, b=60),
                            xaxis=dict(
                                range=[-axis_limit, axis_limit],
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
                        # Prediction / SHAP consistency check
                        # -------------------------------------------------
                        metric1, metric2, metric3 = st.columns(3)
                        if prediction is not None:
                            metric1.metric(
                                "Model Prediction",
                                f"${prediction:,.2f}",
                            )
                        if base_value is not None:
                            metric2.metric(
                                "SHAP Base Value",
                                f"${base_value:,.2f}",
                            )
                        if reconstructed is not None and prediction is not None:
                            metric3.metric(
                                "SHAP Reconstructed",
                                f"${reconstructed:,.2f}",
                            )

                        st.markdown("""
                        - **Red / positive:** this feature pushes the prediction **up**.
                        - **Green / negative:** this feature pushes the prediction **down**.
                        - **Larger absolute value:** stronger influence on this specific prediction.
                        - The **base value** is the model's reference prediction from the training-data background.
                        """)

                        # -------------------------------------------------
                        # Automatic local interpretation
                        # -------------------------------------------------
                        top = df_shap.nlargest(3, "abs_val")
                        if not top.empty:
                            st.markdown("#### Local Interpretation")
                            for _, row in top.iterrows():
                                value = float(row["SHAP Contribution ($)"])
                                direction = "increases" if value >= 0 else "decreases"
                                st.write(
                                    f"• **{row['Feature']}** {direction} the predicted premium "
                                    f"by approximately **${abs(value):,.2f}** for this profile."
                                )

                else:
                    try:
                        error_body = shap_res.json()
                        detail = error_body.get("detail", str(error_body))
                    except Exception:
                        detail = shap_res.text or "Unknown backend error"

                    st.error(
                        f"SHAP backend error ({shap_res.status_code}): {detail}"
                    )

            except requests.RequestException as e:
                st.error(
                    "Could not connect to the FastAPI SHAP endpoint. "
                    f"Make sure the backend is running. Details: {e}"
                )
            except Exception as e:
                st.error(f"Failed to generate SHAP visualization: {str(e)}")

    st.divider()
    st.markdown("### Multi-Variable Risk Analysis")

    np.random.seed(42)
    sample_size = 500
    sim_age = np.random.randint(18, 65, size=sample_size)
    sim_bmi = np.random.normal(30, 6, size=sample_size).clip(15, 50)
    sim_smoker = np.random.choice(
        ["yes", "no"], size=sample_size, p=[0.2, 0.8]
    )
    sim_children = np.random.choice(
        [0, 1, 2, 3, 4], size=sample_size, p=[0.4, 0.3, 0.15, 0.1, 0.05]
    )

    sim_charges = (
        1000
        + (sim_age * 250)
        + (sim_bmi * 320)
        + (np.where(sim_smoker == "yes", 23000, 0))
        + (np.where((sim_bmi >= 30) & (sim_smoker == "yes"), 12000, 0))
        + (sim_children * 500)
        + np.random.normal(0, 1500, size=sample_size)
    )

    df_sim = pd.DataFrame({
        "Age": sim_age,
        "BMI": sim_bmi,
        "Smoker": sim_smoker,
        "Children": sim_children,
        "Charges": sim_charges,
        "Obese_Smoker": np.where(
            (sim_bmi >= 30) & (sim_smoker == "yes"), "Obese Smoker", "Other"
        ),
    })

    col_g1, col_g2 = st.columns(2)

    with col_g1:
        fig_bmi = px.scatter(
            df_sim,
            x="BMI",
            y="Charges",
            color="Smoker",
            hover_data=["Age", "Children"],
            title="BMI vs Charges Distribution (Segmented by Smoking Status)",
            color_discrete_map={"yes": "#EF4444", "no": "#10B981"},
            template="plotly_dark",
        )
        st.plotly_chart(fig_bmi, use_container_width=True)

    with col_g2:
        fig_age = px.scatter(
            df_sim,
            x="Age",
            y="Charges",
            color="Smoker",
            hover_data=["BMI"],
            title="Age vs Insurance Premium Trends",
            color_discrete_map={"yes": "#EF4444", "no": "#10B981"},
            template="plotly_dark",
        )
        st.plotly_chart(fig_age, use_container_width=True)

    col_g3, col_g4 = st.columns(2)

    with col_g3:
        fig_box = px.box(
            df_sim,
            x="Smoker",
            y="Charges",
            color="Smoker",
            title="Insurance Cost Variance by Smoking Status",
            color_discrete_map={"yes": "#EF4444", "no": "#10B981"},
            template="plotly_dark",
        )
        st.plotly_chart(fig_box, use_container_width=True)

    with col_g4:
        fig_risk = px.box(
            df_sim,
            x="Obese_Smoker",
            y="Charges",
            color="Obese_Smoker",
            title="High-Risk Interaction Profile (Obese Smoker vs Other)",
            color_discrete_map={"Obese Smoker": "#DC2626", "Other": "#3B82F6"},
            template="plotly_dark",
        )
        st.plotly_chart(fig_risk, use_container_width=True)

# ---------------------------------------------------------
# TAB 3: ABOUT & DOCUMENTATION
# ---------------------------------------------------------
with tab_about:
    st.subheader("Project Architecture & Documentation")

    st.markdown("""
    ### 📌 Project Overview
    The **Medical Insurance Intelligence System** is an end-to-end Machine Learning web application engineered to predict individual annual healthcare insurance costs based on demographic and personal health indicators.

    ---

    ### 🛠️ Architecture & Technology Stack
    * **Frontend Interface:** Streamlit (Multi-Tab Navigation, Reactive Session State, Dynamic Plotly Analytics)
    * **Backend REST API:** FastAPI (Asynchronous Endpoints, Pydantic Schema Validation, Error Handling)
    * **Machine Learning Framework:** Scikit-Learn
    * **Explainable AI (XAI):** SHAP

    ---

    ### 👨‍💻 Author & Contact Information
    * **Developer:** Mudassir Khan
    * **Specialization:** AI & Data Science
    """)
