import os

import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(
    title="Enterprise Medical Insurance Intelligence API",
    version="2.1.0",
)


# ---------------------------------------------------------
# File Paths
# ---------------------------------------------------------
# Works both when this file is inside a backend/ folder and
# when it is placed directly in the project root.
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)

MODEL_CANDIDATES = [
    os.path.join(CURRENT_DIR, "Trained_Model", "insurance_charges_model.pkl"),
    os.path.join(PARENT_DIR, "Trained_Model", "insurance_charges_model.pkl"),
]

MODEL_PATH = next((p for p in MODEL_CANDIDATES if os.path.exists(p)), None)

if MODEL_PATH is None:
    raise RuntimeError(
        "Could not find Trained_Model/insurance_charges_model.pkl. "
        f"Checked: {MODEL_CANDIDATES}"
    )

MODEL_DIR = os.path.dirname(MODEL_PATH)
SHAP_BACKGROUND_PATH = os.path.join(MODEL_DIR, "shap_background.pkl")
SHAP_FEATURE_NAMES_PATH = os.path.join(MODEL_DIR, "shap_feature_names.pkl")


# ---------------------------------------------------------
# Load Model
# ---------------------------------------------------------
try:
    model_pipeline = joblib.load(MODEL_PATH)
except Exception as e:
    raise RuntimeError(
        f"Failed to load model from path {MODEL_PATH}: {str(e)}"
    )


# ---------------------------------------------------------
# Load SHAP Background / Metadata
# ---------------------------------------------------------
shap_background = None
shap_feature_names = None
shap_background_error = None

try:
    if os.path.exists(SHAP_BACKGROUND_PATH):
        shap_background = joblib.load(SHAP_BACKGROUND_PATH)
    else:
        shap_background_error = (
            "SHAP background file is missing. Run the final training/export "
            "cell in Untitled.ipynb once to create shap_background.pkl."
        )

    if os.path.exists(SHAP_FEATURE_NAMES_PATH):
        shap_feature_names = joblib.load(SHAP_FEATURE_NAMES_PATH)
    else:
        shap_background_error = (
            "SHAP feature-name file is missing. Run the final training/export "
            "cell in Untitled.ipynb once to create shap_feature_names.pkl."
        )
except Exception as e:
    shap_background_error = f"Failed to load SHAP artifacts: {str(e)}"


class InsuranceInputSchema(BaseModel):
    age: int = Field(
        ..., ge=18, le=100, description="Age of the primary beneficiary"
    )
    sex: str = Field(..., description="Gender: male or female")
    bmi: float = Field(..., ge=10.0, le=60.0, description="Body mass index")
    children: int = Field(
        ..., ge=0, le=20, description="Number of dependents covered"
    )
    smoker: str = Field(..., description="Smoking status: yes or no")


def preprocess_features(df: pd.DataFrame) -> pd.DataFrame:
    """Match the feature engineering used during model training."""
    df_transformed = df.copy()

    df_transformed["is_obese"] = np.where(
        df_transformed["bmi"] >= 30.0, "yes", "no"
    )

    df_transformed["has_children"] = np.where(
        df_transformed["children"] > 0, "yes", "no"
    )

    df_transformed["obese_smoker"] = np.where(
        (df_transformed["bmi"] >= 30.0)
        & (df_transformed["smoker"].str.lower() == "yes"),
        "yes",
        "no",
    )

    return df_transformed


def get_pipeline_components():
    """Return the fitted preprocessor and final estimator."""
    if not hasattr(model_pipeline, "named_steps"):
        return None, model_pipeline

    named_steps = model_pipeline.named_steps

    # The training notebook uses ('preprocessor', ...) and ('model', ...).
    preprocessor = named_steps.get("preprocessor")
    if preprocessor is None and len(model_pipeline.steps) > 1:
        preprocessor = model_pipeline.steps[0][1]

    regressor = named_steps.get("model")
    if regressor is None:
        regressor = named_steps.get("regressor")
    if regressor is None:
        regressor = model_pipeline.steps[-1][1]

    return preprocessor, regressor


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model_pipeline is not None,
        "model_path": MODEL_PATH,
        "shap_ready": (
            shap_background is not None and shap_feature_names is not None
        ),
        "shap_background_path": SHAP_BACKGROUND_PATH,
    }


@app.post("/predict")
def predict_charges(data: InsuranceInputSchema):
    try:
        input_dict = data.model_dump()
        input_df = pd.DataFrame([input_dict])
        processed_df = preprocess_features(input_df)

        prediction = model_pipeline.predict(processed_df)[0]

        return {
            "status": "success",
            "prediction_amount": float(np.round(prediction, 2)),
            "processed_input": processed_df.to_dict(orient="records")[0],
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Prediction Error: {str(e)}"
        )


@app.post("/explain")
def explain_prediction(data: InsuranceInputSchema):
    """Return local SHAP contributions for the latest prediction.

    IMPORTANT: the current user row is NOT used as the SHAP background.
    A representative training-data background saved by the notebook is used.
    """
    try:
        if shap_background is None or shap_feature_names is None:
            raise HTTPException(
                status_code=503,
                detail=shap_background_error
                or "SHAP artifacts are not available.",
            )

        input_dict = data.model_dump()
        input_df = pd.DataFrame([input_dict])
        processed_df = preprocess_features(input_df)

        preprocessor, regressor = get_pipeline_components()

        if preprocessor is not None:
            transformed_input = preprocessor.transform(processed_df)
        else:
            transformed_input = processed_df

        if hasattr(transformed_input, "toarray"):
            transformed_input = transformed_input.toarray()
        transformed_input = np.asarray(transformed_input, dtype=float)

        background = shap_background
        if hasattr(background, "toarray"):
            background = background.toarray()
        background = np.asarray(background, dtype=float)

        # Make sure the saved background and current model input have the
        # exact same number of transformed features.
        if background.ndim != 2:
            raise ValueError(
                f"SHAP background must be 2-D, got shape {background.shape}."
            )

        if transformed_input.shape[1] != background.shape[1]:
            raise ValueError(
                "SHAP background/input feature mismatch: "
                f"background has {background.shape[1]} features, "
                f"input has {transformed_input.shape[1]} features. "
                "Re-run the final notebook export cell so SHAP artifacts "
                "are generated from the current trained pipeline."
            )

        if len(shap_feature_names) != transformed_input.shape[1]:
            raise ValueError(
                "SHAP feature-name mismatch: "
                f"got {len(shap_feature_names)} names for "
                f"{transformed_input.shape[1]} transformed features. "
                "Re-run the final notebook export cell."
            )

        # -----------------------------------------------------
        # Correct SHAP setup:
        # background = representative training rows
        # data       = current user's transformed row
        # -----------------------------------------------------
        explainer = shap.Explainer(regressor, background)
        shap_values = explainer(transformed_input)

        values = np.asarray(shap_values.values, dtype=float)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        elif values.ndim > 2:
            values = np.squeeze(values)
            if values.ndim == 1:
                values = values.reshape(1, -1)

        values = values[0]

        base_values = np.asarray(shap_values.base_values, dtype=float).reshape(-1)
        base_value = float(base_values[0]) if len(base_values) else None

        # Verification: SHAP decomposition should reconstruct the model
        # prediction (up to floating-point tolerance).
        prediction = float(model_pipeline.predict(processed_df)[0])
        reconstructed = (
            base_value + float(np.sum(values))
            if base_value is not None
            else None
        )

        return {
            "status": "success",
            "prediction": prediction,
            "base_value": base_value,
            "reconstructed_prediction": reconstructed,
            "shap_values": [float(x) for x in values],
            "feature_names": [str(f) for f in shap_feature_names],
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"SHAP Computation Error: {str(e)}"
        )