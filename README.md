# 🩺 Medical Insurance Charges Prediction & Explainable AI

> An end-to-end Machine Learning application for predicting medical insurance charges with SHAP-based explainability and an interactive Streamlit dashboard.

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://st886kh7ksfrutwute6o64.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![SHAP](https://img.shields.io/badge/XAI-SHAP-8A2BE2)](https://shap.readthedocs.io/)

## 🔗 Project Links

- 🌐 **Live Application:** https://st886kh7ksfrutwute6o64.streamlit.app/
- 💻 **GitHub Repository:** https://github.com/mudassirkhan1249/Insurace-Charges-Predictions-

---

## 📌 Project Overview

The **Medical Insurance Charges Prediction & Explainable AI** project is an end-to-end Machine Learning application designed to estimate an individual's medical insurance charges from demographic and health-related information.

The project combines:

- Machine Learning regression
- Data preprocessing
- Feature engineering
- Model evaluation
- Saved model artifacts
- Interactive Streamlit UI
- SHAP-based local explainability
- Interactive analytical visualizations

The user can enter a beneficiary profile and receive an estimated insurance charge. The application can also explain the prediction using SHAP feature contributions.

---

## 🎯 Objectives

1. Build a Machine Learning model for medical insurance charge prediction.
2. Apply feature engineering to create additional risk-related features.
3. Save the trained model for inference.
4. Build an interactive web application using Streamlit.
5. Add Explainable AI using SHAP.
6. Visualize relationships between insurance charges and important variables.
7. Demonstrate an end-to-end Machine Learning project from data preparation to deployment.

---

## 🧠 Input Features

| Feature | Description |
|---|---|
| `age` | Age of the primary beneficiary |
| `sex` | Biological sex |
| `bmi` | Body Mass Index |
| `children` | Number of children/dependents |
| `smoker` | Smoking status |

### 🔧 Engineered Features

The preprocessing logic creates additional features:

- `is_obese`
- `has_children`
- `obese_smoker`

These features are derived from the original input variables and are used to capture additional relationships that may influence insurance charges.

---

## 🤖 Machine Learning Workflow

```text
Raw Dataset
    ↓
Exploratory Data Analysis
    ↓
Data Preprocessing
    ↓
Feature Engineering
    ↓
Model Training
    ↓
Model Evaluation
    ↓
Model Export
    ↓
Streamlit Application
    ↓
Prediction + SHAP Explainability
```

The trained model is stored in:

```text
Trained_Model/insurance_charges_model.pkl
```

SHAP artifacts are stored in:

```text
Trained_Model/shap_background.pkl
Trained_Model/shap_feature_names.pkl
```

---

## 🔍 Explainable AI with SHAP

A key feature of the project is **local model explainability**.

After generating a prediction, the application can calculate SHAP contributions for the selected profile.

The dashboard provides:

- SHAP contribution for each transformed feature
- Positive and negative feature contributions
- SHAP base value
- Model prediction
- SHAP reconstructed prediction
- Top features influencing the prediction
- Automatic local interpretation

### SHAP Interpretation

```text
Positive SHAP value
        ↓
Feature pushes the prediction higher

Negative SHAP value
        ↓
Feature pushes the prediction lower

Larger absolute SHAP value
        ↓
Stronger influence on the prediction
```

The backend implementation uses a representative training-data background for SHAP rather than using the current user's row as the background.

---

## 📊 Dashboard Features

### 🎯 1. Prediction Hub

Users can enter age, sex, BMI, number of children/dependents, and smoking status.

The application displays:

- Estimated annual premium
- BMI category classification
- High-risk profile indicator

### 📊 2. Explainability & Analytics

This section includes:

- SHAP feature attribution
- Model prediction
- SHAP base value
- SHAP reconstructed prediction
- Automatic local interpretation
- BMI vs. charges visualization
- Age vs. charges visualization
- Smoking-status cost comparison
- Obese-smoker risk comparison

### 📖 3. About & Documentation

The application includes information about project purpose, architecture, technology stack, and developer information.

---

## 🏗️ Project Architecture

```text
Insurace-Charges-Predictions-
│
├── backend/
│   ├── __pycache__/
│   └── main.py
│
├── Data/
│   └── insurance.csv
│
├── frontend/
│   └── app.py
│
├── Plots and charts/
│
├── Trained_Model/
│   ├── insurance_charges_model.pkl
│   ├── shap_background.pkl
│   └── shap_feature_names.pkl
│
├── Insurance.ipynb
├── Model Evaluation.csv
└── requirements.txt
```

---

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| **Python** | Core programming language |
| **Pandas** | Data manipulation |
| **NumPy** | Numerical computation |
| **Scikit-Learn** | Machine Learning |
| **SHAP** | Explainable AI |
| **Streamlit** | Interactive web application |
| **Plotly** | Interactive visualizations |
| **Matplotlib** | Visualization |
| **Seaborn** | Statistical visualization |
| **Joblib** | Model serialization |
| **FastAPI** | Backend/API implementation |

---

## 📁 Important Files

- **`Insurance.ipynb`** — Machine Learning workflow including exploration, preprocessing, feature engineering, training/evaluation, and export.
- **`Trained_Model/insurance_charges_model.pkl`** — Serialized trained Machine Learning pipeline used for prediction.
- **`Trained_Model/shap_background.pkl`** — Representative transformed background data used by the SHAP workflow.
- **`Trained_Model/shap_feature_names.pkl`** — Feature names corresponding to the transformed model input.
- **`frontend/app.py`** — Streamlit dashboard containing prediction, SHAP visualization, analytics, and documentation.
- **`backend/main.py`** — FastAPI implementation containing prediction and SHAP explanation endpoints.
- **`requirements.txt`** — Python dependencies required by the project.

---

## 🚀 Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/mudassirkhan1249/Insurace-Charges-Predictions-.git
cd Insurace-Charges-Predictions-
```

### 2. Create a virtual environment

#### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

#### macOS/Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit application

From the project root:

```bash
streamlit run frontend/app.py
```

---

## 🌐 Live Demo

The project is deployed on Streamlit:

**https://st886kh7ksfrutwute6o64.streamlit.app/**

The live application provides an interactive prediction dashboard together with SHAP explainability and analytical visualizations.

---

## 🔌 Backend Architecture

The repository also contains a FastAPI implementation in:

```text
backend/main.py
```

The backend exposes:

```text
GET  /health
POST /predict
POST /explain
```

The FastAPI implementation loads the serialized model and SHAP artifacts from `Trained_Model`.

For Streamlit deployment, the frontend can be structured to perform inference directly inside the Streamlit application rather than depending on a localhost API.

---

## 🔬 Example Prediction Workflow

```text
User enters beneficiary information
                ↓
        Feature engineering
                ↓
       Trained ML pipeline
                ↓
      Insurance charge estimate
                ↓
        SHAP explanation
                ↓
      Interactive visualization
```

---

## 📈 Portfolio Highlights

This project demonstrates practical Machine Learning skills across the full workflow:

- Data analysis
- Data preprocessing
- Feature engineering
- Regression modeling
- Model evaluation
- Model serialization
- ML inference
- Streamlit application development
- Explainable AI
- Interactive visualization
- API architecture
- Deployment

---

## ⚠️ Disclaimer

This project is an educational Machine Learning application. The predicted insurance charge is an estimate generated by the trained model and should not be considered professional medical, insurance, financial, or underwriting advice.

---

## 👨‍💻 Author

### Mudassir Khan

**Focus:** AI & Data Science

- GitHub: https://github.com/mudassirkhan1249
- Project Repository: https://github.com/mudassirkhan1249/Insurace-Charges-Predictions-

---

## ⭐ Support

If you find this project useful or interesting, consider giving the repository a ⭐ on GitHub.

---

## 📜 License

No explicit license is currently specified in the provided project structure. If the repository is intended for public reuse, add an appropriate `LICENSE` file.
