# 🏥 Medical Insurance Premium Predictor (90.35% Accuracy)

An end-to-end Machine Learning project that predicts affordable medical insurance costs based on individual health metrics and demographic data. By implementing custom feature engineering, the model achieves a high **R2 Score of 0.9035**.

---

## 🚀 Key Highlights & Performance

*   **Super Feature Engineering:** Introduced a high-impact feature (`obese_smoker`) combining BMI and smoking status, which pushed the model baseline performance from ~75% to over 90%.
*   **Feature Optimization:** Eliminated multicollinearity by dropping duplicate features (`has_children`, `is_obese`) and simplified the feature space by removing low-impact demographics.
*   **Production Ready:** Trained artifacts are serialized using `joblib` for instant inference without retraining.

### Model Evaluation Metrics

| Metric | Value |
| :--- | :--- |
| **R2 Score (Accuracy)** | **0.9035** (90.35%) |
| **Mean Absolute Error (MAE)** | \$2,367.65 |
| **Root Mean Squared Error (RMSE)** | \$4,210.89 |

---

## 🛠️ Tech Stack & Workflow

*   **Language:** Python
*   **Libraries:** Pandas, NumPy, Scikit-Learn, Seaborn, Matplotlib, Joblib

### Project Workflow
1.  **Data Preprocessing:** Outlier capping on `bmi` using the Interquartile Range (IQR) method.
2.  **Categorical Encoding:** Converted text fields into structural binary values.
3.  **Feature Selection:** Analyzed data using a Pearson Correlation Heatmap to eliminate multicollinear features.
4.  **Scaling:** Applied `StandardScaler` to normalize numeric variations.
5.  **Model Selection:** Evaluated `LinearRegression` against `RandomForestRegressor`. Selected Linear Regression due to its superior generalization and lower MAE on this feature space.

---

## 📊 Final Mathematical Insights (Coefficients)

The model evaluates insurance costs based on the following feature weights (Intercept baseline: **\$13,030.20**):

*   **`obese_smoker`** (+5793.68) ➔ Highest risk factor
*   **`is_smoker`** (+5330.98) ➔ Significant independent cost driver
*   **`age`** (+3636.14) ➔ Gradual baseline increase per life stage
*   **`children`** (+675.69) ➔ Incremental family dependent cost
*   **`bmi`** (+240.40) ➔ Maintained for granular health tracking

---

## 💻 How to Run Inference

Make sure you have `joblib`, `pandas`, and `scikit-learn` installed. Run the following snippet to predict insurance costs for new inputs:

```python
import joblib
import pandas as pd

# 1. Load the serialized artifacts
model = joblib.load('medical_insurance_lr_model.pkl')
scaler = joblib.load('insurance_scaler.pkl')

# 2. Input profile (Age: 28, BMI: 33.5, Children: 2, Smoker: Yes)
age, bmi, children, is_smoker = 28, 33.5, 2, 1
obese_smoker = 1 if (bmi >= 30 and is_smoker == 1) else 0

# 3. Predict cost
raw_data = pd.DataFrame([[age, bmi, children, is_smoker, obese_smoker]], 
                        columns=['age', 'bmi', 'children', 'is_smoker', 'obese_smoker'])
scaled_data = scaler.transform(raw_data)
predicted_cost = model.predict(scaled_data)[0]

print(f"Predicted Medical Insurance Premium: ${predicted_cost:,.2f}")
```
