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

1.  **Exploratory Data Analysis (EDA):** Visualized distributions and structural trends using interactive plots and a Pearson Correlation Heatmap.
2.  **Data Cleaning:** Handled extreme values and parsed structural inconsistencies across data columns.
3.  **Data Preprocessing:** Handled continuous variance via outlier capping on `bmi` using the Interquartile Range (IQR) method.
4.  **Categorical Encoding:** Converted categorical text fields into highly clean numeric maps and one-hot vectors.
5.  **Feature Selection:** Analyzed multi-collinearity to drop overlapping metrics (`has_children`, `is_obese`) and low-impact fields (`is_male`, regions).
6.  **Data Scaling:** Applied `StandardScaler` to ensure numerical features share a uniform baseline and prevent algorithm bias.
7.  **Model Selection & Training:** Evaluated `LinearRegression` against `RandomForestRegressor`. Selected Linear Regression due to its superior generalization, better interpretability, and lower MAE on this optimized feature space.

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

---

## 🎯 Conclusion & Key Takeaways

This project demonstrates how strategic **Feature Engineering** can significantly boost a model's predictive power over complex algorithms. 

### Key Discoveries:
1. **Domain Knowledge is King:** Creating the `obese_smoker` interaction feature based on real-world medical insurance risks was the turning point. It allowed a simple, interpretable model like Linear Regression to outperform a more complex, non-linear model like Random Forest.
2. **Simplicity Wins:** By dropping low-impact and highly correlated columns, we made the model cleaner, reduced the mean absolute error (MAE), and built a system that relies strictly on objective health risks rather than demographic biases.
3. **Data Leakage Prevention:** By properly splitting the dataset before applying `StandardScaler`, we ensured that our evaluation metrics reflect true performance on completely unseen data.

Ultimately, this project highlights that **understanding the data and the business domain** is often far more valuable than blindly tuning hyperparameters or using heavy algorithms.
