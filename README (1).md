# 🤖 Predictive Modeling Using Machine Learning

> **Course:** Data Engineering | **Author:** Jasmine  
> **Tools:** Python · Scikit-learn · Pandas · Matplotlib · Seaborn · Jupyter Notebook

---

## 📌 Project Overview

This project builds a **Customer Churn Prediction** system using three supervised machine learning algorithms. We train, test, and compare models — evaluating them using accuracy scores, confusion matrices, and ROC curves.

**Problem:** Predict whether a customer will churn (leave) based on their profile and usage data.

---

## 🗂️ Project Structure

```
ml-project/
│
├── data/
│   └── customer_churn.csv         # Dataset (200 customers, 11 features)
│
├── notebooks/
│   └── ML_Predictive_Modeling.ipynb  # Full Jupyter notebook
│
├── outputs/
│   ├── model_comparison.png       # Accuracy, AUC, CV comparison
│   ├── confusion_matrices.png     # All 3 confusion matrices
│   ├── roc_curves.png             # ROC curves overlay
│   ├── feature_importance.png     # Random Forest feature scores
│   ├── churn_analysis.png         # Churn distribution charts
│   └── model_report.txt          # Full classification reports
│
├── ml_model.py                    # Main ML pipeline script
├── requirements.txt               # Python dependencies
└── README.md
```

---

## 📊 Dataset Description

| Column | Type | Description |
|---|---|---|
| `customer_id` | int | Unique identifier |
| `age` | int | Customer age |
| `gender` | string | Male / Female |
| `tenure` | int | Months with company |
| `monthly_charges` | float | Monthly bill (₹) |
| `total_charges` | int | Total amount paid |
| `num_products` | int | Products subscribed |
| `has_internet` | int | Internet service (0/1) |
| `has_phone` | int | Phone service (0/1) |
| `support_calls` | int | Support calls made |
| `churn` | int | Target: 1=Churned, 0=Retained |

**Churn Rate:** 38% of customers churned

---

## 🤖 Models Used

### 1. Logistic Regression
- Baseline linear classifier
- Scaled features using StandardScaler
- Accuracy: **95.00%** | AUC: **1.000**

### 2. Decision Tree
- Non-linear tree-based classifier
- max_depth = 5 to prevent overfitting
- Accuracy: **100.00%** | AUC: **1.000**

### 3. Random Forest
- Ensemble of 100 decision trees
- Best for feature importance ranking
- Accuracy: **97.50%** | AUC: **1.000**

---

## 📈 Model Results

| Model | Test Accuracy | AUC Score | CV Accuracy (5-fold) |
|---|---|---|---|
| Logistic Regression | 95.00% | 1.000 | 96.50% |
| Decision Tree | **100.00%** | 1.000 | 97.50% |
| Random Forest | 97.50% | 1.000 | 97.50% |

✅ **Best Model: Decision Tree — 100% Test Accuracy**

---

## 🔑 Key Features (by importance)

1. **tenure** — longer tenure = lower churn
2. **support_calls** — more calls = higher churn risk
3. **monthly_charges** — higher charges = higher churn
4. **total_charges** — reflects loyalty
5. **num_products** — more products = more retention

---

## 🧪 Evaluation Metrics

- **Accuracy** — % of correct predictions
- **Confusion Matrix** — True/False Positives and Negatives
- **ROC Curve** — Trade-off between sensitivity and specificity
- **AUC Score** — Area under ROC (1.0 = perfect model)
- **5-Fold Cross Validation** — Generalization check

---

## 💡 Key Insights

- Customers with **low tenure** (< 6 months) churn the most
- High **support_calls** (5+) is the strongest churn signal
- Customers with **4 products** almost never churn
- **Monthly charges > ₹80** increases churn probability

---

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/ml-project.git
cd ml-project
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the ML pipeline
```bash
python ml_model.py
```

### 4. Open the Jupyter Notebook
```bash
jupyter notebook notebooks/ML_Predictive_Modeling.ipynb
```

---

## 🛠️ Technologies Used

| Tool | Purpose |
|---|---|
| **Python 3.10+** | Core language |
| **Pandas / NumPy** | Data handling |
| **Scikit-learn** | ML models, metrics, preprocessing |
| **Matplotlib** | Charts and dashboards |
| **Seaborn** | Statistical visualizations |
| **Jupyter Notebook** | Interactive exploration |

---

*Submitted as part of the Data Engineering course — Predictive Modeling project.*
