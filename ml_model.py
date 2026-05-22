"""
ml_model.py
===========
Predictive Modeling — Customer Churn Prediction
Algorithms: Logistic Regression, Decision Tree, Random Forest
Author: Jasmine
Project: Predictive Modeling Using Machine Learning
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_curve, auc, ConfusionMatrixDisplay
)

os.makedirs("outputs", exist_ok=True)

# ── Styling ────────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "font.family"      : "DejaVu Sans",
    "axes.spines.top"  : False,
    "axes.spines.right": False,
    "figure.dpi"       : 120,
})

COLORS = {"Logistic Regression": "#378ADD",
          "Decision Tree"      : "#1D9E75",
          "Random Forest"      : "#534AB7"}

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Load & Inspect
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("  PREDICTIVE MODELING — CUSTOMER CHURN")
print("=" * 60)

df = pd.read_csv("data/customer_churn.csv")
print(f"\n[1] DATA LOADED: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"    Churn rate: {df['churn'].mean()*100:.1f}%")
print(f"\n    Features:\n{df.dtypes.to_string()}")
print(f"\n    Missing values: {df.isnull().sum().sum()}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Preprocessing
# ══════════════════════════════════════════════════════════════════════════════
le = LabelEncoder()
df['gender_enc'] = le.fit_transform(df['gender'])

FEATURES = ['age', 'gender_enc', 'tenure', 'monthly_charges',
            'total_charges', 'num_products', 'has_internet',
            'has_phone', 'support_calls']
TARGET = 'churn'

X = df[FEATURES]
y = df[TARGET]

# Train/Test split — 80/20
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Scale features
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print(f"\n[2] PREPROCESSING DONE")
print(f"    Training set : {X_train.shape[0]} samples")
print(f"    Testing set  : {X_test.shape[0]} samples")
print(f"    Features used: {len(FEATURES)}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Train 3 Models
# ══════════════════════════════════════════════════════════════════════════════
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree"      : DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest"      : RandomForestClassifier(n_estimators=100, random_state=42),
}

results = {}
print(f"\n[3] TRAINING MODELS...")

for name, model in models.items():
    if name == "Logistic Regression":
        model.fit(X_train_sc, y_train)
        y_pred = model.predict(X_test_sc)
        y_prob = model.predict_proba(X_test_sc)[:, 1]
        cv = cross_val_score(model, scaler.transform(X), y, cv=5, scoring='accuracy')
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        cv = cross_val_score(model, X, y, cv=5, scoring='accuracy')

    acc  = accuracy_score(y_test, y_pred)
    cm   = confusion_matrix(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    results[name] = {
        "model"   : model,
        "y_pred"  : y_pred,
        "y_prob"  : y_prob,
        "accuracy": acc,
        "cm"      : cm,
        "fpr"     : fpr,
        "tpr"     : tpr,
        "auc"     : roc_auc,
        "cv_mean" : cv.mean(),
        "cv_std"  : cv.std(),
        "report"  : classification_report(y_test, y_pred),
    }
    print(f"    {name:<22} → Accuracy: {acc*100:.2f}%  AUC: {roc_auc:.3f}  CV: {cv.mean()*100:.2f}%")

best_model = max(results, key=lambda k: results[k]['accuracy'])
print(f"\n    ✓ Best Model: {best_model} ({results[best_model]['accuracy']*100:.2f}%)")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Visualizations
# ══════════════════════════════════════════════════════════════════════════════

# ── Fig 1: Model Comparison Dashboard ─────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Model Performance Comparison", fontsize=15, fontweight="bold")

names = list(results.keys())
accs  = [results[n]["accuracy"]*100 for n in names]
aucs  = [results[n]["auc"] for n in names]
cvs   = [results[n]["cv_mean"]*100 for n in names]
colors = [COLORS[n] for n in names]

# Accuracy
bars = axes[0].bar(names, accs, color=colors, width=0.5, edgecolor="none")
for b in bars:
    axes[0].text(b.get_x()+b.get_width()/2, b.get_height()+0.3,
                 f"{b.get_height():.1f}%", ha="center", fontsize=10, fontweight="bold")
axes[0].set_title("Test Accuracy (%)", fontweight="bold")
axes[0].set_ylim(80, 103)
axes[0].tick_params(axis="x", rotation=12)

# AUC
bars2 = axes[1].bar(names, aucs, color=colors, width=0.5, edgecolor="none")
for b in bars2:
    axes[1].text(b.get_x()+b.get_width()/2, b.get_height()+0.003,
                 f"{b.get_height():.3f}", ha="center", fontsize=10, fontweight="bold")
axes[1].set_title("ROC-AUC Score", fontweight="bold")
axes[1].set_ylim(0.8, 1.05)
axes[1].tick_params(axis="x", rotation=12)

# CV Score
bars3 = axes[2].bar(names, cvs, color=colors, width=0.5, edgecolor="none")
for b in bars3:
    axes[2].text(b.get_x()+b.get_width()/2, b.get_height()+0.3,
                 f"{b.get_height():.1f}%", ha="center", fontsize=10, fontweight="bold")
axes[2].set_title("5-Fold CV Accuracy (%)", fontweight="bold")
axes[2].set_ylim(80, 103)
axes[2].tick_params(axis="x", rotation=12)

plt.tight_layout()
plt.savefig("outputs/model_comparison.png", bbox_inches="tight", dpi=150)
plt.close()
print("\n[4] Saved → outputs/model_comparison.png")

# ── Fig 2: Confusion Matrices (all 3) ─────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle("Confusion Matrices", fontsize=15, fontweight="bold")

for ax, (name, res) in zip(axes, results.items()):
    disp = ConfusionMatrixDisplay(confusion_matrix=res["cm"],
                                   display_labels=["No Churn", "Churn"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(name, fontweight="bold")

plt.tight_layout()
plt.savefig("outputs/confusion_matrices.png", bbox_inches="tight", dpi=150)
plt.close()
print("     Saved → outputs/confusion_matrices.png")

# ── Fig 3: ROC Curves ─────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot([0,1],[0,1],"k--", linewidth=1, alpha=0.5, label="Random (AUC = 0.50)")

for name, res in results.items():
    ax.plot(res["fpr"], res["tpr"], linewidth=2.5,
            color=COLORS[name], label=f"{name} (AUC = {res['auc']:.3f})")

ax.fill_between(results[best_model]["fpr"], results[best_model]["tpr"],
                alpha=0.08, color=COLORS[best_model])
ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate", fontsize=12)
ax.set_title("ROC Curves — All Models", fontsize=14, fontweight="bold")
ax.legend(loc="lower right", fontsize=10)
plt.tight_layout()
plt.savefig("outputs/roc_curves.png", bbox_inches="tight", dpi=150)
plt.close()
print("     Saved → outputs/roc_curves.png")

# ── Fig 4: Feature Importance (Random Forest) ─────────────────────────────────
rf_model = results["Random Forest"]["model"]
importances = pd.Series(rf_model.feature_importances_, index=FEATURES).sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.barh(importances.index, importances.values,
               color=[plt.cm.Blues(0.4 + 0.06*i) for i in range(len(importances))],
               edgecolor="none", height=0.6)
for bar in bars:
    ax.text(bar.get_width()+0.002, bar.get_y()+bar.get_height()/2,
            f"{bar.get_width():.3f}", va="center", fontsize=9)
ax.set_title("Feature Importance — Random Forest", fontsize=13, fontweight="bold")
ax.set_xlabel("Importance Score")
plt.tight_layout()
plt.savefig("outputs/feature_importance.png", bbox_inches="tight", dpi=150)
plt.close()
print("     Saved → outputs/feature_importance.png")

# ── Fig 5: Churn Distribution ─────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle("Churn Analysis", fontsize=13, fontweight="bold")

churn_counts = df['churn'].value_counts()
axes[0].pie(churn_counts, labels=["No Churn","Churn"], autopct="%1.1f%%",
            colors=["#1D9E75","#E24B4A"],
            wedgeprops=dict(width=0.55), startangle=90)
axes[0].set_title("Overall Churn Rate")

sns.boxplot(data=df, x="churn", y="tenure",
            palette=["#1D9E75","#E24B4A"], hue="churn", legend=False, ax=axes[1], width=0.4)
axes[1].set_xticklabels(["No Churn","Churn"])
axes[1].set_title("Tenure vs Churn")
axes[1].set_xlabel("")
axes[1].set_ylabel("Tenure (months)")

plt.tight_layout()
plt.savefig("outputs/churn_analysis.png", bbox_inches="tight", dpi=150)
plt.close()
print("     Saved → outputs/churn_analysis.png")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — Save Report
# ══════════════════════════════════════════════════════════════════════════════
report_text = ""
for name, res in results.items():
    report_text += f"\n{'='*50}\n{name}\n{'='*50}\n"
    report_text += f"Accuracy : {res['accuracy']*100:.2f}%\n"
    report_text += f"AUC Score: {res['auc']:.4f}\n"
    report_text += f"CV Score : {res['cv_mean']*100:.2f}% ± {res['cv_std']*100:.2f}%\n"
    report_text += f"\nClassification Report:\n{res['report']}\n"

with open("outputs/model_report.txt","w") as f:
    f.write(f"PREDICTIVE MODELING REPORT\nBest Model: {best_model}\n")
    f.write(report_text)

print("     Saved → outputs/model_report.txt")
print(f"\n{'='*60}")
print(f"  Done! Best model: {best_model} — {results[best_model]['accuracy']*100:.2f}% accuracy")
print(f"{'='*60}\n")
