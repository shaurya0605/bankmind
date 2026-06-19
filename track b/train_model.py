"""
BankMind Challenge - Track B (ML Engineer)

Trains and compares a Logistic Regression baseline against a Random
Forest model on the UCI Bank Marketing Dataset, evaluates both
properly, and prints sample predictions with probability scores.

Run with: python train_model.py
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
import joblib

DATA_PATH = "bank-full.csv"
RANDOM_STATE = 42

# ---------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------
df = pd.read_csv(DATA_PATH, sep=";")

print("Shape:", df.shape)
print("\nClass distribution:\n", df["y"].value_counts())
print("\nClass distribution (%):\n", df["y"].value_counts(normalize=True) * 100)
print("\nMissing values:\n", df.isnull().sum().sum(), "total missing cells")

# ---------------------------------------------------------------
# 2. Focused EDA / preprocessing
# ---------------------------------------------------------------
# UCI dataset uses "unknown" as a missing-value marker in some columns
# instead of NaN — worth knowing, doesn't need imputation, treated as
# its own category.
categorical_cols = [
    "job", "marital", "education", "default", "housing",
    "loan", "contact", "month", "poutcome"
]
numeric_cols = ["age", "balance", "day", "duration", "campaign", "pdays", "previous"]

df_model = df.copy()

# Encode target
df_model["y"] = df_model["y"].map({"yes": 1, "no": 0})

# Label-encode categoricals (simple, sufficient for this baseline task)
encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df_model[col] = le.fit_transform(df_model[col])
    encoders[col] = le

X = df_model[categorical_cols + numeric_cols]
y = df_model["y"]

# ---------------------------------------------------------------
# 3. Train/test split
# ---------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

print(f"\nTrain size: {len(X_train)}, Test size: {len(X_test)}")

# ---------------------------------------------------------------
# 4. Baseline: Logistic Regression (needs scaled features)
# ---------------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

log_reg = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)
log_reg.fit(X_train_scaled, y_train)
y_pred_lr = log_reg.predict(X_test_scaled)
y_proba_lr = log_reg.predict_proba(X_test_scaled)[:, 1]

# ---------------------------------------------------------------
# 5. Main model: Random Forest (no scaling needed)
# ---------------------------------------------------------------
rf = RandomForestClassifier(
    n_estimators=200, max_depth=10, class_weight="balanced",
    random_state=RANDOM_STATE, n_jobs=-1
)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
y_proba_rf = rf.predict_proba(X_test)[:, 1]

# ---------------------------------------------------------------
# 6. Evaluation
# ---------------------------------------------------------------
def evaluate(name, y_true, y_pred):
    print(f"\n--- {name} ---")
    print("Accuracy: ", round(accuracy_score(y_true, y_pred), 4))
    print("Precision:", round(precision_score(y_true, y_pred), 4))
    print("Recall:   ", round(recall_score(y_true, y_pred), 4))
    print("F1:       ", round(f1_score(y_true, y_pred), 4))
    print("\nClassification report:")
    print(classification_report(y_true, y_pred, target_names=["no", "yes"]))
    print("Confusion matrix:")
    print(confusion_matrix(y_true, y_pred))

evaluate("Logistic Regression (baseline)", y_test, y_pred_lr)
evaluate("Random Forest (main model)", y_test, y_pred_rf)

# ---------------------------------------------------------------
# 7. Feature importance (Random Forest)
# ---------------------------------------------------------------
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nFeature importances (Random Forest):")
print(importances)

# ---------------------------------------------------------------
# 8. Sample predictions: 5 customers (>=2 "yes", >=2 "no")
# ---------------------------------------------------------------
results_df = X_test.copy()
results_df["actual"] = y_test.values
results_df["predicted"] = y_pred_rf
results_df["probability_yes"] = y_proba_rf

# decode categoricals back to readable labels for display
display_df = results_df.copy()
for col in categorical_cols:
    display_df[col] = encoders[col].inverse_transform(display_df[col])

yes_samples = display_df[display_df["predicted"] == 1].head(2)
no_samples = display_df[display_df["predicted"] == 0].head(3)
sample = pd.concat([yes_samples, no_samples])

readable_cols = ["age", "job", "marital", "balance", "housing", "loan",
                  "actual", "predicted", "probability_yes"]

print("\n--- 5 Sample Predictions ---")
print(sample[readable_cols].to_string(index=False))

# ---------------------------------------------------------------
# 9. Save the model
# ---------------------------------------------------------------
joblib.dump(
    {"model": rf, "encoders": encoders, "feature_cols": list(X.columns)},
    "model.pkl"
)
print("\nModel saved to model.pkl")
