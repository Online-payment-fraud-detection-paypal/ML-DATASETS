
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    average_precision_score,
    roc_auc_score,
    precision_recall_curve
)

# 1. Project directory
BASE_DIR = Path(__file__).resolve().parent

# 2. Load processed datasets
print("Loading datasets...")

X_train = pd.read_csv(BASE_DIR / "X_train_after_smote.csv")
y_train = pd.read_csv(
    BASE_DIR / "y_train_after_smote.csv"
).squeeze()

X_test = pd.read_csv(BASE_DIR / "X_test_processed.csv")
y_test = pd.read_csv(
    BASE_DIR / "y_test.csv"
).squeeze()

# Validate data
if len(X_train) != len(y_train):
    raise ValueError("Training features and labels have different lengths.")

if len(X_test) != len(y_test):
    raise ValueError("Testing features and labels have different lengths.")

y_train = pd.to_numeric(y_train).astype(int)
y_test = pd.to_numeric(y_test).astype(int)

if set(y_train.unique()) != {0, 1}:
    raise ValueError("Training labels must contain both 0 and 1.")

if set(y_test.unique()) != {0, 1}:
    raise ValueError("Testing labels must contain both 0 and 1.")

# 3. Create a validation split from the training data
# Test data is kept separate for final evaluation.
X_fit, X_val, y_fit, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.20,
    random_state=42,
    stratify=y_train
)

# 4. Create Random Forest model
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=2,
    class_weight="balanced_subsample",
    random_state=42,
    n_jobs=-1
)

# 5. Train model
print("\nTraining Random Forest model...")
model.fit(X_fit, y_fit)

# 6. Choose threshold using validation probabilities
val_prob = model.predict_proba(X_val)[:, 1]

precision_values, recall_values, thresholds = (
    precision_recall_curve(y_val, val_prob)
)

if len(thresholds) == 0:
    best_threshold = 0.5
else:
    # F1 for each available threshold
    denominator = precision_values[:-1] + recall_values[:-1]

    f1_values = np.divide(
        2 * precision_values[:-1] * recall_values[:-1],
        denominator,
        out=np.zeros_like(denominator),
        where=denominator != 0
    )

    best_index = int(np.argmax(f1_values))
    best_threshold = float(thresholds[best_index])

print(f"Selected validation threshold: {best_threshold:.4f}")
print(
    f"Best validation F1: "
    f"{f1_values[best_index]:.4f}" if len(thresholds) else
    "Using default threshold."
)

# 7. Refit model using all available training data
print("\nRefitting model on all training data...")
model.fit(X_train, y_train)

# 8. Predict on untouched test data
y_prob = model.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= best_threshold).astype(int)

# 9. Evaluate model
print("\n--- RANDOM FOREST RESULTS ---")
print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print(
    f"Precision: "
    f"{precision_score(y_test, y_pred, zero_division=0):.4f}"
)
print(
    f"Recall   : "
    f"{recall_score(y_test, y_pred, zero_division=0):.4f}"
)
print(f"F1 Score : {f1_score(y_test, y_pred, zero_division=0):.4f}")
print(f"PR-AUC   : {average_precision_score(y_test, y_prob):.4f}")
print(f"ROC-AUC  : {roc_auc_score(y_test, y_prob):.4f}")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Non-Fraud", "Fraud"],
        zero_division=0
    )
)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# 10. Save model and threshold
model_path = BASE_DIR / "paypal_fraud_random_forest.joblib"
threshold_path = BASE_DIR / "fraud_threshold.joblib"

joblib.dump(model, model_path)
joblib.dump(best_threshold, threshold_path)

print(f"\nModel saved as: {model_path.name}")
print(f"Threshold saved as: {threshold_path.name}")

# 11. Save predictions
results = pd.DataFrame({
    "Actual_Fraud": y_test.to_numpy(),
    "Predicted_Fraud": y_pred,
    "Fraud_Probability": y_prob,
    "Threshold_Used": best_threshold
})

results.to_csv(
    BASE_DIR / "fraud_predictions.csv",
    index=False
)

print("Predictions saved as fraud_predictions.csv")
print("\nModel training and evaluation completed!")
