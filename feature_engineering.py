
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from imblearn.over_sampling import SMOTE
import joblib


# Project folder and dataset
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "paypal_fraud_cleaned.csv"

print("Project folder:", BASE_DIR)
print("Loading dataset:", DATA_PATH)

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}"
    )

# Load data
df = pd.read_csv(DATA_PATH)
print("\nDataset loaded:", df.shape)

# Remove duplicates and rows without target
df.drop_duplicates(inplace=True)

if "is_fraud" not in df.columns:
    raise ValueError(
        "The dataset must contain an 'is_fraud' column."
    )

df["is_fraud"] = pd.to_numeric(
    df["is_fraud"], errors="coerce"
)
df.dropna(subset=["is_fraud"], inplace=True)
df["is_fraud"] = df["is_fraud"].astype(int)

if not set(df["is_fraud"].unique()).issubset({0, 1}):
    raise ValueError("'is_fraud' must contain only 0 and 1.")

if df["is_fraud"].nunique() != 2:
    raise ValueError(
        "Both fraud (1) and non-fraud (0) records are required."
    )

# Feature engineering: timestamp
if "timestamp" in df.columns:
    timestamp = pd.to_datetime(
        df["timestamp"], errors="coerce"
    )
    df["transaction_hour"] = timestamp.dt.hour
    df["transaction_day"] = timestamp.dt.day
    df["transaction_month"] = timestamp.dt.month
    df["transaction_dayofweek"] = timestamp.dt.dayofweek
    df.drop(columns=["timestamp"], inplace=True)

# Feature engineering: transaction amount
if "amount_usd" in df.columns:
    df["amount_usd"] = pd.to_numeric(
        df["amount_usd"], errors="coerce"
    )
    df["log_amount"] = np.log1p(
        df["amount_usd"].clip(lower=0)
    )

# Remove identifier columns if present
id_columns = [
    "s_no",
    "transaction_id",
    "user_id"
]
df.drop(
    columns=[c for c in id_columns if c in df.columns],
    inplace=True
)

# Separate features and target
X = df.drop(columns=["is_fraud"])
y = df["is_fraud"]

# Remove completely empty feature columns
empty_columns = X.columns[X.isna().all()].tolist()
if empty_columns:
    X.drop(columns=empty_columns, inplace=True)

if X.shape[1] == 0:
    raise ValueError("No usable feature columns found.")

# Identify numeric and categorical columns
numeric_columns = X.select_dtypes(
    include=["number", "bool"]
).columns.tolist()

categorical_columns = [
    col for col in X.columns
    if col not in numeric_columns
]

print("\nNumeric features:", numeric_columns)
print("Categorical features:", categorical_columns)

# Split before resampling to avoid test-data leakage
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Preprocessing pipelines
transformers = []

if numeric_columns:
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    transformers.append(
        ("numeric", numeric_pipeline, numeric_columns)
    )

if categorical_columns:
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        ))
    ])
    transformers.append(
        ("categorical", categorical_pipeline,
         categorical_columns)
    )

preprocessor = ColumnTransformer(
    transformers=transformers,
    remainder="drop"
)

# Fit on training data only
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Apply SMOTE to training data only
if y_train.value_counts().min() < 2:
    raise ValueError(
        "Not enough minority-class samples for SMOTE."
    )

smote = SMOTE(random_state=42)

X_train_resampled, y_train_resampled = smote.fit_resample(
    X_train_processed,
    y_train
)

# Save model-ready files in the same folder
pd.DataFrame(X_train_resampled).to_csv(
    BASE_DIR / "X_train_after_smote.csv",
    index=False
)

pd.Series(y_train_resampled, name="is_fraud").to_csv(
    BASE_DIR / "y_train_after_smote.csv",
    index=False
)

pd.DataFrame(X_test_processed).to_csv(
    BASE_DIR / "X_test_processed.csv",
    index=False
)

pd.Series(y_test, name="is_fraud").to_csv(
    BASE_DIR / "y_test.csv",
    index=False
)

joblib.dump(
    preprocessor,
    BASE_DIR / "feature_preprocessor.joblib"
)

print("\nFeature engineering and preprocessing completed!")
print("Training data after SMOTE:", X_train_resampled.shape)
print("Testing data:", X_test_processed.shape)
print("\nCreated files:")
print("X_train_after_smote.csv")
print("y_train_after_smote.csv")
print("X_test_processed.csv")
print("y_test.csv")
print("feature_preprocessor.joblib")
