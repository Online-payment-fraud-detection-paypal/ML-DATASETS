import pandas as pd

# STEP 1: Load the original dataset
df = pd.read_csv("paypal_fraud_practice_raw.csv")

print("Original dataset shape:", df.shape)

# STEP 2: Check missing values
print("\nMissing values before cleaning:")
print(df.isnull().sum())

# STEP 3: Check and remove duplicate rows
print("\nDuplicate rows:", df.duplicated().sum())

df = df.drop_duplicates().copy()

print("Shape after removing duplicates:", df.shape)

# STEP 4: Convert timestamp into datetime format
df["timestamp"] = pd.to_datetime(
    df["timestamp"], errors="coerce"
)

# STEP 5: Fill missing numerical values with median
numeric_cols = [
    "amount_usd",
    "time_since_last_login_min"
]

for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

# STEP 6: Fill missing categorical values with mode
categorical_cols = [
    "payment_method",
    "login_country",
    "device_type"
]

for col in categorical_cols:
    df[col] = df[col].fillna(
        df[col].mode(dropna=True)[0]
    )

# STEP 7: Handle invalid timestamps, if any
if df["timestamp"].isnull().any():
    df["timestamp"] = df["timestamp"].fillna(
        df["timestamp"].mode()[0]
    )

# STEP 8: Reset serial numbers after removing duplicates
df["s_no"] = range(1, len(df) + 1)

# STEP 9: Save the cleaned dataset
df["timestamp"] = df["timestamp"].dt.strftime(
    "%Y-%m-%d %H:%M:%S"
)

df.to_csv("paypal_fraud_cleaned.csv", index=False)

# STEP 10: Verify the cleaned dataset
print("\nCleaned dataset shape:", df.shape)

print("\nMissing values after cleaning:")
print(df.isnull().sum())

print("\nDuplicate rows after cleaning:")
print(df.duplicated().sum())

print("\nFraud label distribution:")
print(df["is_fraud"].value_counts())

print("\nData cleaning completed successfully!")