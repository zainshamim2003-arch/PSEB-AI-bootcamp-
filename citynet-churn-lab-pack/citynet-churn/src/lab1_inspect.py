
import pandas as pd
from pathlib import Path

# Step 1: Locate the project directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Step 2: Load the dataset
DATA_PATH = BASE_DIR / "data" / "citynet_churn.csv"

df = pd.read_csv(DATA_PATH)

print("\n========== LAB 1: DATA INSPECTION ==========")

# Step 3: Display dataset shape
print("\n1. Dataset Shape:")
print(df.shape)

# Step 4: Display data types
print("\n2. Data Types:")
print(df.dtypes.value_counts().to_dict())

# Step 5: Check missing values
print("\n3. Missing Values:")
missing = df.isna().sum()
print(missing[missing > 0].to_dict())

# Step 6: Check churn distribution
print("\n4. Churn Distribution:")
print(df["churned"].value_counts(normalize=True).round(3).to_dict())

# Step 7: Statistical summary
print("\n5. Numerical Summary:")
print(
    df[
        ["tenure_months", "support_calls_90d", "outage_hours_90d"]
    ].describe().loc[["mean", "min", "max"]].round(1)
)

# Step 8: Churn rate according to support calls
print("\n6. Churn Rate by Support Calls:")
print(
    df.groupby("support_calls_90d")["churned"]
    .mean().round(2).head(6).to_dict()
)

# Step 9: Churn rate according to cancellation requests
print("\n7. Churn Rate by Cancellation Request:")
print(
    df.groupby("cancel_request_logged")["churned"]
    .mean().round(3).to_dict()
)

print("\n========== TODO QUESTIONS ==========")

# TODO 1: Which column has missing values?
print("\nTODO 1: Columns with missing values:")
print(missing[missing > 0])

# TODO 2: What share of customers churned?
churn_percentage = df["churned"].mean() * 100
print(f"\nTODO 2: Churn percentage: {churn_percentage:.2f}%")

if churn_percentage < 40 or churn_percentage > 60:
    print("The dataset is imbalanced.")
else:
    print("The dataset is relatively balanced.")

# TODO 3: Check cancellation request relationship with churn
print("\nTODO 3: Cancellation request and churn relationship:")
print(
    df.groupby("cancel_request_logged")["churned"]
    .mean().round(3)
)

print("\n========== LAB 1 COMPLETED ==========")

