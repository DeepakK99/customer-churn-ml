import pandas as pd

df = pd.read_csv(filepath_or_buffer="data/raw/Bank Customer Churn Prediction.csv")

# print(df.info())
print("Dataset Shape: ", df.shape, sep="")
print("\nDataset Column DataTypes:\n", df.dtypes, sep="")
print("\nMissing values count:\n", df.isnull().sum(), sep="")
print(
    "\nData imbalance on target column 'churn':\n",
    df["churn"].value_counts(normalize=True),
    sep="",
)
print("\nNon Feature columns: ", ["customer_id"], sep="")

print(df.describe())

print(df["country"].value_counts())

print(df["gender"].value_counts())

print(df.groupby("churn").mean(numeric_only=True))

print(df.groupby("country")["churn"].mean())

print(df.groupby("gender")["churn"].mean())

print(df["balance"].value_counts().head(10))

print(pd.crosstab(df["country"], df["churn"], normalize="index"))
