import pandas as pd

df = pd.read_csv(filepath_or_buffer="data/raw/Bank Customer Churn Prediction.csv")

# print(df.info())
print("Dataset Shape: ", df.shape, sep="")
print("\nDataset Column DataTypes:\n", df.dtypes, sep="")
print("\nMissing values count:\n", df.isnull().sum(), sep="")
print(
    "\nData imbalance on target column 'churn':\n",
    df["churn"].value_counts(normalize=True),
    sep=""
)
print("\nNon Feature columns: ", ["customer_id"], sep="")
