import pandas as pd

DATA_PATH = "data/raw/CustomerChurn.xlsx"

df = pd.read_excel(DATA_PATH)

print("Shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

print("\nTarget value counts:")
print(df["Churn"].value_counts())