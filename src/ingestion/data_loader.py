import pandas as pd

def load_telco_data(path="data/raw/CustomerChurn.xlsx"):
    df = pd.read_excel(path)

    # Drop identifier columns
    df = df.drop(columns=["Customer ID", "LoyaltyID"])

    # Convert target to binary
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # Handle Total Charges (some are blank strings)
    df["Total Charges"] = pd.to_numeric(df["Total Charges"], errors="coerce")
    df["Total Charges"].fillna(0, inplace=True)

    # Separate features & target
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    # One-hot encode categorical features
    X = pd.get_dummies(X, drop_first=True)

    return X, y
