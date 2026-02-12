import os
import pandas as pd
import matplotlib.pyplot as plt


def log_feature_importance(model, X_train, output_dir="outputs", prefix="rf"):
    """
    Logs feature importance CSV and PNG plot
    """

    importances = model.feature_importances_
    feature_names = X_train.columns.tolist()

    fi_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values(by="importance", ascending=False)

    os.makedirs(output_dir, exist_ok=True)

    csv_path = f"{output_dir}/{prefix}_feature_importance.csv"
    png_path = f"{output_dir}/{prefix}_feature_importance.png"

    fi_df.to_csv(csv_path, index=False)

    # Plot top 10
    fi_top = fi_df.head(10)

    plt.figure(figsize=(8, 5))
    plt.barh(fi_top["feature"], fi_top["importance"])
    plt.gca().invert_yaxis()
    plt.xlabel("Feature Importance")
    plt.title("Top 10 Feature Importances")
    plt.tight_layout()
    plt.savefig(png_path)
    plt.close()

    return csv_path, png_path