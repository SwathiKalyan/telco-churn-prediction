import shap
import matplotlib.pyplot as plt
import os


def log_shap_summary(model, X_train, output_dir="outputs", prefix="model"):

    os.makedirs(output_dir, exist_ok=True)

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_train)

        # If binary classification, shap_values is list of arrays
        if isinstance(shap_values, list):
            shap_values_to_plot = shap_values[1]
        else:
            shap_values_to_plot = shap_values

    except Exception:
        explainer = shap.Explainer(model.predict_proba, X_train)
        shap_values = explainer(X_train)

        shap_values_to_plot = shap_values.values

    plot_path = f"{output_dir}/{prefix}_shap_summary.png"

    plt.figure()
    shap.summary_plot(
        shap_values_to_plot,
        X_train,
        show=False
    )
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    return plot_path
