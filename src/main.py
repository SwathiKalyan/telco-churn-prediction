import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split

from src.ingestion.data_loader import load_telco_data
from src.models.train_tree import train_gradient_boosting
from src.evaluation.metrics import evaluate_classification

from src.explainability.shap_utils import log_shap_summary


from src.models.train_tree import (
    train_decision_tree,
    train_random_forest,
    train_gradient_boosting,
)

from src.models.train_churn import train_svm



def run_pipeline():
    # Load data
    X, y = load_telco_data()

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    mlflow.set_experiment("telco-churn-prediction")

    with mlflow.start_run():

        models = {
        "decision_tree": lambda: train_decision_tree(X_train, y_train, max_depth=5),
        "random_forest": lambda: train_random_forest(X_train, y_train, n_estimators=100),
        "gradient_boosting": lambda: train_gradient_boosting(
            X_train, y_train, n_estimators=100, learning_rate=0.1, max_depth=3
        ),
        "svm_rbf": lambda: train_svm(X_train, y_train, kernel="rbf", C=1.0),
        }

    best_model = None
    best_auc = 0
    best_model_name = None

    for model_name, train_func in models.items():

        with mlflow.start_run(run_name=model_name):

            model = train_func()

            metrics = evaluate_classification(model, X_test, y_test)

            mlflow.log_param("model_name", model_name)

            for key, value in metrics.items():
                mlflow.log_metric(key, value)

            mlflow.sklearn.log_model(model, "model")

            print(f"{model_name} metrics:", metrics)

            if metrics["roc_auc"] > best_auc:
                best_auc = metrics["roc_auc"]
                best_model = model
                best_model_name = model_name

    print(f"\nBest model based on ROC-AUC: {best_model_name} ({best_auc:.3f})")

    # Compute SHAP for best model
    with mlflow.start_run(run_name=f"{best_model_name}_shap"):

        shap_plot_path = log_shap_summary(
            best_model,
            X_train,
            output_dir="outputs",
            prefix=best_model_name
        )

        mlflow.log_artifact(shap_plot_path)

    from src.explainability.feature_importance import log_feature_importance

    if best_model_name in ["random_forest", "gradient_boosting", "decision_tree"]:
        
        csv_path, png_path = log_feature_importance(
            best_model,
            X_train,
            output_dir="outputs",
            prefix=best_model_name
        )

        mlflow.log_artifact(csv_path)
        mlflow.log_artifact(png_path)

        print("Feature importance logged.")






if __name__ == "__main__":
    run_pipeline()
