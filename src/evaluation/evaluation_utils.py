import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    recall_score,
    precision_score,
    confusion_matrix
)


def evaluate_binary_classifier(model, X_train, y_train, X_test, y_test, threshold):
    """
    Computes AUC, Recall, Precision and Confusion Matrix
    """

    train_probs = model.predict_proba(X_train)[:, 1]
    test_probs = model.predict_proba(X_test)[:, 1]

    metrics = {}

    metrics["train_auc"] = roc_auc_score(y_train, train_probs)
    metrics["val_auc"] = roc_auc_score(y_test, test_probs)

    y_pred = (test_probs >= threshold).astype(int)

    metrics["recall_churn"] = recall_score(y_test, y_pred)
    metrics["precision_churn"] = precision_score(y_test, y_pred)

    cm = confusion_matrix(y_test, y_pred)

    cm_df = pd.DataFrame(
        cm,
        index=["Actual_No", "Actual_Yes"],
        columns=["Pred_No", "Pred_Yes"]
    )

    return metrics, cm_df

def compute_business_cost(cm_df, cost_fn=5000, cost_fp=500):
    fn = cm_df.loc["Actual_Yes", "Pred_No"]
    fp = cm_df.loc["Actual_No", "Pred_Yes"]

    total_cost = (fn * cost_fn) + (fp * cost_fp)

    return total_cost

