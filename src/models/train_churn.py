import mlflow
import mlflow.sklearn
import os

from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from src.ingestion.data_loader import load_telco_data

def train_svm(X_train, y_train, kernel="rbf", C=1.0):
    from sklearn.svm import SVC

    model = SVC(
        kernel=kernel,
        C=C,
        probability=True,   # Important for ROC-AUC
        random_state=42
    )

    model.fit(X_train, y_train)
    return model

