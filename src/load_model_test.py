import mlflow
import mlflow.sklearn
import numpy as np

# 🔴 REPLACE with your real run_id
RUN_ID = "e281324c74e141f29e8c90d3b9c9ff34"

# Load model from MLflow artifacts
model = mlflow.sklearn.load_model(
    f"runs:/{RUN_ID}/model"
)

print("Model loaded successfully!")

# Dummy input (must match training feature count)
X_dummy = np.random.randn(5, 15)

# Run inference
preds = model.predict_proba(X_dummy)[:, 1]

print("Predictions:", preds)