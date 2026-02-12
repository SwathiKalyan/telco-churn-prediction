# Telco Customer Churn Prediction

## Project Overview

This project focuses on predicting customer churn using supervised machine learning techniques.

The objective is to identify customers likely to churn so that proactive retention strategies can be applied. The system compares multiple classification models, evaluates performance using business-relevant metrics, and interprets model behavior using feature importance and SHAP explainability.

Dataset: IBM Telco Customer Churn (Kaggle)

---

## Key Objectives

- Compare multiple supervised learning models
- Handle imbalanced classification problem
- Evaluate using Recall, Precision, F1-score, and ROC-AUC
- Select best model based on ROC-AUC
- Interpret model behavior using SHAP
- Track experiments using MLflow

---

## Models Compared

- Decision Tree
- Random Forest
- Gradient Boosting
- Support Vector Machine (RBF Kernel)

Best model selected based on ROC-AUC:  
**Gradient Boosting (ROC-AUC ≈ 0.84)**

---

## Evaluation Metrics

Since churn prediction is an imbalanced classification problem, multiple metrics were used:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

Model selection was based primarily on ROC-AUC to evaluate ranking capability, with recall analyzed for business trade-offs.

---

## Explainability

To interpret model decisions:

- Global Feature Importance (tree-based importance)
- SHAP summary plots for feature contribution analysis

This enables understanding which customer attributes most influence churn risk.

---

## Pipeline Architecture

1. Data Loading
2. Train/Test Split (Stratified)
3. Model Training
4. Model Evaluation
5. Best Model Selection
6. SHAP Explainability
7. MLflow Logging

---

## Project Structure

telco-churn-prediction/
│
├── data/
│ └── raw/
│ └── CustomerChurn.xlsx
│
├── src/
│ ├── ingestion/
│ ├── models/
│ ├── evaluation/
│ ├── explainability/
│ └── main.py
│
├── requirements.txt
└── README.md


---

## How to Run

From project root:

```bash
python -m src.main

To launch MLflow UI:

mlflow ui

Then open:

http://127.0.0.1:5000

Key Highlights

Implemented modular supervised ML pipeline

Compared multiple models under consistent evaluation framework

Applied SHAP for interpretable ML

Integrated MLflow for experiment tracking

Designed clean, package-based execution structure