from typing import Dict

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import shap
from catboost import CatBoostClassifier
from sklearn.metrics import (
    accuracy_score,
    auc,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def _log_roc_curve(y_true: np.ndarray, y_scores: np.ndarray, artifact_path: str = "metrics/roc_curve.png") -> None:
    """Logs a ROC curve plot as an MLflow artifact.

    Computes the ROC curve from true labels and predicted scores, plots it using matplotlib,
    and logs the plot directly to MLflow without saving an intermediate file.

    Args:
        y_true (np.ndarray): True binary labels.
        y_scores (np.ndarray): Target scores or probability estimates.
        artifact_path (str, optional): Path for the artifact in MLflow. Defaults to "metrics/roc_curve.png".

    Returns:
        None
    """
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
    ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Receiver Operating Characteristic (ROC) Curve")
    ax.legend(loc="lower right")

    mlflow.log_figure(fig, artifact_path)
    plt.close(fig)


def _log_shap_global(model: CatBoostClassifier, x_test: pd.DataFrame) -> None:
    """Logs a global SHAP summary plot as an MLflow artifact.

    Computes SHAP values for the test set, generates a summary plot with matplotlib,
    and logs the plot directly to MLflow.

    Args:
        model (CatBoostClassifier): Trained CatBoost model.
        x_test (pd.DataFrame): Test features to compute SHAP values on.

    Returns:
        None
    """
    explainer = shap.TreeExplainer(model)
    shap_values_plot = explainer.shap_values(x_test)
    fig = plt.figure()
    shap.summary_plot(shap_values_plot, x_test, show=False)
    mlflow.log_figure(fig, "shap/shap_summary_plot.png")
    plt.close(fig)


def compute_classification_metrics(y_true: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, float]:
    """Computes common classification metrics.

    Calculates AUC, accuracy, F1, precision, recall, and log loss from true labels and predicted probabilities.

    Args:
        y_true (np.ndarray): True binary labels.
        y_pred_proba (np.ndarray): Predicted probabilities for the positive class.

    Returns:
        Dict[str, float]: Dictionary of computed metrics.
    """
    y_pred = (y_pred_proba > 0.5).astype(int)
    return {
        "auc": roc_auc_score(y_true, y_pred_proba),
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "logloss": log_loss(y_true, y_pred_proba),
    }
