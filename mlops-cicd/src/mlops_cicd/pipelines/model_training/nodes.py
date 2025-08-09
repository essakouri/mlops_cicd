"""
This is a boilerplate pipeline 'model_training'
generated using Kedro 1.0.0
"""

from functools import partial
from typing import Any, Dict, List

import mlflow
import optuna
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score


def _objective(
    trial: optuna.trial.Trial,
    x_train: pd.DataFrame,
    x_val: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    cat_features: List[int],
    parent_run_id: str,
) -> float:
    """Optuna objective function for CatBoostClassifier with MLflow tracking.

    This function suggests hyperparameters, trains a CatBoostClassifier, evaluates its
    performance using ROC AUC, and logs parameters and metrics to MLflow.

    Args:
        trial (optuna.trial.Trial): Optuna trial object.
        x_train (pandas.DataFrame): Training features.
        x_val (pandas.DataFrame): Validation features.
        y_train (pandas.Series): Training target.
        y_val (pandas.Series): Validation target.
        cat_features (List[int]): Indices of categorical features for CatBoost.
        parent_run_id (str): MLflow parent run ID for nested experiment tracking.

    Returns:
        float: Validation ROC AUC score (to be maximized by Optuna).
    """
    params = {
        "iterations": trial.suggest_int("iterations", 100, 500),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "depth": trial.suggest_int("depth", 4, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True),
        "random_seed": 1997,
        "verbose": 0,
        "loss_function": "Logloss",
    }

    with mlflow.start_run(parent_run_id=parent_run_id, nested=True):
        mlflow.log_params(params)

        model = CatBoostClassifier(**params, cat_features=cat_features)
        model.fit(x_train, y_train, eval_set=(x_val, y_val), use_best_model=True)

        y_pred_train = model.predict_proba(x_train)[:, 1]
        y_pred_val = model.predict_proba(x_val)[:, 1]

        accuracy_train = roc_auc_score(y_train, y_pred_train)
        accuracy_val = roc_auc_score(y_val, y_pred_val)

        mlflow.log_metric("train_auc", accuracy_train)
        mlflow.log_metric("val_auc", accuracy_val)

    return accuracy_val


def bayes_opti(
    x_train: pd.DataFrame,
    x_val: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    features_dict: Dict[str, str],
    n_trials: int,
) -> Dict[str, Any]:
    """Runs Bayesian optimization for CatBoostClassifier using Optuna and MLflow.

    This function extracts categorical features, sets up an MLflow experiment,
    and optimizes CatBoost hyperparameters with Optuna. Returns the best parameters.

    Args:
        x_train (pandas.DataFrame): Training features.
        x_val (pandas.DataFrame): Validation features.
        y_train (pandas.Series): Training target.
        y_val (pandas.Series): Validation target.
        features_dict (dict): Dictionary mapping feature names to types.
        n_trials (int): Number of trials for Optuna optimization.

    Returns:
        dict: The best hyperparameters found during optimization.
    """
    # Extract categorical features (CatBoost expects indices, not names)
    cat_features = [i for i, (k, v) in enumerate(features_dict.items()) if v == "str"]

    # Set MLflow experiment (creates it if it doesn't exist)
    experiment_name = "Bayesian Opti"
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run() as run:
        study = optuna.create_study(direction="maximize")
        study.optimize(
            partial(
                _objective,
                x_train=x_train,
                x_val=x_val,
                y_train=y_train,
                y_val=y_val,
                cat_features=cat_features,
                parent_run_id=run.info.run_id,
            ),
            n_trials=n_trials,
        )
        best_params = study.best_params
    return best_params
