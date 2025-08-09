"""
This is a boilerplate pipeline 'model_training'
generated using Kedro 1.0.0
"""

from kedro.pipeline import node, Pipeline  # noqa
from .nodes import bayes_opti, train_best_model

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=bayes_opti,
                inputs=[
                    "X_train",
                    "X_val",
                    "y_train",
                    "y_val",
                    "params:features_dict",
                    "params:n_trials"
                ],
                outputs="best_params",
                name="bayesian_opti"
            ),
            node(
                func=train_best_model,
                inputs=[
                    "X_train",
                    "X_val",
                    "X_test",
                    "y_train",
                    "y_val",
                    "y_test",
                    "best_params"
                ],
                outputs="model",
                name="train_best_model"
            ),
        ],
)
