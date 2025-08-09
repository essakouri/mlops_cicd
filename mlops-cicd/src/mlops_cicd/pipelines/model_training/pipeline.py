"""
This is a boilerplate pipeline 'model_training'
generated using Kedro 1.0.0
"""

from kedro.pipeline import node, Pipeline  # noqa
from .nodes import bayes_opti

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
            )
        ],
)
