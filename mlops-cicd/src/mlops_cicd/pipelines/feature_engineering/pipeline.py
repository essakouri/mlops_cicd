"""
This is a boilerplate pipeline 'feature_engineering'
generated using Kedro 0.19.14
"""
from kedro.pipeline import node, Pipeline, pipeline  # noqa
from .nodes import refine_wth_columns, create_grouped_features, sep_x_y, cast_types


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
                func=refine_wth_columns,
                inputs=[
                    "df_raw",
                ],
                outputs="df_refined",
                name="adding_new_features",
            ),
            node(
                func=create_grouped_features,
                inputs=[
                    "df_refined",
                ],
                outputs="df_with_grouped",
                name="create_grouped_features",
            ),
            node(
                func=sep_x_y,
                inputs=[
                    "df_with_grouped",
                    "params:features_dict",
                    "params:target_name"
                ],
                outputs=["X", "y"],
                name="separate_x_and_y",
            ),
            node(
                func=cast_types,
                inputs=[
                    "X",
                    "params:features_dict",
                ],
                outputs="X_casted",
                name="casting_types",
            ),
    ],
    inputs="df_raw",
    outputs=["X_casted", "y"])
