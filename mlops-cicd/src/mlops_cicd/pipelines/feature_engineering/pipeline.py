"""
This is a boilerplate pipeline 'feature_engineering'
generated using Kedro 0.19.14
"""
from kedro.pipeline import node, Pipeline, pipeline  # noqa
from .nodes import refine_wth_columns


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
                func=refine_wth_columns,
                inputs=[
                    "df_raw",
                ],
                outputs="df_refined",
                name="Adding new features",
            ),
    ],
    inputs="df_raw",
    outputs="df_refined")
