"""
This is a boilerplate test file for pipeline 'feature_engineering'
generated using Kedro 0.19.14.
Please add your pipeline tests here.

Kedro recommends using `pytest` framework, more info about it can be found
in the official documentation:
https://docs.pytest.org/en/latest/getting-started.html
"""

import pandas as pd
import pytest

from src.mlops_cicd.pipelines.feature_engineering.nodes import process_passenger_id

@pytest.mark.parametrize(
        "df_input, df_output",
        [
            (
                pd.DataFrame({
                    "PassengerId" : ["01_01", "01_02", "02_01"],
                }),
                pd.DataFrame({
                    "PassengerId" : ["01_01", "01_02", "02_01"],
                    "group_id" : ["01", "01", "02"],
                    "pos_in_group" : ["01", "02", "01"]
                 }),
            )
        ]
)
def test_process_passenger_id(df_input, df_output):
    result = process_passenger_id(df_input)
    pd.testing.assert_frame_equal(result, df_output)