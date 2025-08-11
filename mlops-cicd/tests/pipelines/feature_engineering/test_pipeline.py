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
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from src.mlops_cicd.pipelines.feature_engineering.nodes import process_passenger_id, process_cabin, process_name

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

@pytest.mark.parametrize(
    "df_input, df_output",
    [
        (
            pd.DataFrame({
                "Cabin" : ["C/123/B", "A/123/C", "S/340/D"]
            }),
            pd.DataFrame({
                "Cabin" : ["C/123/B", "A/123/C", "S/340/D"],
                "Cabin1" : ["C", "A", "S"],
                "Cabin2" : ["123", "123", "340"],
                "Cabin3" : ["B", "C", "D"]
            })
        )
    ]
)
def test_process_cabin(df_input, df_output):
    result = process_cabin(df_input)
    pd.testing.assert_frame_equal(result, df_output)


@pytest.mark.parametrize(
    "df_input, df_output",
    [
        (
            pd.DataFrame({
                "Name" : ["Ahmed Es", "Tom Ch", "Mahdi Ba"],
            }),
            pd.DataFrame({
                "Name" : ["Ahmed Es", "Tom Ch", "Mahdi Ba"],
                "first_name" : ["Ahmed", "Tom", "Mahdi"],
                "last_name" : ["Es", "Ch", "Ba"],
            })
        )
    ]
)
def test_process_name(df_input, df_output):
    result = process_name(df_input)
    pd.testing.assert_frame_equal(result, df_output)
