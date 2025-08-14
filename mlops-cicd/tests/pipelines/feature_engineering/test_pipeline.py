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

from src.mlops_cicd.pipelines.feature_engineering.nodes import (
    process_passenger_id,
    process_cabin,
    process_name,
    get_sum_of_fees,
    refine_wth_columns,
    create_nbr_by_group,
    create_nbr_by_fam,
    create_nbr_by_cabin,
    create_grouped_features,
    sep_x_y,
    cast_types,
    split_train_val_test,
)

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
        ),
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
        ),
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
        ),
    ]
)
def test_process_name(df_input, df_output):
    result = process_name(df_input)
    pd.testing.assert_frame_equal(result, df_output)

@pytest.mark.parametrize(
    "df_input, df_output",
    [
        (
            pd.DataFrame({
                "FoodCourt": [10, 20],
                "ShoppingMall": [5, 15],
                "Spa": [0, 10],
                "VRDeck": [2, 8],
            }),
            pd.DataFrame({
                "FoodCourt": [10, 20],
                "ShoppingMall": [5, 15],
                "Spa": [0, 10],
                "VRDeck": [2, 8],
                "sum_of_fees": [17, 53],
            })
        ),
        # Add other scenarios if needed
    ]
)
def test_get_sum_of_fees(df_input, df_output):
    result = get_sum_of_fees(df_input)
    pd.testing.assert_frame_equal(result, df_output)

@pytest.mark.parametrize(
    "df_input, df_output",
    [
        (
            pd.DataFrame({
                "PassengerId": ["01_01", "01_02"],
                "Cabin": ["C/123/B", "A/321/A"],
                "Name": ["Ahmed Es", "Tom Ch"],
                "FoodCourt": [10, 20],
                "ShoppingMall": [5, 15],
                "Spa": [0, 10],
                "VRDeck": [2, 8],
            }),
            pd.DataFrame({
                "PassengerId": ["01_01", "01_02"],
                "Cabin": ["C/123/B", "A/321/A"],
                "Name": ["Ahmed Es", "Tom Ch"],
                "FoodCourt": [10, 20],
                "ShoppingMall": [5, 15],
                "Spa": [0, 10],
                "VRDeck": [2, 8],
                "group_id": ["01", "01"],
                "pos_in_group": ["01", "02"],
                "Cabin1": ["C", "A"],
                "Cabin2": ["123", "321"],
                "Cabin3": ["B", "A"],
                "first_name": ["Ahmed", "Tom"],
                "last_name": ["Es", "Ch"],
                "sum_of_fees": [17, 53],
            })
        ),
    ]
)
def test_refine_wth_columns(df_input, df_output):
    result = refine_wth_columns(df_input)
    pd.testing.assert_frame_equal(result, df_output)

@pytest.mark.parametrize(
    "df_input, df_output",
    [
        (
            pd.DataFrame({
                "group_id": ["01", "01", "02"]
            }),
            pd.DataFrame({
                "group_id": ["01", "01", "02"],
                "group_count": [2, 2, 1]
            }),
        ),
    ]
)
def test_create_nbr_by_group(df_input, df_output):
    result = create_nbr_by_group(df_input)
    pd.testing.assert_frame_equal(result, df_output)

@pytest.mark.parametrize(
    "df_input, df_output",
    [
        (
            pd.DataFrame({
                "last_name": ["Smith", "Smith", "Lee", "Doe"]
            }),
            pd.DataFrame({
                "last_name": ["Smith", "Smith", "Lee", "Doe"],
                "family_size": [2, 2, 1, 1]
            }),
        ),
    ]
)
def test_create_nbr_by_fam(df_input, df_output):
    result = create_nbr_by_fam(df_input)
    pd.testing.assert_frame_equal(result, df_output)

@pytest.mark.parametrize(
    "df_input, df_output",
    [
        (
            pd.DataFrame({
                "Cabin1": ["A", "A", "B"]
            }),
            pd.DataFrame({
                "Cabin1": ["A", "A", "B"],
                "cabin_count": [2, 2, 1]
            }),
        ),
    ]
)
def test_create_nbr_by_cabin(df_input, df_output):
    result = create_nbr_by_cabin(df_input)
    pd.testing.assert_frame_equal(result, df_output)

@pytest.mark.parametrize(
    "df_input, df_output",
    [
        (
            pd.DataFrame({
                "group_id": ["01", "01", "02"],
                "last_name": ["Smith", "Smith", "Lee"],
                "Cabin1": ["A", "A", "B"]
            }),
            pd.DataFrame({
                "group_id": ["01", "01", "02"],
                "last_name": ["Smith", "Smith", "Lee"],
                "Cabin1": ["A", "A", "B"],
                "group_count": [2, 2, 1],
                "family_size": [2, 2, 1],
                "cabin_count": [2, 2, 1]
            }),
        ),
    ]
)
def test_create_grouped_features(df_input, df_output):
    result = create_grouped_features(df_input)
    pd.testing.assert_frame_equal(result, df_output)

@pytest.mark.parametrize(
    "df_input, feats_dict, target_name, expected_X, expected_y",
    [
        (
            pd.DataFrame({
                "f1": [1, 2],
                "f2": [3, 4],
                "target": [0, 1]
            }),
            {"f1": None, "f2": None}, "target",
            pd.DataFrame({"f1": [1, 2], "f2": [3, 4]}),
            pd.Series([0, 1], name="target"),
        ),
    ]
)
def test_sep_x_y(df_input, feats_dict, target_name, expected_X, expected_y):
    X, y = sep_x_y(df_input, feats_dict, target_name)
    pd.testing.assert_frame_equal(X, expected_X)
    pd.testing.assert_series_equal(y, expected_y, check_names=True)

@pytest.mark.parametrize(
    "df_input, feats_dict, df_output",
    [
        (
            pd.DataFrame({
                "f1": [1.2, None, 3.5],
                "f2": [None, 2, 3],
            }),
            {"f1": "int", "f2": "float"},
            pd.DataFrame({
                "f1": [1, -1, 3],
                "f2": [-1.0, 2.0, 3.0],
            }),
        ),
    ]
)
def test_cast_types(df_input, feats_dict, df_output):
    result = cast_types(df_input, feats_dict)
    pd.testing.assert_frame_equal(result, df_output)

def test_split_train_val_test():
    X = pd.DataFrame({
        "a": range(20),
        "b": range(100, 120)
    })
    y = pd.Series(range(20))
    X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_test(X, y)
    # Check that splits add up
    assert len(X_train) + len(X_val) + len(X_test) == 20
    assert len(y_train) + len(y_val) + len(y_test) == 20
    # Check no overlap
    train_idx = set(X_train.index)
    val_idx = set(X_val.index)
    test_idx = set(X_test.index)
    assert train_idx.isdisjoint(val_idx)
    assert val_idx.isdisjoint(test_idx)
    assert train_idx.isdisjoint(test_idx)
