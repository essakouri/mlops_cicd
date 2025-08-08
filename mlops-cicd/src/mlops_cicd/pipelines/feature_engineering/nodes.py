"""
This is a boilerplate pipeline 'feature_engineering'
generated using Kedro 0.19.14
"""

import pandas as pd
import numpy as np

from typing import List, Tuple, Dict
from sklearn.model_selection import train_test_split


def process_passenger_id(df: pd.DataFrame, colname: str = "PassengerId") -> pd.DataFrame:
    """Splits the specified column into 'group_id' and 'pos_in_group' columns.

    This function takes a DataFrame and splits the values of the specified column
    (by default, 'PassengerId') on the underscore character '_', creating two new
    columns: 'group_id' and 'pos_in_group'. The original DataFrame is modified in-place.

    Args:
        df (pandas.DataFrame): The input DataFrame containing the passenger ID column.
        colname (str, optional): The name of the column to split. Defaults to "PassengerId".

    Returns:
        pandas.DataFrame: The DataFrame with added 'group_id' and 'pos_in_group' columns.

    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame({'PassengerId': ['123_1', '123_2']})
        >>> process_passenger_id(df)
        Index(['PassengerId', 'group_id', 'pos_in_group'], dtype='object')
        PassengerId group_id pos_in_group
        0      123_1     123           1
        1      123_2     123           2
    """
    df[["group_id", "pos_in_group"]] = df[colname].str.split("_", expand=True)
    return df


def process_cabin(df: pd.DataFrame, colname: str = "Cabin") -> pd.DataFrame:
    """Splits the specified cabin column into 'Cabin1', 'Cabin2', and 'Cabin3' columns.

    This function splits the values of the given column (by default, 'Cabin')
    on the '/' character, creating three new columns: 'Cabin1', 'Cabin2', and 'Cabin3'.
    The original DataFrame is modified in-place.

    Args:
        df (pandas.DataFrame): The input DataFrame containing the cabin information.
        colname (str, optional): The name of the column to split. Defaults to 'Cabin'.

    Returns:
        pandas.DataFrame: The DataFrame with added 'Cabin1', 'Cabin2', and 'Cabin3' columns.

    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame({'Cabin': ['C/123/B', 'D/456/A']})
        >>> process_cabin(df)
          Cabin Cabin1 Cabin2 Cabin3
        0  C/123/B     C    123      B
        1  D/456/A     D    456      A
    """
    df[["Cabin1", "Cabin2", "Cabin3"]] = df[colname].str.split("/", expand=True)
    return df


def process_name(df: pd.DataFrame, colname: str = "Name") -> pd.DataFrame:
    """Splits the specified name column into 'first_name' and 'last_name' columns.

    This function splits the values of the given column (by default, 'Name')
    on the space character, creating two new columns: 'first_name' and 'last_name'.
    The original DataFrame is modified in-place.

    Args:
        df (pandas.DataFrame): The input DataFrame containing the name information.
        colname (str, optional): The name of the column to split. Defaults to 'Name'.

    Returns:
        pandas.DataFrame: The DataFrame with added 'first_name' and 'last_name' columns.

    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame({'Name': ['John Doe', 'Jane Smith']})
        >>> process_name(df)
             Name first_name last_name
        0   John Doe       John       Doe
        1 Jane Smith       Jane     Smith
    """
    df[["first_name", "last_name"]] = df[colname].str.split(" ", expand=True)
    return df


def get_sum_of_fees(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates the sum of specified fee columns and adds it as a new column.

    This function computes the sum of the 'FoodCourt', 'ShoppingMall', 'Spa',
    and 'VRDeck' columns for each row in the DataFrame and adds the result in
    a new column named 'sum_of_fees'.

    Args:
        df (pandas.DataFrame): The input DataFrame containing the fee columns.

    Returns:
        pandas.DataFrame: The DataFrame with an additional 'sum_of_fees' column.

    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame({
        ...     'FoodCourt': [10, 20],
        ...     'ShoppingMall': [5, 15],
        ...     'Spa': [0, 10],
        ...     'VRDeck': [2, 8]
        ... })
        >>> get_sum_of_fees(df)
           FoodCourt  ShoppingMall  Spa  VRDeck  sum_of_fees
        0         10             5    0       2           17
        1         20            15   10       8           53
    """
    df["sum_of_fees"] = df["FoodCourt"] + df["ShoppingMall"] + df["Spa"] + df["VRDeck"]
    return df


def refine_wth_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Applies passenger, cabin, and name processing functions to a DataFrame.

    This function applies the following processing functions sequentially
    to the input DataFrame:
    - process_passenger_id: Splits the passenger ID column.
    - process_cabin: Splits the cabin column.
    - process_name: Splits the name column.

    Args:
        df (pandas.DataFrame): The input DataFrame to process.

    Returns:
        pandas.DataFrame: The processed DataFrame with new columns added.
    """
    df = process_passenger_id(df)
    df = process_cabin(df)
    df = process_name(df)
    df = get_sum_of_fees(df)
    return df


def create_nbr_by_group(df: pd.DataFrame) -> pd.DataFrame:
    """Adds a 'group_count' column with the size of each group identified by 'group_id'.

    Args:
        df (pandas.DataFrame): Input DataFrame, must have a 'group_id' column.

    Returns:
        pandas.DataFrame: DataFrame with an additional 'group_count' column.
    """
    grouped = df.groupby(by="group_id").size().reset_index(name="group_count")
    df = df.merge(grouped, on="group_id", how="left")
    return df


def create_nbr_by_fam(df: pd.DataFrame) -> pd.DataFrame:
    """Adds a 'family_size' column with the number of occurrences for each last name.

    Args:
        df (pandas.DataFrame): Input DataFrame, must have a 'last_name' column.

    Returns:
        pandas.DataFrame: DataFrame with an additional 'family_size' column.
    """
    grouped = df.groupby(by="last_name").size().reset_index(name="family_size")
    df = df.merge(grouped, on="last_name", how="left")
    return df


def create_nbr_by_cabin(df: pd.DataFrame) -> pd.DataFrame:
    """Adds a 'cabin_count' column with the size of each group identified by 'Cabin1'.

    Args:
        df (pandas.DataFrame): Input DataFrame, must have a 'Cabin1' column.

    Returns:
        pandas.DataFrame: DataFrame with an additional 'cabin_count' column.
    """
    grouped = df.groupby(by="Cabin1").size().reset_index(name="cabin_count")
    df = df.merge(grouped, on="Cabin1", how="left")
    return df


def create_grouped_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds grouped features: 'group_count', 'family_size', and 'cabin_count'.

    Applies the three counting functions in sequence to the input DataFrame.

    Args:
        df (pandas.DataFrame): Input DataFrame.

    Returns:
        pandas.DataFrame: DataFrame with 'group_count', 'family_size', and 'cabin_count' columns added.
    """
    df = create_nbr_by_group(df)
    df = create_nbr_by_fam(df)
    df = create_nbr_by_cabin(df)
    return df


def sep_x_y(df: pd.DataFrame, feats_dict: Dict[str, any], target_name: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Splits a DataFrame into features X and target y.

    Args:
        df (pandas.DataFrame): The input DataFrame.
        feats_dict (Dict[str, Any]): Dictionary whose keys are feature column names.
        target_name (str): Name of the target column (y).

    Returns:
        Tuple[pandas.DataFrame, pandas.Series]:
            - X: DataFrame containing feature columns.
            - y: Series containing the target column.
    """
    return df[list(feats_dict.keys())], df[target_name]


def cast_types(X, feats_dict):
    # TODO: change when creating fillna fct
    for i in feats_dict.keys():
        X[i] = X[i].fillna(-1).astype(feats_dict[i])
    return X


def split_train_val_test(
    X: pd.DataFrame, y: pd.Series
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Splits datasets into train, validation, and test sets (70/15/15 by default).

    First splits into train (70%) and temp (30%), then splits temp into
    validation (15%) and test (15%).

    Args:
        X (pd.DataFrame): Features.
        y (pd.Series): Target.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
            X_train, X_val, X_test, y_train, y_val, y_test

    Example:
        >>> X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_test(X, y)
        >>> print(X_train.shape, X_val.shape, X_test.shape)
    """
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, train_size=0.70, random_state=1997)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, train_size=0.50, random_state=1997)
    return X_train, X_val, X_test, y_train, y_val, y_test
