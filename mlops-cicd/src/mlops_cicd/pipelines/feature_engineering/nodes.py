"""
This is a boilerplate pipeline 'feature_engineering'
generated using Kedro 0.19.14
"""
import pandas as pd
import numpy as np

def process_passenger_id(
    df: pd.DataFrame,
    colname: str = "PassengerId"
) -> pd.DataFrame:
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

def process_cabin(
    df: pd.DataFrame,
    colname: str = "Cabin"
) -> pd.DataFrame:
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

def process_name(
    df: pd.DataFrame,
    colname: str = "Name"
) -> pd.DataFrame:
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
    return df