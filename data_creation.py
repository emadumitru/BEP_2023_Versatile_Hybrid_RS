from collections import Counter
from typing import Any, Dict, Tuple
import os
import numpy as np
import pandas as pd

# Define list of tasks and data characteristics
tasks = ["Data Manipulation", "Data Visualization", "Data Cleaning and Preprocessing",
         "Programming Concepts", "Exploratory Data Analysis", "Object-Oriented Programming"]
data_char = ['crow', 'ccol', 'object', 'float64', 'int64', 'bool', 'cnan', 'categorical', 'string']

# Define paths and loading Boolean flags
PATH_FINAL = 'data/data_final/'
LOAD_CHAR = 1
LOAD_25_CSVS = 1
LOAD_MAPPING = 1


# Functions for inferring data types and column types
def infer_type(value: Any) -> str:
    """
    Infer the type of given value.

    :param value: The value to infer the type from.
    :return: The inferred type of the value out of (bool, int64, float64, and str).
    """
    try:
        bool_val = value.lower()
        if bool_val == 'true' or bool_val == 'false':
            return 'bool'
    except AttributeError:
        pass

    try:
        int(value)
        return 'int64'
    except ValueError:
        pass

    try:
        float(value)
        return 'float64'
    except ValueError:
        pass

    return 'string'


def get_object_column_types(df: pd.DataFrame) -> Dict[str, str]:
    """
    Get the inferred types of object columns in a DataFrame.

    :param df: The DataFrame to analyze.
    :return: A dictionary mapping column names to their inferred types.
    """
    object_columns = df.select_dtypes(include='object').columns
    column_types = {}

    for column in object_columns:
        types = [infer_type(value) for value in df[column]]
        most_common_type = Counter(types).most_common(1)[0][0]

        if most_common_type == 'string':
            unique_values_ratio = df[column].nunique() / len(df)
            if unique_values_ratio <= 0.2:
                most_common_type = "categorical"
            if df[column].nunique() <= 3 and len(df) < 20:
                most_common_type = "categorical"

        column_types[column] = most_common_type

    return column_types


# Function to get dataset characteristics
def get_datasets_characteristics(data_path: str, df: pd.DataFrame = pd.DataFrame(columns=data_char)) -> pd.DataFrame:
    """
    Retrieve characteristics of datasets from a given path and update DataFrame of characteristics to contain them.

    :param data_path: The path to the dataset directory.
    :param df: DataFrame containing dataset characteristics.
    :return: DataFrame with the updated dataset characteristics.
    """
    # Get list of all CSV files in the provided directory
    list_csv = os.listdir(data_path)

    # Load each CSV and compute its characteristics
    for file in list_csv:
        csv = pd.read_csv(data_path + file)
        dtypes = csv.dtypes.unique()

        object_column_type_dict = get_object_column_types(csv)

        df.loc[file, 'crow'] = len(csv)
        df.loc[file, 'ccol'] = len(csv.columns)
        for dtype in dtypes:
            num_cols = len(csv.select_dtypes(include=[dtype]).columns)
            df.loc[file, str(dtype)] = num_cols
        df.loc[file, 'cnan'] = csv.isnull().sum().sum()

        for item in set(object_column_type_dict.values()):
            if item == str:
                item2 = 'string'
                df.loc[file, item2] = list(object_column_type_dict.values()).count(item)
            elif item == 'int64' or item == 'float64':
                df.loc[file, item] += list(object_column_type_dict.values()).count(item)
            else:
                df.loc[file, item] = list(object_column_type_dict.values()).count(item)

    return df.fillna(0)


# Function to expand dataframe to a target size
def expand_dataframe(df_25: pd.DataFrame, target: int) -> pd.DataFrame:
    """
    Expand a DataFrame to reach a target number of rows.

    :param df_25: The input DataFrame with 25% of the target number of rows.
    :param target: The target number of rows.
    :return: The expanded DataFrame.
    """
    num_rows_to_add = target - len(df_25)
    df = pd.DataFrame()

    for column_name in df_25.columns:
        median = df_25[column_name].median()
        std_dev = df_25[column_name].std()
        if column_name == 'bool':
            median = median + 0.7

        fake_values = np.random.normal(loc=median, scale=std_dev, size=num_rows_to_add).astype(int)

        min_val = df_25[column_name].min()
        max_val = df_25[column_name].max()
        fake_values = np.clip(fake_values, min_val, max_val)

        df = pd.concat([df, pd.DataFrame(fake_values, columns=[column_name])], axis=1)

    df = pd.concat([df_25, df], ignore_index=True)
    df['ccol'] = df[['object', 'float64', 'int64', 'bool', 'categorical', 'string']].sum(axis=1)

    return df


# Functions to check task suitability for a given CSV
def check_csv_for_tasks(csv_description: pd.Series) -> Dict[str, bool]:
    """
    Check the CSV mapping to various programming tasks.

    :param csv_description: Series containing the CSV characteristics.
    :return: Dictionary with task names as keys and boolean values indicating task suitability.
    """

    def check_data_manipulation(df: pd.Series) -> bool:
        """
        Check if the CSV is suitable for data manipulation task.

        :param df: DataFrame containing the CSV characteristics.
        :return: True if the CSV is suitable for data manipulation task, False otherwise.
        """
        row_col = df.loc['crow'] >= 250 and df.loc['ccol'] >= 10
        return row_col or df.loc['int64'] >= 5

    def check_data_visualization(df: pd.Series) -> bool:
        """
        Check if the CSV is suitable for data visualization task.

        :param df: DataFrame containing the CSV characteristics.
        :return: True if the CSV is suitable for data visualization task, False otherwise.
        """
        numeric_ratio = (df.loc['float64'] + df.loc['int64']) / df.loc['ccol']
        return (numeric_ratio >= 0.2 and df.loc['ccol'] >= 4) or df.loc['float64'] >= 7

    def check_data_cleaning(df: pd.Series) -> bool:
        """
        Check if the CSV is suitable for data cleaning and preprocessing task.

        :param df: DataFrame containing the CSV characteristics.
        :return: True if the CSV is suitable for data cleaning and preprocessing task, False otherwise.
        """
        missing_value_ratio = (df.loc['cnan'] + 1) / (df.loc['crow'] * df.loc['ccol'])
        return (missing_value_ratio >= 0.1 or df.loc['object'] >= 5) and df.loc['crow'] >= 100

    def check_programming_concepts(df: pd.Series) -> bool:
        """
        Check if the CSV is suitable for programming concepts task.

        :param df: DataFrame containing the CSV characteristics.
        :return: True if the CSV is suitable for programming concepts task, False otherwise.
        """
        descriptors = df.loc['object'] + df.loc['categorical'] + df.loc['string']
        return df.loc['crow'] >= 50 and descriptors >= 5

    def check_exploratory_data_analysis(df: pd.Series) -> bool:
        """
        Check if the CSV is suitable for exploratory data analysis task.

        :param df: DataFrame containing the CSV characteristics.
        :return: True if the CSV is suitable for exploratory data analysis task, False otherwise.
        """
        row_col = df.loc['crow'] >= 200 and df.loc['ccol'] >= 3
        return (row_col and df.loc['bool'] >= 1) or df.loc['cnan'] >= 1000

    def check_object_oriented_programming(df: pd.Series) -> bool:
        """
        Check if the CSV is suitable for object-oriented programming task.

        :param df: DataFrame containing the CSV characteristics.
        :return: True if the CSV is suitable for object-oriented programming task, False otherwise.
        """
        categorical_ratio = df.loc['object'] / df.loc['ccol']
        row_col = df.loc['crow'] >= 150 and df.loc['ccol'] >= 4
        return (categorical_ratio >= 0.15 and row_col) or df.loc['object'] >= 10

    checks = {"Data Manipulation": check_data_manipulation,
              "Data Visualization": check_data_visualization,
              "Data Cleaning and Preprocessing": check_data_cleaning,
              "Programming Concepts": check_programming_concepts,
              "Exploratory Data Analysis": check_exploratory_data_analysis,
              "Object-Oriented Programming": check_object_oriented_programming}

    results = {}
    for task, check_func in checks.items():
        results[task] = check_func(csv_description)

    return results


# Function to check all CSVs in a dataframe for various tasks
def check_all_csvs_in_df(df_initial: pd.DataFrame) -> pd.DataFrame:
    """
    Check all CSVs in a DataFrame for various tasks and update the DataFrame with task suitability information.

    :param df_initial: Initial DataFrame containing CSVs' characteristics.
    :return: Updated DataFrame with task suitability information.
    """
    df = df_initial.copy()
    for file in df.index.tolist():
        suitability = check_csv_for_tasks(df.loc[file])
        for task in suitability:
            df.loc[file, task] = int(suitability[task])
    return df.astype(int)


# Main function to generate final data
def generate_topic_data(load_char: bool = LOAD_CHAR, load_csv25: bool = LOAD_25_CSVS, load_mapping: bool = LOAD_MAPPING,
                        target_rows: int = 1000) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate final data to be used in the tool creation and testing.
    Generates CSVs' characteristics and their mapping to 6 programming tasks.

    :param load_char: Boolean flag indicating whether to load fully existing characteristics data or generate new data.
    :param load_csv25: Boolean flag indicating whether to load existing 25 CSVs' characteristics or generate new data.
    :param load_mapping: Boolean flag indicating whether to load existing mapping data or generate new data.
    :param target_rows: Target row count for expanding the DataFrame.
    :return: Tuple of DataFrames: (characteristics DataFrame, mapping DataFrame).
    """
    if load_char:
        df_char = pd.read_csv(PATH_FINAL + 'full_characteristics.csv').astype(int)
    else:
        if load_csv25:
            df_25 = pd.read_csv(PATH_FINAL + 'characteristics_25.csv', index_col='Unnamed: 0')
        else:
            df_25 = pd.DataFrame(columns=data_char)
            df_25 = get_datasets_characteristics('data/data_initial/course/', df_25)
            df_25 = get_datasets_characteristics('data/data_initial/Kaggle/', df_25)
            df_25 = df_25.fillna(0)
            df_25.to_csv(PATH_FINAL + 'characteristics_25.csv', index=True)

        df_char = expand_dataframe(df_25.reset_index(drop=True), target_rows)
        df_char.to_csv(PATH_FINAL + 'characteristics_1000_temp.csv', index=False)

    if load_mapping:
        df_map = pd.read_csv(PATH_FINAL + 'full_mapping.csv').astype(int)
    else:
        df_map = check_all_csvs_in_df(df_char).fillna(0)
        df_map = df_map[tasks]
        df_map.to_csv(PATH_FINAL + 'mapping_1000_temp.csv', index=False)

    return df_char, df_map


generate_topic_data(True, True, False, 1000)