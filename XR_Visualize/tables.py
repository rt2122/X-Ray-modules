import pandas as pd
import numpy as np


def format_floats_table(df: pd.DataFrame, fmt: str) -> pd.DataFrame:
    """
    Replace every float in table with formated string.
    """
    new_df = df.copy()
    for column in list(df):
        arr = df[column]
        non_zero = arr[arr != 0]
        non_zero = np.array(non_zero)[0]
        if int(non_zero) != non_zero:
            new_df[column] = np.array(map(lambda x: fmt.format(x), list(arr)))
    return new_df
