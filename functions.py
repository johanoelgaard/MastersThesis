import numpy as np
import pandas as pd

def first_valid(row, columns):
    """
    Returns the first non-NaN value in the row from the specified columns.
    
    Parameters:
      row (pd.Series): A row from the DataFrame.
      columns (iterable): An iterable of column names to check.
      
    Returns:
      The first non-NaN value or np.nan if all are NaN.
    """
    for col in columns:
        if pd.notnull(row[col]):
            return row[col]
    return np.nan