import numpy as np
import pandas as pd
import re
from decimal import Decimal

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

def extract_nace(row, cols):
    # Format a Decimal to a plain string (no scientific notation or extra zeros)
    def format_plain(d):
        s = format(d, 'f')
        return s.rstrip('0').rstrip('.') if '.' in s else s

    # Expand a numeric range from start_str to end_str into a list of codes
    def decimal_range(start_str, end_str):
        start_dec, end_dec = Decimal(start_str), Decimal(end_str)
        # Determine the number of decimal places to use for the step
        step_places = max(abs(start_dec.as_tuple().exponent), abs(end_dec.as_tuple().exponent))
        step = Decimal("1") / (Decimal("10") ** step_places)
        values = []
        while start_dec <= end_dec:
            values.append(format_plain(start_dec))
            start_dec += step
        return values

    # Parse an individual segment such as "20.1 to 20.3", "30-30.3", or "26"
    def parse_segment(segment):
        segment = segment.strip()
        # Handle subtraction of codes (e.g. "30 less 30.3")
        if " less " in segment:
            left, right = segment.split(" less ", 1)
            return parse_segment(left) - parse_segment(right)
        # Match a range with "to" or "-" as a delimiter
        m_range = re.match(
            r'^(\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*(?:to|-)\s*(\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)$', segment)
        if m_range:
            start, end = m_range.groups()
            return set(decimal_range(start, end))
        # Match a single code
        m_single = re.match(r'^(\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)$', segment)
        if m_single:
            return {format_plain(Decimal(m_single.group(1)))}
        return set()

    # Extract NACE codes from a text cell (searching for the "nace rv2" pattern)
    def extract_from_text(text):
        match = re.search(r'nace rv2\s*([\d\.,\s\-\w]+)', text, re.IGNORECASE)
        if not match:
            return set()
        codes = set()
        for part in match.group(1).split(','):
            if part.strip():
                codes |= parse_segment(part)
        return codes

    # Loop through the specified columns in the row and combine codes
    combined = set()
    for col in cols:
        cell_value = row.get(col, "")
        if pd.notna(cell_value) and isinstance(cell_value, str):
            combined |= extract_from_text(cell_value)
    # Sort by numeric value using Decimal
    return sorted(combined, key=lambda x: Decimal(x))



def rolling_beta(g):
    g = g.sort_index()
    cov = g['stkre'].rolling('365D').cov(g['mktre'], ddof=0)
    var = g['mktre'].rolling('365D').var()
    return cov.divide(var)


def expand_monthly(row):
    # build a monthly DateTimeIndex from timestamp up to (but not including) period_end
    return pd.date_range(
        start=row.timestamp,
        end=row.period_end - pd.DateOffset(days=1),
        freq='ME'
    )
