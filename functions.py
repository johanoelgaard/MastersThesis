import numpy as np
import pandas as pd
import re
from decimal import Decimal
from numpy import linalg as la
from tabulate import tabulate
from scipy import stats
from scipy.stats import chi2
import math

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

def estimate(y, x, z=None, id=None, transform='', T=None, robust=False):
    """
    Performs regression using OLS

    Args:
        y (np.ndarray): Dependent variable (2D shape).
        x (np.ndarray): Independent variables (2D shape).
        z (np.ndarray, optional): Instrumental variables for 2SLS. Defaults to None.
        id (np.ndarray, optional): Panel data id. Defaults to None.
        transform (str, optional): Data transformation ('', 'fd', 'be', 'fe', 're'). Defaults to ''.
        T (int, optional): Number of time periods (for panel data). Defaults to None.
        robust (bool, optional): Whether to compute robust standard errors. Defaults to False.

    Returns:
        dict: Regression results containing coefficients, standard errors, etc.
    """
    b_hat = est_ols(y, x)
 
    residual = y - x @ b_hat
    SSR = residual.T @ residual
    SST = (y - np.mean(y)).T @ (y - np.mean(y))
    R2 = 1 - SSR / SST

    sigma2, cov, se, df = variance(transform, SSR, x, T, residual, robust)
    t_values = b_hat / se
    wald = (b_hat - 0) ** 2 / se ** 2

    results = {
        'b_hat': b_hat,
        'se': se,
        'sigma2': sigma2,
        't_values': t_values,
        'R2': R2,
        'cov': cov,
        'wald': wald,
        'df': df
    }

    return results

def est_ols( y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Estimates y on x by ordinary least squares, returns coefficents

    Args:
        >> y (np.ndarray): Dependent variable (Needs to have shape 2D shape)
        >> x (np.ndarray): Independent variable (Needs to have shape 2D shape)

    Returns:
        np.array: Estimated beta coefficients.
    """
    return la.inv(x.T@x)@(x.T@y)

def variance(transform, SSR, x, T, residual, robust=False):
    """
    Computes covariance matrix and standard errors.

    Args:
        transform (str): Data transformation ('', 'fd', 'be', 'fe', 're').
        SSR (float): Sum of squared residuals.
        x (np.ndarray): Independent variables.
        T (int): Number of time periods (for panel data).
        residual (np.ndarray): Residuals from regression.
        robust (bool, optional): Whether to compute robust standard errors. Defaults to False.

    Returns:
        tuple: sigma2, covariance matrix, standard errors, degrees of freedom.
    """
    K = x.shape[1]
    if transform in ('', 'fd', 'be'):
        N = x.shape[0]
    else:
        N = x.shape[0] / T

    if transform in ('', 'fd', 'be'):
        df = N - K
        sigma2 = SSR / df
    elif transform.lower() == 'fe':
        df = N * (T - 1) - K
        sigma2 = SSR / df
    elif transform.lower() == 're':
        df = T * N - K
        sigma2 = SSR / df
    else:
        raise ValueError('Invalid transform provided.')

    XtX_inv = la.inv(x.T @ x)
    if robust:
        cov = XtX_inv @ (x.T @ np.diagflat(residual**2) @ x) @ XtX_inv
    else:
        cov = sigma2 * XtX_inv

    se = np.sqrt(np.diag(cov)).reshape(-1, 1)
    return sigma2, cov, se, df

def print_table(
        labels: tuple,
        results: dict,
        headers=["", "Beta", "Se", "t-values"],
        title="Results",
        _lambda:float=None,
        **kwargs
    ) -> None:
    """Prints a nice looking table, must at least have coefficients, 
    standard errors and t-values. The number of coefficients must be the
    same length as the labels.

    Args:
        >> labels (tuple): Touple with first a label for y, and then a list of 
        labels for x.
        >> results (dict): The results from a regression. Needs to be in a 
        dictionary with at least the following keys:
            'b_hat', 'se', 't_values', 'R2', 'sigma2'
        >> headers (list, optional): Column headers. Defaults to 
        ["", "Beta", "Se", "t-values"].
        >> title (str, optional): Table title. Defaults to "Results".
        _lambda (float, optional): Only used with Random effects. 
        Defaults to None.
    """
    
    # Unpack the labels
    label_y, label_x = labels
    
    # Create table, using the label for x to get a variable's coefficient,
    # standard error and t_value.
    table = []
    for i, name in enumerate(label_x):
        row = [
            name, 
            results.get('b_hat')[i], 
            results.get('se')[i], 
            results.get('t_values')[i]
        ]
        table.append(row)
    
    # Print the table
    print(title)
    print(f"Dependent variable: {label_y}\n")
    print(tabulate(table, headers, **kwargs))
    
    # Print extra statistics of the model.
    print(f"R\u00b2 = {results.get('R2').item():.3f}")
    print(f"\u03C3\u00b2 = {results.get('sigma2').item():.3f}")
    if _lambda: 
        print(f'\u03bb = {_lambda.item():.3f}')


def perm( Q_T: np.ndarray, A: np.ndarray) -> np.ndarray:
    """Takes a transformation matrix and performs the transformation on 
    the given vector or matrix.

    Args:
        Q_T (np.ndarray): The transformation matrix. Needs to have the same
        dimensions as number of years a person is in the sample.
        
        A (np.ndarray): The vector or matrix that is to be transformed. Has
        to be a 2d array.

    Returns:
        np.array: Returns the transformed vector or matrix.
    """
    # We can infer t from the shape of the transformation matrix.
    M,T = Q_T.shape 
    N = int(A.shape[0]/T)
    K = A.shape[1]

    # initialize output 
    Z = np.empty((M*N, K))
    
    for i in range(N): 
        ii_A = slice(i*T, (i+1)*T)
        ii_Z = slice(i*M, (i+1)*M)
        Z[ii_Z, :] = Q_T @ A[ii_A, :]

    return Z

# Create function to check rank of demeaned matrix, and return its eigenvalues.
def check_rank(x):
    print(f'Rank of demeaned x: {la.matrix_rank(x)}')
    lambdas, V = la.eig(x.T@x)
    np.set_printoptions(suppress=True)  # This is just to print nicely.
    print(f'Eigenvalues of within-transformed x: {lambdas.round(decimals=0)}')
    print(V) 

# Create transformation matrix
def demeaning_matrix(T):
    Q_T = np.eye(T) - np.tile(1/T, (T, T))
    return Q_T

# Create transformation matrix
def fd_matrix(T):
    D_T = np.eye(T) - np.eye(T, k=-1)
    D_T = D_T[1:]
    return D_T

# define the wald test
def wald(b, cov, r):
    # define the restriction matrix
    R = np.ones((1, b.shape[0]))

    beta_sum = R @ b

    # Calculate the Wald statistic
    W = (R @ b - r).T @ la.inv(R @ cov @ R.T) @ (R @ b - r)
    
    # Calculate the p-value
    p_val = 1 - chi2.cdf(W.item(), R.shape[0])

    return beta_sum, W, p_val

def wald_joint(b, cov, R, r):
    """
    Performs a joint Wald test for multiple linear restrictions.

    Args:
        b (np.ndarray): Estimated coefficients (k x 1).
        cov (np.ndarray): Covariance matrix of coefficients (k x k).
        R (np.ndarray): Restriction matrix (m x k), where m is the number of restrictions.
        r (np.ndarray): Restriction values (m x 1).

    Returns:
        tuple: (Wald statistic, p-value)
    """
    # Ensure R and r are numpy arrays
    R = np.array(R)
    r = np.array(r).reshape(-1, 1)

    # Compute the Wald statistic
    W = (R @ b - r).T @ la.inv(R @ cov @ R.T) @ (R @ b - r)
    W = W.item()  # Convert from 1x1 array to scalar

    # Degrees of freedom is the number of restrictions
    df = R.shape[0]

    # Compute the p-value
    p_val = 1 - chi2.cdf(W, df)

    return W, p_val


# function to calculate the serial correlation
def serial_corr(y, x, T):
    # calculate the residuals
    b_hat = est_ols(y, x)
    e = y - x@b_hat
    
    # create a lag transformation matrix
    L_T = np.eye(T, k=-1)
    L_T = L_T[1:]

    # lag residuals
    e_l = perm(L_T, e)

    # create a transformation matrix that removes the first observation of each individual
    I_T = np.eye(T, k=0)
    I_T = I_T[1:]
    
    # remove first observation of each individual
    e = perm(I_T, e)
    
    # calculate the serial correlation
    return estimate(e, e_l,T=T-1,robust=True)


def diebold_mariano_test(actuals, forecast1, forecast2, loss_function='mse', h=1):
    """
    Performs the Diebold-Mariano test for predictive accuracy.

    Args:
        actuals (array): Actual observed values.
        forecast1 (array): First forecast to compare.
        forecast2 (array): Second forecast to compare.
        loss_function (str): The loss function to use ('mse' or 'mae').
        h (int): Forecast horizon, default is 1 (single-step forecast).

    Returns:
        DM statistic and p-value.
    """
    # Compute forecast errors
    error1 = actuals - forecast1
    error2 = actuals - forecast2

    # Choose the loss function
    if loss_function == 'mse':
        diff = error1**2 - error2**2
    elif loss_function == 'mae':
        diff = np.abs(error1) - np.abs(error2)
    else:
        raise ValueError("Unsupported loss function. Use 'mse' or 'mae'.")

    # Compute mean and variance of the loss differential
    mean_diff = np.mean(diff)
    n = len(diff)
    variance_diff = np.var(diff, ddof=1)

    # Correct variance for autocorrelation if h > 1
    if h > 1:
        autocov = np.correlate(diff, diff, mode='full') / n
        variance_diff += 2 * sum(autocov[n-1:n-1+h])

    # Compute DM statistic
    dm_stat = mean_diff / np.sqrt(variance_diff / n)

    # Compute p-value
    dof = n - 1  # Degrees of freedom
    p_value = 2 * (1 - t.cdf(abs(dm_stat), df=dof))  # Two-tailed test

    return dm_stat, p_value


def latex_table(models, metrics, p_values=False):
    """
    Generates a LaTeX table in wide format comparing models based on given metrics.

    Args:
    models (list of str): Names of the models (e.g., ['Naive', 'SARIMA', 'SARIMAX', 'LSTM']).
    metrics (dict): Dictionary with metric names as keys (e.g., 'RMSE', 'MAE') and 
                    lists of metric values as values. For p-values, use tuples where the second
                    value is the p-value (e.g., {'RMSE': [123, (101, 0.05), (95, 0.03), (85, 0.01)], ...}).

    Returns:
    str: A LaTeX table string in wide format.
    """
    # Start the table
    table = "\\begin{tabular}{l" + "c" * len(models) + "}\n\hline\hline \\\\ [-1.8ex]\n"

    # Add the header
    table += " & " + " & ".join(models) + " \\\\ \n \hline \n"

    # Add rows for each metric
    for metric, values in metrics.items():
        # Main metric values row
        row = f"{metric} & "
        row_values = []
        for value in values:
            if isinstance(value, tuple):  # If it's a tuple, extract the main value
                main_value, _ = value
                row_values.append(f"{main_value:.5f}")
            else:
                row_values.append(f"{value:.5f}")
        row += " & ".join(row_values) + " \\\\ \n"
        table += row
        if p_values:    
            # P-values row
            p_row = "" if metric.strip() == "" else " & "
            p_values = []
            for value in values:
                if isinstance(value, tuple):  # If it's a tuple, extract the p-value
                    _, p_value = value
                    p_values.append(f"({p_value:.5f})")
                else:
                    p_values.append("-")
            p_row += " & ".join(p_values) + " \\\\ \n"
            table += p_row

    # Close the table
    table += "\hline\hline\n\end{tabular}"

    return table

def latex_table_grouped(models, metrics, p_values=False):
    """
    Generates a LaTeX table in wide format with grouped metrics using multirow and rotatebox.

    Args:
    models (list of str): Names of the models (e.g., ['Naive', 'SARIMA', 'SARIMAX', 'LSTM']).
    metrics (dict): Dictionary with keys in the format '*GROUP*Label' and values as lists of metric values.
                    Values can be either floats or tuples (value, p-value).
    p_values (bool): Whether to include a second row with p-values.

    Returns:
    str: A LaTeX table string.
    """
    from collections import defaultdict

    # Organize metrics by group
    grouped = defaultdict(list)
    for key, values in metrics.items():
        try:
            _, group, label = key.split("*")
        except ValueError:
            raise ValueError(f"Metric key '{key}' must be in the format '*GROUP*Label'")
        grouped[group].append((label, values))

    table = "\\begin{tabular}{cl" + "c" * len(models) + "}\n\\hline\\hline \\\\ [-1.8ex]\n"
    table += " &  & " + " & ".join(models) + " \\\\ \n \\hline \n"

    for group, entries in grouped.items():
        n_rows = len(entries)
        for i, (label, values) in enumerate(entries):
            row = ""
            if i == 0:
                row += f"\\multirow[c]{{{n_rows}}}{{*}}{{\\rotatebox{{90}}{{{group}}}}}"
            else:
                row += " "
            row += f"& {label} & "
            value_strs = []
            for value in values:
                if isinstance(value, tuple):
                    main_value, _ = value
                    value_strs.append(f"{main_value:.5f}")
                else:
                    value_strs.append(f"{value:.5f}")
            row += " & ".join(value_strs) + " \\\\ \n"
            table += row

            if p_values:
                p_row = " &  & "
                p_strs = []
                for value in values:
                    if isinstance(value, tuple):
                        _, p = value
                        p_strs.append(f"({p:.5f})")
                    else:
                        p_strs.append("-")
                p_row += " & ".join(p_strs) + " \\\\ \n"
                table += p_row

        table += "\\hline"

    table += "\\hline\n\\end{tabular}"

    return table


def latex_table_nested(models, metrics, p_values=False):
    """
    Generates a LaTeX table with two vertical groupings (rotated), using \multirow, \rotatebox, and \cline.

    Args:
    models (list): List of model names.
    metrics (dict): Keys must be '*Outer**Inner*Label'. Values are lists of values or tuples.
    p_values (bool): Whether to add p-value rows under the estimates.

    Returns:
    str: LaTeX table as a string.
    """
    from collections import defaultdict

    # Organize metrics into nested structure: {Outer: {Inner: [(Label, Values), ...]}}
    nested = defaultdict(lambda: defaultdict(list))
    for key, values in metrics.items():
        try:
            _, outer, inner, label = key.split("*")
        except ValueError:
            raise ValueError(f"Metric key '{key}' must be in the format '*Outer*Inner*Label'")
        nested[outer][inner].append((label, values))

    col_count = len(models) + 3  # Two grouping columns + label + model columns

    table = f"\\begin{{tabular}}{{ccl{'c' * len(models)}}}\n\\hline\\hline \\\\ [-1.8ex]\n"
    table += " &  &  & " + " & ".join(models) + " \\\\ \n\\hline \n"

    for outer_group, inner_groups in nested.items():
        outer_row_count = sum(len(items) for items in inner_groups.values())
        outer_started = False
        for j, (inner_group, rows) in enumerate(inner_groups.items()):
            inner_row_count = len(rows)
            idx = 0
            for i, (label, values) in enumerate(rows):
                row = ""
                if not outer_started:
                    row += f"\\multirow[c]{{{outer_row_count}}}{{*}}{{\\rotatebox{{90}}{{{outer_group}}}}}"
                    outer_started = True
                else:
                    row += " "
                if i == 0:
                    row += f" & \\multirow[c]{{{inner_row_count}}}{{*}}{{\\rotatebox{{90}}{{{inner_group}}}}}"
                else:
                    row += " & "
                row += f" & {label} & "
                formatted_vals = []
                for val in values:
                    if isinstance(val, tuple):
                        formatted_vals.append(f"{val[0]:.5f}")
                    elif isinstance(val, (int, float)):
                        formatted_vals.append(f"{val:.5f}")
                    else:
                        formatted_vals.append("-")
                row += " & ".join(formatted_vals) + " \\\\ \n"
                table += row

                if p_values:
                    p_row = " &  &  & "
                    p_strs = []
                    for val in values:
                        if isinstance(val, tuple):
                            p_strs.append(f"({val[1]:.5f})")
                        else:
                            p_strs.append("-")
                    p_row += " & ".join(p_strs) + " \\\\ \n"
                    table += p_row
                idx += 1
                # print(f'{i}: {inner_group}, {inner_row_count-1}, {j}: {outer_group}, {outer_row_count-1}')
                if i == inner_row_count-1 and j != outer_row_count:          # Not last inner block
                    table += f"\\cline{{2-{col_count}}}\n"
                    # print(row)
        table += "\\hline\n"
    table += "\\hline\n\\end{tabular}"
    return table

def cumret(series, upper, lower):
    """
    Cumulative return from months -upper … -lower (inclusive),
    skipping the most recent `lower-1` months.
    """
    window = upper - lower + 1
    return (
        (1 + series)
        .shift(lower)                    # skip `lower-1` most-recent months
        .rolling(window, min_periods=window)
        .apply(np.prod, raw=True) - 1
    )

def student_t_pdf(x, nu, mu, sigma):
    num = math.gamma((nu + 1) / 2)
    den = math.sqrt(nu * math.pi) * math.gamma(nu / 2) * sigma
    return num / den * (1 + ((x - mu) / sigma) ** 2 / nu) ** (-(nu + 1) / 2)