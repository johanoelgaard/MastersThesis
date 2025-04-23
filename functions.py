import numpy as np
import pandas as pd
import re
from decimal import Decimal
from numpy import linalg as la
from tabulate import tabulate
from scipy import stats
from scipy.stats import chi2

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

def export_to_latex(
    results_list: list,
    col_headers: list,
    var_names_list: list,
    filename: str = None,
    label_x: list = None,
    **kwargs
) -> None:
    """
    Exports regression results to a LaTeX table and saves it as a .txt file.

    Args:
        results_list (list): List of result dictionaries from the estimate function.
        col_headers (list): List of column headers (e.g., ['OLS', 'FE', 'FD']).
        var_names_list (list): List of lists of variable names for each result.
        filename (str): The filename to save the LaTeX table (should end with .txt).
        label_x (list, optional): List of variable names to include in the table in desired order.
    """
    num_models = len(results_list)

    # Generate the list of all variables to include in the table
    if label_x is None:
        label_x = var_names_list[0]  # default to variables from the first model

    # Start constructing the LaTeX table
    lines = []

    lines.append("\\begin{tabular}{" + "l" + "c" * num_models + "}")
    lines.append("\\hline\\hline\\\\[-1.8ex]")
    header_row = [""] + col_headers
    lines.append(" & ".join(header_row) + " \\\\")
    lines.append("\\hline")

    # For each variable in label_x
    for var_name in label_x:
        estimate_row = [var_name]
        se_row = ['']
        for result, var_names in zip(results_list, var_names_list):
            if var_name in var_names:
                idx = var_names.index(var_name)
                b_hat = result.get('b_hat')[idx][0]
                se = result.get('se')[idx][0]
                wald = result.get('wald')[idx][0]

                # Calculate p-value from wald
                p_value = 1 - chi2.cdf(wald, result.get('b_hat')[idx].shape[0])

                # Determine the number of stars based on p-value
                if p_value < 0.01:
                    stars = '***'
                elif p_value < 0.05:
                    stars = '**'
                elif p_value < 0.10:
                    stars = '*'
                else:
                    stars = ''

                # Append coefficient with stars
                estimate_row.append(f"{b_hat:.4f}{stars}")
                # Append standard error in parentheses
                se_row.append(f"({se:.4f})")
            else:
                # Variable not in this model
                estimate_row.append("")
                se_row.append("")

        # Write estimate row
        lines.append(" & ".join(estimate_row) + " \\\\[-1ex]")
        # Write standard error row
        lines.append(" & \\footnotesize".join(se_row) + " \\\\")

    # Additional statistics
    lines.append("\\hline")
    statistics = ["R-squared"] + [f"{result.get('R2').item():.3f}" for result in results_list]
    lines.append(" & ".join(statistics) + " \\\\")
    lines.append("\\hline\\hline")

    # End of table
    lines.append("\\end{tabular}")

    # Save to file if filename is provided else return the table as a string
    if filename:
        with open(filename, 'w') as f:
            f.write("\n".join(lines))
        print(f"LaTeX table saved to {filename}")
    else:
        return "\n".join(lines)

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