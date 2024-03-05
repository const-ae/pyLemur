import numpy as np

def row_groups(matrix):
    """
    Returns the row groups of a matrix.

    Parameters
    ----------
    matrix : np.ndarray
        The matrix to find the row groups of.

    Returns
    -------
    np.ndarray
        A vector of integers representing the row groups of the matrix.
    """
    _, inv = np.unique(matrix, axis = 0, return_inverse=True)
    return inv
    