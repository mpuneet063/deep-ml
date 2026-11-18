import numpy as np

def cholesky_decomposition(A):
    """
    Perform Cholesky decomposition on a symmetric positive-definite matrix.
    
    Args:
        A: A symmetric positive-definite matrix (2D list or numpy array)
    
    Returns:
        L: Lower triangular matrix such that A = L @ L.T as a 2D list,
           or -1 if decomposition is not possible
    """
    # Your code here
    # check for possiblity of cholesky_decomposition
    A = np.array(A)
    n,m = A.shape
    if n != m:
        return -1
    if A.all() != A.T.all():
        return -1
    eigv = np.linalg.eigvals(A)
    for e in eigv:
        if e < 0:
            return -1
    return np.linalg.cholesky(A)