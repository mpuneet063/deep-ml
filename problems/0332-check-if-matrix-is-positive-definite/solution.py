import numpy as np

def check_positive_definite(matrix: list) -> dict:
    """
    Check if a matrix is positive definite and compute its eigenvalues.
    
    Args:
        matrix: A 2D list representing a square matrix
        
    Returns:
        dict with 'is_positive_definite' (bool) and 'eigenvalues' (list of floats sorted ascending)
    """
    # Your code here
    n = len(matrix)

    evals = np.real(np.linalg.eigvals(matrix).round(4))

    es = [e for e in evals if e > 1e-10]

    return {
        'is_positive_definite': len(es) == n,
        'eigenvalues': sorted(list(evals))
    }