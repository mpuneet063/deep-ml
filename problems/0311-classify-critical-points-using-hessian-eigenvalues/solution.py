import numpy as np

def classify_critical_point(hessian: np.ndarray, tol: float = 1e-10):
    evals = np.linalg.eigvals(hessian)

    if np.all(evals > 0):
        return -1
    elif np.all(evals < 0):
        return 1
    elif np.any(np.abs(evals) < tol):
        return None
    else:
        return 0