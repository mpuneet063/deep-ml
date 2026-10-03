import numpy as np

def calculate_correlation_matrix(X, Y=None):
    X = np.array(X, dtype=float)
    Y = X if Y is None else np.array(Y, dtype=float)
    Xc = X - X.mean(axis=0)
    Yc = Y - Y.mean(axis=0)
    Dx = np.sum(Xc**2, axis=0)
    Dy = np.sum(Yc**2, axis=0)
    return (Xc.T @ Yc) / np.sqrt(Dx[:, None] * Dy[None, :])