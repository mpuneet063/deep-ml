import numpy as np

def pairwise_cosine_similarity(X):
    X = np.asarray(X)
    dot = X @ X.T
    norms = np.linalg.norm(X, axis=1)
    return dot / (np.outer(norms, norms) + 1e-6)