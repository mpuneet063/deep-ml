import numpy as np

def orthonormal_basis(vectors: list[list[float]], tol: float = 1e-10) -> list[np.ndarray]:
    # Your code here
    q = []
    if vectors[0] != [0,0]:
        q.append(vectors[0] / np.linalg.norm(vectors[0]))
    for i in range(1, len(vectors)):
        vj = vectors[i] - sum((q[k] @ vectors[i]) * q[k] for k in range(len(q)))
        if np.linalg.norm(vj) != 0:
            q.append(vj / (np.linalg.norm(vj)))

    return q