import numpy as np

def cramers_rule(A, b):
    # Your code here
    A, b = np.array(A), np.array(b)
    detA = np.linalg.det(A)
    if np.isclose(detA, 0):
        return -1
    x = []
    
    for i in range(len(A)):
        Ai = A.copy()
        Ai[:, i] = b
        detAi = np.linalg.det(Ai)
        x.append(detAi / detA)

    return x