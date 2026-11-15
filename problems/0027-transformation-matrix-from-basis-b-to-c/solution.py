import numpy as np

def transform_basis(B: list[list[int]], C: list[list[int]]) -> list[list[float]]:
    B, C = np.array(B), np.array(C)
    C_inv = np.linalg.inv(C)
    P = np.dot(C_inv, B)
    return P