import sympy as sp
import numpy as np

def rref(matrix):
	rref_matrix, _ = sp.Matrix(matrix).rref()
	return np.array(rref_matrix).astype(float)