import sympy as sp
import numpy as np

def matrix_image(A):
	# Write your code here
	_, pivot_cols = sp.Matrix(A).rref()
	return np.asarray(A)[:, list(pivot_cols)]