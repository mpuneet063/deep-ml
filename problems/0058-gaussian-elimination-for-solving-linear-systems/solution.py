import numpy as np

def gaussian_elimination(A, b):
	"""
	Solves the system Ax = b using Gaussian Elimination with partial pivoting.
    
	:param A: Coefficient matrix
	:param b: Right-hand side vector
	:return: Solution vector x
	"""
	n = len(A)
	aug = np.column_stack((A,b)).astype(float)

	for i in range(n):
		pivot_row = i
		max_val = abs(aug[i][i])
		for k in range(i+1,n):
			if abs(aug[k][i]) > max_val:
				max_val = aug[k][i]
				pivot_row = k

		if pivot_row != i:
			aug[i], aug[pivot_row] = aug[pivot_row].copy(), aug[i].copy()

		if abs(aug[i][i]) < 1e-10:
			return "No unique solution"

		pivot = aug[i][i]
		aug[i] = aug[i] / pivot

		for j in range(i+1, n):
			factor = aug[j][i]
			aug[j] -= factor * aug[i]

	x = np.zeros_like(b)
	for i in range(n-1,-1,-1):
		x[i] = aug[i][-1]
		for j in range(i+1,n):
			x[i] -= aug[i][j]*x[j]

	return x