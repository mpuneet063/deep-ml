import numpy as np

def lu_decomposition(A: list) -> tuple:
	"""
	Perform LU decomposition on a square matrix using Doolittle's method.
	
	Args:
		A: Square matrix as a list of lists
	
	Returns:
		tuple: (L, U) where L is lower triangular with 1s on diagonal,
		       U is upper triangular, and A = L @ U
	"""
	# Your code here
	n = len(A)
	L , U = [[0]*n for _ in range(n)], [[0]*n for _ in range(n)]

	for i in range(n):
		L[i][i] = 1

		# calculate row i of U
		for j in range(i, n):
			U[i][j] = A[i][j] - sum(L[i][k]*U[k][j] for k in range(i))

		# calculate column i of L
		for j in range(i+1, n):
			L[j][i] = (A[j][i] - sum(L[j][k]*U[k][i] for k in range(i))) / U[i][i]
	L, U = np.array(L), np.array(U)
	return (L, U)