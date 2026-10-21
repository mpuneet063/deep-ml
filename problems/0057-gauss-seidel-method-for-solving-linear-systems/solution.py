import numpy as np

def gauss_seidel(A, b, n, x_ini=None):
	A = np.asarray(A, dtype=float)
	b = np.asarray(b, dtype=float)
	m = len(b)

	x = np.zeros(m) if x_ini is None else np.array(x_ini, dtype=float)

	tol = 1e-10
	for _ in range(n):
		x_old = x.copy()

		for i in range(m):
			x[i] = (
				b[i]
				- np.dot(A[i, :i], x[:i])
				- np.dot(A[i, i+1:], x_old[i+1:])
			) / A[i, i]

		if np.linalg.norm(x - x_old, ord=np.inf) < tol:
			return x

	return x